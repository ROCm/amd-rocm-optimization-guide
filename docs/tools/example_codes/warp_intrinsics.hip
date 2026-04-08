// MIT License
//
// Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include <hip/hip_runtime.h>

#include <limits>
#include <cmath>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#define HIP_CHECK(expression)                      \
    {                                              \
        const hipError_t status = expression;      \
        if (status != hipSuccess)                  \
        {                                          \
            std::cerr << "HIP error "              \
                      << status << ": "            \
                      << hipGetErrorString(status) \
                      << " at " << __FILE__ << ":" \
                      << __LINE__ << std::endl;    \
        }                                          \
    }

// DPP broadcast instructions (row_bcast:15 and row_bcast:31) are available on
// GFX8 and GFX9 (CDNA) only. On GFX10+ (RDNA, wave32) the cross-row step
// falls back to ds_swizzle and the warp size is 32.
#if defined(__GFX8__) || defined(__GFX9__)
#define HAS_DPP_BROADCAST
#endif

constexpr int block_size  = 256; // multiple warps per block; warp size queried at runtime
constexpr int input_size  = 1 << 24; // 16 M elements; matches reduction.hip for timer comparison
constexpr int timing_runs = 100;

// The lane operation kernels (readlane, rotate, swizzle, ballot) use
// a smaller input so that their CPU reference simulations remain fast.
constexpr int lane_input_size = block_size * 4;

// [Sphinx shfl down reduce start]
template<int WarpSize>
__device__ float shfl_down_reduce(float val)
{
    // Halve the active offset each step until all lanes have contributed.
    // WarpSize is a compile-time constant, allowing the compiler to fully
    // unroll this loop.
    for (int offset = WarpSize / 2; offset > 0; offset >>= 1)
    {
        val += __shfl_down(val, offset);
    }
    return val;
}

template<int WarpSize>
__global__ __launch_bounds__(block_size)
void reduce_shfl_down(const float *input, float *output, int n)
{
    int gid    = blockIdx.x * blockDim.x + threadIdx.x;
    int lane   = threadIdx.x % WarpSize;
    int warpid = threadIdx.x / WarpSize;

    float val = (gid < n) ? input[gid] : 0.0f;
    val = shfl_down_reduce<WarpSize>(val);

    // Collect one partial result per warp into shared memory, then reduce
    // those with the first warp.
    extern __shared__ float sdata[];
    int num_warps = blockDim.x / WarpSize;
    if (lane == 0)
    {
        sdata[warpid] = val;
    }
    __syncthreads();

    if (warpid == 0)
    {
        val = (lane < num_warps) ? sdata[lane] : 0.0f;
        val = shfl_down_reduce<WarpSize>(val);
    }

    if (threadIdx.x == 0)
    {
        output[blockIdx.x] = val;
    }
}
// [Sphinx shfl down reduce end]

// [Sphinx dpp reduce start]
// DPP control words for butterfly reduction steps.
// quad_perm and row_ror steps work on both CDNA and RDNA.
// row_bcast steps are CDNA only; ds_swizzle mask 0x1e0 is the RDNA fallback.
constexpr int dpp_quad_perm_1032 = 0xb1;
constexpr int dpp_quad_perm_2301 = 0x4e;
constexpr int dpp_row_ror4       = 0x124;
constexpr int dpp_row_ror8       = 0x128;
#ifdef HAS_DPP_BROADCAST
constexpr int dpp_row_bcast15    = 0x142;
constexpr int dpp_row_bcast31    = 0x143;
#else
constexpr int swizzle_bcast15    = 0x1e0;
#endif
template<int WarpSize>
__device__ float dpp_warp_reduce(float val)
{
    // Reinterpret float bits as int for DPP and swizzle intrinsics.
    auto as_int   = [](float f) { return __builtin_bit_cast(int, f); };
    auto as_float = [](int i)   { return __builtin_bit_cast(float, i); };

    // Steps 1-2: combine adjacent pairs within groups of 4.
    val += as_float(__builtin_amdgcn_mov_dpp(as_int(val), dpp_quad_perm_1032, 0xf, 0xf, false));
    val += as_float(__builtin_amdgcn_mov_dpp(as_int(val), dpp_quad_perm_2301, 0xf, 0xf, false));
    // Steps 3-4: combine groups of 4 into a row of 16.
    val += as_float(__builtin_amdgcn_mov_dpp(as_int(val), dpp_row_ror4, 0xf, 0xf, false));
    val += as_float(__builtin_amdgcn_mov_dpp(as_int(val), dpp_row_ror8, 0xf, 0xf, false));

#ifdef HAS_DPP_BROADCAST
    // Steps 5-6 (CDNA): combine two rows of 16 into the full wave64 result.
    val += as_float(__builtin_amdgcn_mov_dpp(as_int(val), dpp_row_bcast15, 0xa, 0xf, false));
    val += as_float(__builtin_amdgcn_mov_dpp(as_int(val), dpp_row_bcast31, 0xc, 0xf, false));
#else
    // Step 5 (RDNA): ds_swizzle combines the two rows of the wave32 warp.
    val += as_float(__builtin_amdgcn_ds_swizzle(as_int(val), swizzle_bcast15));
#endif

    return val;
}

template<int WarpSize>
__global__ __launch_bounds__(block_size)
void reduce_dpp(const float *input, float *output, int n)
{
    int gid    = blockIdx.x * blockDim.x + threadIdx.x;
    int lane   = threadIdx.x % WarpSize;
    int warpid = threadIdx.x / WarpSize;

    float val = (gid < n) ? input[gid] : 0.0f;
    val = dpp_warp_reduce<WarpSize>(val);

    extern __shared__ float sdata[];
    int num_warps = blockDim.x / WarpSize;
    if (lane == WarpSize - 1)
    {
        sdata[warpid] = val;
    }
    __syncthreads();

    if (warpid == 0)
    {
        val = (lane < num_warps) ? sdata[lane] : 0.0f;
        val = dpp_warp_reduce<WarpSize>(val);
    }

    if (threadIdx.x == WarpSize - 1)
    {
        output[blockIdx.x] = val;
    }
}
// [Sphinx dpp reduce end]

// [Sphinx wave reduce start]
template<int WarpSize>
__global__ __launch_bounds__(block_size)
void reduce_wave(const unsigned int *input, unsigned int *output, int n)
{
    int gid    = blockIdx.x * blockDim.x + threadIdx.x;
    int lane   = threadIdx.x % WarpSize;
    int warpid = threadIdx.x / WarpSize;

    unsigned int val = (gid < n) ? input[gid] : 0u;

    // Reduces val across all active lanes in a single instruction.
    // The result is broadcast to every lane.
    unsigned int sum = __builtin_amdgcn_wave_reduce_add_u32(val, 0);

    extern __shared__ unsigned int sdataui[];
    int num_warps = blockDim.x / WarpSize;
    if (lane == 0)
    {
        sdataui[warpid] = sum;
    }
    __syncthreads();

    if (warpid == 0)
    {
        sum = (lane < num_warps) ? sdataui[lane] : 0u;
        sum = __builtin_amdgcn_wave_reduce_add_u32(sum, 0);
    }

    if (threadIdx.x == 0)
    {
        output[blockIdx.x] = sum;
    }
}
// [Sphinx wave reduce end]

// [Sphinx readlane start]
__global__ void broadcast_readlane(const int *input, int *output, int n)
{
    int gid  = blockIdx.x * blockDim.x + threadIdx.x;
    int lane = threadIdx.x % warpSize;
    int val  = (gid < n) ? input[gid] : 0;

    // Find the warp maximum via __shfl_down reduction.
    int max_val = val;
    for (int offset = warpSize / 2; offset > 0; offset >>= 1)
    {
        max_val = max(max_val, __shfl_down(max_val, offset));
    }

    // Broadcast the maximum from lane 0 to all lanes.
    // The lane index must be uniform; use readfirstlane to promote a
    // runtime-derived index to a uniform value.
    int broadcast = __builtin_amdgcn_readlane(max_val, 0);

    if (gid < n)
    {
        output[gid] = broadcast;
    }

    (void)lane;
}
// [Sphinx readlane end]

// [Sphinx warp rotate start]
// wave_ror:1 (CDNA): rotates the entire warp right by one lane.
// row_ror:1 (RDNA): rotates within each row of 16; ds_bpermute fixes wrap-around.
#ifdef HAS_DPP_BROADCAST
constexpr int dpp_wave_ror1 = 0x134;
#else
constexpr int dpp_row_ror1  = 0x121;
#endif
__device__ int warp_rotate_right(int val)
{
#ifdef HAS_DPP_BROADCAST
    // CDNA: single wave_ror:1 rotates all lanes atomically.
    return __builtin_amdgcn_mov_dpp(val, dpp_wave_ror1, 0xf, 0xf, false);
#else
    // RDNA: row_ror:1 rotates within each row of 16.
    int rotated = __builtin_amdgcn_mov_dpp(val, dpp_row_ror1, 0xf, 0xf, false);

    // ds_bpermute (byte addressing: lane k -> addr k*4) fixes up the
    // wrap-around value at lane 0 of each row.
    int lane    = threadIdx.x % warpSize;
    int addr    = (lane == 0) ? 124 : (lane == 16) ? 60 : lane * 4;
    int wrapped = __builtin_amdgcn_ds_bpermute(addr, val);

    return (lane == 0 || lane == 16) ? wrapped : rotated;
#endif
}

__global__ void rotate_warp(const int *input, int *output, int n)
{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int val = (gid < n) ? input[gid] : 0;

    // Rotate right warpSize times, XORing each step with the step index.
    for (int i = 0; i < warpSize; ++i)
    {
        val = warp_rotate_right(val);
        val ^= i;
    }

    if (gid < n)
    {
        output[gid] = val;
    }
}
// [Sphinx warp rotate end]

// [Sphinx warp rotate shfl start]
__device__ int warp_rotate_right_shfl(int val)
{
    int lane = threadIdx.x % warpSize;
    // Rotate right: each lane reads from (lane - 1 + warpSize) % warpSize.
    return __shfl(val, (lane - 1 + warpSize) % warpSize);
}

__global__ void rotate_warp_shfl(const int *input, int *output, int n)
{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int val = (gid < n) ? input[gid] : 0;

    // Rotate right warpSize times, XORing each step with the step index.
    // Produces the same result as rotate_warp.
    for (int i = 0; i < warpSize; ++i)
    {
        val = warp_rotate_right_shfl(val);
        val ^= i;
    }

    if (gid < n)
    {
        output[gid] = val;
    }
}
// [Sphinx warp rotate shfl end]

// [Sphinx ds swizzle start]
// ds_swizzle mask layout: bits [14:10]=and_mask, [9:5]=or_mask, [4:0]=xor_mask.
// Source lane = (lane & and_mask) | or_mask ^ xor_mask.
// 0x101F: and_mask=0x1F, xor_mask=0x04 — swaps neighboring groups of 4.
constexpr int swizzle_swap_groups4 = 0x101F;
__global__ void swizzle_swap_groups(const int *input, int *output, int n)
{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int val = (gid < n) ? input[gid] : 0;

    int swapped = __builtin_amdgcn_ds_swizzle(val, swizzle_swap_groups4);

    if (gid < n)
    {
        output[gid] = swapped;
    }
}
// [Sphinx ds swizzle end]

// [Sphinx ballot mbcnt start]
__global__ void compaction_index(const int *input, int *output, int n)
{
    int gid    = blockIdx.x * blockDim.x + threadIdx.x;
    int lane   = threadIdx.x % warpSize;
    int warpid = threadIdx.x / warpSize;
    int val    = (gid < n) ? input[gid] : 0;

    bool keep  = (gid < n) && (val > 0);
    int  index = 0;

#ifdef HAS_DPP_BROADCAST
    // wave64 (CDNA): chain mbcnt_lo and mbcnt_hi over the 64-bit mask.
    unsigned long long mask = __builtin_amdgcn_ballot_w64(keep);
    unsigned int lo         = static_cast<unsigned int>(mask);
    unsigned int hi         = static_cast<unsigned int>(mask >> 32);
    index                   = __builtin_amdgcn_mbcnt_lo(lo, 0);
    index                   = __builtin_amdgcn_mbcnt_hi(hi, index);
#else
    // wave32 (RDNA): mbcnt_lo over the 32-bit mask suffices.
    unsigned int mask = __builtin_amdgcn_ballot_w32(keep);
    index             = __builtin_amdgcn_mbcnt_lo(mask, 0);
#endif

    // Compute per-warp base offsets for contiguous compacted output.
    extern __shared__ int sdatai[];
    int num_warps  = blockDim.x / warpSize;
    int warp_count = __builtin_amdgcn_readlane(index + (keep ? 1 : 0), warpSize - 1);
    if (lane == 0)
    {
        sdatai[warpid] = warp_count;
    }
    __syncthreads();

    int warp_base = 0;
    for (int w = 0; w < warpid; ++w)
    {
        warp_base += sdatai[w];
    }

    if (keep)
    {
        output[blockIdx.x * blockDim.x + warp_base + index] = val;
    }

    (void)num_warps;
}
// [Sphinx ballot mbcnt end]


template<typename T>
std::vector<T> cpu_reference(const std::vector<T> &input, int logical_block_size)
{
    int n          = static_cast<int>(input.size());
    int num_chunks = (n + logical_block_size - 1) / logical_block_size;

    std::vector<T> partials(num_chunks, T{});
    for (int b = 0; b < num_chunks; ++b)
    {
        int start = b * logical_block_size;
        int end   = std::min(start + logical_block_size, n);
        for (int i = start; i < end; ++i)
        {
            partials[b] += input[i];
        }
    }
    return partials;
}

#ifdef ENABLE_TIMERS
void time_kernel(const std::string &label, const std::function<void()> &launch)
{
    hipEvent_t start, stop;
    HIP_CHECK(hipEventCreate(&start));
    HIP_CHECK(hipEventCreate(&stop));
    float total_ms = 0.0f;
    for (int run = 0; run < timing_runs; ++run)
    {
        HIP_CHECK(hipEventRecord(start));
        launch();
        HIP_CHECK(hipGetLastError());
        HIP_CHECK(hipEventRecord(stop));
        HIP_CHECK(hipEventSynchronize(stop));
        float ms = 0.0f;
        HIP_CHECK(hipEventElapsedTime(&ms, start, stop));
        total_ms += ms;
    }
    std::cout << "    [timer] " << label << ": avg " << total_ms / timing_runs
              << " ms over " << timing_runs << " runs" << std::endl;
    HIP_CHECK(hipEventDestroy(start));
    HIP_CHECK(hipEventDestroy(stop));
}
#endif

int main()
{
    int deviceid     = 0;
    int warpsizehost = 0;
    HIP_CHECK(hipDeviceGetAttribute(&warpsizehost, hipDeviceAttributeWarpSize, deviceid));
    std::cout << "Warp size: " << warpsizehost << std::endl;

    int numwarps      = block_size / warpsizehost;
    int smem_float    = numwarps * sizeof(float);
    int smem_int      = numwarps * sizeof(int);
    int nb            = (input_size + block_size - 1) / block_size;
    int lane_nb       = lane_input_size / block_size;

    // Reduction inputs: 16 M random floats, matching reduction.hip exactly.
    std::vector<float> h_float(input_size);
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (float &v : h_float)
    {
        v = dist(gen);
    }
    std::vector<float> ref_float = cpu_reference(h_float, block_size);

    // Unsigned int inputs for the wave_reduce kernel.
    std::vector<unsigned int> h_uint(input_size);
    std::uniform_int_distribution<unsigned int> udist(0u, 255u);
    for (unsigned int &v : h_uint)
    {
        v = udist(gen);
    }
    std::vector<unsigned int> ref_uint = cpu_reference(h_uint, block_size);

    // Lane operation inputs: small int array so CPU reference simulation is fast.
    std::vector<int> h_int(lane_input_size);
    for (int i = 0; i < lane_input_size; ++i)
    {
        h_int[i] = i - lane_input_size / 2; // mix of negative and positive for ballot test
    }

    float        *d_float = nullptr;
    float        *d_fout  = nullptr;
    unsigned int *d_uint  = nullptr;
    unsigned int *d_uout  = nullptr;
    int          *d_int   = nullptr;
    int          *d_iout  = nullptr;

    HIP_CHECK(hipMalloc(&d_float, sizeof(float)        * input_size));
    HIP_CHECK(hipMalloc(&d_fout,  sizeof(float)        * nb));
    HIP_CHECK(hipMalloc(&d_uint,  sizeof(unsigned int) * input_size));
    HIP_CHECK(hipMalloc(&d_uout,  sizeof(unsigned int) * nb));
    HIP_CHECK(hipMalloc(&d_int,   sizeof(int)          * lane_input_size));
    HIP_CHECK(hipMalloc(&d_iout,  sizeof(int)          * lane_input_size));

    HIP_CHECK(hipMemcpy(d_float, h_float.data(), sizeof(float)        * input_size, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_uint,  h_uint.data(),  sizeof(unsigned int) * input_size, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_int,   h_int.data(),   sizeof(int)          * lane_input_size, hipMemcpyHostToDevice));

    std::cout << "Running warp intrinsic kernels..." << std::endl;

    // shfl_down reduction
    if (warpsizehost == 64)
    {
        reduce_shfl_down<64><<<nb, block_size, smem_float>>>(d_float, d_fout, input_size);
    }
    else
    {
        reduce_shfl_down<32><<<nb, block_size, smem_float>>>(d_float, d_fout, input_size);
    }
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());
    {
        std::vector<float> h_out(nb);
        HIP_CHECK(hipMemcpy(h_out.data(), d_fout, sizeof(float) * nb, hipMemcpyDeviceToHost));
        bool passed = true;
        const float tolerance = 1e-3f;
        for (int i = 0; i < nb; ++i)
        {
            if (std::fabs(h_out[i] - ref_float[i]) > tolerance)
            {
                passed = false;
                break;
            }
        }
        std::cout << "  shfl_down reduction:  " << (passed ? "PASSED" : "FAILED") << std::endl;
    }
#ifdef ENABLE_TIMERS
    time_kernel("shfl_down reduction", [&]()
    {
        if (warpsizehost == 64)
        {
            reduce_shfl_down<64><<<nb, block_size, smem_float>>>(d_float, d_fout, input_size);
        }
        else
        {
            reduce_shfl_down<32><<<nb, block_size, smem_float>>>(d_float, d_fout, input_size);
        }
    });
#endif

    // DPP reduction
    if (warpsizehost == 64)
    {
        reduce_dpp<64><<<nb, block_size, smem_float>>>(d_float, d_fout, input_size);
    }
    else
    {
        reduce_dpp<32><<<nb, block_size, smem_float>>>(d_float, d_fout, input_size);
    }
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());
    {
        std::vector<float> h_out(nb);
        HIP_CHECK(hipMemcpy(h_out.data(), d_fout, sizeof(float) * nb, hipMemcpyDeviceToHost));
        bool passed = true;
        const float tolerance = 1e-3f;
        for (int i = 0; i < nb; ++i)
        {
            if (std::fabs(h_out[i] - ref_float[i]) > tolerance)
            {
                passed = false;
                break;
            }
        }
        std::cout << "  DPP reduction:        " << (passed ? "PASSED" : "FAILED") << std::endl;
    }
#ifdef ENABLE_TIMERS
    time_kernel("DPP reduction", [&]()
    {
        if (warpsizehost == 64)
        {
            reduce_dpp<64><<<nb, block_size, smem_float>>>(d_float, d_fout, input_size);
        }
        else
        {
            reduce_dpp<32><<<nb, block_size, smem_float>>>(d_float, d_fout, input_size);
        }
    });
#endif

    // wave_reduce (unsigned int)
    int smem_uint = numwarps * sizeof(unsigned int);
    if (warpsizehost == 64)
    {
        reduce_wave<64><<<nb, block_size, smem_uint>>>(d_uint, d_uout, input_size);
    }
    else
    {
        reduce_wave<32><<<nb, block_size, smem_uint>>>(d_uint, d_uout, input_size);
    }
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());
    {
        std::vector<unsigned int> h_out(nb);
        HIP_CHECK(hipMemcpy(h_out.data(), d_uout, sizeof(unsigned int) * nb, hipMemcpyDeviceToHost));
        bool passed = true;
        for (int i = 0; i < nb; ++i)
        {
            if (h_out[i] != ref_uint[i])
            {
                passed = false;
                break;
            }
        }
        std::cout << "  wave_reduce:          " << (passed ? "PASSED" : "FAILED") << std::endl;
    }
#ifdef ENABLE_TIMERS
    time_kernel("wave_reduce", [&]()
    {
        if (warpsizehost == 64)
        {
            reduce_wave<64><<<nb, block_size, smem_uint>>>(d_uint, d_uout, input_size);
        }
        else
        {
            reduce_wave<32><<<nb, block_size, smem_uint>>>(d_uint, d_uout, input_size);
        }
    });
#endif

    // readlane broadcast
    broadcast_readlane<<<lane_nb, block_size>>>(d_int, d_iout, lane_input_size);
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());
    {
        std::vector<int> h_out(lane_input_size);
        HIP_CHECK(hipMemcpy(h_out.data(), d_iout, sizeof(int) * lane_input_size,
                            hipMemcpyDeviceToHost));
        bool passed = true;
        for (int b = 0; b < lane_nb && passed; ++b)
        {
            for (int w = 0; w < numwarps && passed; ++w)
            {
                int warp_max = std::numeric_limits<int>::min();
                for (int i = 0; i < warpsizehost; ++i)
                {
                    warp_max = std::max(warp_max, h_int[b * block_size + w * warpsizehost + i]);
                }
                for (int i = 0; i < warpsizehost; ++i)
                {
                    if (h_out[b * block_size + w * warpsizehost + i] != warp_max)
                    {
                        passed = false;
                        break;
                    }
                }
            }
        }
        std::cout << "  readlane broadcast:   " << (passed ? "PASSED" : "FAILED") << std::endl;
    }

    // warp rotation
    rotate_warp<<<lane_nb, block_size>>>(d_int, d_iout, lane_input_size);
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());
    {
        std::vector<int> h_out(lane_input_size);
        HIP_CHECK(hipMemcpy(h_out.data(), d_iout, sizeof(int) * lane_input_size,
                            hipMemcpyDeviceToHost));
        bool passed = true;
        for (int b = 0; b < lane_nb && passed; ++b)
        {
            for (int w = 0; w < numwarps && passed; ++w)
            {
                // Simulate rotation+XOR on the CPU to match the GPU kernel.
                std::vector<int> lane_val(warpsizehost);
                for (int l = 0; l < warpsizehost; ++l)
                {
                    lane_val[l] = h_int[b * block_size + w * warpsizehost + l];
                }
                for (int i = 0; i < warpsizehost; ++i)
                {
                    std::vector<int> next(warpsizehost);
                    for (int l = 0; l < warpsizehost; ++l)
                    {
                        next[l] = lane_val[(l - 1 + warpsizehost) % warpsizehost] ^ i;
                    }
                    lane_val = next;
                }
                for (int l = 0; l < warpsizehost; ++l)
                {
                    if (h_out[b * block_size + w * warpsizehost + l] != lane_val[l])
                    {
                        passed = false;
                        break;
                    }
                }
            }
        }
        std::cout << "  warp rotation (DPP):  " << (passed ? "PASSED" : "FAILED") << std::endl;
    }

    // warp rotation (shuffle)
    rotate_warp_shfl<<<lane_nb, block_size>>>(d_int, d_iout, lane_input_size);
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());
    {
        std::vector<int> h_out(lane_input_size);
        HIP_CHECK(hipMemcpy(h_out.data(), d_iout, sizeof(int) * lane_input_size,
                            hipMemcpyDeviceToHost));
        bool passed = true;
        for (int b = 0; b < lane_nb && passed; ++b)
        {
            for (int w = 0; w < numwarps && passed; ++w)
            {
                // Same CPU simulation as rotate_warp — the two kernels produce
                // identical results and can be verified against the same reference.
                std::vector<int> lane_val(warpsizehost);
                for (int l = 0; l < warpsizehost; ++l)
                {
                    lane_val[l] = h_int[b * block_size + w * warpsizehost + l];
                }
                for (int i = 0; i < warpsizehost; ++i)
                {
                    std::vector<int> next(warpsizehost);
                    for (int l = 0; l < warpsizehost; ++l)
                    {
                        next[l] = lane_val[(l - 1 + warpsizehost) % warpsizehost] ^ i;
                    }
                    lane_val = next;
                }
                for (int l = 0; l < warpsizehost; ++l)
                {
                    if (h_out[b * block_size + w * warpsizehost + l] != lane_val[l])
                    {
                        passed = false;
                        break;
                    }
                }
            }
        }
        std::cout << "  warp rotation (shfl): " << (passed ? "PASSED" : "FAILED") << std::endl;
    }

    // ds_swizzle neighboring group swap
    swizzle_swap_groups<<<lane_nb, block_size>>>(d_int, d_iout, lane_input_size);
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());
    {
        std::vector<int> h_out(lane_input_size);
        HIP_CHECK(hipMemcpy(h_out.data(), d_iout, sizeof(int) * lane_input_size,
                            hipMemcpyDeviceToHost));
        bool passed = true;
        // xor_mask=0x04: each lane reads from lane_id ^ 4 within its warp.
        for (int b = 0; b < lane_nb && passed; ++b)
        {
            for (int w = 0; w < numwarps && passed; ++w)
            {
                for (int lane = 0; lane < warpsizehost; ++lane)
                {
                    int src      = lane ^ 4;
                    int expected = h_int[b * block_size + w * warpsizehost + src];
                    if (h_out[b * block_size + w * warpsizehost + lane] != expected)
                    {
                        passed = false;
                        break;
                    }
                }
            }
        }
        std::cout << "  ds_swizzle grp swap:  " << (passed ? "PASSED" : "FAILED") << std::endl;
    }

    // ballot + mbcnt compaction index
    HIP_CHECK(hipMemset(d_iout, 0, sizeof(int) * lane_input_size));
    compaction_index<<<lane_nb, block_size, smem_int>>>(d_int, d_iout, lane_input_size);
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());
    {
        std::vector<int> h_out(lane_input_size);
        HIP_CHECK(hipMemcpy(h_out.data(), d_iout, sizeof(int) * lane_input_size,
                            hipMemcpyDeviceToHost));
        bool passed = true;
        for (int b = 0; b < lane_nb && passed; ++b)
        {
            std::vector<int> expected;
            for (int i = 0; i < block_size; ++i)
            {
                int v = h_int[b * block_size + i];
                if (v > 0)
                {
                    expected.push_back(v);
                }
            }
            for (int i = 0; i < (int)expected.size(); ++i)
            {
                if (h_out[b * block_size + i] != expected[i])
                {
                    passed = false;
                    break;
                }
            }
        }
        std::cout << "  ballot+mbcnt compact: " << (passed ? "PASSED" : "FAILED") << std::endl;
    }

    std::cout << "Execution completed successfully." << std::endl;

    HIP_CHECK(hipFree(d_float));
    HIP_CHECK(hipFree(d_fout));
    HIP_CHECK(hipFree(d_uint));
    HIP_CHECK(hipFree(d_uout));
    HIP_CHECK(hipFree(d_int));
    HIP_CHECK(hipFree(d_iout));

    return EXIT_SUCCESS;
}
