// MIT License
//
// Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
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

constexpr int input_size  = 1 << 24; // 16 M elements
constexpr int block_size  = 256;
constexpr int timing_runs = 100;

// [Sphinx reduction naive kernel start]
__global__ void reduction_naive(const float *input, float *output, int n)
{
    extern __shared__ float sdata[];

    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + tid;

    sdata[tid] = (gid < n) ? input[gid] : 0.0f;
    __syncthreads();

    for (int stride = 1; stride < blockDim.x; stride *= 2)
    {
        if (tid % (2 * stride) == 0)
        {
            sdata[tid] += sdata[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0)
    {
        output[blockIdx.x] = sdata[0];
    }
}
// [Sphinx reduction naive kernel end]

// [Sphinx reduction sequential kernel start]
__global__ void reduction_sequential(const float *input, float *output, int n)
{
    extern __shared__ float sdata[];

    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + tid;

    sdata[tid] = (gid < n) ? input[gid] : 0.0f;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1)
    {
        if (tid < stride)
        {
            sdata[tid] += sdata[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0)
    {
        output[blockIdx.x] = sdata[0];
    }
}
// [Sphinx reduction sequential kernel end]

// [Sphinx reduction warp reduce helper start]
template<int WarpSize>
__device__ __forceinline__ void warp_reduce(float *sdata, int tid, int lane)
{
    if (WarpSize == 64) { if (lane < 32) { sdata[tid] += sdata[tid + 32]; } }
                          if (lane < 16) { sdata[tid] += sdata[tid + 16]; }
                          if (lane <  8) { sdata[tid] += sdata[tid +  8]; }
                          if (lane <  4) { sdata[tid] += sdata[tid +  4]; }
                          if (lane <  2) { sdata[tid] += sdata[tid +  2]; }
                          if (lane <  1) { sdata[tid] += sdata[tid +  1]; }
}
// [Sphinx reduction warp reduce helper end]

// [Sphinx reduction warp kernel start]
template<int WarpSize>
__global__ __launch_bounds__(block_size)
void reduction_warp(const float *input, float *output, int n)
{
    extern __shared__ float sdata[];

    int tid    = threadIdx.x;
    int gid    = blockIdx.x * blockDim.x + tid;
    int lane   = tid % WarpSize;
    int warpid = tid / WarpSize;

    sdata[tid] = (gid < n) ? input[gid] : 0.0f;

    warp_reduce<WarpSize>(sdata, tid, lane);

    int num_warps = blockDim.x / WarpSize;
    if (lane == 0)
    {
        sdata[blockDim.x + warpid] = sdata[tid];
    }
    __syncthreads();

    if (warpid == 0)
    {
        sdata[tid] = (lane < num_warps) ? sdata[blockDim.x + lane] : 0.0f;
        warp_reduce<WarpSize>(sdata, tid, lane);
    }

    if (tid == 0)
    {
        output[blockIdx.x] = sdata[0];
    }
}
// [Sphinx reduction warp kernel end]

// [Sphinx reduction float4 kernel start]
template<int WarpSize>
__global__ __launch_bounds__(block_size)
void reduction_float4(const float *input, float *output, int n)
{
    extern __shared__ float sdata[];

    int tid    = threadIdx.x;
    int gid    = blockIdx.x * (blockDim.x * 4) + tid;
    int lane   = tid % WarpSize;
    int warpid = tid / WarpSize;

    float val = 0.0f;
    if (gid + 3 * blockDim.x < n)
    {
        float4 v = reinterpret_cast<const float4 *>(input)[blockIdx.x * blockDim.x + tid];
        val = v.x + v.y + v.z + v.w;
    }
    else
    {
        if (gid                  < n) { val += input[gid]; }
        if (gid +     blockDim.x < n) { val += input[gid +     blockDim.x]; }
        if (gid + 2 * blockDim.x < n) { val += input[gid + 2 * blockDim.x]; }
        if (gid + 3 * blockDim.x < n) { val += input[gid + 3 * blockDim.x]; }
    }

    int num_warps = blockDim.x / WarpSize;
    sdata[tid] = val;
    warp_reduce<WarpSize>(sdata, tid, lane);

    if (lane == 0)
    {
        sdata[blockDim.x + warpid] = sdata[tid];
    }
    __syncthreads();

    if (warpid == 0)
    {
        sdata[tid] = (lane < num_warps) ? sdata[blockDim.x + lane] : 0.0f;
        warp_reduce<WarpSize>(sdata, tid, lane);
    }

    if (tid == 0)
    {
        output[blockIdx.x] = sdata[0];
    }
}
// [Sphinx reduction float4 kernel end]

std::vector<float> cpu_reference(const std::vector<float> &input, int logical_block_size)
{
    int n          = static_cast<int>(input.size());
    int num_chunks = (n + logical_block_size - 1) / logical_block_size;

    std::vector<float> partials(num_chunks, 0.0f);
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

void verify(const std::vector<float> &ref, const float *d_output, int num_blocks)
{
    std::vector<float> h_output(num_blocks);
    HIP_CHECK(hipMemcpy(h_output.data(), d_output, sizeof(float) * num_blocks,
                        hipMemcpyDeviceToHost));

    for (int b = 0; b < num_blocks; ++b)
    {
        float rel_err = std::fabs(h_output[b] - ref[b]) / (std::fabs(ref[b]) + 1.0f);
        if (rel_err >= 1e-3f)
        {
            std::cout << "FAILED (block " << b
                      << ": result=" << h_output[b]
                      << ", ref=" << ref[b] << ")" << std::endl;
            return;
        }
    }
    std::cout << "PASSED" << std::endl;
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
    int deviceid      = 0;
    int warpsizehost  = 0;
    HIP_CHECK(hipDeviceGetAttribute(&warpsizehost, hipDeviceAttributeWarpSize, deviceid));
    std::cout << "Wavefront size: " << warpsizehost << std::endl;

    std::vector<float> h_input(input_size);
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (float &v : h_input)
    {
        v = dist(gen);
    }

    std::vector<float> ref_bs1 = cpu_reference(h_input, block_size);
    std::vector<float> ref_bs4 = cpu_reference(h_input, block_size * 4);

    int nb1 = static_cast<int>(ref_bs1.size());
    int nb4 = static_cast<int>(ref_bs4.size());

    int num_warps  = block_size / warpsizehost;
    int smem_block = block_size * sizeof(float);
    int smem_warp  = (block_size + num_warps) * sizeof(float);

    float *d_input  = nullptr;
    float *d_output = nullptr;

    HIP_CHECK(hipMalloc(&d_input,  sizeof(float) * input_size));
    HIP_CHECK(hipMalloc(&d_output, sizeof(float) * nb1));
    HIP_CHECK(hipMemcpy(d_input, h_input.data(), sizeof(float) * input_size,
                        hipMemcpyHostToDevice));

    std::cout << "Running reduction kernels..." << std::endl;

    // Naive kernel
    reduction_naive<<<nb1, block_size, smem_block>>>(d_input, d_output, input_size);
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());
    std::cout << "  Naive: ";
    verify(ref_bs1, d_output, nb1);
#ifdef ENABLE_TIMERS
    time_kernel("naive", [&]()
    {
        reduction_naive<<<nb1, block_size, smem_block>>>(d_input, d_output, input_size);
    });
#endif

    // Sequential addressing kernel
    reduction_sequential<<<nb1, block_size, smem_block>>>(d_input, d_output, input_size);
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());
    std::cout << "  Sequential: ";
    verify(ref_bs1, d_output, nb1);
#ifdef ENABLE_TIMERS
    time_kernel("sequential", [&]()
    {
        reduction_sequential<<<nb1, block_size, smem_block>>>(d_input, d_output, input_size);
    });
#endif

    // Warp reduction kernel
    if (warpsizehost == 64)
    {
        reduction_warp<64><<<nb1, block_size, smem_warp>>>(d_input, d_output, input_size);
    }
    else
    {
        reduction_warp<32><<<nb1, block_size, smem_warp>>>(d_input, d_output, input_size);
    }
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());
    std::cout << "  Warp reduction: ";
    verify(ref_bs1, d_output, nb1);
#ifdef ENABLE_TIMERS
    time_kernel("warp reduction", [&]()
    {
        if (warpsizehost == 64)
        {
            reduction_warp<64><<<nb1, block_size, smem_warp>>>(d_input, d_output, input_size);
        }
        else
        {
            reduction_warp<32><<<nb1, block_size, smem_warp>>>(d_input, d_output, input_size);
        }
    });
#endif

    // float4 vectorized loads kernel
    if (warpsizehost == 64)
    {
        reduction_float4<64><<<nb4, block_size, smem_warp>>>(d_input, d_output, input_size);
    }
    else
    {
        reduction_float4<32><<<nb4, block_size, smem_warp>>>(d_input, d_output, input_size);
    }
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());
    std::cout << "  float4 vectorized loads: ";
    verify(ref_bs4, d_output, nb4);
#ifdef ENABLE_TIMERS
    time_kernel("float4 vectorized loads", [&]()
    {
        if (warpsizehost == 64)
        {
            reduction_float4<64><<<nb4, block_size, smem_warp>>>(d_input, d_output, input_size);
        }
        else
        {
            reduction_float4<32><<<nb4, block_size, smem_warp>>>(d_input, d_output, input_size);
        }
    });
#endif

    // Device-level reduction
    // [Sphinx reduction device level start]
    {
        float *d_partial_a = nullptr;
        float *d_partial_b = nullptr;
        HIP_CHECK(hipMalloc(&d_partial_a, sizeof(float) * nb1));
        HIP_CHECK(hipMalloc(&d_partial_b, sizeof(float) * nb1));

        int          current = input_size;
        const float *src     = d_input;
        float       *dst     = d_partial_a;

        while (current > 1)
        {
            int nb = (current + block_size * 4 - 1) / (block_size * 4);
            if (warpsizehost == 64)
            {
                reduction_float4<64><<<nb, block_size, smem_warp>>>(src, dst, current);
            }
            else
            {
                reduction_float4<32><<<nb, block_size, smem_warp>>>(src, dst, current);
            }
            HIP_CHECK(hipGetLastError());
            current = nb;
            if (current > 1)
            {
                src = dst;
                dst = (dst == d_partial_a) ? d_partial_b : d_partial_a;
            }
        }

        HIP_CHECK(hipDeviceSynchronize());
        float result = 0.0f;
        HIP_CHECK(hipMemcpy(&result, dst, sizeof(float), hipMemcpyDeviceToHost));

        // Compute the scalar reference by reducing the per-block CPU partials.
        float ref_scalar = 0.0f;
        for (float v : ref_bs4)
        {
            ref_scalar += v;
        }
        float rel_err = std::fabs(result - ref_scalar) / (std::fabs(ref_scalar) + 1.0f);

        std::cout << "  Device reduction: ";
        std::cout << (rel_err < 1e-3f ? "PASSED" : "FAILED") << std::endl;

        HIP_CHECK(hipFree(d_partial_a));
        HIP_CHECK(hipFree(d_partial_b));
    }
    // [Sphinx reduction device level end]

    std::cout << "Execution completed successfully." << std::endl;

    HIP_CHECK(hipFree(d_input));
    HIP_CHECK(hipFree(d_output));

    return EXIT_SUCCESS;
}
