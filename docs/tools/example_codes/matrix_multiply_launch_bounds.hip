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
#include <iostream>
#include <random>
#include <vector>

#define HIP_CHECK(expression)                                                   \
    {                                                                           \
        const hipError_t status = (expression);                                 \
        if(status != hipSuccess)                                                \
        {                                                                       \
            std::cerr << "HIP error " << status << ": "                         \
                      << hipGetErrorString(status) << " at " << __FILE__ << ":" \
                      << __LINE__ << std::endl;                                 \
            std::exit(EXIT_FAILURE);                                            \
        }                                                                       \
    }

constexpr int M = 4096;
constexpr int N = 4096;
constexpr int K = 4096;

constexpr int WARMUP_RUNS = 3;
constexpr int TIMING_RUNS = 10;

// Tile sizes (same as Steps 3–5)
constexpr int BLOCK_TILE_M  = 128;
constexpr int BLOCK_TILE_N  = 128;
constexpr int K_TILE_SIZE   = 16;
constexpr int THREAD_TILE_M = 8;
constexpr int THREAD_TILE_N = 8;
constexpr int BLOCK_DIM_X   = BLOCK_TILE_N / THREAD_TILE_N; // 16
constexpr int BLOCK_DIM_Y   = BLOCK_TILE_M / THREAD_TILE_M; // 16
constexpr int BLOCK_THREADS = BLOCK_DIM_X * BLOCK_DIM_Y;    // 256

// ===========================================================================
// Register pressure and occupancy
//
// The register file on AMD GPUs is shared among all wavefronts that are
// resident on a compute unit (CU) at the same time.  Occupancy is the number
// of resident wavefronts divided by the hardware maximum.  High occupancy
// helps hide memory and instruction latency by giving the scheduler more
// wavefronts to choose from.  However, occupancy is limited by the number of
// vector general-purpose registers (VGPRs) each wavefront requires: doubling
// the per-wavefront VGPR count halves the maximum occupancy.
//
// A register-tiled kernel like the one in Step 3 allocates large per-thread
// accumulators (acc[TM][TN] = 8*8 = 64 floats = 64 VGPRs just for the
// accumulator) as well as fragment arrays and index variables.  The total
// VGPR count can easily reach 128–200+, which may reduce occupancy to 1–2
// wavefronts per CU.  At that point, the CU cannot hide the latency of MFMA
// or global memory instructions through wavefront switching.
//
// Two AMD-specific mechanisms let the programmer influence this trade-off:
//
// 1. __launch_bounds__(max_threads_per_block [, min_waves_per_eu])
//    ---------------------------------------------------------------
//    The first argument tells the compiler the maximum block size that will
//    ever be used to launch this kernel.  The compiler uses this to set
//    an upper bound on the VGPR count that preserves the corresponding
//    minimum occupancy.
//
//    The optional second argument (min_waves_per_eu) requests a MINIMUM
//    number of wavefronts per CU.  The compiler will limit VGPR allocation
//    so that at least this many wavefronts can be resident simultaneously.
//    If the kernel's register pressure exceeds what this constraint allows,
//    the compiler will spill registers to scratch memory (LDS or VGPR spill
//    to global).  Spilling has a measurable cost, so the optimal value must
//    be found by profiling – it is a tuning knob, not a correctness concern.
//
//    For a 256-thread block (4 wavefronts of 64 or 8 wavefronts of 32):
//      __launch_bounds__(256, 2)  ← at least 2 wavefronts per CU resident
//
// 2. [[clang::amdgpu_waves_per_eu(min_waves, max_waves)]]
//    -------------------------------------------------------
//    This Clang attribute directly constrains the number of wavefronts per
//    EU that the compiler targets.  It is more precise than __launch_bounds__
//    because it operates at the wavefront (not thread-block) granularity and
//    applies regardless of the runtime block size.
//
//    Both min and max can be set to the same value to pin occupancy exactly.
//    This is useful when profiling has established the optimal wavefront
//    count and you want to prevent the compiler from choosing differently
//    on a future recompilation.
//
//    Example: [[clang::amdgpu_waves_per_eu(4, 8)]]
//      The compiler will allocate VGPRs so that at least 4 and at most 8
//      wavefronts per CU can be resident.
//
// Interaction between the two mechanisms
// ---------------------------------------
// __launch_bounds__ and amdgpu_waves_per_eu can be combined; the compiler
// takes the most restrictive constraint.  In practice, choose one mechanism
// per kernel to avoid confusion.  __launch_bounds__ is more portable
// (supported by both CUDA and HIP); amdgpu_waves_per_eu is AMD-specific but
// more expressive.
//
// Profiling guidance (rocprofv3)
// --------------------------------
// Profile the kernel with rocprofv3 --kernel-trace and inspect the CSV output:
//   - VGPR_Count    : per-kernel VGPR allocation
//   - Scratch_Size  : non-zero indicates register spilling
//   - Wave_Size     : wavefront width (32 or 64)
//   - LDS_Block_Size: shared memory allocation per block
//
// Increase min_waves_per_eu / reduce max_waves_per_eu until the product of
// "occupancy" and "compute utilisation" is maximised.  A kernel with 4
// wavefronts at 95% utilisation is usually better than one with 8 wavefronts
// at 40% utilisation.
// ===========================================================================

// ---------------------------------------------------------------------------
// Shared LDS load / compute body extracted into a __device__ function so that
// the three kernel variants below share the same logic.  The compiler inlines
// this function into each kernel.
// ---------------------------------------------------------------------------
__device__ void gemm_body(const float* __restrict__ A,
                          const float* __restrict__ B,
                          float* __restrict__ C,
                          int m,
                          int n,
                          int k,
                          float (&tile_a)[BLOCK_TILE_M][K_TILE_SIZE],
                          float (&tile_b_T)[BLOCK_TILE_N][K_TILE_SIZE])
{
    const int tx          = threadIdx.x;
    const int ty          = threadIdx.y;
    const int tid         = ty * blockDim.x + tx;
    const int num_threads = blockDim.x * blockDim.y;

    const int thread_row      = ty * THREAD_TILE_M;
    const int thread_col      = tx * THREAD_TILE_N;
    const int block_row_start = blockIdx.y * BLOCK_TILE_M;
    const int block_col_start = blockIdx.x * BLOCK_TILE_N;

    float acc[THREAD_TILE_M][THREAD_TILE_N] = {};

    const int num_tiles = (k + K_TILE_SIZE - 1) / K_TILE_SIZE;

    for(int t = 0; t < num_tiles; ++t)
    {
        // Cooperative load of tile_a
        {
            const int tile_elems = BLOCK_TILE_M * K_TILE_SIZE;
            for(int idx = tid; idx < tile_elems; idx += num_threads)
            {
                const int tile_row   = idx / K_TILE_SIZE;
                const int tile_col   = idx % K_TILE_SIZE;
                const int global_row = block_row_start + tile_row;
                const int global_col = t * K_TILE_SIZE + tile_col;
                tile_a[tile_row][tile_col] =
                    (global_row < m && global_col < k)
                        ? A[global_row * k + global_col]
                        : 0.0f;
            }
        }

        // Cooperative load of tile_b_T
        {
            const int tile_elems = BLOCK_TILE_N * K_TILE_SIZE;
            for(int idx = tid; idx < tile_elems; idx += num_threads)
            {
                const int tile_col_idx = idx / K_TILE_SIZE;
                const int tile_k_idx   = idx % K_TILE_SIZE;
                const int global_row   = t * K_TILE_SIZE + tile_k_idx;
                const int global_col   = block_col_start + tile_col_idx;
                tile_b_T[tile_col_idx][tile_k_idx] =
                    (global_row < k && global_col < n)
                        ? B[global_row * n + global_col]
                        : 0.0f;
            }
        }

        __syncthreads();

        // Outer-product accumulation
        for(int ki = 0; ki < K_TILE_SIZE; ++ki)
        {
            float a_frag[THREAD_TILE_M];
            #pragma unroll
            for(int i = 0; i < THREAD_TILE_M; ++i)
                a_frag[i] = tile_a[thread_row + i][ki];

            float b_frag[THREAD_TILE_N];
            #pragma unroll
            for(int j = 0; j < THREAD_TILE_N; ++j)
                b_frag[j] = tile_b_T[thread_col + j][ki];

            #pragma unroll
            for(int i = 0; i < THREAD_TILE_M; ++i)
            {
                #pragma unroll
                for(int j = 0; j < THREAD_TILE_N; ++j)
                    acc[i][j] += a_frag[i] * b_frag[j];
            }
        }

        __syncthreads();
    }

    // Store
    #pragma unroll
    for(int i = 0; i < THREAD_TILE_M; ++i)
    {
        #pragma unroll
        for(int j = 0; j < THREAD_TILE_N; ++j)
        {
            const int out_row = blockIdx.y * BLOCK_TILE_M + threadIdx.y * THREAD_TILE_M + i;
            const int out_col = blockIdx.x * BLOCK_TILE_N + threadIdx.x * THREAD_TILE_N + j;
            if(out_row < m && out_col < n)
                C[out_row * n + out_col] = acc[i][j];
        }
    }
}

// ---------------------------------------------------------------------------
// Variant 1: no occupancy hint (compiler chooses VGPR allocation freely).
// ---------------------------------------------------------------------------
// [Sphinx no hint kernel start]
__global__ void matrix_multiply_no_hint(const float* __restrict__ A,
                                        const float* __restrict__ B,
                                        float* __restrict__ C,
                                        int m,
                                        int n,
                                        int k)
{
    __shared__ float tile_a[BLOCK_TILE_M][K_TILE_SIZE];
    __shared__ float tile_b_T[BLOCK_TILE_N][K_TILE_SIZE];
    gemm_body(A, B, C, m, n, k, tile_a, tile_b_T);
}
// [Sphinx no hint kernel end]

// ---------------------------------------------------------------------------
// Variant 2: __launch_bounds__
//
// Informs the compiler of the maximum thread-block size and requests a minimum
// number of wavefronts per CU.  The compiler limits VGPR allocation so that
// at least `min_waves` wavefronts can reside on a CU simultaneously.
//
// For 256 threads per block on CDNA (wavefront = 64):
//   4 wavefronts per block × 1 block = 4 wavefronts per CU minimum.
// For 256 threads per block on RDNA (wavefront = 32):
//   8 wavefronts per block × 1 block = 8 wavefronts per CU minimum.
//
// The min_waves_per_eu value below is illustrative.  Profile with rocprofv3
// to find the value that maximises performance on your target architecture.
// ---------------------------------------------------------------------------
// [Sphinx launch bounds kernel start]
constexpr int MIN_WAVES_PER_EU = 2; // tune this with rocprofv3

__global__
__launch_bounds__(BLOCK_THREADS, MIN_WAVES_PER_EU)
void matrix_multiply_launch_bounds(const float* __restrict__ A,
                                   const float* __restrict__ B,
                                   float* __restrict__ C,
                                   int m,
                                   int n,
                                   int k)
{
    __shared__ float tile_a[BLOCK_TILE_M][K_TILE_SIZE];
    __shared__ float tile_b_T[BLOCK_TILE_N][K_TILE_SIZE];
    gemm_body(A, B, C, m, n, k, tile_a, tile_b_T);
}
// [Sphinx launch bounds kernel end]

// ---------------------------------------------------------------------------
// Variant 3: [[clang::amdgpu_waves_per_eu(min, max)]]
//
// This AMD-specific Clang attribute directly pins the wavefront occupancy
// target used during register allocation.  It is independent of the runtime
// block size, which makes it more predictable than __launch_bounds__ when the
// block size varies.
//
// Setting min == max pins occupancy to exactly that value.  Setting a range
// gives the compiler freedom to choose the best allocation within the range.
//
// Use rocprofv3 --kernel-trace to compare VGPR_Count and Scratch_Size between
// variants and verify that the chosen range does not introduce register spilling.
// ---------------------------------------------------------------------------
// [Sphinx amdgpu waves per eu kernel start]
// tune WAVES_PER_EU_MIN and WAVES_PER_EU_MAX with rocprofv3 --kernel-trace
// (compare VGPR_Count across variants).
constexpr int WAVES_PER_EU_MIN = 2;
constexpr int WAVES_PER_EU_MAX = 4;

__global__ void matrix_multiply_waves_per_eu
[[clang::amdgpu_waves_per_eu(WAVES_PER_EU_MIN, WAVES_PER_EU_MAX)]]
(const float* __restrict__ A,
 const float* __restrict__ B,
 float* __restrict__ C,
 int m,
 int n,
 int k)
{
    __shared__ float tile_a[BLOCK_TILE_M][K_TILE_SIZE];
    __shared__ float tile_b_T[BLOCK_TILE_N][K_TILE_SIZE];
    gemm_body(A, B, C, m, n, k, tile_a, tile_b_T);
}
// [Sphinx amdgpu waves per eu kernel end]

// ---------------------------------------------------------------------------
// Shared helpers
// ---------------------------------------------------------------------------
template<typename KernelLaunch>
float time_kernel_ms(KernelLaunch launch, int warmup_runs, int timing_runs)
{
    for(int i = 0; i < warmup_runs; ++i)
    {
        launch();
        HIP_CHECK(hipDeviceSynchronize());
    }

    hipEvent_t ev_start, ev_stop;
    HIP_CHECK(hipEventCreate(&ev_start));
    HIP_CHECK(hipEventCreate(&ev_stop));

    HIP_CHECK(hipEventRecord(ev_start));
    for(int i = 0; i < timing_runs; ++i)
    {
        launch();
    }
    HIP_CHECK(hipEventRecord(ev_stop));
    HIP_CHECK(hipEventSynchronize(ev_stop));

    float elapsed_ms = 0.0f;
    HIP_CHECK(hipEventElapsedTime(&elapsed_ms, ev_start, ev_stop));

    HIP_CHECK(hipEventDestroy(ev_start));
    HIP_CHECK(hipEventDestroy(ev_stop));

    return elapsed_ms / static_cast<float>(timing_runs);
}

bool verify_result(const std::vector<float>& C,
                   const std::vector<float>& A,
                   int                       m,
                   int                       n,
                   int                       k,
                   float                     tolerance = 1.0e-3f)
{
    const int check_cols = std::min(k, n);
    for(int i = 0; i < m; ++i)
    {
        for(int j = 0; j < check_cols; ++j)
        {
            if(std::fabs(C[i * n + j] - A[i * k + j]) > tolerance)
            {
                std::cerr << "Mismatch at [" << i << "][" << j << "]: "
                          << "expected " << A[i * k + j]
                          << ", got " << C[i * n + j] << "\n";
                return false;
            }
        }
    }
    return true;
}

void print_metrics(const char* label,
                   bool        passed,
                   float       avg_ms,
                   long long   m,
                   long long   n,
                   long long   k)
{
    const double bytes_min =
        sizeof(float) * static_cast<double>(m * k + k * n + m * n);
    const double flops = 2.0 * static_cast<double>(m) * static_cast<double>(n)
                         * static_cast<double>(k);

    std::cout << label << ": " << (passed ? "PASSED" : "FAILED") << "\n"
              << "  Average kernel time  : " << avg_ms << " ms\n"
              << "  Eff. bandwidth       : "
              << bytes_min / (avg_ms * 1.0e-3) / 1.0e9 << " GB/s\n"
              << "  Arithmetic throughput: "
              << flops / (avg_ms * 1.0e-3) / 1.0e12 << " TFLOPS\n\n";
}

int main()
{
    hipDeviceProp_t props{};
    HIP_CHECK(hipGetDeviceProperties(&props, 0));
    std::cout << "Device       : " << props.name << "\n"
              << "Warp size    : " << props.warpSize << "\n"
              << "Matrix dims  : A(" << M << "x" << K << ") * B(" << K << "x" << N << ")\n"
              << "Block tile   : " << BLOCK_TILE_M << "x" << BLOCK_TILE_N
              << "  K-strip: " << K_TILE_SIZE << "\n"
              << "Thread tile  : " << THREAD_TILE_M << "x" << THREAD_TILE_N << "\n"
              << "Block threads: " << BLOCK_THREADS << "\n\n";

    // ------------------------------------------------------------------
    // Host buffers
    // ------------------------------------------------------------------
    std::vector<float> h_A(M * K);
    std::vector<float> h_B(K * N, 0.0f);
    std::vector<float> h_C(M * N, 0.0f);

    std::mt19937                          gen(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for(float& v : h_A)
    {
        v = dist(gen);
    }
    for(int i = 0; i < std::min(K, N); ++i)
    {
        h_B[i * N + i] = 1.0f;
    }

    // ------------------------------------------------------------------
    // Device allocation and upload
    // ------------------------------------------------------------------
    float *d_A, *d_B, *d_C;
    HIP_CHECK(hipMalloc(&d_A, sizeof(float) * M * K));
    HIP_CHECK(hipMalloc(&d_B, sizeof(float) * K * N));
    HIP_CHECK(hipMalloc(&d_C, sizeof(float) * M * N));

    HIP_CHECK(hipMemcpy(d_A, h_A.data(), sizeof(float) * M * K, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_B, h_B.data(), sizeof(float) * K * N, hipMemcpyHostToDevice));

    const dim3 block(BLOCK_DIM_X, BLOCK_DIM_Y);
    const dim3 grid((N + BLOCK_TILE_N - 1) / BLOCK_TILE_N,
                    (M + BLOCK_TILE_M - 1) / BLOCK_TILE_M);

    // ------------------------------------------------------------------
    // Run all three variants
    // ------------------------------------------------------------------
    {
        auto launch = [&]()
        {
            matrix_multiply_no_hint<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
            HIP_CHECK(hipGetLastError());
        };
        const float ms = time_kernel_ms(launch, WARMUP_RUNS, TIMING_RUNS);
        HIP_CHECK(hipMemcpy(h_C.data(), d_C, sizeof(float) * M * N, hipMemcpyDeviceToHost));
        const bool ok = verify_result(h_C, h_A, M, N, K);
        print_metrics("No occupancy hint (baseline)          ", ok, ms, M, N, K);
    }

    {
        auto launch = [&]()
        {
            matrix_multiply_launch_bounds<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
            HIP_CHECK(hipGetLastError());
        };
        const float ms = time_kernel_ms(launch, WARMUP_RUNS, TIMING_RUNS);
        HIP_CHECK(hipMemcpy(h_C.data(), d_C, sizeof(float) * M * N, hipMemcpyDeviceToHost));
        const bool ok = verify_result(h_C, h_A, M, N, K);
        print_metrics("__launch_bounds__                     ", ok, ms, M, N, K);
    }

    {
        auto launch = [&]()
        {
            matrix_multiply_waves_per_eu<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
            HIP_CHECK(hipGetLastError());
        };
        const float ms = time_kernel_ms(launch, WARMUP_RUNS, TIMING_RUNS);
        HIP_CHECK(hipMemcpy(h_C.data(), d_C, sizeof(float) * M * N, hipMemcpyDeviceToHost));
        const bool ok = verify_result(h_C, h_A, M, N, K);
        print_metrics("[[clang::amdgpu_waves_per_eu(2, 4)]]  ", ok, ms, M, N, K);
    }

    // ------------------------------------------------------------------
    // Cleanup
    // ------------------------------------------------------------------
    HIP_CHECK(hipFree(d_A));
    HIP_CHECK(hipFree(d_B));
    HIP_CHECK(hipFree(d_C));

    return EXIT_SUCCESS;
}
