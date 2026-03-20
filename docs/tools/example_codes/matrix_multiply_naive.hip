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

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <random>
#include <vector>

// ---------------------------------------------------------------------------
// Error-checking wrapper.  Aborts on the first HIP error.
// ---------------------------------------------------------------------------
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

// ---------------------------------------------------------------------------
// Problem dimensions.  All three are multiples of the block size so that no
// boundary-handling is needed in the kernel, keeping the code readable.
// ---------------------------------------------------------------------------
constexpr int M = 4096; // Rows of A and C
constexpr int N = 4096; // Columns of B and C
constexpr int K = 4096; // Columns of A / rows of B

// ---------------------------------------------------------------------------
// Kernel launch geometry.
// A 2-D thread block of BLOCK_SIZE x BLOCK_SIZE threads is a natural fit
// for a 2-D output matrix.
// ---------------------------------------------------------------------------
// [Sphinx naive block size start]
constexpr int BLOCK_SIZE = 16; // Thread-block edge length (square block)
// [Sphinx naive block size end]

// Number of timing repetitions.
constexpr int WARMUP_RUNS = 3;
constexpr int TIMING_RUNS = 10;

// ---------------------------------------------------------------------------
// Naive matrix-multiplication kernel: C = A * B  (all FP32, row-major)
//
// Thread mapping
// --------------
// The 2-D grid covers the M x N output matrix one thread per element:
//
//   col = blockIdx.x * BLOCK_SIZE + threadIdx.x   (output column)
//   row = blockIdx.y * BLOCK_SIZE + threadIdx.y   (output row)
//
// Memory access pattern
// ---------------------
// A[row * K + i]:
//   For a fixed iteration i, threads with the same threadIdx.y share the
//   same row and therefore broadcast-read the same A element within a
//   wavefront.  However, threads with different threadIdx.y values (which
//   belong to the same wavefront on 64-wide CDNA hardware) read from
//   distinct rows, producing scattered, non-coalesced accesses.
//   Moreover, every column of blocks re-reads each row of A, for a total
//   of (N / BLOCK_SIZE) global loads per A element.
//
// B[i * N + col]:
//   Consecutive threadIdx.x values map to consecutive columns, so reads
//   along the threadIdx.x dimension are coalesced.  The stride between
//   successive k-iterations is N elements (one full row), however, which
//   prevents any useful prefetch or stride-1 streaming.
//   Every row of blocks re-reads each column of B, for a total of
//   (M / BLOCK_SIZE) global loads per B element.
//
// Summary: the total DRAM traffic is proportional to
//   A_traffic = M * K * (N / BLOCK_SIZE)
//   B_traffic = K * N * (M / BLOCK_SIZE)
// which is BLOCK_SIZE times the combined matrix footprint.  For BLOCK_SIZE=16
// and matrices that overflow the L2 cache, nearly every load misses to DRAM.
// ---------------------------------------------------------------------------
// [Sphinx naive kernel start]
__global__ void matrix_multiply_naive(const float* __restrict__ A,
                                       const float* __restrict__ B,
                                       float* __restrict__ C,
                                       int m,
                                       int n,
                                       int k)
{
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = blockIdx.y * blockDim.y + threadIdx.y;

    if(row < m && col < n)
    {
        float sum = 0.0f;
        for(int i = 0; i < k; ++i)
        {
            sum += A[row * k + i] * B[i * n + col];
        }
        C[row * n + col] = sum;
    }
}
// [Sphinx naive kernel end]

// ---------------------------------------------------------------------------
// Timing helper.
// Runs `launch` WARMUP_RUNS times to let the GPU reach steady state, then
// measures the average wall time over TIMING_RUNS launches using HIP events.
// The kernel lambda is intentionally kept generic so the same helper is
// reused unchanged in every subsequent step.
// ---------------------------------------------------------------------------
// [Sphinx timing helper start]
template<typename KernelLaunch>
float time_kernel_ms(KernelLaunch launch, int warmup_runs, int timing_runs)
{
    // Warm-up: driver initialisation and clock settling
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
// [Sphinx timing helper end]

// ---------------------------------------------------------------------------
// Correctness check.
// Initialising B as the identity matrix means the mathematically expected
// result is C == A (up to floating-point rounding).  This avoids a full
// CPU reference multiplication at the cost of restricting B to a specific
// pattern, which is sufficient to exercise all code paths in the kernel.
// ---------------------------------------------------------------------------
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

// ---------------------------------------------------------------------------
// Performance metrics.
// "Effective bandwidth" and "arithmetic throughput" are computed from the
// minimum possible data movement (each matrix element read/written once) and
// the number of floating-point operations (2 per multiply-add).  The naive
// kernel will appear far below the hardware peak for both metrics because its
// actual DRAM traffic is much larger than the theoretical minimum.
// ---------------------------------------------------------------------------
void print_metrics(const char* label,
                   bool        passed,
                   float       avg_ms,
                   long long   m,
                   long long   n,
                   long long   k)
{
    // Minimum data moved: read A, read B, write C (each element once)
    const double bytes_min =
        sizeof(float) * static_cast<double>(m * k + k * n + m * n);
    // Total floating-point ops: one multiply and one add per k-step per output element
    const double flops = 2.0 * static_cast<double>(m) * static_cast<double>(n)
                         * static_cast<double>(k);

    std::cout << label << ": " << (passed ? "PASSED" : "FAILED") << "\n"
              << "  Average kernel time : " << avg_ms << " ms\n"
              << "  Eff. bandwidth      : "
              << bytes_min / (avg_ms * 1.0e-3) / 1.0e9 << " GB/s"
              << "  (lower bound; actual DRAM traffic is higher)\n"
              << "  Arithmetic throughput: "
              << flops / (avg_ms * 1.0e-3) / 1.0e12 << " TFLOPS\n\n";
}

int main()
{
    // ------------------------------------------------------------------
    // Print device information
    // ------------------------------------------------------------------
    hipDeviceProp_t props{};
    HIP_CHECK(hipGetDeviceProperties(&props, 0));
    std::cout << "Device      : " << props.name << "\n"
              << "Warp size   : " << props.warpSize << "\n"
              << "Matrix dims : A(" << M << "x" << K << ") * B(" << K << "x" << N << ")\n"
              << "Block size  : " << BLOCK_SIZE << "x" << BLOCK_SIZE << "\n\n";

    // ------------------------------------------------------------------
    // Host buffers
    // A: random values in [-1, 1]
    // B: identity matrix  =>  expected result C == A
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

    // ------------------------------------------------------------------
    // Launch geometry
    // ------------------------------------------------------------------
    // [Sphinx naive launch config start]
    const dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    const dim3 grid((N + BLOCK_SIZE - 1) / BLOCK_SIZE,
                    (M + BLOCK_SIZE - 1) / BLOCK_SIZE);
    // [Sphinx naive launch config end]

    std::cout << "Grid  : " << grid.x << " x " << grid.y << " blocks\n"
              << "Block : " << block.x << " x " << block.y << " threads\n\n";

    // ------------------------------------------------------------------
    // Warm-up, time, verify
    // ------------------------------------------------------------------
    auto launch_naive = [&]()
    {
        matrix_multiply_naive<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
        HIP_CHECK(hipGetLastError());
    };

    const float avg_ms = time_kernel_ms(launch_naive, WARMUP_RUNS, TIMING_RUNS);

    HIP_CHECK(hipMemcpy(h_C.data(), d_C, sizeof(float) * M * N, hipMemcpyDeviceToHost));
    const bool passed = verify_result(h_C, h_A, M, N, K);

    print_metrics("Naive", passed, avg_ms, M, N, K);

    // ------------------------------------------------------------------
    // Cleanup
    // ------------------------------------------------------------------
    HIP_CHECK(hipFree(d_A));
    HIP_CHECK(hipFree(d_B));
    HIP_CHECK(hipFree(d_C));

    return passed ? EXIT_SUCCESS : EXIT_FAILURE;
}
