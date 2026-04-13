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

// ---------------------------------------------------------------------------
// LDS tile size.
//
// TILE_SIZE has two roles:
//   1. It sets the edge length of each square LDS tile for A and B.
//   2. It must equal the thread-block edge length so that exactly one thread
//      per tile element is available to cooperatively load that element.
//
// LDS capacity check (per block, both tiles combined):
//   2 * TILE_SIZE * TILE_SIZE * sizeof(float) = 2 * 16 * 16 * 4 = 2 048 bytes
// This is well within the minimum LDS size of 32 KiB per workgroup on all
// ROCm-supported architectures.
// ---------------------------------------------------------------------------
// [Sphinx LDS tile size start]
constexpr int TILE_SIZE = 16;
// [Sphinx LDS tile size end]

// ---------------------------------------------------------------------------
// Naive kernel (baseline for comparison).
// See matrix_multiply_naive.hip for a detailed description.
// ---------------------------------------------------------------------------
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

// ---------------------------------------------------------------------------
// LDS-tiled matrix-multiplication kernel: C = A * B  (all FP32, row-major)
//
// Tiling strategy
// ---------------
// The K dimension is split into tiles of width TILE_SIZE.  For each tile:
//
//  1. Load phase: all TILE_SIZE*TILE_SIZE threads in the block cooperate to
//     copy a TILE_SIZE x TILE_SIZE sub-matrix of A and a TILE_SIZE x TILE_SIZE
//     sub-matrix of B from global memory into LDS.
//
//  2. Barrier: __syncthreads() ensures every thread sees the complete tile
//     before any thread begins computing from it.
//
//  3. Compute phase: each thread accumulates TILE_SIZE multiply-add operations
//     using only LDS as the data source.
//
//  4. Barrier: __syncthreads() ensures no thread starts overwriting an LDS
//     tile that a neighbour is still reading.
//
// Data reuse
// ----------
// Each global memory element is loaded exactly once into LDS and then reused
// TILE_SIZE times by other threads in the same block.  This reduces the total
// DRAM traffic by a factor of TILE_SIZE compared with the naive kernel.
//
// LDS access patterns
// -------------------
// tile_a[ty][tx]:
//   During load, thread (tx, ty) writes to row ty. During compute, each
//   thread reads a full row of tile_a sequentially (index i = 0..TILE_SIZE-1).
//   At any compute step i, all threads in the wavefront access tile_a[ty][i]:
//   the same row but different threadIdx.y values → different LDS rows, each
//   accessed at column i → no bank conflict.
//
// tile_b[ty][tx]:
//   During load, thread (tx, ty) writes to row ty, column tx.
//   During compute, each thread reads tile_b[i][tx] for i = 0..TILE_SIZE-1.
//   At step i, threads access tile_b[i][0], tile_b[i][1], …, tile_b[i][15]:
//   consecutive columns of the same row → consecutive LDS addresses → no bank
//   conflict for TILE_SIZE ≤ number of LDS banks.
//   NOTE: if TILE_SIZE were increased to equal the LDS bank count (typically
//   32), stride-equal accesses would cause conflicts.  The register-tiling
//   step (matrix_multiply_register_tiling.hip) resolves this with a transposed
//   tile_b layout that is also better suited to the outer-product accumulation.
// ---------------------------------------------------------------------------
// [Sphinx LDS kernel start]
__global__ void matrix_multiply_lds(const float* __restrict__ A,
                                    const float* __restrict__ B,
                                    float* __restrict__ C,
                                    int m,
                                    int n,
                                    int k)
{
    // [Sphinx LDS shared memory start]
    // Allocate two LDS tiles: one strip of A (row-major) and one of B (row-major).
    // Each tile holds TILE_SIZE x TILE_SIZE float elements = 1 KiB.
    __shared__ float tile_a[TILE_SIZE][TILE_SIZE];
    __shared__ float tile_b[TILE_SIZE][TILE_SIZE];
    // [Sphinx LDS shared memory end]

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    // Global row and column that this thread is responsible for in C
    const int row = blockIdx.y * TILE_SIZE + ty;
    const int col = blockIdx.x * TILE_SIZE + tx;

    float sum = 0.0f;

    const int num_tiles = (k + TILE_SIZE - 1) / TILE_SIZE;

    for(int t = 0; t < num_tiles; ++t)
    {
        // [Sphinx LDS load phase start]
        // Cooperative load: thread (tx, ty) loads one element of each tile.
        //
        // tile_a ← A[row][t*TILE_SIZE + tx]
        //   Row comes from the block's output row; column strides across K.
        const int a_col = t * TILE_SIZE + tx;
        tile_a[ty][tx]  = (row < m && a_col < k) ? A[row * k + a_col] : 0.0f;

        // tile_b ← B[t*TILE_SIZE + ty][col]
        //   Row strides across K; column comes from the block's output column.
        const int b_row = t * TILE_SIZE + ty;
        tile_b[ty][tx]  = (b_row < k && col < n) ? B[b_row * n + col] : 0.0f;

        // Ensure all threads have written their element before any thread reads.
        __syncthreads();
        // [Sphinx LDS load phase end]

        // [Sphinx LDS compute phase start]
        // Accumulate the partial dot product for this tile from LDS.
        // Both tile_a and tile_b reside in LDS; no global memory is touched here.
        for(int i = 0; i < TILE_SIZE; ++i)
        {
            sum += tile_a[ty][i] * tile_b[i][tx];
        }

        // Ensure all threads have finished reading before the next iteration
        // overwrites the tiles.
        __syncthreads();
        // [Sphinx LDS compute phase end]
    }

    if(row < m && col < n)
    {
        C[row * n + col] = sum;
    }
}
// [Sphinx LDS kernel end]

// ---------------------------------------------------------------------------
// Shared helpers (identical to matrix_multiply_naive.hip)
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
    std::cout << "Device      : " << props.name << "\n"
              << "Warp size   : " << props.warpSize << "\n"
              << "Matrix dims : A(" << M << "x" << K << ") * B(" << K << "x" << N << ")\n"
              << "Tile size   : " << TILE_SIZE << "x" << TILE_SIZE << "\n\n";

    // ------------------------------------------------------------------
    // Host buffers: random A, identity B
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
    // Naive baseline
    // ------------------------------------------------------------------
    const dim3 block_naive(TILE_SIZE, TILE_SIZE);
    const dim3 grid_naive((N + TILE_SIZE - 1) / TILE_SIZE,
                          (M + TILE_SIZE - 1) / TILE_SIZE);

    auto launch_naive = [&]()
    {
        matrix_multiply_naive<<<grid_naive, block_naive>>>(d_A, d_B, d_C, M, N, K);
        HIP_CHECK(hipGetLastError());
    };

    const float naive_ms = time_kernel_ms(launch_naive, WARMUP_RUNS, TIMING_RUNS);
    HIP_CHECK(hipMemcpy(h_C.data(), d_C, sizeof(float) * M * N, hipMemcpyDeviceToHost));
    const bool naive_ok = verify_result(h_C, h_A, M, N, K);
    print_metrics("Naive (baseline)", naive_ok, naive_ms, M, N, K);

    // ------------------------------------------------------------------
    // LDS-tiled kernel
    // The block dimensions must match TILE_SIZE so that each thread loads
    // exactly one element during the cooperative load phase.
    // ------------------------------------------------------------------
    const dim3 block_lds(TILE_SIZE, TILE_SIZE);
    const dim3 grid_lds((N + TILE_SIZE - 1) / TILE_SIZE,
                        (M + TILE_SIZE - 1) / TILE_SIZE);

    auto launch_lds = [&]()
    {
        matrix_multiply_lds<<<grid_lds, block_lds>>>(d_A, d_B, d_C, M, N, K);
        HIP_CHECK(hipGetLastError());
    };

    const float lds_ms = time_kernel_ms(launch_lds, WARMUP_RUNS, TIMING_RUNS);
    HIP_CHECK(hipMemcpy(h_C.data(), d_C, sizeof(float) * M * N, hipMemcpyDeviceToHost));
    const bool lds_ok = verify_result(h_C, h_A, M, N, K);
    print_metrics("LDS tiling", lds_ok, lds_ms, M, N, K);

    std::cout << "Speedup (LDS / naive): " << naive_ms / lds_ms << "x\n";

    // ------------------------------------------------------------------
    // Cleanup
    // ------------------------------------------------------------------
    HIP_CHECK(hipFree(d_A));
    HIP_CHECK(hipFree(d_B));
    HIP_CHECK(hipFree(d_C));

    return (naive_ok && lds_ok) ? EXIT_SUCCESS : EXIT_FAILURE;
}
