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
// Register-tiling parameters
//
// The block tile (BLOCK_TILE_M x BLOCK_TILE_N) is the region of C that one
// thread block computes.  It is sub-divided among threads so that each thread
// computes a THREAD_TILE_M x THREAD_TILE_N sub-tile of C entirely in registers.
//
// K is traversed in strips of K_TILE_SIZE to bound LDS usage.
//
// Thread-block shape: (BLOCK_TILE_N / THREAD_TILE_N) x (BLOCK_TILE_M / THREAD_TILE_M)
//   = (128/8) x (128/8) = 16 x 16 = 256 threads per block.
//
// LDS usage per block:
//   tile_a: BLOCK_TILE_M * K_TILE_SIZE * 4 = 128 * 16 * 4 = 8 192 bytes
//   tile_b: K_TILE_SIZE * BLOCK_TILE_N * 4 = 16 * 128 * 4 = 8 192 bytes (transposed)
//   Total: 16 384 bytes = 16 KiB  (well within the 32–64 KiB LDS per block)
// ---------------------------------------------------------------------------
// [Sphinx register tiling params start]
constexpr int BLOCK_TILE_M  = 128; // Rows of C computed by one block
constexpr int BLOCK_TILE_N  = 128; // Columns of C computed by one block
constexpr int K_TILE_SIZE   = 16;  // K-strip width loaded into LDS each iteration
constexpr int THREAD_TILE_M = 8;   // Rows of C computed by one thread
constexpr int THREAD_TILE_N = 8;   // Columns of C computed by one thread
// [Sphinx register tiling params end]

// Derived block dimensions (number of threads along each axis)
constexpr int BLOCK_DIM_X = BLOCK_TILE_N / THREAD_TILE_N; // 16
constexpr int BLOCK_DIM_Y = BLOCK_TILE_M / THREAD_TILE_M; // 16

// ---------------------------------------------------------------------------
// LDS baseline (from Step 2) – included as a performance reference.
// ---------------------------------------------------------------------------
constexpr int LDS_TILE_SIZE = 16;

__global__ void matrix_multiply_lds(const float* __restrict__ A,
                                     const float* __restrict__ B,
                                     float* __restrict__ C,
                                     int m,
                                     int n,
                                     int k)
{
    __shared__ float tile_a[LDS_TILE_SIZE][LDS_TILE_SIZE];
    __shared__ float tile_b[LDS_TILE_SIZE][LDS_TILE_SIZE];

    const int tx  = threadIdx.x;
    const int ty  = threadIdx.y;
    const int row = blockIdx.y * LDS_TILE_SIZE + ty;
    const int col = blockIdx.x * LDS_TILE_SIZE + tx;

    float sum = 0.0f;
    for(int t = 0; t < (k + LDS_TILE_SIZE - 1) / LDS_TILE_SIZE; ++t)
    {
        const int a_col = t * LDS_TILE_SIZE + tx;
        tile_a[ty][tx]  = (row < m && a_col < k) ? A[row * k + a_col] : 0.0f;
        const int b_row = t * LDS_TILE_SIZE + ty;
        tile_b[ty][tx]  = (b_row < k && col < n) ? B[b_row * n + col] : 0.0f;
        __syncthreads();
        for(int i = 0; i < LDS_TILE_SIZE; ++i)
            sum += tile_a[ty][i] * tile_b[i][tx];
        __syncthreads();
    }
    if(row < m && col < n)
        C[row * n + col] = sum;
}

// ---------------------------------------------------------------------------
// Register-tiled matrix-multiplication kernel: C = A * B  (FP32, row-major)
//
// Why register tiling?
// --------------------
// In the LDS-tiled kernel (Step 2) each thread computes a single output
// element, reading TILE_SIZE elements from tile_a and TILE_SIZE elements from
// tile_b per K-strip.  Two improvements are possible:
//
//  1. Compute multiple output elements per thread.  If each thread computes a
//     TM x TN sub-tile of C, it amortises the LDS load cost across TM*TN
//     outputs instead of 1.  The outer product of a column fragment of A
//     (length TM) and a row fragment of B (length TN) produces a full TM x TN
//     block of C contributions in TM + TN LDS reads instead of TM*TN reads.
//
//  2. Keep those fragments in registers.  Registers are the fastest on-chip
//     storage and have no bank-conflict semantics.
//
// Transposed tile_b layout
// ------------------------
// In the LDS kernel, tile_b is stored row-major: tile_b[k_idx][col].
// The outer-product computation reads tile_b[k_idx][thread_col + j] for each
// j in 0..TN-1.  If TN > 1, this is a sequential read with stride 1 along
// the column axis – still row-major, no issue.
//
// However, tile_b is *loaded* as B[b_row][col], which is already row-major.
// Switching to a *column-major* layout for tile_b – stored as
// tile_b_T[col][k_idx] – lets us read the B fragment as a contiguous column
// during load and avoids the bank-conflict risk that arises when TILE_SIZE
// equals the LDS bank count.  During the compute phase we access
// tile_b_T[thread_col + j][k_idx], which is again a stride-1 read (now along
// the first dimension).
//
// Register accumulation pattern (outer product)
// ---------------------------------------------
// For each k-index within the K-strip:
//
//   Load a_frag[0..TM-1]  = tile_a  [thread_row + i][k_idx]   (column of A tile)
//   Load b_frag[0..TN-1]  = tile_b_T[thread_col + j][k_idx]   (row of B tile, transposed)
//
//   for i in 0..TM-1:
//     for j in 0..TN-1:
//       acc[i][j] += a_frag[i] * b_frag[j]       (outer product in registers)
//
// Each LDS element is read once into a register fragment and reused TN (or TM)
// times.  The register-to-register FMA instructions replace repeated LDS reads
// with much faster register file operations.
// ---------------------------------------------------------------------------
// [Sphinx register tiling kernel start]
__global__ void matrix_multiply_register_tiling(const float* __restrict__ A,
                                                  const float* __restrict__ B,
                                                  float* __restrict__ C,
                                                  int m,
                                                  int n,
                                                  int k)
{
    // [Sphinx register tiling shared memory start]
    // tile_a: row-major strip of A, shape [BLOCK_TILE_M][K_TILE_SIZE]
    __shared__ float tile_a[BLOCK_TILE_M][K_TILE_SIZE];
    // tile_b_T: column-major strip of B (transposed in LDS), shape [BLOCK_TILE_N][K_TILE_SIZE]
    // Storing transposed eliminates the bank-conflict risk that arises when
    // TILE_SIZE equals the LDS bank count, and produces stride-1 reads during
    // both the load phase and the compute phase.
    __shared__ float tile_b_T[BLOCK_TILE_N][K_TILE_SIZE];
    // [Sphinx register tiling shared memory end]

    const int tx  = threadIdx.x; // 0..BLOCK_DIM_X-1  (column tile index)
    const int ty  = threadIdx.y; // 0..BLOCK_DIM_Y-1  (row tile index)
    const int tid = ty * blockDim.x + tx;

    // First row / column of the output sub-tile owned by this thread
    const int thread_row = ty * THREAD_TILE_M; // within the block tile
    const int thread_col = tx * THREAD_TILE_N;

    // Top-left corner of the block tile in the global matrix
    const int block_row_start = blockIdx.y * BLOCK_TILE_M;
    const int block_col_start = blockIdx.x * BLOCK_TILE_N;

    // Register accumulator: THREAD_TILE_M x THREAD_TILE_N partial results
    float acc[THREAD_TILE_M][THREAD_TILE_N] = {};

    const int num_tiles    = (k + K_TILE_SIZE - 1) / K_TILE_SIZE;
    const int num_threads  = BLOCK_DIM_X * BLOCK_DIM_Y;

    for(int t = 0; t < num_tiles; ++t)
    {
        // [Sphinx register tiling load phase start]
        // --- Cooperative load of tile_a (BLOCK_TILE_M x K_TILE_SIZE) ---
        // Each thread loads one or more elements by striding through the flat
        // tile index space with step = num_threads.
        {
            const int tile_elems = BLOCK_TILE_M * K_TILE_SIZE;
            for(int idx = tid; idx < tile_elems; idx += num_threads)
            {
                const int tile_row  = idx / K_TILE_SIZE;
                const int tile_col  = idx % K_TILE_SIZE;
                const int global_row = block_row_start + tile_row;
                const int global_col = t * K_TILE_SIZE + tile_col;
                tile_a[tile_row][tile_col] =
                    (global_row < m && global_col < k)
                        ? A[global_row * k + global_col]
                        : 0.0f;
            }
        }

        // --- Cooperative load of tile_b_T (BLOCK_TILE_N x K_TILE_SIZE, transposed) ---
        // We load B[b_row][b_col] into tile_b_T[b_col - block_col_start][b_row - t*K].
        // Equivalently: thread loads the element at (col_within_tile, k_within_tile)
        // of the transposed tile, reading B in row-major order.
        {
            const int tile_elems = BLOCK_TILE_N * K_TILE_SIZE;
            for(int idx = tid; idx < tile_elems; idx += num_threads)
            {
                const int tile_col_idx = idx / K_TILE_SIZE; // column within block tile
                const int tile_k_idx   = idx % K_TILE_SIZE; // k index within strip
                const int global_row   = t * K_TILE_SIZE + tile_k_idx;
                const int global_col   = block_col_start + tile_col_idx;
                tile_b_T[tile_col_idx][tile_k_idx] =
                    (global_row < k && global_col < n)
                        ? B[global_row * n + global_col]
                        : 0.0f;
            }
        }

        __syncthreads();
        // [Sphinx register tiling load phase end]

        // [Sphinx register tiling compute phase start]
        // Outer-product accumulation entirely in registers.
        // For each k-index within the strip:
        //   - Load a column fragment of tile_a into registers (a_frag).
        //   - Load a row fragment of tile_b_T into registers (b_frag).
        //   - Accumulate the TM x TN outer product into acc.
        for(int ki = 0; ki < K_TILE_SIZE; ++ki)
        {
            float a_frag[THREAD_TILE_M];
            #pragma unroll
            for(int i = 0; i < THREAD_TILE_M; ++i)
            {
                a_frag[i] = tile_a[thread_row + i][ki];
            }

            float b_frag[THREAD_TILE_N];
            #pragma unroll
            for(int j = 0; j < THREAD_TILE_N; ++j)
            {
                b_frag[j] = tile_b_T[thread_col + j][ki];
            }

            #pragma unroll
            for(int i = 0; i < THREAD_TILE_M; ++i)
            {
                #pragma unroll
                for(int j = 0; j < THREAD_TILE_N; ++j)
                {
                    acc[i][j] += a_frag[i] * b_frag[j];
                }
            }
        }

        __syncthreads();
        // [Sphinx register tiling compute phase end]
    }

    // [Sphinx register tiling store start]
    // Write the register accumulator to global memory.
    // Each store is a coalesced write: threads with consecutive tx values write
    // to consecutive columns (thread_col strides by THREAD_TILE_N = 8, but the
    // inner j loop produces 8 consecutive output columns per thread, so adjacent
    // threads in x cover adjacent 8-wide strips → effectively coalesced).
    #pragma unroll
    for(int i = 0; i < THREAD_TILE_M; ++i)
    {
        #pragma unroll
        for(int j = 0; j < THREAD_TILE_N; ++j)
        {
            const int out_row = block_row_start + thread_row + i;
            const int out_col = block_col_start + thread_col + j;
            if(out_row < m && out_col < n)
            {
                C[out_row * n + out_col] = acc[i][j];
            }
        }
    }
    // [Sphinx register tiling store end]
}
// [Sphinx register tiling kernel end]

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
              << "Block dims   : " << BLOCK_DIM_X << "x" << BLOCK_DIM_Y << " threads\n\n";

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

    // ------------------------------------------------------------------
    // LDS baseline
    // ------------------------------------------------------------------
    {
        const dim3 block(LDS_TILE_SIZE, LDS_TILE_SIZE);
        const dim3 grid((N + LDS_TILE_SIZE - 1) / LDS_TILE_SIZE,
                        (M + LDS_TILE_SIZE - 1) / LDS_TILE_SIZE);

        auto launch = [&]()
        {
            matrix_multiply_lds<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
            HIP_CHECK(hipGetLastError());
        };

        const float ms = time_kernel_ms(launch, WARMUP_RUNS, TIMING_RUNS);
        HIP_CHECK(hipMemcpy(h_C.data(), d_C, sizeof(float) * M * N, hipMemcpyDeviceToHost));
        const bool ok = verify_result(h_C, h_A, M, N, K);
        print_metrics("LDS tiling (baseline)", ok, ms, M, N, K);
    }

    // ------------------------------------------------------------------
    // Register-tiled kernel
    // ------------------------------------------------------------------
    {
        const dim3 block(BLOCK_DIM_X, BLOCK_DIM_Y);
        const dim3 grid((N + BLOCK_TILE_N - 1) / BLOCK_TILE_N,
                        (M + BLOCK_TILE_M - 1) / BLOCK_TILE_M);

        auto launch = [&]()
        {
            matrix_multiply_register_tiling<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
            HIP_CHECK(hipGetLastError());
        };

        const float ms = time_kernel_ms(launch, WARMUP_RUNS, TIMING_RUNS);
        HIP_CHECK(hipMemcpy(h_C.data(), d_C, sizeof(float) * M * N, hipMemcpyDeviceToHost));
        const bool ok = verify_result(h_C, h_A, M, N, K);
        print_metrics("Register tiling", ok, ms, M, N, K);
    }

    // ------------------------------------------------------------------
    // Cleanup
    // ------------------------------------------------------------------
    HIP_CHECK(hipFree(d_A));
    HIP_CHECK(hipFree(d_B));
    HIP_CHECK(hipFree(d_C));

    return EXIT_SUCCESS;
}
