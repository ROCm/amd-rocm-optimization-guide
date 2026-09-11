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
#include <type_traits>
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

// Tile sizes (same as Step 3 for a fair comparison)
constexpr int BLOCK_TILE_M  = 128;
constexpr int BLOCK_TILE_N  = 128;
constexpr int K_TILE_SIZE   = 16;
constexpr int THREAD_TILE_M = 8;
constexpr int THREAD_TILE_N = 8;
constexpr int BLOCK_DIM_X   = BLOCK_TILE_N / THREAD_TILE_N; // 16
constexpr int BLOCK_DIM_Y   = BLOCK_TILE_M / THREAD_TILE_M; // 16

// ===========================================================================
// TilePolicy concept (C++17, enforced through static_assert in the kernel)
//
// A TilePolicy controls three aspects of the K-loop:
//
//  SharedStorage  – the LDS layout (one or two buffer pairs)
//  prologue()     – called once before the loop; may preload the first tile
//  prefetch()     – called at the top of each iteration with the NEXT tile index;
//                   may issue a load or be a no-op
//  acquire()      – called after prefetch(); ensures the CURRENT tile is ready
//  release()      – called after compute(); protects LDS before the next prefetch
//  get_tile_a()   – returns a pointer to the A tile in the correct LDS buffer
//  get_tile_b_T() – returns a pointer to the transposed B tile in LDS
//  num_buffers    – compile-time constant: 1 (single) or 2 (double)
//
// The kernel loop structure is identical for all policies:
//
//   policy.prologue(...)
//   for t in 0..num_tiles-1:
//       policy.prefetch(t + 1, ...)   // issue next load
//       policy.acquire(t)             // current tile is now safe to read
//       compute(policy.get_tile_a(t), policy.get_tile_b_T(t), ...)
//       policy.release(t)             // done with this buffer
//   write accumulator to global C
//
// SingleBufferTilePolicy:
//   prologue  → no-op
//   prefetch  → called with tile_idx = t+1; loads tile t (= tile_idx-1) into
//               buf[0].  The load must complete before compute begins, which
//               is enforced by acquire()'s __syncthreads().
//   acquire   → __syncthreads() (wait for the prefetch load just issued)
//   release   → __syncthreads() (protect buf[0] before next prefetch)
//
//   The compiler cannot overlap the load with the preceding compute because
//   both operations share the single buffer and are separated by two
//   __syncthreads() calls.  The double-buffer policy provides the second
//   buffer that allows real overlap.
//
// SoftwareDoubleBufferTilePolicy:
//   prologue  → load tile 0 into buf[0], __syncthreads()
//   prefetch  → load tile next_idx into buf[next_idx % 2] (no sync)
//   acquire   → no-op (current buf is ready from prologue or previous release)
//   release   → __syncthreads() (covers both: compute done + prefetch done)
//
// ===========================================================================

// ---------------------------------------------------------------------------
// Single-buffer tile policy
// ---------------------------------------------------------------------------
// [Sphinx single buffer policy start]
struct SingleBufferTilePolicy
{
    static constexpr int num_buffers = 1;

    // LDS holds one pair of tiles.
    struct SharedStorage
    {
        float tile_a[BLOCK_TILE_M][K_TILE_SIZE];
        float tile_b_T[BLOCK_TILE_N][K_TILE_SIZE];
    };

    // No preloading: the first prefetch() call loads tile 0.
    static __device__ void prologue(SharedStorage& /*smem*/,
                                    const float* /*A*/,
                                    const float* /*B*/,
                                    int /*block_row_start*/,
                                    int /*block_col_start*/,
                                    int /*m*/,
                                    int /*n*/,
                                    int /*k*/,
                                    int /*tid*/,
                                    int /*num_threads*/,
                                    int /*num_tiles*/)
    {}

    // Load tile `tile_idx - 1` (the CURRENT tile t) into the single buffer.
    // The kernel calls prefetch(t + 1), so tile_idx = t + 1.  We load
    // tile t = tile_idx - 1 here; acquire() then __syncthreads() to ensure
    // it is visible before compute begins.
    static __device__ void prefetch(SharedStorage&  smem,
                                    const float*    A,
                                    const float*    B,
                                    int             tile_idx,
                                    int             block_row_start,
                                    int             block_col_start,
                                    int             m,
                                    int             n,
                                    int             k,
                                    int             tid,
                                    int             num_threads,
                                    int             /*num_tiles*/)
    {
        // tile_idx = t + 1, so the tile to load is t = tile_idx - 1.
        // The loop runs t in [0, num_tiles), so tile_idx in [1, num_tiles],
        // and load_idx in [0, num_tiles - 1] - always in range.
        const int load_idx = tile_idx - 1;

        const int tile_elems_a = BLOCK_TILE_M * K_TILE_SIZE;
        for(int idx = tid; idx < tile_elems_a; idx += num_threads)
        {
            const int tile_row   = idx / K_TILE_SIZE;
            const int tile_col   = idx % K_TILE_SIZE;
            const int global_row = block_row_start + tile_row;
            const int global_col = load_idx * K_TILE_SIZE + tile_col;
            smem.tile_a[tile_row][tile_col] =
                (global_row < m && global_col < k)
                    ? A[global_row * k + global_col]
                    : 0.0f;
        }

        const int tile_elems_b = BLOCK_TILE_N * K_TILE_SIZE;
        for(int idx = tid; idx < tile_elems_b; idx += num_threads)
        {
            const int tile_col_idx = idx / K_TILE_SIZE;
            const int tile_k_idx   = idx % K_TILE_SIZE;
            const int global_row   = load_idx * K_TILE_SIZE + tile_k_idx;
            const int global_col   = block_col_start + tile_col_idx;
            smem.tile_b_T[tile_col_idx][tile_k_idx] =
                (global_row < k && global_col < n)
                    ? B[global_row * n + global_col]
                    : 0.0f;
        }
    }

    // Wait for the prefetch just issued to complete.
    static __device__ void acquire(int /*tile_idx*/)
    {
        __syncthreads();
    }

    // Protect the buffer: no thread may start the next prefetch until all
    // threads have finished reading the current tile.
    static __device__ void release(int /*tile_idx*/)
    {
        __syncthreads();
    }

    // Always return buffer index 0.
    static __device__ int buf_idx(int /*tile_idx*/)
    {
        return 0;
    }

    static __device__ const float* get_tile_a(const SharedStorage& smem, int /*buf*/)
    {
        return &smem.tile_a[0][0];
    }

    static __device__ const float* get_tile_b_T(const SharedStorage& smem, int /*buf*/)
    {
        return &smem.tile_b_T[0][0];
    }
};
// [Sphinx single buffer policy end]

// ---------------------------------------------------------------------------
// Software double-buffer tile policy
//
// Key idea: while the GPU is computing from buffer A, the memory system is
// filling buffer B.  On the next iteration the roles swap.  There is only one
// __syncthreads() per iteration (in release()), which serves double duty:
//
//   1. It ensures the current compute pass has finished reading from buf[t%2],
//      so the next prefetch may safely overwrite it.
//   2. Because threads issue the global-memory loads for tile t+1 before the
//      sync, and those loads are in-flight while compute runs, the sync also
//      waits for those loads to reach LDS.
//
// The net effect is that global-memory latency is hidden behind compute
// (assuming compute takes longer than the load).
// ---------------------------------------------------------------------------
// [Sphinx double buffer policy start]
struct SoftwareDoubleBufferTilePolicy
{
    static constexpr int num_buffers = 2;

    // LDS holds TWO pairs of tiles, indexed by [buf][...].
    struct SharedStorage
    {
        float tile_a[2][BLOCK_TILE_M][K_TILE_SIZE];
        float tile_b_T[2][BLOCK_TILE_N][K_TILE_SIZE];
    };

    // Load tile 0 into buf[0] and synchronise so that the first iteration
    // can call acquire() as a no-op.
    static __device__ void prologue(SharedStorage&  smem,
                                    const float*    A,
                                    const float*    B,
                                    int             block_row_start,
                                    int             block_col_start,
                                    int             m,
                                    int             n,
                                    int             k,
                                    int             tid,
                                    int             num_threads,
                                    int             /*num_tiles*/)
    {
        const int tile_elems_a = BLOCK_TILE_M * K_TILE_SIZE;
        for(int idx = tid; idx < tile_elems_a; idx += num_threads)
        {
            const int tile_row   = idx / K_TILE_SIZE;
            const int tile_col   = idx % K_TILE_SIZE;
            const int global_row = block_row_start + tile_row;
            const int global_col = tile_col; // tile 0: K offset = 0
            smem.tile_a[0][tile_row][tile_col] =
                (global_row < m && global_col < k)
                    ? A[global_row * k + global_col]
                    : 0.0f;
        }

        const int tile_elems_b = BLOCK_TILE_N * K_TILE_SIZE;
        for(int idx = tid; idx < tile_elems_b; idx += num_threads)
        {
            const int tile_col_idx = idx / K_TILE_SIZE;
            const int tile_k_idx   = idx % K_TILE_SIZE;
            const int global_row   = tile_k_idx; // tile 0: K offset = 0
            const int global_col   = block_col_start + tile_col_idx;
            smem.tile_b_T[0][tile_col_idx][tile_k_idx] =
                (global_row < k && global_col < n)
                    ? B[global_row * n + global_col]
                    : 0.0f;
        }

        // Ensure tile 0 is fully in LDS before the main loop begins.
        __syncthreads();
    }

    // Issue the load for tile `tile_idx` into the alternate buffer.
    // This load runs concurrently with the compute on the current buffer.
    static __device__ void prefetch(SharedStorage&  smem,
                                    const float*    A,
                                    const float*    B,
                                    int             tile_idx,
                                    int             block_row_start,
                                    int             block_col_start,
                                    int             m,
                                    int             n,
                                    int             k,
                                    int             tid,
                                    int             num_threads,
                                    int             num_tiles)
    {
        if(tile_idx >= num_tiles)
        {
            return;
        }

        // Write into the buffer that is NOT currently being read.
        const int dst_buf = tile_idx % 2;

        const int tile_elems_a = BLOCK_TILE_M * K_TILE_SIZE;
        for(int idx = tid; idx < tile_elems_a; idx += num_threads)
        {
            const int tile_row   = idx / K_TILE_SIZE;
            const int tile_col   = idx % K_TILE_SIZE;
            const int global_row = block_row_start + tile_row;
            const int global_col = tile_idx * K_TILE_SIZE + tile_col;
            smem.tile_a[dst_buf][tile_row][tile_col] =
                (global_row < m && global_col < k)
                    ? A[global_row * k + global_col]
                    : 0.0f;
        }

        const int tile_elems_b = BLOCK_TILE_N * K_TILE_SIZE;
        for(int idx = tid; idx < tile_elems_b; idx += num_threads)
        {
            const int tile_col_idx = idx / K_TILE_SIZE;
            const int tile_k_idx   = idx % K_TILE_SIZE;
            const int global_row   = tile_idx * K_TILE_SIZE + tile_k_idx;
            const int global_col   = block_col_start + tile_col_idx;
            smem.tile_b_T[dst_buf][tile_col_idx][tile_k_idx] =
                (global_row < k && global_col < n)
                    ? B[global_row * n + global_col]
                    : 0.0f;
        }
        // No sync here: the load runs concurrently with compute on the other buffer.
    }

    // The current buffer is already filled (by prologue or the previous release).
    // No synchronisation is needed here.
    static __device__ void acquire(int /*tile_idx*/) {}

    // Single __syncthreads() serves dual purpose:
    //   - Ensures every thread has finished reading buf[tile_idx % 2]
    //     (so the next prefetch may safely overwrite it).
    //   - Ensures the prefetch issued at the top of this iteration
    //     (loading into buf[(tile_idx+1) % 2]) has completed.
    static __device__ void release(int /*tile_idx*/)
    {
        __syncthreads();
    }

    static __device__ int buf_idx(int tile_idx)
    {
        return tile_idx % 2;
    }

    static __device__ const float* get_tile_a(const SharedStorage& smem, int buf)
    {
        return &smem.tile_a[buf][0][0];
    }

    static __device__ const float* get_tile_b_T(const SharedStorage& smem, int buf)
    {
        return &smem.tile_b_T[buf][0][0];
    }
};
// [Sphinx double buffer policy end]

// ---------------------------------------------------------------------------
// C++17 static_assert validation for TilePolicy requirements.
// Placed in a helper struct so it can be instantiated from the kernel template
// without needing C++20 concepts.
// ---------------------------------------------------------------------------
// [Sphinx tile policy static assert start]
template<typename TilePolicy>
struct TilePolicyTraits
{
    static_assert(TilePolicy::num_buffers == 1 || TilePolicy::num_buffers == 2,
                  "TilePolicy::num_buffers must be 1 (single buffer) or 2 (double buffer)");

    // SharedStorage must be a non-empty, trivially-destructible aggregate
    // that the compiler can place in __shared__ memory.
    static_assert(sizeof(typename TilePolicy::SharedStorage) > 0,
                  "TilePolicy::SharedStorage must have non-zero size");
    static_assert(std::is_trivially_destructible_v<typename TilePolicy::SharedStorage>,
                  "TilePolicy::SharedStorage must be trivially destructible "
                  "(required for __shared__ variables)");
};
// [Sphinx tile policy static assert end]

// ---------------------------------------------------------------------------
// Template kernel – identical loop structure for all TilePolicy variants.
// ---------------------------------------------------------------------------
// [Sphinx double buffer kernel start]
template<typename TilePolicy>
__global__ void matrix_multiply_buffered(const float* __restrict__ A,
                                         const float* __restrict__ B,
                                         float* __restrict__ C,
                                         int m,
                                         int n,
                                         int k)
{
    // Instantiate the trait checker; all static_asserts fire at compile time.
    (void)TilePolicyTraits<TilePolicy>{};

    __shared__ typename TilePolicy::SharedStorage smem;

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

    // [Sphinx prologue start]
    // Prologue: SingleBuffer → no-op; DoubleBuffer → preload tile 0 + sync.
    TilePolicy::prologue(smem, A, B, block_row_start, block_col_start,
                         m, n, k, tid, num_threads, num_tiles);
    // [Sphinx prologue end]

    for(int t = 0; t < num_tiles; ++t)
    {
        // [Sphinx prefetch acquire start]
        // Issue the load for the NEXT tile into the alternate buffer (or the
        // single buffer for SingleBufferTilePolicy).
        TilePolicy::prefetch(smem, A, B, t + 1, block_row_start, block_col_start,
                             m, n, k, tid, num_threads, num_tiles);

        // Ensure the CURRENT tile (tile t) is fully in LDS before reading it.
        // SingleBuffer: __syncthreads() after the load just issued above.
        // DoubleBuffer: no-op (tile t was already in LDS from the previous release).
        TilePolicy::acquire(t);
        // [Sphinx prefetch acquire end]

        // Outer-product accumulation from LDS (register tiling, same as Step 3).
        const float* ta  = TilePolicy::get_tile_a(smem, TilePolicy::buf_idx(t));
        const float* tb_T = TilePolicy::get_tile_b_T(smem, TilePolicy::buf_idx(t));

        for(int ki = 0; ki < K_TILE_SIZE; ++ki)
        {
            float a_frag[THREAD_TILE_M];
            #pragma unroll
            for(int i = 0; i < THREAD_TILE_M; ++i)
            {
                // tile_a is laid out as [BLOCK_TILE_M][K_TILE_SIZE]
                a_frag[i] = ta[(thread_row + i) * K_TILE_SIZE + ki];
            }

            float b_frag[THREAD_TILE_N];
            #pragma unroll
            for(int j = 0; j < THREAD_TILE_N; ++j)
            {
                // tile_b_T is laid out as [BLOCK_TILE_N][K_TILE_SIZE]
                b_frag[j] = tb_T[(thread_col + j) * K_TILE_SIZE + ki];
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

        // [Sphinx release start]
        // SingleBuffer: second __syncthreads() – protects LDS before prefetch
        //               overwrites it in the next iteration.
        // DoubleBuffer: single __syncthreads() – ensures:
        //   (a) compute has finished reading buf[t%2], so the next prefetch
        //       may safely write into it.
        //   (b) the prefetch issued above (for tile t+1 into buf[(t+1)%2])
        //       has completed before the next iteration's acquire() returns.
        TilePolicy::release(t);
        // [Sphinx release end]
    }

    // Write accumulator to global C
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
}
// [Sphinx double buffer kernel end]

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
              << "Thread tile  : " << THREAD_TILE_M << "x" << THREAD_TILE_N << "\n\n";

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
    // Single-buffer variant (same algorithm as Step 3, via policy)
    // ------------------------------------------------------------------
    {
        auto launch = [&]()
        {
            matrix_multiply_buffered<SingleBufferTilePolicy>
                <<<grid, block>>>(d_A, d_B, d_C, M, N, K);
            HIP_CHECK(hipGetLastError());
        };

        const float ms = time_kernel_ms(launch, WARMUP_RUNS, TIMING_RUNS);
        HIP_CHECK(hipMemcpy(h_C.data(), d_C, sizeof(float) * M * N, hipMemcpyDeviceToHost));
        const bool ok = verify_result(h_C, h_A, M, N, K);
        print_metrics("Single buffer (baseline)", ok, ms, M, N, K);
    }

    // ------------------------------------------------------------------
    // Software double-buffer variant
    // ------------------------------------------------------------------
    {
        auto launch = [&]()
        {
            matrix_multiply_buffered<SoftwareDoubleBufferTilePolicy>
                <<<grid, block>>>(d_A, d_B, d_C, M, N, K);
            HIP_CHECK(hipGetLastError());
        };

        const float ms = time_kernel_ms(launch, WARMUP_RUNS, TIMING_RUNS);
        HIP_CHECK(hipMemcpy(h_C.data(), d_C, sizeof(float) * M * N, hipMemcpyDeviceToHost));
        const bool ok = verify_result(h_C, h_A, M, N, K);
        print_metrics("Software double buffer", ok, ms, M, N, K);
    }

    // ------------------------------------------------------------------
    // Cleanup
    // ------------------------------------------------------------------
    HIP_CHECK(hipFree(d_A));
    HIP_CHECK(hipFree(d_B));
    HIP_CHECK(hipFree(d_C));

    return EXIT_SUCCESS;
}
