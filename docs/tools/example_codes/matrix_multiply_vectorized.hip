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
#include <cstdint>
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

// Tile sizes (same as Steps 3–4)
constexpr int BLOCK_TILE_M  = 128;
constexpr int BLOCK_TILE_N  = 128;
constexpr int K_TILE_SIZE   = 16;
constexpr int THREAD_TILE_M = 8;
constexpr int THREAD_TILE_N = 8;
constexpr int BLOCK_DIM_X   = BLOCK_TILE_N / THREAD_TILE_N; // 16
constexpr int BLOCK_DIM_Y   = BLOCK_TILE_M / THREAD_TILE_M; // 16

// ===========================================================================
// Vectorized load policies
//
// Replacing N scalar 32-bit loads with a single vector load (float2 = 64-bit,
// float4 = 128-bit) reduces the number of load instructions issued per thread.
// This benefits performance in two ways:
//
//   1. Instruction-count reduction: fewer instructions ⇒ less instruction fetch
//      and dispatch overhead.
//
//   2. Memory-system efficiency: wider loads saturate the memory bus in fewer
//      transactions, improving effective bandwidth when the access pattern is
//      coalesced and the base pointer is sufficiently aligned.
//
// Alignment requirements
// ----------------------
// A vector load of width W bytes requires the source address to be W-byte
// aligned.  For float2 (8 bytes) the address must satisfy addr % 8 == 0.
// For float4 (16 bytes) it must satisfy addr % 16 == 0.
//
// In this example the matrices are allocated through hipMalloc, which returns
// 256-byte-aligned pointers, so the *base* pointer satisfies both requirements.
// Each thread must additionally ensure that its *per-thread offset* preserves
// the alignment.  This is guaranteed when:
//
//   float2: the number of elements loaded before this thread's chunk is a
//           multiple of 2 (i.e. tid * ELEMENTS_PER_THREAD % 2 == 0).
//   float4: the same count must be a multiple of 4.
//
// In the cooperative load below, threads stride through the tile treated as
// a flat 1-D array of vector elements.  The flat vector index vi maps to
// (row, col) in the tile via:
//   flat_elem = vi * LoadWidth
//   row       = flat_elem / K_TILE_SIZE
//   col       = flat_elem % K_TILE_SIZE
//
// Because K_TILE_SIZE % LoadWidth == 0, a vector load never crosses a row
// boundary.  Each load reads LoadWidth contiguous floats within a single
// row of A, which IS contiguous in global memory (row-major, stride K).
//
// For float4 (LoadWidth = 4) this requires:
//   K_TILE_SIZE % 4 == 0                     (16 % 4 == 0  ✓)
//   BLOCK_TILE_M * K_TILE_SIZE % 4 == 0      (2048 % 4 == 0  ✓)
//
// If these conditions are not met at compile time, the static_asserts in
// load_tiles will fire.  Runtime alignment of the base pointer is checked
// in main() via a simple modulo test.
//
// The static_assert below validates the compile-time conditions.  Runtime
// alignment of the base pointer is checked in main() via a simple modulo test.
// ===========================================================================

// ---------------------------------------------------------------------------
// Compile-time load width selector.
// Width is the number of floats loaded per vector instruction (1, 2, or 4).
// ---------------------------------------------------------------------------
// [Sphinx vector type start]
template<int Width>
struct VectorType;

template<>
struct VectorType<1>
{
    using type                      = float;
    static constexpr int width      = 1;
    static constexpr int byte_align = alignof(float); // 4 bytes
};

template<>
struct VectorType<2>
{
    using type                      = float2;
    static constexpr int width      = 2;
    static constexpr int byte_align = 8; // float2 must be 8-byte aligned
};

template<>
struct VectorType<4>
{
    using type                      = float4;
    static constexpr int width      = 4;
    static constexpr int byte_align = 16; // float4 must be 16-byte aligned
};
// [Sphinx vector type end]

// ---------------------------------------------------------------------------
// Vectorized cooperative tile load (global → LDS).
//
// The tile is treated as a 1-D array of `VectorType<Width>::type` elements.
// Each thread loads (tile_elems / Width / num_threads) vectors per call,
// striding by num_threads vectors across the flat tile.
//
// The target LDS array is also addressed as a plain float* (via reinterpret),
// so no extra LDS layout change is needed: the LDS tile remains
// [BLOCK_TILE_M][K_TILE_SIZE] for tile_a and [BLOCK_TILE_N][K_TILE_SIZE] for
// tile_b_T, in row-major order.
// ---------------------------------------------------------------------------
// [Sphinx vector load function start]
template<int Width>
__device__ void vectorized_load(float* __restrict__       dst,
                                const float* __restrict__ src,
                                int                       tile_elems,
                                int                       tid,
                                int                       num_threads)
{
    using Vec = typename VectorType<Width>::type;

    static_assert(sizeof(Vec) == Width * sizeof(float),
                  "Vector type size mismatch");

    // Number of complete vector elements in the tile.
    // The static_asserts in the kernel guarantee tile_elems % Width == 0.
    const int vec_elems = tile_elems / Width;

    const Vec* src_vec = reinterpret_cast<const Vec*>(src);
    Vec*       dst_vec = reinterpret_cast<Vec*>(dst);

    for(int idx = tid; idx < vec_elems; idx += num_threads)
    {
        dst_vec[idx] = src_vec[idx];
    }
}
// [Sphinx vector load function end]

// ---------------------------------------------------------------------------
// Vectorized double-buffer tile policy parameterised by load width.
// Inherits the same prologue / prefetch / acquire / release protocol as Step 4.
// ---------------------------------------------------------------------------
// [Sphinx vectorized policy start]
template<int LoadWidth>
struct VectorizedDoubleBufferTilePolicy
{
    static constexpr int num_buffers      = 2;
    static constexpr int elements_per_vec = LoadWidth;

    struct SharedStorage
    {
        float tile_a[2][BLOCK_TILE_M][K_TILE_SIZE];
        float tile_b_T[2][BLOCK_TILE_N][K_TILE_SIZE];
    };

    static __device__ void load_tiles(SharedStorage&  smem,
                                      int             dst_buf,
                                      const float*    A,
                                      const float*    B,
                                      int             tile_idx,
                                      int             block_row_start,
                                      int             block_col_start,
                                      int             m,
                                      int             n,
                                      int             k,
                                      int             tid,
                                      int             num_threads)
    {
        // Cooperative flat-strided load: every thread strides through
        // the flat element array of each tile.  For tile_a the elements
        // within each row are contiguous in global memory (row-major A,
        // stride K), so we vectorise along the column (k) dimension.
        // tile_b_T gathers from stride-N columns of B, which is not
        // contiguous — scalar fallback is used there.

        const int tile_k_start = tile_idx * K_TILE_SIZE;

        static_assert(K_TILE_SIZE % LoadWidth == 0,
                      "K_TILE_SIZE must be divisible by the vector load width");
        static_assert(BLOCK_TILE_M * K_TILE_SIZE % LoadWidth == 0,
                      "tile_a element count must be divisible by vector load width");
        static_assert(BLOCK_TILE_N * K_TILE_SIZE % LoadWidth == 0,
                      "tile_b_T element count must be divisible by vector load width");

        // tile_a load: flat-strided across BLOCK_TILE_M * K_TILE_SIZE elements.
        // Each vector load reads LoadWidth contiguous floats.  Because
        // K_TILE_SIZE % LoadWidth == 0, a vector load never crosses a row
        // boundary in the LDS tile — the flat vector index maps cleanly
        // to (row, col) within the tile.
        {
            using Vec                       = typename VectorType<LoadWidth>::type;
            constexpr int tile_a_elems      = BLOCK_TILE_M * K_TILE_SIZE;
            constexpr int tile_a_vec_elems  = tile_a_elems / LoadWidth;

            Vec*       dst_vec = reinterpret_cast<Vec*>(&smem.tile_a[dst_buf][0][0]);

            for(int vi = tid; vi < tile_a_vec_elems; vi += num_threads)
            {
                // Convert flat vector index to (row, col) in the tile.
                const int flat_elem  = vi * LoadWidth;
                const int tile_row   = flat_elem / K_TILE_SIZE;
                const int tile_col   = flat_elem % K_TILE_SIZE;
                const int global_row = block_row_start + tile_row;
                const int global_col = tile_k_start + tile_col;

                if(global_row < m && global_col + LoadWidth - 1 < k)
                {
                    const Vec* src_vec = reinterpret_cast<const Vec*>(
                        A + global_row * k + global_col);
                    dst_vec[vi] = *src_vec;
                }
                else
                {
                    // Out-of-bounds: zero-fill element by element.
                    float* dst_scalar = &smem.tile_a[dst_buf][0][0] + flat_elem;
                    for(int w = 0; w < LoadWidth; ++w)
                    {
                        const int gr = global_row;
                        const int gc = global_col + w;
                        dst_scalar[w] = (gr < m && gc < k)
                                            ? A[gr * k + gc]
                                            : 0.0f;
                    }
                }
            }
        }

        // tile_b_T load: flat-strided, scalar.
        // B is row-major and we need column tile_col of the K-strip:
        //   B[tile_k_start + ki][block_col_start + tile_col]
        // These elements are stride-N apart in global memory (not contiguous),
        // so vectorisation is not possible here.  Architecture-specific
        // intrinsics can address this in follow-up sections.
        //
        // tile_b_T is stored [BLOCK_TILE_N][K_TILE_SIZE] in LDS, so the flat
        // index maps to:
        //   tile_col = flat_idx / K_TILE_SIZE
        //   ki       = flat_idx % K_TILE_SIZE
        {
            constexpr int tile_b_elems = BLOCK_TILE_N * K_TILE_SIZE;

            for(int idx = tid; idx < tile_b_elems; idx += num_threads)
            {
                const int tile_col   = idx / K_TILE_SIZE;
                const int ki         = idx % K_TILE_SIZE;
                const int global_row = tile_k_start + ki;
                const int global_col = block_col_start + tile_col;

                smem.tile_b_T[dst_buf][tile_col][ki] =
                    (global_row < k && global_col < n)
                        ? B[global_row * n + global_col]
                        : 0.0f;
            }
        }
    }

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
        load_tiles(smem, 0, A, B, 0, block_row_start, block_col_start,
                   m, n, k, tid, num_threads);
        __syncthreads();
    }

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
            return;
        load_tiles(smem, tile_idx % 2, A, B, tile_idx, block_row_start,
                   block_col_start, m, n, k, tid, num_threads);
    }

    static __device__ void acquire(int /*tile_idx*/) {}

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
// [Sphinx vectorized policy end]

// ---------------------------------------------------------------------------
// Template kernel (same body as Step 4; policy is the only variable).
// ---------------------------------------------------------------------------
template<typename TilePolicy>
__global__ void matrix_multiply_vectorized(const float* __restrict__ A,
                                           const float* __restrict__ B,
                                           float* __restrict__ C,
                                           int m,
                                           int n,
                                           int k)
{
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

    TilePolicy::prologue(smem, A, B, block_row_start, block_col_start,
                         m, n, k, tid, num_threads, num_tiles);

    for(int t = 0; t < num_tiles; ++t)
    {
        TilePolicy::prefetch(smem, A, B, t + 1, block_row_start, block_col_start,
                             m, n, k, tid, num_threads, num_tiles);
        TilePolicy::acquire(t);

        const float* ta   = TilePolicy::get_tile_a(smem, TilePolicy::buf_idx(t));
        const float* tb_T = TilePolicy::get_tile_b_T(smem, TilePolicy::buf_idx(t));

        for(int ki = 0; ki < K_TILE_SIZE; ++ki)
        {
            float a_frag[THREAD_TILE_M];
            #pragma unroll
            for(int i = 0; i < THREAD_TILE_M; ++i)
            {
                a_frag[i] = ta[(thread_row + i) * K_TILE_SIZE + ki];
            }

            float b_frag[THREAD_TILE_N];
            #pragma unroll
            for(int j = 0; j < THREAD_TILE_N; ++j)
            {
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

        TilePolicy::release(t);
    }

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

// ---------------------------------------------------------------------------
// Runtime alignment check helper
// ---------------------------------------------------------------------------
// [Sphinx alignment check start]
void check_alignment(const void* ptr, int required_bytes, const char* name)
{
    const uintptr_t addr = reinterpret_cast<uintptr_t>(ptr);
    if(addr % static_cast<uintptr_t>(required_bytes) != 0)
    {
        std::cerr << "WARNING: " << name << " pointer " << ptr
                  << " is not " << required_bytes << "-byte aligned. "
                  << "Vectorised loads of width " << required_bytes
                  << " bytes will produce undefined behaviour.\n";
    }
}
// [Sphinx alignment check end]

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

    // Verify alignment: hipMalloc guarantees ≥ 256-byte alignment, so both
    // float2 (8-byte) and float4 (16-byte) requirements are satisfied here.
    check_alignment(d_A, VectorType<4>::byte_align, "d_A");
    check_alignment(d_B, VectorType<4>::byte_align, "d_B");

    HIP_CHECK(hipMemcpy(d_A, h_A.data(), sizeof(float) * M * K, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_B, h_B.data(), sizeof(float) * K * N, hipMemcpyHostToDevice));

    const dim3 block(BLOCK_DIM_X, BLOCK_DIM_Y);
    const dim3 grid((N + BLOCK_TILE_N - 1) / BLOCK_TILE_N,
                    (M + BLOCK_TILE_M - 1) / BLOCK_TILE_M);

    // ------------------------------------------------------------------
    // Scalar load (float1) – same as Step 4 double buffer, for comparison
    // ------------------------------------------------------------------
    {
        using Policy = VectorizedDoubleBufferTilePolicy<1>;
        auto launch  = [&]()
        {
            matrix_multiply_vectorized<Policy><<<grid, block>>>(d_A, d_B, d_C, M, N, K);
            HIP_CHECK(hipGetLastError());
        };
        const float ms = time_kernel_ms(launch, WARMUP_RUNS, TIMING_RUNS);
        HIP_CHECK(hipMemcpy(h_C.data(), d_C, sizeof(float) * M * N, hipMemcpyDeviceToHost));
        const bool ok = verify_result(h_C, h_A, M, N, K);
        print_metrics("Scalar load  (float1, baseline)", ok, ms, M, N, K);
    }

    // ------------------------------------------------------------------
    // float2 loads (64-bit, 8-byte aligned)
    // ------------------------------------------------------------------
    {
        using Policy = VectorizedDoubleBufferTilePolicy<2>;
        auto launch  = [&]()
        {
            matrix_multiply_vectorized<Policy><<<grid, block>>>(d_A, d_B, d_C, M, N, K);
            HIP_CHECK(hipGetLastError());
        };
        const float ms = time_kernel_ms(launch, WARMUP_RUNS, TIMING_RUNS);
        HIP_CHECK(hipMemcpy(h_C.data(), d_C, sizeof(float) * M * N, hipMemcpyDeviceToHost));
        const bool ok = verify_result(h_C, h_A, M, N, K);
        print_metrics("Vector load  (float2, 64-bit)  ", ok, ms, M, N, K);
    }

    // ------------------------------------------------------------------
    // float4 loads (128-bit, 16-byte aligned)
    // ------------------------------------------------------------------
    {
        using Policy = VectorizedDoubleBufferTilePolicy<4>;
        auto launch  = [&]()
        {
            matrix_multiply_vectorized<Policy><<<grid, block>>>(d_A, d_B, d_C, M, N, K);
            HIP_CHECK(hipGetLastError());
        };
        const float ms = time_kernel_ms(launch, WARMUP_RUNS, TIMING_RUNS);
        HIP_CHECK(hipMemcpy(h_C.data(), d_C, sizeof(float) * M * N, hipMemcpyDeviceToHost));
        const bool ok = verify_result(h_C, h_A, M, N, K);
        print_metrics("Vector load  (float4, 128-bit) ", ok, ms, M, N, K);
    }

    // ------------------------------------------------------------------
    // Cleanup
    // ------------------------------------------------------------------
    HIP_CHECK(hipFree(d_A));
    HIP_CHECK(hipFree(d_B));
    HIP_CHECK(hipFree(d_C));

    return EXIT_SUCCESS;
}
