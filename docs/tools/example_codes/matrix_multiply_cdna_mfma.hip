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

// ---------------------------------------------------------------------------
// matrix_multiply_cdna_mfma.hip  --  MFMA ComputePolicy for CDNA (gfx908, MI100)
//
// This file is a companion to matrix_multiply_generic.hip (Step 7) and
// provides a concrete ComputePolicy implementation that uses the MFMA
// instruction v_mfma_f32_32x32x1f32 available on CDNA GPUs (gfx908, MI100).
//
// The policy is a drop-in replacement for ScalarFMAPolicy: the generic
// kernel template (matrix_multiply_generic) is reused without modification.
// Only the ComputePolicy type alias changes.
//
// Architecture: CDNA (gfx908, MI100) only.
// Compile with: amdclang++ -O3 -std=c++17 --offload-arch=gfx908 \
//                   matrix_multiply_cdna_mfma.hip -o mm_cdna_mfma
//
// The code is also guarded by a preprocessor check so that it compiles on
// other architectures (ScalarFMAPolicy is used as a fallback) without
// emitting undefined-behaviour warnings or hard errors.
// ---------------------------------------------------------------------------

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

// ===========================================================================
// Section 1: TilePolicy and ComputePolicy interfaces
//             (copied verbatim from matrix_multiply_generic.hip)
// ===========================================================================

// [Sphinx tile policy traits start]
template<typename TilePolicy>
struct TilePolicyTraits
{
    static_assert(TilePolicy::num_buffers == 1 || TilePolicy::num_buffers == 2,
                  "TilePolicy::num_buffers must be 1 (single) or 2 (double-buffer)");
    static_assert(TilePolicy::block_tile_m > 0,
                  "TilePolicy::block_tile_m must be a positive integer");
    static_assert(TilePolicy::block_tile_n > 0,
                  "TilePolicy::block_tile_n must be a positive integer");
    static_assert(TilePolicy::k_tile_size > 0,
                  "TilePolicy::k_tile_size must be a positive integer");
    static_assert(std::is_trivially_destructible_v<typename TilePolicy::SharedStorage>,
                  "TilePolicy::SharedStorage must be trivially destructible");
    static_assert(sizeof(typename TilePolicy::SharedStorage) > 0,
                  "TilePolicy::SharedStorage must have non-zero size");
};
// [Sphinx tile policy traits end]

// [Sphinx compute policy traits start]
template<typename ComputePolicy>
struct ComputePolicyTraits
{
    static_assert(ComputePolicy::thread_tile_m > 0,
                  "ComputePolicy::thread_tile_m must be a positive integer");
    static_assert(ComputePolicy::thread_tile_n > 0,
                  "ComputePolicy::thread_tile_n must be a positive integer");
    static_assert(ComputePolicy::effective_lanes > 0,
                  "ComputePolicy::effective_lanes must be a positive integer");
    static_assert(ComputePolicy::k_step > 0,
                  "ComputePolicy::k_step must be a positive integer");
    static_assert(std::is_trivially_constructible_v<typename ComputePolicy::Accumulator>,
                  "ComputePolicy::Accumulator must be trivially constructible");
    static_assert(sizeof(typename ComputePolicy::elem_a) > 0,
                  "ComputePolicy::elem_a must have non-zero size");
    static_assert(sizeof(typename ComputePolicy::elem_b) > 0,
                  "ComputePolicy::elem_b must have non-zero size");
};
// [Sphinx compute policy traits end]

// ===========================================================================
// Section 2: Cooperative tile load helpers
//             (identical to matrix_multiply_generic.hip)
// ===========================================================================

__device__ inline void cooperative_load_tile_a(float*       dest,
                                               const float* A,
                                               int          block_row_start,
                                               int          block_tile_m,
                                               int          t,
                                               int          k_tile_size,
                                               int          m,
                                               int          k,
                                               int          tid,
                                               int          num_threads)
{
    const int tile_elems = block_tile_m * k_tile_size;
    for(int idx = tid; idx < tile_elems; idx += num_threads)
    {
        const int tile_row   = idx / k_tile_size;
        const int tile_col   = idx % k_tile_size;
        const int global_row = block_row_start + tile_row;
        const int global_col = t * k_tile_size + tile_col;
        dest[tile_row * k_tile_size + tile_col] =
            (global_row < m && global_col < k) ? A[global_row * k + global_col] : 0.0f;
    }
}

__device__ inline void cooperative_load_tile_b_T(float*       dest,
                                                 const float* B,
                                                 int          block_col_start,
                                                 int          block_tile_n,
                                                 int          t,
                                                 int          k_tile_size,
                                                 int          k,
                                                 int          n,
                                                 int          tid,
                                                 int          num_threads)
{
    const int tile_elems = block_tile_n * k_tile_size;
    for(int idx = tid; idx < tile_elems; idx += num_threads)
    {
        const int tile_col_idx = idx / k_tile_size;
        const int tile_k_idx   = idx % k_tile_size;
        const int global_row   = t * k_tile_size + tile_k_idx;
        const int global_col   = block_col_start + tile_col_idx;
        dest[tile_col_idx * k_tile_size + tile_k_idx] =
            (global_row < k && global_col < n) ? B[global_row * n + global_col] : 0.0f;
    }
}

// ===========================================================================
// Section 3: TilePolicy implementations
//             (single-buffer only; double-buffer can be added identically)
// ===========================================================================

// [Sphinx single buffer policy start]
template<int BlockTileM_, int BlockTileN_, int KTileSize_>
struct SingleBufferTilePolicy
{
    static constexpr int num_buffers  = 1;
    static constexpr int block_tile_m = BlockTileM_;
    static constexpr int block_tile_n = BlockTileN_;
    static constexpr int k_tile_size  = KTileSize_;

    struct SharedStorage
    {
        float tile_a[BlockTileM_][KTileSize_];
        float tile_b_T[BlockTileN_][KTileSize_];
    };

    __device__ static void prologue(SharedStorage& /*smem*/,
                                    const float* /*A*/, const float* /*B*/,
                                    int /*block_row_start*/, int /*block_col_start*/,
                                    int /*m*/, int /*n*/, int /*k*/,
                                    int /*tid*/, int /*num_threads*/) {}

    __device__ static void prefetch(SharedStorage& smem,
                                    const float* A, const float* B,
                                    int          t_next,
                                    int block_row_start, int block_col_start,
                                    int m, int n, int k,
                                    int tid, int num_threads)
    {
        const int t = t_next;
        cooperative_load_tile_a(&smem.tile_a[0][0], A, block_row_start,
                                block_tile_m, t, k_tile_size, m, k, tid, num_threads);
        cooperative_load_tile_b_T(&smem.tile_b_T[0][0], B, block_col_start,
                                   block_tile_n, t, k_tile_size, k, n, tid, num_threads);
    }

    __device__ static void acquire(int /*t*/) { __syncthreads(); }
    __device__ static void release(int /*t*/) { __syncthreads(); }
    __device__ static int  buf_idx(int /*t*/) { return 0; }

    __device__ static float* get_tile_a(SharedStorage& smem, int /*buf*/)
    {
        return &smem.tile_a[0][0];
    }
    __device__ static float* get_tile_b_T(SharedStorage& smem, int /*buf*/)
    {
        return &smem.tile_b_T[0][0];
    }
};
// [Sphinx single buffer policy end]

// ===========================================================================
// Section 4: MfmaCdnaPolicy -- MFMA ComputePolicy for gfx908 (CDNA, MI100)
// ===========================================================================
//
// This policy uses v_mfma_f32_32x32x1f32 to compute one 32x32 FP32 output
// tile per wavefront per intrinsic call.  Each block contains 2x2 = 4
// wavefronts, giving a 64x64 block tile.
//
// Accumulator layout (32x32 tile, 2 blocks):
//   Given lane L and accVGPR index G (0..31):
//     block b  = G / 16
//     g_local  = G mod 16
//     i = 8*(g_local/4) + 4*(L/32) + (g_local mod 4)
//     j = L mod 32
//
// Fragment distribution:
//   A rows 0..31 map to lanes 0..31 (block 0) and mirror to lanes 32..63
//   (block 1).  Each lane holds one scalar A element: tile_a[row][ki].
//   B cols 0..31 map to lanes 0..31, mirrored to lanes 32..63.
// ===========================================================================

// [Sphinx mfma cdna policy start]
// MfmaCdnaPolicy is defined for all compilation passes so that the host and
// device agree on the template instantiation.  The intrinsic body inside mma()
// is guarded by __gfx908__, which is defined only during the device pass.
struct MfmaCdnaPolicy
{
    // ── ComputePolicy constants ──────────────────────────────────────────────
    // One wavefront owns a 32×32 output tile.
    // All 64 lanes contribute unique output elements (no lane mirroring).
    static constexpr int thread_tile_m  = 32;
    static constexpr int thread_tile_n  = 32;
    static constexpr int effective_lanes = 64;
    // v_mfma_f32_32x32x1f32 consumes one K-index per call (K=1 per instruction).
    static constexpr int k_step          = 1;
    // MFMA takes one scalar per lane.
    static constexpr int frag_size_m     = 1;
    static constexpr int frag_size_n     = 1;

    using elem_a = float;
    using elem_b = float;

    // ── Accumulator ──────────────────────────────────────────────────────────
    // v32float holds 32 accVGPRs per lane:
    //   elements [0,15]  → block 0 accVGPRs 0–15
    //   elements [16,31] → block 1 accVGPRs 16–31
    using v32float = float [[clang::ext_vector_type(32)]];

    struct Accumulator
    {
        v32float regs;
    };

    __device__ static void zero(Accumulator& acc)
    {
        acc.regs = v32float{};
    }

    // ── thread_tile_offset ───────────────────────────────────────────────────
    // The entire 32×32 tile belongs to one wavefront; multiple wavefronts
    // in a block cover different 32×32 sub-tiles.  Each wavefront's top-left
    // corner is determined by its wavefront index within the block.
    //
    //   wavefront id within block: wid = tid / 64
    //   wid_x = wid % (block_tile_n / 32)
    //   wid_y = wid / (block_tile_n / 32)
    //   *thread_row = wid_y * 32
    //   *thread_col = wid_x * 32
    __device__ static void thread_tile_offset(int  tid,
                                               int  /*lane_id*/,
                                               int* thread_row,
                                               int* thread_col)
    {
        // block_tile_n / 32 = number of wavefronts along the N dimension.
        // For a 64×64 block tile this is 64/32 = 2.
        constexpr int waves_n = 2; // block_tile_n / thread_tile_n
        const int wid   = tid / 64;
        const int wid_x = wid % waves_n;
        const int wid_y = wid / waves_n;
        *thread_row = wid_y * 32;
        *thread_col = wid_x * 32;
    }

    // ── Fragment loads ────────────────────────────────────────────────────────
    // load_a: each lane reads its own A element from LDS.
    //
    // The 32×32 MFMA A-operand distributes 32 rows across the lower 32 lanes:
    //   lane L in [0,31]:  provides A-row (tile_a_row + L)
    //   lane L in [32,63]: mirrors the same 32 rows for the second block
    // Indexing: tile_a[tile_a_row + (lane_id mod 32)][ki].
    __device__ static void load_a(const float* tile_a_ptr,
                                  int          tile_a_row,
                                  int          ki,
                                  int          k_tile_size,
                                  int          lane_id,
                                  elem_a     (&frag)[1])
    {
        const int row = tile_a_row + (lane_id % 32);
        frag[0] = tile_a_ptr[row * k_tile_size + ki];
    }

    // load_b: each lane reads its own B element from LDS.
    //
    // The 32×32 MFMA B-operand distributes 32 cols across the lower 32 lanes:
    //   lane L in [0,31]:  provides B-col (tile_b_col + L)
    //   lane L in [32,63]: mirrors the same 32 cols for the second block
    // tile_b_T is column-major: tile_b_T[(tile_b_col + (lane_id mod 32))][ki].
    __device__ static void load_b(const float* tile_b_T_ptr,
                                  int          tile_b_col,
                                  int          ki,
                                  int          k_tile_size,
                                  int          lane_id,
                                  elem_b     (&frag)[1])
    {
        const int col = tile_b_col + (lane_id % 32);
        frag[0] = tile_b_T_ptr[col * k_tile_size + ki];
    }

    // ── mma ───────────────────────────────────────────────────────────────────
    // Issue one v_mfma_f32_32x32x1f32 instruction.
    // cbsz=0: use both blocks (no A-broadcast).
    // abid=0: select block 0 as A source.
    // blgp=0: no B lane-group permutation.
    __device__ static void mma(Accumulator&       acc,
                                const elem_a     (&a_frag)[1],
                                const elem_b     (&b_frag)[1])
    {
#if defined(__gfx908__)
        acc.regs = __builtin_amdgcn_mfma_f32_32x32x1f32(
            a_frag[0], b_frag[0], acc.regs,
            /*cbsz=*/0, /*abid=*/0, /*blgp=*/0);
#endif
    }

    // ── store_c ───────────────────────────────────────────────────────────────
    // Scatter the 32 accVGPR values to their global-memory positions.
    //
    // Inverse layout (lane L, accVGPR index G in [0,31]):
    //   g_local = G mod 16        -- accVGPR within its block
    //   i = 8*(g_local/4) + 4*(L/32) + (g_local mod 4)
    //   j = L mod 32
    __device__ static void store_c(const Accumulator& acc,
                                   float*             C,
                                   int                out_row_base,
                                   int                out_col_base,
                                   int                m,
                                   int                n,
                                   int                lane_id)
    {
        const int L = lane_id;
        const int j = L % 32;

        #pragma unroll
        for(int G = 0; G < 32; ++G)
        {
            const int g_local = G % 16;
            const int i = 8 * (g_local / 4)
                        + 4 * (L / 32)
                        + (g_local % 4);

            const int r = out_row_base + i;
            const int c = out_col_base + j;
            if(r < m && c < n)
                C[r * n + c] = acc.regs[G];
        }
    }
};
// [Sphinx mfma cdna policy end]

// ===========================================================================
// Section 5: ScalarFMAPolicy -- portable fallback for non-CDNA targets
// ===========================================================================

// [Sphinx scalar fma policy start]
template<int ThreadTileM_, int ThreadTileN_, int BlockDimX_>
struct ScalarFMAPolicy
{
    static constexpr int thread_tile_m   = ThreadTileM_;
    static constexpr int thread_tile_n   = ThreadTileN_;
    static constexpr int effective_lanes = 64;
    static constexpr int k_step          = 1;
    static constexpr int frag_size_m     = ThreadTileM_;
    static constexpr int frag_size_n     = ThreadTileN_;

    using elem_a = float;
    using elem_b = float;

    struct Accumulator
    {
        float data[ThreadTileM_][ThreadTileN_];
    };

    __device__ static void zero(Accumulator& acc)
    {
        #pragma unroll
        for(int i = 0; i < ThreadTileM_; ++i)
            #pragma unroll
            for(int j = 0; j < ThreadTileN_; ++j)
                acc.data[i][j] = 0.0f;
    }

    __device__ static void thread_tile_offset(int tid,
                                              int /*lane_id*/,
                                              int* thread_row,
                                              int* thread_col)
    {
        const int tx = tid % BlockDimX_;
        const int ty = tid / BlockDimX_;
        *thread_row  = ty * ThreadTileM_;
        *thread_col  = tx * ThreadTileN_;
    }

    __device__ static void load_a(const float* tile_a_ptr,
                                  int          tile_a_row,
                                  int          ki,
                                  int          k_tile_size,
                                  int          /*lane_id*/,
                                  elem_a (&frag)[ThreadTileM_])
    {
        #pragma unroll
        for(int i = 0; i < ThreadTileM_; ++i)
            frag[i] = tile_a_ptr[(tile_a_row + i) * k_tile_size + ki];
    }

    __device__ static void load_b(const float* tile_b_T_ptr,
                                  int          tile_b_col,
                                  int          ki,
                                  int          k_tile_size,
                                  int          /*lane_id*/,
                                  elem_b (&frag)[ThreadTileN_])
    {
        #pragma unroll
        for(int j = 0; j < ThreadTileN_; ++j)
            frag[j] = tile_b_T_ptr[(tile_b_col + j) * k_tile_size + ki];
    }

    __device__ static void mma(Accumulator&       acc,
                               const elem_a (&a_frag)[ThreadTileM_],
                               const elem_b (&b_frag)[ThreadTileN_])
    {
        #pragma unroll
        for(int i = 0; i < ThreadTileM_; ++i)
            #pragma unroll
            for(int j = 0; j < ThreadTileN_; ++j)
                acc.data[i][j] += a_frag[i] * b_frag[j];
    }

    __device__ static void store_c(const Accumulator& acc,
                                   float*             C,
                                   int                out_row_base,
                                   int                out_col_base,
                                   int                m,
                                   int                n,
                                   int                /*lane_id*/)
    {
        #pragma unroll
        for(int i = 0; i < ThreadTileM_; ++i)
            #pragma unroll
            for(int j = 0; j < ThreadTileN_; ++j)
            {
                const int r = out_row_base + i;
                const int c = out_col_base + j;
                if(r < m && c < n)
                    C[r * n + c] = acc.data[i][j];
            }
    }
};
// [Sphinx scalar fma policy end]

// ===========================================================================
// Section 6: Generic kernel template
//             (identical to matrix_multiply_generic.hip)
// ===========================================================================

// [Sphinx gemm kernel start]
template<typename TilePolicy, typename ComputePolicy>
__global__ void
    __launch_bounds__(TilePolicy::block_tile_m / ComputePolicy::thread_tile_m
                          * TilePolicy::block_tile_n / ComputePolicy::thread_tile_n
                          * 64) // threads per block = wavefronts * wavefront_size
    matrix_multiply_generic(const float* __restrict__ A,
                            const float* __restrict__ B,
                            float* __restrict__       C,
                            int                       m,
                            int                       n,
                            int                       k)
{
    (void)TilePolicyTraits<TilePolicy>{};
    (void)ComputePolicyTraits<ComputePolicy>{};

    __shared__ typename TilePolicy::SharedStorage smem;

    const int tid         = threadIdx.y * blockDim.x + threadIdx.x;
    const int lane_id     = tid % 64;
    const int num_threads = blockDim.x * blockDim.y;

    const int block_row_start = blockIdx.y * TilePolicy::block_tile_m;
    const int block_col_start = blockIdx.x * TilePolicy::block_tile_n;

    int thread_row = 0, thread_col = 0;
    ComputePolicy::thread_tile_offset(tid, lane_id, &thread_row, &thread_col);

    typename ComputePolicy::Accumulator acc;
    ComputePolicy::zero(acc);

    const int num_tiles = (k + TilePolicy::k_tile_size - 1) / TilePolicy::k_tile_size;

    TilePolicy::prologue(smem, A, B,
                         block_row_start, block_col_start,
                         m, n, k, tid, num_threads);

    for(int t = 0; t < num_tiles; ++t)
    {
        TilePolicy::prefetch(smem, A, B, t + (TilePolicy::num_buffers == 1 ? 0 : 1),
                             block_row_start, block_col_start,
                             m, n, k, tid, num_threads);

        TilePolicy::acquire(t);

        const int buf = TilePolicy::buf_idx(t);
        const float* tile_a_ptr   = TilePolicy::get_tile_a(smem, buf);
        const float* tile_b_T_ptr = TilePolicy::get_tile_b_T(smem, buf);

        static_assert(TilePolicy::k_tile_size % ComputePolicy::k_step == 0,
                      "k_tile_size must be divisible by ComputePolicy::k_step");

        for(int ki = 0; ki < TilePolicy::k_tile_size; ki += ComputePolicy::k_step)
        {
            typename ComputePolicy::elem_a a_frag[ComputePolicy::frag_size_m];
            typename ComputePolicy::elem_b b_frag[ComputePolicy::frag_size_n];

            ComputePolicy::load_a(tile_a_ptr, thread_row, ki,
                                   TilePolicy::k_tile_size, lane_id, a_frag);
            ComputePolicy::load_b(tile_b_T_ptr, thread_col, ki,
                                   TilePolicy::k_tile_size, lane_id, b_frag);
            ComputePolicy::mma(acc, a_frag, b_frag);
        }

        TilePolicy::release(t);
    }

    ComputePolicy::store_c(acc, C,
                            block_row_start + thread_row,
                            block_col_start + thread_col,
                            m, n, lane_id);
}
// [Sphinx gemm kernel end]

// ===========================================================================
// Section 7: Policy aliases and launch configuration
// ===========================================================================
//
// Block tile: 64x64 (2x2 wavefronts, each computing a 32x32 MFMA tile).
// K-strip  : 16 (k_tile_size=16, k_step=1 → 16 MFMA calls per tile).
//
// Thread grid:
//   Each block spans BLOCK_TILE_M rows and BLOCK_TILE_N cols of C.
//   Blocks per grid: ceil(M/64) x ceil(N/64).
//   Threads per block: (BLOCK_TILE_M/32) * (BLOCK_TILE_N/32) wavefronts * 64 lanes
//                    = 2 * 2 * 64 = 256.
// ===========================================================================

// [Sphinx mfma policy aliases start]
// TilePolicy: 64x64 block tile, 16-element K-strip, single-buffered.
using MfmaTilePolicy = SingleBufferTilePolicy<64, 64, 16>;
// [Sphinx mfma policy aliases end]

// [Sphinx mfma launch config start]
// Block tile dimensions shared by both dispatch paths.
constexpr int BLOCK_TILE_M = 64;
constexpr int BLOCK_TILE_N = 64;
// [Sphinx mfma launch config end]

// ===========================================================================
// Section 8: Helpers (timing, verification, metrics)
// ===========================================================================

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
        launch();
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
        for(int j = 0; j < check_cols; ++j)
            if(std::fabs(C[i * n + j] - A[i * k + j]) > tolerance)
            {
                std::cerr << "Mismatch at [" << i << "][" << j << "]: "
                          << "expected " << A[i * k + j]
                          << ", got " << C[i * n + j] << "\n";
                return false;
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

// ===========================================================================
// Section 9: main
// ===========================================================================
int main()
{
    hipDeviceProp_t props{};
    HIP_CHECK(hipGetDeviceProperties(&props, 0));

    // Select the MFMA or fallback policy at runtime based on the device
    // GCN architecture name.  Both kernel instantiations are compiled into
    // the device binary; only the matching one is launched.
    const bool is_gfx908 =
        std::string(props.gcnArchName).rfind("gfx908", 0) == 0;
    const int threads_per_block = is_gfx908 ? 256 : 64;

    std::cout << "Device       : " << props.name << "\n"
              << "GCN arch     : " << props.gcnArchName << "\n"
              << "Warp size    : " << props.warpSize << "\n"
              << "Matrix dims  : A(" << M << "x" << K << ") * B(" << K << "x" << N << ")\n"
              << "Block tile   : " << BLOCK_TILE_M << "x" << BLOCK_TILE_N
              << "  K-strip: " << MfmaTilePolicy::k_tile_size << "\n"
              << "Threads/block: " << threads_per_block << "\n\n";

    std::vector<float> h_A(M * K);
    std::vector<float> h_B(K * N, 0.0f);
    std::vector<float> h_C(M * N, 0.0f);

    std::mt19937                          gen(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for(float& v : h_A)
        v = dist(gen);
    for(int i = 0; i < std::min(K, N); ++i)
        h_B[i * N + i] = 1.0f;

    float *d_A, *d_B, *d_C;
    HIP_CHECK(hipMalloc(&d_A, sizeof(float) * M * K));
    HIP_CHECK(hipMalloc(&d_B, sizeof(float) * K * N));
    HIP_CHECK(hipMalloc(&d_C, sizeof(float) * M * N));

    HIP_CHECK(hipMemcpy(d_A, h_A.data(), sizeof(float) * M * K, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_B, h_B.data(), sizeof(float) * K * N, hipMemcpyHostToDevice));

    // [Sphinx mfma kernel launch start]
    // Grid: one block per 64×64 output tile.
    const dim3 grid((N + BLOCK_TILE_N - 1) / BLOCK_TILE_N,
                    (M + BLOCK_TILE_M - 1) / BLOCK_TILE_M);

    float       ms;
    bool        ok;
    const char* policy_label;

    if(is_gfx908)
    {
        // CDNA (MI100): 2×2 wavefronts of 64 lanes = 256 threads per block.
        auto launch = [&]()
        {
            matrix_multiply_generic<MfmaTilePolicy, MfmaCdnaPolicy>
                <<<grid, dim3(256)>>>(d_A, d_B, d_C, M, N, K);
            HIP_CHECK(hipGetLastError());
        };
        ms           = time_kernel_ms(launch, WARMUP_RUNS, TIMING_RUNS);
        policy_label = "GemmKernel<MfmaTilePolicy, MfmaCdnaPolicy> [gfx908]";
    }
    else
    {
        // Other targets: scalar fallback, 8×8 thread tiles = 64 threads per block.
        auto launch = [&]()
        {
            matrix_multiply_generic<MfmaTilePolicy, ScalarFMAPolicy<8, 8, 8>>
                <<<grid, dim3(64)>>>(d_A, d_B, d_C, M, N, K);
            HIP_CHECK(hipGetLastError());
        };
        ms           = time_kernel_ms(launch, WARMUP_RUNS, TIMING_RUNS);
        policy_label = "GemmKernel<MfmaTilePolicy, ScalarFMAPolicy> [fallback]";
    }
    // [Sphinx mfma kernel launch end]

    HIP_CHECK(hipMemcpy(h_C.data(), d_C, sizeof(float) * M * N, hipMemcpyDeviceToHost));
    ok = verify_result(h_C, h_A, M, N, K);
    print_metrics(policy_label, ok, ms, M, N, K);

    HIP_CHECK(hipFree(d_A));
    HIP_CHECK(hipFree(d_B));
    HIP_CHECK(hipFree(d_C));

    return EXIT_SUCCESS;
}
