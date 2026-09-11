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
// matrix_multiply_cdna2_mfma.hip  --  FP64 MFMA ComputePolicy for CDNA2 (gfx90a, MI200)
//
// This file is a companion to matrix_multiply_generic.hip (Step 7) and
// provides a concrete ComputePolicy implementation that uses the MFMA
// instruction v_mfma_f64_16x16x4f64 available on CDNA2 GPUs (gfx90a, MI200).
//
// The policy is a drop-in replacement for ScalarFMAPolicy: the generic
// kernel template (matrix_multiply_generic) is reused without modification,
// adapted here to work with double-precision element types.
//
// Architecture: CDNA2 (gfx90a, MI200) only.
// Compile with: amdclang++ -O3 -std=c++17 --offload-arch=gfx90a \
//                   matrix_multiply_cdna2_mfma.hip -o mm_cdna2_mfma
//
// The code is guarded by a preprocessor check so that it compiles on other
// architectures (ScalarFMADPolicy is used as a fallback) without emitting
// undefined-behaviour warnings or hard errors.
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
// Section 2: Cooperative tile load helpers (FP64 variant)
// ===========================================================================

__device__ inline void cooperative_load_tile_a(double*       dest,
                                               const double* A,
                                               int           block_row_start,
                                               int           block_tile_m,
                                               int           t,
                                               int           k_tile_size,
                                               int           m,
                                               int           k,
                                               int           tid,
                                               int           num_threads)
{
    const int tile_elems = block_tile_m * k_tile_size;
    for(int idx = tid; idx < tile_elems; idx += num_threads)
    {
        const int tile_row   = idx / k_tile_size;
        const int tile_col   = idx % k_tile_size;
        const int global_row = block_row_start + tile_row;
        const int global_col = t * k_tile_size + tile_col;
        dest[tile_row * k_tile_size + tile_col] =
            (global_row < m && global_col < k) ? A[global_row * k + global_col] : 0.0;
    }
}

__device__ inline void cooperative_load_tile_b_T(double*       dest,
                                                 const double* B,
                                                 int           block_col_start,
                                                 int           block_tile_n,
                                                 int           t,
                                                 int           k_tile_size,
                                                 int           k,
                                                 int           n,
                                                 int           tid,
                                                 int           num_threads)
{
    const int tile_elems = block_tile_n * k_tile_size;
    for(int idx = tid; idx < tile_elems; idx += num_threads)
    {
        const int tile_col_idx = idx / k_tile_size;
        const int tile_k_idx   = idx % k_tile_size;
        const int global_row   = t * k_tile_size + tile_k_idx;
        const int global_col   = block_col_start + tile_col_idx;
        dest[tile_col_idx * k_tile_size + tile_k_idx] =
            (global_row < k && global_col < n) ? B[global_row * n + global_col] : 0.0;
    }
}

// ===========================================================================
// Section 3: TilePolicy implementation (FP64 shared memory)
//
// The 16×16 FP64 MFMA instruction has k_step=4, meaning each call consumes
// four K-positions.  One call per K-strip suffices when k_tile_size=4.
// The 32×32 block tile uses 2×2 wavefronts, so the shared-memory tile is
// 32×32 with a K-strip of 4.
// ===========================================================================

// [Sphinx single buffer policy start]
template<int BlockTileM_, int BlockTileN_, int KTileSize_>
struct SingleBufferTilePolicyD
{
    static constexpr int num_buffers  = 1;
    static constexpr int block_tile_m = BlockTileM_;
    static constexpr int block_tile_n = BlockTileN_;
    static constexpr int k_tile_size  = KTileSize_;

    struct SharedStorage
    {
        double tile_a[BlockTileM_][KTileSize_];
        double tile_b_T[BlockTileN_][KTileSize_];
    };

    __device__ static void prologue(SharedStorage& /*smem*/,
                                    const double* /*A*/, const double* /*B*/,
                                    int /*block_row_start*/, int /*block_col_start*/,
                                    int /*m*/, int /*n*/, int /*k*/,
                                    int /*tid*/, int /*num_threads*/) {}

    __device__ static void prefetch(SharedStorage& smem,
                                    const double* A, const double* B,
                                    int           t_next,
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

    __device__ static double* get_tile_a(SharedStorage& smem, int /*buf*/)
    {
        return &smem.tile_a[0][0];
    }
    __device__ static double* get_tile_b_T(SharedStorage& smem, int /*buf*/)
    {
        return &smem.tile_b_T[0][0];
    }
};
// [Sphinx single buffer policy end]

// ===========================================================================
// Section 4: MfmaCdna2F64Policy -- MFMA ComputePolicy for gfx90a (CDNA2, MI200)
// ===========================================================================
//
// This policy uses v_mfma_f64_16x16x4f64 to compute one 16x16 FP64 output
// tile per wavefront.  Each block contains 2x2 = 4 wavefronts, giving a
// 32x32 block tile.
//
// Accumulator layout (16x16 tile, 1 block):
//   Each lane holds 4 FP64 values in accVGPR pairs 0-3
//   (physical registers v[1:0], v[3:2], v[5:4], v[7:6]).
//
//   Given lane L and accVGPR pair index k (0-3):
//     i = 4k + floor(L / 16)
//     j = L mod 16
//
// Fragment distribution:
//   v_mfma_f64_16x16x4f64 has a 16x4 A-operand and a 4x16 B-operand.
//   Each lane provides one double for A (its row in [0,15]) and one double
//   for B (its column in [0,15]).
//
//   Within each 16-lane group (lanes 16*g .. 16*g+15, g in [0,3]):
//     lane L provides A-row (tile_a_row + L mod 16) at K-position ki + g
//     lane L provides B-col (tile_b_col + L mod 16) at K-position ki + g
// ===========================================================================

// [Sphinx mfma cdna2 policy start]
struct MfmaCdna2F64Policy
{
    // -- ComputePolicy constants ----------------------------------------------
    // One wavefront owns a 16×16 output tile.
    // All 64 lanes hold unique output elements (4 elements per lane).
    static constexpr int thread_tile_m   = 16;
    static constexpr int thread_tile_n   = 16;
    static constexpr int frag_size_m     = 1;
    static constexpr int frag_size_n     = 1;
    static constexpr int effective_lanes = 64;
    // v_mfma_f64_16x16x4f64 consumes 4 K-positions per call (k_step=4).
    // A single call with cbsz=0 uses all 64 lanes as 4 groups of 16,
    // each contributing one double for one of the 4 K-positions.
    static constexpr int k_step          = 4;

    using elem_a = double;
    using elem_b = double;

    // -- Accumulator ----------------------------------------------------------
    // v4double holds 4 FP64 accVGPR pairs per lane.
    // accVGPR pair k = floor(i/4), physical regs v[2k+1:2k].
    using v4double = double [[clang::ext_vector_type(4)]];

    struct Accumulator
    {
        v4double regs;
    };

    __device__ static void zero(Accumulator& acc)
    {
        acc.regs = v4double{};
    }

    // -- thread_tile_offset ---------------------------------------------------
    // The entire 16×16 tile belongs to one wavefront.  Multiple wavefronts
    // in a block cover different 16×16 sub-tiles.
    //
    //   wavefront id within block: wid = tid / 64
    //   waves_n = block_tile_n / thread_tile_n = 32 / 16 = 2
    //   wid_x = wid % waves_n
    //   wid_y = wid / waves_n
    //   *thread_row = wid_y * 16
    //   *thread_col = wid_x * 16
    __device__ static void thread_tile_offset(int  tid,
                                              int  /*lane_id*/,
                                              int* thread_row,
                                              int* thread_col)
    {
        constexpr int waves_n = 2; // block_tile_n / thread_tile_n = 32 / 16
        const int wid   = tid / 64;
        const int wid_x = wid % waves_n;
        const int wid_y = wid / waves_n;
        *thread_row = wid_y * 16;
        *thread_col = wid_x * 16;
    }

    // -- Fragment loads --------------------------------------------------------
    // load_a: each lane reads one double from LDS.
    //
    // v_mfma_f64_16x16x4f64 (cbsz=0, abid=0, blgp=0) interprets the 64
    // lanes as 4 groups of 16 (indexed by g = lane_id / 16).  Each group
    // covers one of the 4 K-positions: group g provides data for ki + g.
    // Within group g, lane offset (lane_id mod 16) selects one of the 16
    // A-rows.
    //
    //   A-row index: tile_a_row + (lane_id mod 16)
    //   K-position : ki + (lane_id / 16)
    __device__ static void load_a(const double* tile_a_ptr,
                                  int           tile_a_row,
                                  int           ki,
                                  int           k_tile_size,
                                  int           lane_id,
                                  elem_a      (&frag)[1])
    {
        const int row    = tile_a_row + (lane_id % 16);
        const int k_off  = ki + (lane_id / 16);
        frag[0] = tile_a_ptr[row * k_tile_size + k_off];
    }

    // load_b: each lane reads one double from LDS (symmetric to load_a).
    //
    // tile_b_T is column-major: tile_b_T[(tile_b_col + col)][k_off].
    //   B-col index: tile_b_col + (lane_id mod 16)
    //   K-position : ki + (lane_id / 16)
    __device__ static void load_b(const double* tile_b_T_ptr,
                                  int           tile_b_col,
                                  int           ki,
                                  int           k_tile_size,
                                  int           lane_id,
                                  elem_b      (&frag)[1])
    {
        const int col   = tile_b_col + (lane_id % 16);
        const int k_off = ki + (lane_id / 16);
        frag[0] = tile_b_T_ptr[col * k_tile_size + k_off];
    }

    // -- mma -------------------------------------------------------------------
    // Issue one v_mfma_f64_16x16x4f64 instruction.
    // All modifier parameters must be 0 for this instruction.
    // cbsz=0, abid=0: standard 64-lane A-operand, no broadcast.
    // blgp=0: no B lane-group permutation.
    __device__ static void mma(Accumulator&       acc,
                               const elem_a     (&a_frag)[1],
                               const elem_b     (&b_frag)[1])
    {
#if defined(__gfx90a__)
        acc.regs = __builtin_amdgcn_mfma_f64_16x16x4f64(
            a_frag[0], b_frag[0], acc.regs,
            /*cbsz=*/0, /*abid=*/0, /*blgp=*/0);
#endif
    }

    // -- store_c ---------------------------------------------------------------
    // Scatter the 4 accVGPR pair values to their global-memory positions.
    //
    // Inverse layout (lane L, accVGPR pair index k in [0,3]):
    //   i = 4k + floor(L / 16)
    //   j = L mod 16
    __device__ static void store_c(const Accumulator& acc,
                                   double*            C,
                                   int                out_row_base,
                                   int                out_col_base,
                                   int                m,
                                   int                n,
                                   int                lane_id)
    {
        const int L = lane_id;
        const int j = L % 16;

        #pragma unroll
        for(int k = 0; k < 4; ++k)
        {
            const int i = 4 * k + (L / 16);
            const int r = out_row_base + i;
            const int c = out_col_base + j;
            if(r < m && c < n)
                C[r * n + c] = acc.regs[k];
        }
    }
};
// [Sphinx mfma cdna2 policy end]

// ===========================================================================
// Section 5: ScalarFMADPolicy -- portable FP64 fallback for non-CDNA2 targets
// ===========================================================================

// [Sphinx scalar fmad policy start]
template<int ThreadTileM_, int ThreadTileN_, int BlockDimX_>
struct ScalarFMADPolicy
{
    static constexpr int thread_tile_m   = ThreadTileM_;
    static constexpr int thread_tile_n   = ThreadTileN_;
    static constexpr int frag_size_m     = ThreadTileM_;
    static constexpr int frag_size_n     = ThreadTileN_;
    static constexpr int effective_lanes = 64;
    static constexpr int k_step          = 1;

    using elem_a = double;
    using elem_b = double;

    struct Accumulator
    {
        double data[ThreadTileM_][ThreadTileN_];
    };

    __device__ static void zero(Accumulator& acc)
    {
        #pragma unroll
        for(int i = 0; i < ThreadTileM_; ++i)
            #pragma unroll
            for(int j = 0; j < ThreadTileN_; ++j)
                acc.data[i][j] = 0.0;
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

    __device__ static void load_a(const double* tile_a_ptr,
                                  int           tile_a_row,
                                  int           ki,
                                  int           k_tile_size,
                                  int           /*lane_id*/,
                                  elem_a      (&frag)[ThreadTileM_])
    {
        #pragma unroll
        for(int i = 0; i < ThreadTileM_; ++i)
            frag[i] = tile_a_ptr[(tile_a_row + i) * k_tile_size + ki];
    }

    __device__ static void load_b(const double* tile_b_T_ptr,
                                  int           tile_b_col,
                                  int           ki,
                                  int           k_tile_size,
                                  int           /*lane_id*/,
                                  elem_b      (&frag)[ThreadTileN_])
    {
        #pragma unroll
        for(int j = 0; j < ThreadTileN_; ++j)
            frag[j] = tile_b_T_ptr[(tile_b_col + j) * k_tile_size + ki];
    }

    __device__ static void mma(Accumulator&       acc,
                               const elem_a     (&a_frag)[ThreadTileM_],
                               const elem_b     (&b_frag)[ThreadTileN_])
    {
        #pragma unroll
        for(int i = 0; i < ThreadTileM_; ++i)
            #pragma unroll
            for(int j = 0; j < ThreadTileN_; ++j)
                acc.data[i][j] += a_frag[i] * b_frag[j];
    }

    __device__ static void store_c(const Accumulator& acc,
                                   double*            C,
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
// [Sphinx scalar fmad policy end]

// ===========================================================================
// Section 6: Generic kernel template (FP64 variant)
// ===========================================================================

// [Sphinx gemm kernel start]
template<typename TilePolicy, typename ComputePolicy>
__global__ void
    __launch_bounds__(TilePolicy::block_tile_m / ComputePolicy::thread_tile_m
                          * TilePolicy::block_tile_n / ComputePolicy::thread_tile_n
                          * 64) // threads per block = wavefronts * wavefront_size
    matrix_multiply_generic(const double* __restrict__ A,
                            const double* __restrict__ B,
                            double* __restrict__       C,
                            int                        m,
                            int                        n,
                            int                        k)
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
        const double* tile_a_ptr   = TilePolicy::get_tile_a(smem, buf);
        const double* tile_b_T_ptr = TilePolicy::get_tile_b_T(smem, buf);

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
// Block tile: 32x32 (2x2 wavefronts, each computing a 16x16 MFMA tile).
// K-strip  :  4 (k_tile_size=4, k_step=4 → 1 MFMA call per tile).
//
// Thread grid:
//   Each block spans BLOCK_TILE_M rows and BLOCK_TILE_N cols of C.
//   Blocks per grid: ceil(M/32) x ceil(N/32).
//   Threads per block: (BLOCK_TILE_M/16) * (BLOCK_TILE_N/16) wavefronts * 64 lanes
//                    = 2 * 2 * 64 = 256.
// ===========================================================================

// [Sphinx mfma policy aliases start]
// TilePolicy: 32x32 block tile, 4-element K-strip, single-buffered.
using MfmaTilePolicy = SingleBufferTilePolicyD<32, 32, 4>;
// [Sphinx mfma policy aliases end]

// [Sphinx mfma launch config start]
// Launch parameters:
//
// CDNA2 (gfx90a): 2x2 wavefronts of 64 lanes = 256 threads per block.
//   block_tile_m = 32, thread_tile_m = 16  → 2 wavefronts along M
//   block_tile_n = 32, thread_tile_n = 16  → 2 wavefronts along N
//
// Fallback (scalar): 8x8 thread tiles in a 32x32 block → 16 threads per block,
//   padded to one full wavefront of 64.
//
// The kernel to launch and its thread count are selected at runtime based on
// the device architecture string returned by hipGetDeviceProperties.
//
// Grid: one block per 32x32 output tile.
constexpr int BLOCK_TILE_M = 32;
constexpr int BLOCK_TILE_N = 32;
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

bool verify_result(const std::vector<double>& C,
                   const std::vector<double>& A,
                   int                        m,
                   int                        n,
                   int                        k,
                   double                     tolerance = 1.0e-9)
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
        sizeof(double) * static_cast<double>(m * k + k * n + m * n);
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
    const bool is_gfx90a =
        std::string(props.gcnArchName).rfind("gfx90a", 0) == 0;
    const int threads_per_block = is_gfx90a ? 256 : 64;
    std::cout << "Device       : " << props.name << "\n"
              << "GCN arch     : " << props.gcnArchName << "\n"
              << "Warp size    : " << props.warpSize << "\n"
              << "Matrix dims  : A(" << M << "x" << K << ") * B(" << K << "x" << N << ")\n"
              << "Block tile   : " << BLOCK_TILE_M << "x" << BLOCK_TILE_N
              << "  K-strip: " << MfmaTilePolicy::k_tile_size << "\n"
              << "Threads/block: " << threads_per_block << "\n\n";

    std::vector<double> h_A(M * K);
    std::vector<double> h_B(K * N, 0.0);
    std::vector<double> h_C(M * N, 0.0);

    std::mt19937                           gen(42);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    for(double& v : h_A)
        v = dist(gen);
    // Identity matrix: multiplying A * I should give A in C.
    for(int i = 0; i < std::min(K, N); ++i)
        h_B[i * N + i] = 1.0;

    double *d_A, *d_B, *d_C;
    HIP_CHECK(hipMalloc(&d_A, sizeof(double) * M * K));
    HIP_CHECK(hipMalloc(&d_B, sizeof(double) * K * N));
    HIP_CHECK(hipMalloc(&d_C, sizeof(double) * M * N));

    HIP_CHECK(hipMemcpy(d_A, h_A.data(), sizeof(double) * M * K, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_B, h_B.data(), sizeof(double) * K * N, hipMemcpyHostToDevice));

    // [Sphinx mfma kernel launch start]
    // Grid: one block per 32×32 output tile.
    const dim3 grid((N + BLOCK_TILE_N - 1) / BLOCK_TILE_N,
                    (M + BLOCK_TILE_M - 1) / BLOCK_TILE_M);

    float       ms = 0.0f;
    const char* policy_label = nullptr;

    if(is_gfx90a)
    {
        auto launch = [&]()
        {
            matrix_multiply_generic<MfmaTilePolicy, MfmaCdna2F64Policy>
                <<<grid, dim3(256)>>>(d_A, d_B, d_C, M, N, K);
            HIP_CHECK(hipGetLastError());
        };
        ms           = time_kernel_ms(launch, WARMUP_RUNS, TIMING_RUNS);
        policy_label = "GemmKernel<MfmaTilePolicy, MfmaCdna2F64Policy> [gfx90a]";
    }
    else
    {
        auto launch = [&]()
        {
            matrix_multiply_generic<MfmaTilePolicy, ScalarFMADPolicy<8, 8, 8>>
                <<<grid, dim3(64)>>>(d_A, d_B, d_C, M, N, K);
            HIP_CHECK(hipGetLastError());
        };
        ms           = time_kernel_ms(launch, WARMUP_RUNS, TIMING_RUNS);
        policy_label = "GemmKernel<MfmaTilePolicy, ScalarFMADPolicy> [fallback]";
    }
    // [Sphinx mfma kernel launch end]

    HIP_CHECK(hipMemcpy(h_C.data(), d_C, sizeof(double) * M * N, hipMemcpyDeviceToHost));
    const bool ok = verify_result(h_C, h_A, M, N, K);
    print_metrics(policy_label, ok, ms, M, N, K);

    HIP_CHECK(hipFree(d_A));
    HIP_CHECK(hipFree(d_B));
    HIP_CHECK(hipFree(d_C));

    return EXIT_SUCCESS;
}
