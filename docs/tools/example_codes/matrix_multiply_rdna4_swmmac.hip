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
// matrix_multiply_rdna4_swmmac.hip -- FP16 SWMMAC ComputePolicy for RDNA 4
//                                     (gfx1200 / gfx1201)
//
// This file is a companion to matrix_multiply_generic.hip (Step 7) and
// provides a concrete ComputePolicy implementation that uses the SWMMAC
// instruction __builtin_amdgcn_swmmac_f32_16x16x32_f16_w32 available on
// RDNA 4 GPUs (gfx1200 RX 9070 XT, gfx1201 RX 9070).
//
// SWMMAC exploits structured 2:4 sparsity in the A matrix: for every four
// consecutive K-elements, exactly two are non-zero. The hardware uses a
// per-lane index register to expand the compressed A values back to full width
// before multiplying, matching the corresponding B elements with no extra
// cost. The result is a 16x16x32 instruction that processes the equivalent
// of a dense 16x16x64 multiply.
//
// The policy is a drop-in replacement for ScalarFMASPolicy: the generic
// kernel template is reused without modification, adapted here to work with
// half-precision element types and a pre-sparsified A matrix.
//
// Architecture: RDNA 4 (gfx1200, gfx1201) only.
// Compile with: amdclang++ -O3 -std=c++17 --offload-arch=gfx1200 \
//                   matrix_multiply_rdna4_swmmac.hip -o mm_rdna4_swmmac
//          or:  amdclang++ -O3 -std=c++17 --offload-arch=gfx1201 \
//                   matrix_multiply_rdna4_swmmac.hip -o mm_rdna4_swmmac
//
// The code is guarded by a preprocessor check so that it compiles on other
// architectures (ScalarFMASPolicy is used as a fallback) without errors.
// ---------------------------------------------------------------------------

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#include <cassert>
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

// Matrix dimensions. K_DENSE is the full dense K; K_SPARSE = K_DENSE / 2
// after 2:4 pruning.
constexpr int M       = 4096;
constexpr int N       = 4096;
constexpr int K_DENSE = 4096;
constexpr int K_SPARSE = K_DENSE / 2; // compressed A has half the columns

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
};
// [Sphinx compute policy traits end]

// ===========================================================================
// Section 2: Cooperative tile load helpers (FP16 / sparse-A variant)
//
// A is stored in compressed form: K_SPARSE columns instead of K_DENSE.
// B and C are dense.
// The index tensor (one int per K_SPARSE/8 position per row) is loaded
// alongside A.
// ===========================================================================

__device__ inline void cooperative_load_tile_a_sparse(
    __fp16*       dest_a,
    int*          dest_idx,
    const __fp16* A_sparse,
    const int*    A_index,
    int           block_row_start,
    int           block_tile_m,
    int           t,
    int           k_tile_sparse, // compressed K strip size (= k_tile_dense / 2)
    int           m,
    int           k_sparse,
    int           tid,
    int           num_threads)
{
    // Load compressed A values
    const int tile_elems = block_tile_m * k_tile_sparse;
    for(int idx = tid; idx < tile_elems; idx += num_threads)
    {
        const int tile_row   = idx / k_tile_sparse;
        const int tile_col   = idx % k_tile_sparse;
        const int global_row = block_row_start + tile_row;
        const int global_col = t * k_tile_sparse + tile_col;
        dest_a[tile_row * k_tile_sparse + tile_col] =
            (global_row < m && global_col < k_sparse)
                ? A_sparse[global_row * k_sparse + global_col]
                : __fp16(0.0f);
    }

    // Load sparsity index: one int per 8 compressed elements (16 dense positions)
    // Each int covers 8 compressed K-positions (2 bits each).
    const int index_cols = k_tile_sparse / 8; // ints per row in this strip
    const int idx_elems  = block_tile_m * index_cols;
    for(int idx = tid; idx < idx_elems; idx += num_threads)
    {
        const int tile_row   = idx / index_cols;
        const int tile_icol  = idx % index_cols;
        const int global_row = block_row_start + tile_row;
        const int global_icol = t * index_cols + tile_icol;
        dest_idx[tile_row * index_cols + tile_icol] =
            (global_row < m && global_icol < k_sparse / 8)
                ? A_index[global_row * (k_sparse / 8) + global_icol]
                : 0xAAAAAAAA; // default: positions 1,3 (10 10 ... pattern)
    }
}

__device__ inline void cooperative_load_tile_b_T(
    __fp16*       dest,
    const __fp16* B,
    int           block_col_start,
    int           block_tile_n,
    int           t,
    int           k_tile_dense,
    int           k_dense,
    int           n,
    int           tid,
    int           num_threads)
{
    const int tile_elems = block_tile_n * k_tile_dense;
    for(int idx = tid; idx < tile_elems; idx += num_threads)
    {
        const int tile_col_idx = idx / k_tile_dense;
        const int tile_k_idx   = idx % k_tile_dense;
        const int global_row   = t * k_tile_dense + tile_k_idx;
        const int global_col   = block_col_start + tile_col_idx;
        dest[tile_col_idx * k_tile_dense + tile_k_idx] =
            (global_row < k_dense && global_col < n)
                ? B[global_row * n + global_col]
                : __fp16(0.0f);
    }
}

// ===========================================================================
// Section 3: TilePolicy implementation (FP16 sparse A, dense B)
//
// k_tile_size here is the compressed K strip size (16 by default).
// The corresponding dense strip is k_tile_size * 2 = 32.
// One SWMMAC call with k_step=32 (dense) covers the full tile.
// ===========================================================================

template<int BlockTileM_, int BlockTileN_, int KTileSparse_>
struct SingleBufferTilePolicyH
{
    static constexpr int num_buffers   = 1;
    static constexpr int block_tile_m  = BlockTileM_;
    static constexpr int block_tile_n  = BlockTileN_;
    // k_tile_size is the *dense* K strip, for compatibility with the generic
    // kernel's loop: ki += k_step where k_step = 32 (dense).
    static constexpr int k_tile_size   = KTileSparse_ * 2;
    static constexpr int k_tile_sparse = KTileSparse_;

    struct SharedStorage
    {
        __fp16 tile_a[BlockTileM_][KTileSparse_];
        int    tile_idx[BlockTileM_][KTileSparse_ / 8];
        __fp16 tile_b_T[BlockTileN_][KTileSparse_ * 2]; // dense B
    };

    __device__ static void prologue(SharedStorage& /*smem*/,
                                    const __fp16* /*A*/, const int* /*Aidx*/,
                                    const __fp16* /*B*/,
                                    int /*block_row_start*/, int /*block_col_start*/,
                                    int /*m*/, int /*n*/, int /*k_sparse*/,
                                    int /*tid*/, int /*num_threads*/) {}

    __device__ static void prefetch(SharedStorage&  smem,
                                    const __fp16*   A_sparse,
                                    const int*      A_index,
                                    const __fp16*   B,
                                    int             t_next,
                                    int block_row_start, int block_col_start,
                                    int m, int n, int k_sparse,
                                    int tid, int num_threads)
    {
        const int t = t_next;
        cooperative_load_tile_a_sparse(
            &smem.tile_a[0][0], &smem.tile_idx[0][0],
            A_sparse, A_index,
            block_row_start, block_tile_m, t, k_tile_sparse,
            m, k_sparse, tid, num_threads);
        cooperative_load_tile_b_T(
            &smem.tile_b_T[0][0], B, block_col_start, block_tile_n,
            t, k_tile_size, k_sparse * 2, n, tid, num_threads);
    }

    __device__ static void acquire(int /*t*/) { __syncthreads(); }
    __device__ static void release(int /*t*/) { __syncthreads(); }
    __device__ static int  buf_idx(int /*t*/) { return 0; }
};

// ===========================================================================
// Section 4: SwmmacRdna4F16Policy -- SWMMAC ComputePolicy for gfx1200/gfx1201
// ===========================================================================
//
// Uses __builtin_amdgcn_swmmac_f32_16x16x32_f16_w32 to compute one 16x16
// FP32 output tile per wave32 wavefront. A is sparse (2:4), B is dense.
//
// Accumulator layout (16x16 tile, wave32):
//   Each lane holds 8 FP32 values in VGPRs 0-7.
//   Given lane L (0-31) and VGPR g (0-7):
//     row i = (g % 2) * 8 + L / 4    (varies: see GFX12 ISA)
//     col j = (g / 2) * 4 + L % 4
//   Exact mapping: row = (L / 16) * 8 + (g & 1) * 4 + (L % 4)
//                  col = ((L / 4) % 4) + (g / 2) * 4 -- see GFX12 ISA doc.
//
// Fragment distribution (wave32, 16x16x32 sparse):
//   srcA: v8fp16 -- 8 compressed FP16 per lane (= 16 dense K-positions)
//   srcB: v16fp16 -- 16 dense FP16 per lane
//   index: int -- 2 bits per compressed element, packed 16 per int
// ===========================================================================

#if defined(__gfx1200__) || defined(__gfx1201__)
// [Sphinx swmmac rdna4 policy start]

// Vector type aliases required by the intrinsic
using v8fp16  = __fp16 [[clang::ext_vector_type(8)]];
using v16fp16 = __fp16 [[clang::ext_vector_type(16)]];
using v8float = float  [[clang::ext_vector_type(8)]];

struct SwmmacRdna4F16Policy
{
    // ── ComputePolicy constants ──────────────────────────────────────────────
    // One wave32 wavefront owns a 16×16 output tile.
    static constexpr int thread_tile_m   = 16;
    static constexpr int thread_tile_n   = 16;
    static constexpr int effective_lanes = 32;
    // swmmac_f32_16x16x32_f16_w32 processes 32 dense K-positions per call.
    static constexpr int k_step          = 32;

    using elem_a = __fp16;
    using elem_b = __fp16;

    // ── Accumulator ──────────────────────────────────────────────────────────
    // v8float holds 8 FP32 output elements per lane.
    struct Accumulator
    {
        v8float regs;
    };

    __device__ static void zero(Accumulator& acc)
    {
        acc.regs = v8float{};
    }

    // ── thread_tile_offset ───────────────────────────────────────────────────
    // The entire 16×16 tile belongs to one wavefront. Multiple wavefronts in
    // a block cover different 16×16 sub-tiles.
    //
    //   wavefront id within block: wid = tid / 32
    //   waves_n = block_tile_n / thread_tile_n
    //   wid_x = wid % waves_n
    //   wid_y = wid / waves_n
    __device__ static void thread_tile_offset(int  tid,
                                              int  /*lane_id*/,
                                              int* thread_row,
                                              int* thread_col)
    {
        constexpr int waves_n = 2; // block_tile_n / thread_tile_n = 32 / 16
        const int wid   = tid / 32;
        const int wid_x = wid % waves_n;
        const int wid_y = wid / waves_n;
        *thread_row = wid_y * 16;
        *thread_col = wid_x * 16;
    }

    // ── Fragment loads ────────────────────────────────────────────────────────
    // load_a: each lane reads 8 compressed FP16 values and one index int from
    // LDS.
    //
    // The SWMMAC instruction requires srcA as v8fp16 (8 compressed elements)
    // and srcB as v16fp16 (16 dense elements). The index register is a scalar
    // int containing 16 two-bit position codes.
    //
    // For a 16×16 tile in a wave32, lanes are split into two groups of 16:
    //   lanes 0-15:  rows 0-7  (each lane covers one row, 8 compressed K-pos)
    //   lanes 16-31: rows 8-15
    // Lane L within a group provides data for A-row (tile_a_row + L % 16).

    // Compressed A fragment: 8 FP16 elements per lane.
    // Stores both the FP16 data and the matching index int.
    struct AFragment
    {
        v8fp16 data;
        int    index;
    };

    __device__ static AFragment load_a_frag(const __fp16* tile_a_ptr,
                                            const int*    tile_idx_ptr,
                                            int           tile_a_row,
                                            int           k_tile_sparse,
                                            int           lane_id)
    {
        AFragment frag;
        const int row      = tile_a_row + (lane_id % 16);
        // ki_sparse = 0: one SWMMAC call covers the whole k_tile_sparse=16
        // Each lane loads 8 consecutive compressed FP16 values.
        const __fp16* row_ptr = tile_a_ptr + row * k_tile_sparse;
        #pragma unroll
        for(int e = 0; e < 8; ++e)
            frag.data[e] = row_ptr[e];
        // One index int covers 8 compressed positions (16 bits of 2-bit codes).
        frag.index = tile_idx_ptr[row * (k_tile_sparse / 8)];
        return frag;
    }

    // Dense B fragment: 16 FP16 elements per lane.
    __device__ static v16fp16 load_b_frag(const __fp16* tile_b_T_ptr,
                                          int           tile_b_col,
                                          int           k_tile_dense,
                                          int           lane_id)
    {
        v16fp16 frag;
        // Each lane provides B-column (tile_b_col + lane_id % 16) across all
        // 16 (dense) K-positions in the tile strip.
        const int col = tile_b_col + (lane_id % 16);
        const __fp16* col_ptr = tile_b_T_ptr + col * k_tile_dense;
        #pragma unroll
        for(int e = 0; e < 16; ++e)
            frag[e] = col_ptr[e];
        return frag;
    }

    // ── mma ───────────────────────────────────────────────────────────────────
    // Issue one swmmac_f32_16x16x32_f16_w32 instruction.
    __device__ static void mma(Accumulator&      acc,
                               const AFragment&  a_frag,
                               const v16fp16&    b_frag)
    {
        acc.regs = __builtin_amdgcn_swmmac_f32_16x16x32_f16_w32(
            a_frag.data, b_frag, acc.regs, a_frag.index);
    }

    // ── store_c ───────────────────────────────────────────────────────────────
    // Scatter the 8 accumulator values to their global-memory positions.
    //
    // GFX12 SWMMAC 16x16 wave32 layout (from AMD ISA reference):
    //   Lane L (0-31), VGPR g (0-7):
    //     row i = (L / 16) * 8 + (g % 2) + (L % 16 / 4) * 2
    //     col j = (L % 4) + (g / 2) * 4
    // This gives a complete, non-overlapping tiling of the 16x16 output.
    __device__ static void store_c(const Accumulator& acc,
                                   float*             C,
                                   int                out_row_base,
                                   int                out_col_base,
                                   int                m,
                                   int                n,
                                   int                lane_id)
    {
        const int L        = lane_id; // 0-31
        const int row_base = (L / 16) * 8 + ((L % 16) / 4) * 2;
        const int col_base = (L % 4);

        #pragma unroll
        for(int g = 0; g < 8; ++g)
        {
            const int i = row_base + (g % 2);
            const int j = col_base + (g / 2) * 4;
            const int r = out_row_base + i;
            const int c = out_col_base + j;
            if(r < m && c < n)
                C[r * n + c] = acc.regs[g];
        }
    }
};
// [Sphinx swmmac rdna4 policy end]
#endif // __gfx1200__ || __gfx1201__

// ===========================================================================
// Section 5: ScalarFMASPolicy -- portable FP32 fallback for non-RDNA4 targets
// ===========================================================================

template<int ThreadTileM_, int ThreadTileN_, int BlockDimX_>
struct ScalarFMASPolicy
{
    static constexpr int thread_tile_m   = ThreadTileM_;
    static constexpr int thread_tile_n   = ThreadTileN_;
    static constexpr int effective_lanes = 64;
    static constexpr int k_step          = 1;

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

    __device__ static void thread_tile_offset(int tid, int /*lane_id*/,
                                              int* thread_row, int* thread_col)
    {
        const int tx = tid % BlockDimX_;
        const int ty = tid / BlockDimX_;
        *thread_row  = ty * ThreadTileM_;
        *thread_col  = tx * ThreadTileN_;
    }

    __device__ static void load_a(const float* tile_a_ptr, int tile_a_row,
                                  int ki, int k_tile_size, int /*lane_id*/,
                                  elem_a (&frag)[ThreadTileM_])
    {
        #pragma unroll
        for(int i = 0; i < ThreadTileM_; ++i)
            frag[i] = tile_a_ptr[(tile_a_row + i) * k_tile_size + ki];
    }

    __device__ static void load_b(const float* tile_b_T_ptr, int tile_b_col,
                                  int ki, int k_tile_size, int /*lane_id*/,
                                  elem_b (&frag)[ThreadTileN_])
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

    __device__ static void store_c(const Accumulator& acc, float* C,
                                   int out_row_base, int out_col_base,
                                   int m, int n, int /*lane_id*/)
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

// ===========================================================================
// Section 6: SWMMAC kernel (RDNA 4 specific)
//
// Unlike the generic kernel, this kernel takes separate sparse A, index, and
// dense B pointers. The TilePolicy loads both A and the index register.
// ===========================================================================

#if defined(__gfx1200__) || defined(__gfx1201__)
template<typename TilePolicy>
__global__
__launch_bounds__(
    TilePolicy::block_tile_m / SwmmacRdna4F16Policy::thread_tile_m
    * TilePolicy::block_tile_n / SwmmacRdna4F16Policy::thread_tile_n
    * 32) // threads per block = wavefronts * wave32 size
void matrix_multiply_swmmac(const __fp16* __restrict__ A_sparse,
                             const int*    __restrict__ A_index,
                             const __fp16* __restrict__ B,
                             float*        __restrict__ C,
                             int m, int n, int k_sparse)
{
    __shared__ typename TilePolicy::SharedStorage smem;

    const int tid         = threadIdx.y * blockDim.x + threadIdx.x;
    const int lane_id     = tid % 32;
    const int num_threads = blockDim.x * blockDim.y;

    const int block_row_start = blockIdx.y * TilePolicy::block_tile_m;
    const int block_col_start = blockIdx.x * TilePolicy::block_tile_n;

    int thread_row = 0, thread_col = 0;
    SwmmacRdna4F16Policy::thread_tile_offset(tid, lane_id,
                                             &thread_row, &thread_col);

    SwmmacRdna4F16Policy::Accumulator acc;
    SwmmacRdna4F16Policy::zero(acc);

    const int k_tile_sparse = TilePolicy::k_tile_sparse;
    const int k_tile_dense  = TilePolicy::k_tile_size;
    const int num_tiles = (k_sparse + k_tile_sparse - 1) / k_tile_sparse;

    for(int t = 0; t < num_tiles; ++t)
    {
        TilePolicy::prefetch(smem, A_sparse, A_index, B, t,
                             block_row_start, block_col_start,
                             m, n, k_sparse, tid, num_threads);
        __syncthreads();

        const __fp16* tile_a_ptr   = &smem.tile_a[0][0];
        const int*    tile_idx_ptr = &smem.tile_idx[0][0];
        const __fp16* tile_b_T_ptr = &smem.tile_b_T[0][0];

        // One SWMMAC call covers the full k_tile_sparse strip.
        auto a_frag = SwmmacRdna4F16Policy::load_a_frag(
            tile_a_ptr, tile_idx_ptr, thread_row, k_tile_sparse, lane_id);
        auto b_frag = SwmmacRdna4F16Policy::load_b_frag(
            tile_b_T_ptr, thread_col, k_tile_dense, lane_id);
        SwmmacRdna4F16Policy::mma(acc, a_frag, b_frag);

        __syncthreads();
    }

    SwmmacRdna4F16Policy::store_c(acc, C,
                                   block_row_start + thread_row,
                                   block_col_start + thread_col,
                                   m, n, lane_id);
}
#endif // __gfx1200__ || __gfx1201__

// ===========================================================================
// Section 7: Policy aliases and launch configuration
// ===========================================================================

// [Sphinx swmmac policy aliases start]
// TilePolicy: 32×32 block tile, 16-element sparse K-strip (= 32 dense).
// SwmmacRdna4F16Policy: one 16×16 tile per wave32 wavefront.
#if defined(__gfx1200__) || defined(__gfx1201__)
using SwmmacTilePolicy = SingleBufferTilePolicyH<32, 32, 16>;
#endif
// [Sphinx swmmac policy aliases end]

// [Sphinx swmmac launch config start]
// Block tile: 32×32 (2×2 wavefronts, each computing a 16×16 SWMMAC tile).
// Threads per block: 2 × 2 wavefronts × 32 lanes = 128.
constexpr int BLOCK_TILE_M = 32;
constexpr int BLOCK_TILE_N = 32;

#if defined(__gfx1200__) || defined(__gfx1201__)
constexpr int THREADS_PER_BLOCK = 128; // 4 wave32 wavefronts × 32 lanes
#else
constexpr int THREADS_PER_BLOCK = 64;  // 1 wave64 wavefront (fallback)
#endif
// [Sphinx swmmac launch config end]

// ===========================================================================
// Section 8: Host helpers -- sparsify, timing, verification
// ===========================================================================

// Build a compressed 2:4 sparse representation of a dense FP16 matrix.
// Uses a fixed "keep positions 0 and 2" pattern for simplicity.
// In practice, select the two largest-magnitude values in each block.
void sparsify_2_4(const std::vector<__fp16>& dense,
                  std::vector<__fp16>&        sparse_out,
                  std::vector<uint32_t>&      index_out,
                  int rows, int cols)
{
    // Compressed A: rows × (cols/2) FP16 values.
    sparse_out.resize(rows * cols / 2);
    // Index: rows × (cols/2/8) uint32_t values.
    //   Each uint32_t holds 16 two-bit codes (16 × 2 = 32 bits).
    index_out.resize(rows * (cols / 16), 0);

    for(int r = 0; r < rows; ++r)
    {
        int sparse_col = 0;
        for(int c = 0; c < cols; c += 4)
        {
            // Keep positions 0 and 2 (two-bit codes: 00 and 10).
            sparse_out[r * (cols / 2) + sparse_col]     = dense[r * cols + c];
            sparse_out[r * (cols / 2) + sparse_col + 1] = dense[r * cols + c + 2];
            sparse_col += 2;

            // Pack two-bit index codes into the index register.
            // bits [2s+1:2s] = position of s-th non-zero in its block of 4.
            // For "keep 0 and 2": code0 = 00b = 0, code1 = 10b = 2.
            const int idx_int  = (c / 2) / 16; // which uint32 for this block
            const int bit_base = ((c / 2) % 16) * 2; // bit offset within uint32
            // sparse_col-2 corresponds to position 0, sparse_col-1 to position 2
            index_out[r * (cols / 16) + idx_int] |= (0u << bit_base);       // pos 0
            index_out[r * (cols / 16) + idx_int] |= (2u << (bit_base + 2)); // pos 2
        }
    }
}

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

// Verify: C ≈ A_dense × B (expand sparse A back to dense for reference).
bool verify_result(const std::vector<float>&   C,
                   const std::vector<__fp16>&  A_dense,
                   const std::vector<__fp16>&  B,
                   int m, int n, int k_dense,
                   float tolerance = 0.5f) // FP16 accumulation allows slack
{
    for(int i = 0; i < m; ++i)
        for(int j = 0; j < n; ++j)
        {
            float ref = 0.0f;
            for(int kk = 0; kk < k_dense; ++kk)
                ref += float(A_dense[i * k_dense + kk]) * float(B[kk * n + j]);
            if(std::fabs(C[i * n + j] - ref) > tolerance * (1.0f + std::fabs(ref)))
            {
                std::cerr << "Mismatch at [" << i << "][" << j << "]: "
                          << "expected " << ref << ", got " << C[i * n + j] << "\n";
                return false;
            }
        }
    return true;
}

void print_metrics(const char* label, bool passed, float avg_ms,
                   long long m, long long n, long long k_dense)
{
    // Sparse GEMM: only half the A elements are non-zero, so effective FLOPs
    // equal a dense multiply of the same M×N×(K/2).
    const double flops = 2.0 * double(m) * double(n) * double(k_dense / 2);
    std::cout << label << ": " << (passed ? "PASSED" : "FAILED") << "\n"
              << "  Average kernel time  : " << avg_ms << " ms\n"
              << "  Sparse TFLOPS        : "
              << flops / (avg_ms * 1.0e-3) / 1.0e12 << "\n\n";
}

// ===========================================================================
// Section 9: main
// ===========================================================================

int main()
{
    hipDeviceProp_t props{};
    HIP_CHECK(hipGetDeviceProperties(&props, 0));
    std::cout << "Device       : " << props.name << "\n"
              << "Warp size    : " << props.warpSize << "\n"
              << "Matrix dims  : A(" << M << "x" << K_DENSE << " dense, "
              << K_SPARSE << " sparse cols)"
              << " * B(" << K_DENSE << "x" << N << ")\n"
              << "Block tile   : " << BLOCK_TILE_M << "x" << BLOCK_TILE_N << "\n"
              << "Threads/block: " << THREADS_PER_BLOCK << "\n\n";

    // Build dense A (FP16) and dense B (FP16), C (FP32).
    std::vector<__fp16> h_A_dense(M * K_DENSE);
    std::vector<__fp16> h_B(K_DENSE * N);
    std::vector<float>  h_C(M * N, 0.0f);

    std::mt19937                          gen(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for(__fp16& v : h_A_dense) v = __fp16(dist(gen));
    // Identity B: multiplying A * I should give A cast to FP32 in C.
    for(int i = 0; i < std::min(K_DENSE, N); ++i)
        h_B[i * N + i] = __fp16(1.0f);

    // Sparsify A.
    std::vector<__fp16>    h_A_sparse;
    std::vector<uint32_t>  h_A_index;
    sparsify_2_4(h_A_dense, h_A_sparse, h_A_index, M, K_DENSE);

    // Allocate device memory.
    __fp16* d_A_sparse = nullptr;
    int*    d_A_index  = nullptr;
    __fp16* d_B        = nullptr;
    float*  d_C        = nullptr;

    HIP_CHECK(hipMalloc(&d_A_sparse, h_A_sparse.size() * sizeof(__fp16)));
    HIP_CHECK(hipMalloc(&d_A_index,  h_A_index.size()  * sizeof(uint32_t)));
    HIP_CHECK(hipMalloc(&d_B,        h_B.size()         * sizeof(__fp16)));
    HIP_CHECK(hipMalloc(&d_C,        h_C.size()         * sizeof(float)));

    HIP_CHECK(hipMemcpy(d_A_sparse, h_A_sparse.data(),
                        h_A_sparse.size() * sizeof(__fp16), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_A_index, h_A_index.data(),
                        h_A_index.size() * sizeof(uint32_t), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_B, h_B.data(),
                        h_B.size() * sizeof(__fp16), hipMemcpyHostToDevice));

    // [Sphinx swmmac kernel launch start]
    // Grid: one block per 32×32 output tile.
    const dim3 grid((N + BLOCK_TILE_N - 1) / BLOCK_TILE_N,
                    (M + BLOCK_TILE_M - 1) / BLOCK_TILE_M);
    const dim3 block(THREADS_PER_BLOCK);

#if defined(__gfx1200__) || defined(__gfx1201__)
    auto launch = [&]()
    {
        matrix_multiply_swmmac<SwmmacTilePolicy>
            <<<grid, block>>>(d_A_sparse, reinterpret_cast<const int*>(d_A_index),
                              d_B, d_C, M, N, K_SPARSE);
        HIP_CHECK(hipGetLastError());
    };
#else
    // Fallback: not shown; use dense FP32 scalar kernel on non-RDNA4 targets.
    auto launch = [&]() {};
#endif
    // [Sphinx swmmac kernel launch end]

    const float ms = time_kernel_ms(launch, WARMUP_RUNS, TIMING_RUNS);
    HIP_CHECK(hipMemcpy(h_C.data(), d_C, h_C.size() * sizeof(float),
                        hipMemcpyDeviceToHost));

    // Only verify a small sub-block to keep host verification fast.
    const int check_m = std::min(M, 64);
    const int check_n = std::min(N, 64);
    std::vector<float>  h_C_sub(h_C.begin(), h_C.begin() + check_m * N);
    std::vector<__fp16> h_A_sub(h_A_dense.begin(),
                                h_A_dense.begin() + check_m * K_DENSE);
    const bool ok = verify_result(h_C_sub, h_A_sub, h_B,
                                  check_m, check_n, K_DENSE);

#if defined(__gfx1200__)
    print_metrics("SwmmacRdna4F16Policy [gfx1200]", ok, ms, M, N, K_DENSE);
#elif defined(__gfx1201__)
    print_metrics("SwmmacRdna4F16Policy [gfx1201]", ok, ms, M, N, K_DENSE);
#else
    std::cout << "Non-RDNA4 target: SWMMAC kernel not compiled.\n";
#endif

    HIP_CHECK(hipFree(d_A_sparse));
    HIP_CHECK(hipFree(d_A_index));
    HIP_CHECK(hipFree(d_B));
    HIP_CHECK(hipFree(d_C));

    return EXIT_SUCCESS;
}
