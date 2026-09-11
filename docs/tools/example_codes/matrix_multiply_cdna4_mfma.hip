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
// matrix_multiply_cdna4_mfma.hip  --  FP16 MFMA ComputePolicy for CDNA4
//                                     (gfx950, MI350)
//
// This file is a companion to matrix_multiply_generic.hip (Step 7) and
// provides a concrete ComputePolicy implementation that uses the MFMA
// instruction v_mfma_f32_16x16x32_f16 available on CDNA4 GPUs
// (gfx950, MI350).
//
// The instruction computes a 16x16 FP32 output tile per wavefront.  Each
// call consumes K=32 FP16 elements: the 64 lanes form 4 groups of 16
// (g = lane_id / 16).  Each group covers 8 consecutive K-positions
// (ki + 8*g through ki + 8*g + 7), so each lane supplies 8 FP16 values.
//
// The policy is a drop-in replacement for ScalarFMAPolicy: the generic
// kernel template (matrix_multiply_generic) is reused without modification.
//
// Architecture: CDNA4 (gfx950, MI350) only.
// Compile with: amdclang++ -O3 -std=c++17 --offload-arch=gfx950 \
//                   matrix_multiply_cdna4_mfma.hip -o mm_cdna4_mfma
//
// The code is guarded by a preprocessor check so that it compiles on other
// architectures (ScalarFMAPolicy is used as a fallback) without emitting
// undefined-behaviour warnings or hard errors.
// ---------------------------------------------------------------------------

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

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
// Section 2: Cooperative tile load helpers (FP16 shared memory variant)
// ===========================================================================

__device__ inline void cooperative_load_tile_a_f16(_Float16*       dest,
                                                   const _Float16* A,
                                                   int             block_row_start,
                                                   int             block_tile_m,
                                                   int             t,
                                                   int             k_tile_size,
                                                   int             m,
                                                   int             k,
                                                   int             tid,
                                                   int             num_threads)
{
    const int tile_elems = block_tile_m * k_tile_size;
    for(int idx = tid; idx < tile_elems; idx += num_threads)
    {
        const int tile_row   = idx / k_tile_size;
        const int tile_col   = idx % k_tile_size;
        const int global_row = block_row_start + tile_row;
        const int global_col = t * k_tile_size + tile_col;
        dest[tile_row * k_tile_size + tile_col] =
            (global_row < m && global_col < k)
                ? A[global_row * k + global_col]
                : static_cast<_Float16>(0.0f);
    }
}

__device__ inline void cooperative_load_tile_b_T_f16(_Float16*       dest,
                                                     const _Float16* B,
                                                     int             block_col_start,
                                                     int             block_tile_n,
                                                     int             t,
                                                     int             k_tile_size,
                                                     int             k,
                                                     int             n,
                                                     int             tid,
                                                     int             num_threads)
{
    const int tile_elems = block_tile_n * k_tile_size;
    for(int idx = tid; idx < tile_elems; idx += num_threads)
    {
        const int tile_col_idx = idx / k_tile_size;
        const int tile_k_idx   = idx % k_tile_size;
        const int global_row   = t * k_tile_size + tile_k_idx;
        const int global_col   = block_col_start + tile_col_idx;
        dest[tile_col_idx * k_tile_size + tile_k_idx] =
            (global_row < k && global_col < n)
                ? B[global_row * n + global_col]
                : static_cast<_Float16>(0.0f);
    }
}

// ===========================================================================
// Section 3: TilePolicy implementation (FP16 shared memory)
//
// The 16×16 FP16 MFMA instruction has k_step=32, meaning each call consumes
// 32 K-positions.  One call per K-strip suffices when k_tile_size=32.
// The 32×32 block tile uses 2×2 wavefronts, so the shared-memory tile is
// 32×32 in M/N with a K-strip of 32.
// ===========================================================================

// FP32 tile policy for the non-CDNA4 scalar fallback path.
template<int BlockTileM_, int BlockTileN_, int KTileSize_>
struct SingleBufferTilePolicyF
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
        const int t          = t_next;
        const int tile_elems = block_tile_m * k_tile_size;
        for(int idx = tid; idx < tile_elems; idx += num_threads)
        {
            const int tile_row   = idx / k_tile_size;
            const int tile_col   = idx % k_tile_size;
            const int global_row = block_row_start + tile_row;
            const int global_col = t * k_tile_size + tile_col;
            smem.tile_a[tile_row][tile_col] =
                (global_row < m && global_col < k)
                    ? A[global_row * k + global_col] : 0.0f;
        }
        const int tile_b_elems = block_tile_n * k_tile_size;
        for(int idx = tid; idx < tile_b_elems; idx += num_threads)
        {
            const int tile_col_idx = idx / k_tile_size;
            const int tile_k_idx   = idx % k_tile_size;
            const int global_row   = t * k_tile_size + tile_k_idx;
            const int global_col   = block_col_start + tile_col_idx;
            smem.tile_b_T[tile_col_idx][tile_k_idx] =
                (global_row < k && global_col < n)
                    ? B[global_row * n + global_col] : 0.0f;
        }
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

// [Sphinx single buffer policy f16 start]
template<int BlockTileM_, int BlockTileN_, int KTileSize_>
struct SingleBufferTilePolicyF16
{
    static constexpr int num_buffers  = 1;
    static constexpr int block_tile_m = BlockTileM_;
    static constexpr int block_tile_n = BlockTileN_;
    static constexpr int k_tile_size  = KTileSize_;

    struct SharedStorage
    {
        _Float16 tile_a[BlockTileM_][KTileSize_];
        _Float16 tile_b_T[BlockTileN_][KTileSize_];
    };

    __device__ static void prologue(SharedStorage& /*smem*/,
                                    const _Float16* /*A*/, const _Float16* /*B*/,
                                    int /*block_row_start*/, int /*block_col_start*/,
                                    int /*m*/, int /*n*/, int /*k*/,
                                    int /*tid*/, int /*num_threads*/) {}

    __device__ static void prefetch(SharedStorage&  smem,
                                    const _Float16* A,
                                    const _Float16* B,
                                    int             t_next,
                                    int block_row_start, int block_col_start,
                                    int m, int n, int k,
                                    int tid, int num_threads)
    {
        const int t = t_next;
        cooperative_load_tile_a_f16(
            &smem.tile_a[0][0], A, block_row_start,
            block_tile_m, t, k_tile_size, m, k, tid, num_threads);
        cooperative_load_tile_b_T_f16(
            &smem.tile_b_T[0][0], B, block_col_start,
            block_tile_n, t, k_tile_size, k, n, tid, num_threads);
    }

    __device__ static void acquire(int /*t*/) { __syncthreads(); }
    __device__ static void release(int /*t*/) { __syncthreads(); }
    __device__ static int  buf_idx(int /*t*/) { return 0; }

    __device__ static _Float16* get_tile_a(SharedStorage& smem, int /*buf*/)
    {
        return &smem.tile_a[0][0];
    }
    __device__ static _Float16* get_tile_b_T(SharedStorage& smem, int /*buf*/)
    {
        return &smem.tile_b_T[0][0];
    }
};
// [Sphinx single buffer policy f16 end]

// ===========================================================================
// Section 4: MfmaCdna4F16Policy -- MFMA ComputePolicy for gfx950 (CDNA4)
// ===========================================================================
//
// This policy uses v_mfma_f32_16x16x32_f16 to compute one 16×16 FP32 output
// tile per wavefront.  Each block contains 2×2 = 4 wavefronts, giving a
// 32×32 block tile.
//
// Accumulator layout (16×16 tile, 1 block):
//   Each lane holds 4 FP32 values in accVGPRs 0--3.
//
//   Given lane L and accVGPR index G (0--3):
//     i = 4 * floor(L / 16) + (G mod 4)
//     j = L mod 16
//
// Fragment distribution:
//   v_mfma_f32_16x16x32_f16 (1-block) has a 16×32 A-operand and a 32×16
//   B-operand.  Each lane provides eight FP16 values for A (its row in
//   [0,15]) and eight FP16 values for B (its column in [0,15]).
//
//   The 64 lanes form 4 groups of 16 (g = lane_id / 16).  Each group covers
//   8 consecutive K-positions: group g provides A and B data for
//   K-positions ki + 8*g through ki + 8*g + 7.
//
//   Within group g, lane offset (lane_id mod 16) selects one of the 16
//   A-rows (or B-columns):
//     lane L, elem e (0--7): A[tile_a_row + lane_id%16][ki + 8*(lane_id/16) + e]
// ===========================================================================

// [Sphinx mfma cdna4 policy start]
struct MfmaCdna4F16Policy
{
    // -- ComputePolicy constants ----------------------------------------------
    // One wavefront owns a 16×16 output tile.
    // All 64 lanes hold unique output elements (4 elements per lane).
    static constexpr int thread_tile_m   = 16;
    static constexpr int thread_tile_n   = 16;
    static constexpr int frag_size_m     = 8;
    static constexpr int frag_size_n     = 8;
    static constexpr int effective_lanes = 64;
    // v_mfma_f32_16x16x32_f16 consumes 32 K-positions per call (k_step=32).
    // Each lane provides eight FP16 values (v8half) for A and eight for B.
    // The 64 lanes split into 4 groups of 16; each group covers 8 K-positions.
    static constexpr int k_step          = 32;

    using elem_a = _Float16;
    using elem_b = _Float16;

    // -- Accumulator ----------------------------------------------------------
    // v4float holds 4 FP32 accVGPRs per lane (accVGPRs 0--3).
    using v4float = float [[clang::ext_vector_type(4)]];

    struct Accumulator
    {
        v4float regs;
    };

    __device__ static void zero(Accumulator& acc)
    {
        acc.regs = v4float{};
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
    // load_a: each lane reads eight FP16 values from LDS into a v8half.
    //
    // v_mfma_f32_16x16x32_f16 (1-block) interprets the 64 lanes as 4 groups
    // of 16 (indexed by g = lane_id / 16).  Each group covers 8 consecutive
    // K-positions: group g provides data for ki + 8*g through ki + 8*g + 7.
    // Within group g, lane offset (lane_id mod 16) selects one of the 16
    // A-rows.
    //
    //   A-row index : tile_a_row + (lane_id mod 16)
    //   K-position e: ki + 8*(lane_id / 16) + e   (e = 0..7)
    __device__ static void load_a(const _Float16* tile_a_ptr,
                                  int             tile_a_row,
                                  int             ki,
                                  int             k_tile_size,
                                  int             lane_id,
                                  elem_a        (&frag)[8])
    {
        const int row   = tile_a_row + (lane_id % 16);
        const int k_off = ki + 8 * (lane_id / 16);
        #pragma unroll
        for(int e = 0; e < 8; ++e)
            frag[e] = tile_a_ptr[row * k_tile_size + k_off + e];
    }

    // load_b: each lane reads eight FP16 values from LDS (symmetric to load_a).
    //
    // tile_b_T is stored transposed: tile_b_T[(tile_b_col + col)][k_off].
    //   B-col index : tile_b_col + (lane_id mod 16)
    //   K-position e: ki + 8*(lane_id / 16) + e   (e = 0..7)
    __device__ static void load_b(const _Float16* tile_b_T_ptr,
                                  int             tile_b_col,
                                  int             ki,
                                  int             k_tile_size,
                                  int             lane_id,
                                  elem_b        (&frag)[8])
    {
        const int col   = tile_b_col + (lane_id % 16);
        const int k_off = ki + 8 * (lane_id / 16);
        #pragma unroll
        for(int e = 0; e < 8; ++e)
            frag[e] = tile_b_T_ptr[col * k_tile_size + k_off + e];
    }

    // -- mma -------------------------------------------------------------------
    // Issue one v_mfma_f32_16x16x32_f16 instruction.
    // This intrinsic is single-block only; all modifier parameters must be 0.
    // cbsz=0, abid=0: single-block A-operand, no broadcast.
    // blgp=0: not supported; must be 0.
    __device__ static void mma(Accumulator&       acc,
                               const elem_a     (&a_frag)[8],
                               const elem_b     (&b_frag)[8])
    {
#if defined(__gfx950__)
        using v8half = _Float16 [[clang::ext_vector_type(8)]];
        const v8half a_vec = {a_frag[0], a_frag[1], a_frag[2], a_frag[3],
                              a_frag[4], a_frag[5], a_frag[6], a_frag[7]};
        const v8half b_vec = {b_frag[0], b_frag[1], b_frag[2], b_frag[3],
                              b_frag[4], b_frag[5], b_frag[6], b_frag[7]};
        acc.regs = __builtin_amdgcn_mfma_f32_16x16x32_f16(
            a_vec, b_vec, acc.regs,
            /*cbsz=*/0, /*abid=*/0, /*blgp=*/0);
#endif
    }

    // -- store_c ---------------------------------------------------------------
    // Scatter the 4 accVGPR values to their global-memory positions.
    //
    // Inverse layout (lane L, accVGPR index G in [0,3]):
    //   i = 4 * floor(L / 16) + (G mod 4)
    //   j = L mod 16
    __device__ static void store_c(const Accumulator& acc,
                                   float*             C,
                                   int                out_row_base,
                                   int                out_col_base,
                                   int                m,
                                   int                n,
                                   int                lane_id)
    {
        const int L = lane_id;
        const int j = L % 16;

        #pragma unroll
        for(int G = 0; G < 4; ++G)
        {
            const int i = 4 * (L / 16) + (G % 4);
            const int r = out_row_base + i;
            const int c = out_col_base + j;
            if(r < m && c < n)
                C[r * n + c] = acc.regs[G];
        }
    }
};
// [Sphinx mfma cdna4 policy end]

// ===========================================================================
// Section 4b: TilePolicy with row-major B storage for ds_read_tr path
// ===========================================================================
//
// ds_read_tr16_b64_v4f16 expects B to be stored in LDS in row-major order
// (K rows × N columns): B[k][n].  The standard FP16 tile policy stores B
// transposed (N rows × K columns) to support scalar lane indexing.  This
// policy stores B in its natural row-major layout instead, so that
// ds_read_tr can perform the column-to-lane scatter in hardware.
//
// The A tile is stored identically to SingleBufferTilePolicyF16 (row-major
// in the M×K sense).
// ===========================================================================

// [Sphinx tile policy f16 row b start]
template<int BlockTileM_, int BlockTileN_, int KTileSize_>
struct SingleBufferTilePolicyF16RowB
{
    static constexpr int num_buffers  = 1;
    static constexpr int block_tile_m = BlockTileM_;
    static constexpr int block_tile_n = BlockTileN_;
    static constexpr int k_tile_size  = KTileSize_;

    struct SharedStorage
    {
        _Float16 tile_a[BlockTileM_][KTileSize_];
        // B stored row-major: tile_b[k][n], so that ds_read_tr can transpose
        // K-rows into per-lane N-column fragments in hardware.
        _Float16 tile_b[KTileSize_][BlockTileN_];
    };

    __device__ static void prologue(SharedStorage& /*smem*/,
                                    const _Float16* /*A*/, const _Float16* /*B*/,
                                    int /*block_row_start*/, int /*block_col_start*/,
                                    int /*m*/, int /*n*/, int /*k*/,
                                    int /*tid*/, int /*num_threads*/) {}

    __device__ static void prefetch(SharedStorage&  smem,
                                    const _Float16* A,
                                    const _Float16* B,
                                    int             t_next,
                                    int block_row_start, int block_col_start,
                                    int m, int n, int k,
                                    int tid, int num_threads)
    {
        const int t = t_next;

        // Load A tile: row-major, M rows × K cols.
        const int tile_a_elems = block_tile_m * k_tile_size;
        for(int idx = tid; idx < tile_a_elems; idx += num_threads)
        {
            const int tile_row   = idx / k_tile_size;
            const int tile_col   = idx % k_tile_size;
            const int global_row = block_row_start + tile_row;
            const int global_col = t * k_tile_size + tile_col;
            smem.tile_a[tile_row][tile_col] =
                (global_row < m && global_col < k)
                    ? A[global_row * k + global_col]
                    : static_cast<_Float16>(0.0f);
        }

        // Load B tile: row-major K × N layout in LDS.
        const int tile_b_elems = k_tile_size * block_tile_n;
        for(int idx = tid; idx < tile_b_elems; idx += num_threads)
        {
            const int tile_k_idx   = idx / block_tile_n;
            const int tile_col_idx = idx % block_tile_n;
            const int global_row   = t * k_tile_size + tile_k_idx;
            const int global_col   = block_col_start + tile_col_idx;
            smem.tile_b[tile_k_idx][tile_col_idx] =
                (global_row < k && global_col < n)
                    ? B[global_row * n + global_col]
                    : static_cast<_Float16>(0.0f);
        }
    }

    __device__ static void acquire(int /*t*/) { __syncthreads(); }
    __device__ static void release(int /*t*/) { __syncthreads(); }
    __device__ static int  buf_idx(int /*t*/) { return 0; }

    __device__ static _Float16* get_tile_a(SharedStorage& smem, int /*buf*/)
    {
        return &smem.tile_a[0][0];
    }
    // Returns the row-major B tile base pointer; the ds_read_tr policy will
    // compute per-lane pointers into this region.
    __device__ static _Float16* get_tile_b_T(SharedStorage& smem, int /*buf*/)
    {
        return &smem.tile_b[0][0];
    }
};
// [Sphinx tile policy f16 row b end]

// ===========================================================================
// Section 4c: MfmaCdna4F16TrPolicy -- FP16 MFMA with ds_read_tr B loads
// ===========================================================================
//
// This policy uses the same v_mfma_f32_16x16x32_f16 instruction as
// MfmaCdna4F16Policy, but replaces the scalar B-operand load loop with two
// ds_read_tr16_b64_v4f16 calls.
//
// ds_read_tr16_b64_v4f16 loads 64 bits (four FP16 values) per lane and
// performs a 16-element transpose across adjacent lanes as part of the load.
// One call covers the first 4 K-positions per lane group; a second call
// with a pointer offset of 4 * BlockTileN * sizeof(__fp16) covers the
// remaining 4 K-positions.
//
// For k_step=32 (the full MFMA K depth) the B fragment is loaded in two
// ds_read_tr16_b64_v4f16 calls.  Each call contributes four FP16 values
// (v4half), giving eight FP16 values per lane in total -- exactly the
// B operand width of v_mfma_f32_16x16x32_f16.
//
// A-operand loads remain scalar (identical to MfmaCdna4F16Policy::load_a).
// ===========================================================================

// [Sphinx mfma cdna4 tr policy start]
struct MfmaCdna4F16TrPolicy
{
    // -- ComputePolicy constants ----------------------------------------------
    static constexpr int thread_tile_m   = 16;
    static constexpr int thread_tile_n   = 16;
    static constexpr int frag_size_m     = 8;
    static constexpr int frag_size_n     = 8;
    static constexpr int effective_lanes = 64;
    // k_step=32: one v_mfma_f32_16x16x32_f16 call consumes 32 K-positions,
    // filled by two ds_read_tr16_b64_v4f16 calls.
    static constexpr int k_step          = 32;

    using elem_a = _Float16;
    using elem_b = _Float16;

    // Accumulator and zero() are identical to MfmaCdna4F16Policy.
    using v4float = float [[clang::ext_vector_type(4)]];

    struct Accumulator
    {
        v4float regs;
    };

    __device__ static void zero(Accumulator& acc)
    {
        acc.regs = v4float{};
    }

    // thread_tile_offset is identical to MfmaCdna4F16Policy.
    __device__ static void thread_tile_offset(int  tid,
                                              int  /*lane_id*/,
                                              int* thread_row,
                                              int* thread_col)
    {
        constexpr int waves_n = 2;
        const int wid   = tid / 64;
        const int wid_x = wid % waves_n;
        const int wid_y = wid / waves_n;
        *thread_row = wid_y * 16;
        *thread_col = wid_x * 16;
    }

    // -- load_a ---------------------------------------------------------------
    // Scalar load, identical to MfmaCdna4F16Policy::load_a.
    __device__ static void load_a(const _Float16* tile_a_ptr,
                                  int             tile_a_row,
                                  int             ki,
                                  int             k_tile_size,
                                  int             lane_id,
                                  elem_a        (&frag)[8])
    {
        const int row   = tile_a_row + (lane_id % 16);
        const int k_off = ki + 8 * (lane_id / 16);
        #pragma unroll
        for(int e = 0; e < 8; ++e)
            frag[e] = tile_a_ptr[row * k_tile_size + k_off + e];
    }

    // -- load_b ---------------------------------------------------------------
    // Hardware-transposed load using ds_read_tr16_b64_v4f16.
    //
    // tile_b_ptr points to the base of the row-major B tile in LDS:
    //   tile_b[k][n], dimensions KTileSize × BlockTileN.
    //
    // Each lane belongs to one of four 16-lane groups (g = lane_id / 16),
    // handling K-positions ki + 8*g through ki + 8*g + 7.  Two ds_read_tr
    // calls fill the eight-element B fragment per lane:
    //
    //   local_tx = lane_id % 16
    //   k_base   = ki + 8 * (lane_id / 16)
    //
    //   Call 1: ptr = &tile_b[k_base + local_tx/4][tile_b_col + (local_tx%4)*4]
    //           -> frag[0..3]
    //   Call 2: ptr = &tile_b[k_base + 4 + local_tx/4][tile_b_col + (local_tx%4)*4]
    //           -> frag[4..7]
    //
    // The stride between the two calls is 4 rows of the B tile (4 * BlockTileN
    // elements).  The hardware transpose gathers strided K elements across
    // adjacent lanes so that each lane receives its own column's data.
    __device__ static void load_b(const _Float16* tile_b_ptr,
                                  int             tile_b_col,
                                  int             ki,
                                  int             block_tile_n,
                                  int             lane_id,
                                  elem_b        (&frag)[8])
    {
#if defined(__gfx950__)
        using v4half = __fp16 [[clang::ext_vector_type(4)]];

        const int local_tx = lane_id % 16;
        const int k_base   = ki + 8 * (lane_id / 16);

        __fp16* base0 =
            const_cast<__fp16*>(reinterpret_cast<const __fp16*>(tile_b_ptr))
            + (k_base + local_tx / 4) * block_tile_n
            + tile_b_col + (local_tx % 4) * 4;

        __fp16* base1 =
            const_cast<__fp16*>(reinterpret_cast<const __fp16*>(tile_b_ptr))
            + (k_base + 4 + local_tx / 4) * block_tile_n
            + tile_b_col + (local_tx % 4) * 4;

        const v4half r0 = __builtin_amdgcn_ds_read_tr16_b64_v4f16(
            (__attribute__((address_space(3))) v4half*)(base0));
        const v4half r1 = __builtin_amdgcn_ds_read_tr16_b64_v4f16(
            (__attribute__((address_space(3))) v4half*)(base1));

        frag[0] = r0[0]; frag[1] = r0[1]; frag[2] = r0[2]; frag[3] = r0[3];
        frag[4] = r1[0]; frag[5] = r1[1]; frag[6] = r1[2]; frag[7] = r1[3];
#endif
    }

    // mma and store_c are identical to MfmaCdna4F16Policy.
    __device__ static void mma(Accumulator&       acc,
                               const elem_a     (&a_frag)[8],
                               const elem_b     (&b_frag)[8])
    {
#if defined(__gfx950__)
        using v8half = _Float16 [[clang::ext_vector_type(8)]];
        const v8half a_vec = {a_frag[0], a_frag[1], a_frag[2], a_frag[3],
                              a_frag[4], a_frag[5], a_frag[6], a_frag[7]};
        const v8half b_vec = {b_frag[0], b_frag[1], b_frag[2], b_frag[3],
                              b_frag[4], b_frag[5], b_frag[6], b_frag[7]};
        acc.regs = __builtin_amdgcn_mfma_f32_16x16x32_f16(
            a_vec, b_vec, acc.regs,
            /*cbsz=*/0, /*abid=*/0, /*blgp=*/0);
#endif
    }

    __device__ static void store_c(const Accumulator& acc,
                                   float*             C,
                                   int                out_row_base,
                                   int                out_col_base,
                                   int                m,
                                   int                n,
                                   int                lane_id)
    {
        const int L = lane_id;
        const int j = L % 16;
        #pragma unroll
        for(int G = 0; G < 4; ++G)
        {
            const int i = 4 * (L / 16) + (G % 4);
            const int r = out_row_base + i;
            const int c = out_col_base + j;
            if(r < m && c < n)
                C[r * n + c] = acc.regs[G];
        }
    }
};
// [Sphinx mfma cdna4 tr policy end]

// ===========================================================================
// Section 5: ScalarFMAPolicy -- portable FP32 fallback for non-CDNA4 targets
// ===========================================================================

// [Sphinx scalar fma policy start]
template<int ThreadTileM_, int ThreadTileN_, int BlockDimX_>
struct ScalarFMAPolicy
{
    static constexpr int thread_tile_m   = ThreadTileM_;
    static constexpr int thread_tile_n   = ThreadTileN_;
    static constexpr int frag_size_m     = ThreadTileM_;
    static constexpr int frag_size_n     = ThreadTileN_;
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
                                  elem_a     (&frag)[ThreadTileM_])
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
                                  elem_b     (&frag)[ThreadTileN_])
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
// ===========================================================================
//
// The kernel is templated on ElemIn (the shared-memory / operand element
// type) so that it works with both the FP16 CDNA4 policy and the FP32
// fallback policy.
// ===========================================================================

// [Sphinx gemm kernel start]
template<typename TilePolicy, typename ComputePolicy, typename ElemIn>
__global__ void
    __launch_bounds__(TilePolicy::block_tile_m / ComputePolicy::thread_tile_m
                          * TilePolicy::block_tile_n / ComputePolicy::thread_tile_n
                          * 64) // threads per block = wavefronts * wavefront_size
    matrix_multiply_generic(const ElemIn* __restrict__ A,
                            const ElemIn* __restrict__ B,
                            float* __restrict__        C,
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
        const ElemIn* tile_a_ptr   = TilePolicy::get_tile_a(smem, buf);
        const ElemIn* tile_b_T_ptr = TilePolicy::get_tile_b_T(smem, buf);

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
// Block tile: 32×32 (2×2 wavefronts, each computing a 16×16 MFMA tile).
// K-strip  :  32 (k_tile_size=32, k_step=32 → 1 MFMA call per tile).
//
// Thread grid:
//   Each block spans BLOCK_TILE_M rows and BLOCK_TILE_N cols of C.
//   Blocks per grid: ceil(M/32) × ceil(N/32).
//   Threads per block: (BLOCK_TILE_M/16) * (BLOCK_TILE_N/16) wavefronts * 64 lanes
//                    = 2 * 2 * 64 = 256.
// ===========================================================================

// [Sphinx mfma policy aliases start]
// TilePolicy for the standard FP16 path (transposed B in LDS).
using MfmaTilePolicy    = SingleBufferTilePolicyF16<32, 32, 32>;
// TilePolicy for the ds_read_tr path (row-major B in LDS).
using MfmaTilePolicyTrB = SingleBufferTilePolicyF16RowB<32, 32, 32>;
// [Sphinx mfma policy aliases end]

// [Sphinx mfma launch config start]
// Launch parameters:
//
// CDNA4 (gfx950): 2×2 wavefronts of 64 lanes = 256 threads per block.
//   block_tile_m = 32, thread_tile_m = 16  → 2 wavefronts along M
//   block_tile_n = 32, thread_tile_n = 16  → 2 wavefronts along N
//
// Fallback (scalar): 8×8 thread tiles in a 32×32 block → 16 threads per
//   block, padded to one full wavefront of 64.
//
// The kernel to launch and its thread count are selected at runtime based
// on the device architecture string returned by hipGetDeviceProperties.
//
// Grid: one block per 32×32 output tile.
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

// Verify by multiplying A * I (identity) and comparing with A.
// A element type is templated so this works for both FP16 and FP32 paths.
template<typename ElemA>
bool verify_result(const std::vector<float>&   C,
                   const std::vector<ElemA>&   A,
                   int                         m,
                   int                         n,
                   int                         k,
                   float                       tolerance = 1.0e-2f)
{
    const int check_cols = std::min(k, n);
    for(int i = 0; i < m; ++i)
        for(int j = 0; j < check_cols; ++j)
        {
            const float a_val = static_cast<float>(A[i * k + j]);
            if(std::fabs(C[i * n + j] - a_val) > tolerance)
            {
                std::cerr << "Mismatch at [" << i << "][" << j << "]: "
                          << "expected " << a_val
                          << ", got " << C[i * n + j] << "\n";
                return false;
            }
        }
    return true;
}

void print_metrics(const char* label,
                   bool        passed,
                   float       avg_ms,
                   std::size_t elem_bytes,
                   long long   m,
                   long long   n,
                   long long   k)
{
    const double bytes_min =
        static_cast<double>(elem_bytes) * static_cast<double>(m * k + k * n)
        + sizeof(float) * static_cast<double>(m * n);
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
    const bool is_gfx950 =
        std::string(props.gcnArchName).rfind("gfx950", 0) == 0;
    const int threads_per_block = is_gfx950 ? 256 : 64;
    std::cout << "Device       : " << props.name << "\n"
              << "GCN arch     : " << props.gcnArchName << "\n"
              << "Warp size    : " << props.warpSize << "\n"
              << "Matrix dims  : A(" << M << "x" << K << ") * B("
              << K << "x" << N << ")\n"
              << "Block tile   : " << BLOCK_TILE_M << "x" << BLOCK_TILE_N
              << "  K-strip: " << MfmaTilePolicy::k_tile_size << "\n"
              << "Threads/block: " << threads_per_block << "\n\n";

    // A and B are always stored as FP16 on the host; the scalar fallback
    // path casts to float inside the kernel via the FP32 tile policy.
    std::vector<_Float16> h_A(M * K);
    std::vector<_Float16> h_B(K * N, static_cast<_Float16>(0.0f));
    std::vector<float>    h_C(M * N, 0.0f);

    std::mt19937                          gen(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for(_Float16& v : h_A)
        v = static_cast<_Float16>(dist(gen));
    // Identity matrix: multiplying A * I should give A in C.
    for(int i = 0; i < std::min(K, N); ++i)
        h_B[i * N + i] = static_cast<_Float16>(1.0f);

    _Float16* d_A;
    _Float16* d_B;
    float*    d_C;
    HIP_CHECK(hipMalloc(&d_A, sizeof(_Float16) * M * K));
    HIP_CHECK(hipMalloc(&d_B, sizeof(_Float16) * K * N));
    HIP_CHECK(hipMalloc(&d_C, sizeof(float) * M * N));

    HIP_CHECK(hipMemcpy(d_A, h_A.data(), sizeof(_Float16) * M * K,
                        hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_B, h_B.data(), sizeof(_Float16) * K * N,
                        hipMemcpyHostToDevice));

    // [Sphinx mfma kernel launch start]
    // Grid: one block per 32×32 output tile.
    const dim3 grid((N + BLOCK_TILE_N - 1) / BLOCK_TILE_N,
                    (M + BLOCK_TILE_M - 1) / BLOCK_TILE_M);

    float       ms = 0.0f;
    const char* policy_label = nullptr;

    if(is_gfx950)
    {
        // Standard path: scalar B load from transposed LDS tile.
        auto launch = [&]()
        {
            matrix_multiply_generic<MfmaTilePolicy,
                                    MfmaCdna4F16Policy,
                                    _Float16>
                <<<grid, dim3(256)>>>(d_A, d_B, d_C, M, N, K);
            HIP_CHECK(hipGetLastError());
        };
        ms           = time_kernel_ms(launch, WARMUP_RUNS, TIMING_RUNS);
        policy_label = "GemmKernel<MfmaTilePolicy, MfmaCdna4F16Policy> [gfx950]";

        // ds_read_tr path: hardware-transposed B load from row-major LDS tile.
        // Re-use the same device buffers; d_B was already uploaded.
        HIP_CHECK(hipMemset(d_C, 0, sizeof(float) * M * N));
        auto launch_tr = [&]()
        {
            matrix_multiply_generic<MfmaTilePolicyTrB,
                                    MfmaCdna4F16TrPolicy,
                                    _Float16>
                <<<grid, dim3(256)>>>(d_A, d_B, d_C, M, N, K);
            HIP_CHECK(hipGetLastError());
        };
        const float ms_tr = time_kernel_ms(launch_tr, WARMUP_RUNS, TIMING_RUNS);
        HIP_CHECK(hipMemcpy(h_C.data(), d_C, sizeof(float) * M * N,
                            hipMemcpyDeviceToHost));
        const bool ok_tr = verify_result(h_C, h_A, M, N, K);
        print_metrics(
            "GemmKernel<MfmaTilePolicyTrB, MfmaCdna4F16TrPolicy> [gfx950]",
            ok_tr, ms_tr, sizeof(_Float16), M, N, K);
    }
    else
    {
        // Scalar FP32 fallback: upload B as FP32.
        std::vector<float> h_B_f32(K * N, 0.0f);
        for(int i = 0; i < std::min(K, N); ++i)
            h_B_f32[i * N + i] = 1.0f;
        float* d_B_f32;
        HIP_CHECK(hipMalloc(&d_B_f32, sizeof(float) * K * N));
        HIP_CHECK(hipMemcpy(d_B_f32, h_B_f32.data(), sizeof(float) * K * N,
                            hipMemcpyHostToDevice));
        float* d_A_f32;
        HIP_CHECK(hipMalloc(&d_A_f32, sizeof(float) * M * K));
        std::vector<float> h_A_f32(M * K);
        for(int i = 0; i < M * K; ++i)
            h_A_f32[i] = static_cast<float>(h_A[i]);
        HIP_CHECK(hipMemcpy(d_A_f32, h_A_f32.data(), sizeof(float) * M * K,
                            hipMemcpyHostToDevice));

        auto launch = [&]()
        {
            matrix_multiply_generic<SingleBufferTilePolicyF<32, 32, 8>,
                                    ScalarFMAPolicy<8, 8, 8>,
                                    float>
                <<<grid, dim3(64)>>>(d_A_f32, d_B_f32, d_C, M, N, K);
            HIP_CHECK(hipGetLastError());
        };
        ms           = time_kernel_ms(launch, WARMUP_RUNS, TIMING_RUNS);
        policy_label = "GemmKernel<SingleBufferTilePolicyF, ScalarFMAPolicy> [fallback]";
        HIP_CHECK(hipFree(d_A_f32));
        HIP_CHECK(hipFree(d_B_f32));
    }
    // [Sphinx mfma kernel launch end]

    HIP_CHECK(hipMemcpy(h_C.data(), d_C, sizeof(float) * M * N,
                        hipMemcpyDeviceToHost));
    const bool ok = verify_result(h_C, h_A, M, N, K);
    print_metrics(policy_label, ok, ms, sizeof(_Float16), M, N, K);

    HIP_CHECK(hipFree(d_A));
    HIP_CHECK(hipFree(d_B));
    HIP_CHECK(hipFree(d_C));

    return EXIT_SUCCESS;
}
