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
// matrix_multiply_rdna3_wmma.hip  --  FP16 WMMA ComputePolicy for RDNA 3/3.5
//                                     (gfx11, RX 7000 series)
//
// This file is a companion to matrix_multiply_generic.hip (Step 7) and
// provides a concrete ComputePolicy implementation that uses the WMMA
// instruction v_wmma_f32_16x16x16_f16 available on RDNA 3/3.5 GPUs
// (gfx1100--gfx1103, gfx1150--gfx1153).
//
// The instruction computes a 16x16 FP32 output tile per wavefront.  Each
// call consumes K=16 FP16 elements.  The 32-lane wavefront uses matrix
// replication: lanes 0--15 and 16--31 behave identically with respect to
// A/B fragment indexing (lane_id mod 16 selects the row/column).  Each
// lane supplies 16 FP16 values for both the A and B operands.
//
// Accumulator layout (16x16 tile, wave32):
//   Each lane holds 8 FP32 values in VGPRs 0--7.
//
//   Given lane L and VGPR index G (0--7):
//     row = G * 2 + (L / 16)
//     col = L mod 16
//
// Fragment distribution:
//   v_wmma_f32_16x16x16_f16 takes two v16half operands and a v8float
//   accumulator.  Each lane provides 16 FP16 values for A (one full
//   row of the 16x16 A-tile) and 16 FP16 values for B (one full column
//   of the 16x16 B-tile).
//
//   Due to matrix replication on RDNA 3, lane offset (lane_id mod 16)
//   selects the row for A and the column for B:
//     lane L, elem e (0--15): A[tile_a_row + L%16][ki + e]
//     lane L, elem e (0--15): B_T[tile_b_col + L%16][ki + e]
//
// The policy is a drop-in replacement for ScalarFMAPolicy: the generic
// kernel template (matrix_multiply_generic) is reused without modification.
//
// Architecture: RDNA 3/3.5 (gfx11) only.
// Compile with: amdclang++ -O3 -std=c++17 --offload-arch=gfx1100 \
//                   matrix_multiply_rdna3_wmma.hip -o mm_rdna3_wmma
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
#include <string>
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
// Section 3: TilePolicy implementations
//
// The 16x16 FP16 WMMA instruction has k_step=16, meaning each call consumes
// 16 K-positions.  One call per K-strip suffices when k_tile_size=16.
// The 32x32 block tile uses 2x2 wavefronts (wave32), so the shared-memory
// tile is 32x32 in M/N with a K-strip of 16.
// ===========================================================================

// FP32 tile policy for the scalar fallback path.
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
// Section 4: WmmaRdna3F16Policy -- WMMA ComputePolicy for gfx11 (RDNA 3/3.5)
// ===========================================================================
//
// This policy uses v_wmma_f32_16x16x16_f16 to compute one 16x16 FP32 output
// tile per wavefront (wave32).  Each block contains 2x2 = 4 wavefronts,
// giving a 32x32 block tile.
//
// Accumulator layout (16x16 tile, wave32):
//   Each lane holds 8 FP32 values in VGPRs 0--7.
//
//   Given lane L and VGPR index G (0--7):
//     row = G * 2 + (L / 16)
//     col = L mod 16
//
// Fragment distribution:
//   v_wmma_f32_16x16x16_f16 takes two v16half operands (A and B) and a
//   v8float accumulator.  Each lane provides 16 FP16 values for A (one
//   full row) and 16 FP16 values for B (one full column).
//
//   Due to matrix replication on RDNA 3, lanes 0--15 and 16--31 behave
//   identically: lane offset (lane_id mod 16) selects the row for A and
//   the column for B:
//     lane L, elem e (0--15): A[tile_a_row + L%16][ki + e]
//     lane L, elem e (0--15): B_T[tile_b_col + L%16][ki + e]
// ===========================================================================

// [Sphinx wmma rdna3 policy start]
struct WmmaRdna3F16Policy
{
    // -- ComputePolicy constants ----------------------------------------------
    // One wavefront (wave32) owns a 16x16 output tile.
    // All 32 lanes hold unique output elements (8 elements per lane).
    static constexpr int thread_tile_m   = 16;
    static constexpr int thread_tile_n   = 16;
    static constexpr int effective_lanes = 32;
    // v_wmma_f32_16x16x16_f16 consumes 16 K-positions per call (k_step=16).
    // Each lane provides 16 FP16 values (v16half) for A and 16 for B.
    // Lanes 0--15 and 16--31 mirror each other (matrix replication).
    static constexpr int k_step          = 16;

    // -- Vector types --------------------------------------------------------
    using v16half = _Float16 [[clang::ext_vector_type(16)]];
    using v8float = float    [[clang::ext_vector_type(8)]];

    using elem_a = _Float16;
    using elem_b = _Float16;

    // -- Accumulator ----------------------------------------------------------
    // v8float holds 8 FP32 VGPRs per lane (VGPRs 0--7).
    struct Accumulator
    {
        v8float regs;
    };

    __device__ static void zero(Accumulator& acc)
    {
        acc.regs = v8float{};
    }

    // -- thread_tile_offset ---------------------------------------------------
    // The entire 16x16 tile belongs to one wavefront (wave32).  Multiple
    // wavefronts in a block cover different 16x16 sub-tiles.
    //
    //   wavefront id within block: wid = tid / 32
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
        const int wid   = tid / 32;
        const int wid_x = wid % waves_n;
        const int wid_y = wid / waves_n;
        *thread_row = wid_y * 16;
        *thread_col = wid_x * 16;
    }

    // -- Fragment loads --------------------------------------------------------
    // load_a: each lane reads 16 FP16 values from LDS into the fragment
    // array.  The 16-element _Float16 array has the same layout as v16half
    // and is reinterpreted as such in mma().
    //
    // v_wmma_f32_16x16x16_f16 (wave32) uses matrix replication: lanes 0--15
    // and 16--31 index identically via (lane_id mod 16).  Each lane loads
    // one full row of the 16x16 A-tile.
    //
    //   A-row index : tile_a_row + (lane_id mod 16)
    //   K-position e: ki + e   (e = 0..15)
    __device__ static void load_a(const _Float16* tile_a_ptr,
                                  int             tile_a_row,
                                  int             ki,
                                  int             k_tile_size,
                                  int             lane_id,
                                  elem_a        (&frag)[16])
    {
        const int row = tile_a_row + (lane_id % 16);
        #pragma unroll
        for(int e = 0; e < 16; ++e)
            frag[e] = tile_a_ptr[row * k_tile_size + ki + e];
    }

    // load_b: each lane reads 16 FP16 values from LDS (symmetric to load_a).
    //
    // tile_b_T is stored transposed: tile_b_T[(tile_b_col + col)][k_off].
    //   B-col index : tile_b_col + (lane_id mod 16)
    //   K-position e: ki + e   (e = 0..15)
    __device__ static void load_b(const _Float16* tile_b_T_ptr,
                                  int             tile_b_col,
                                  int             ki,
                                  int             k_tile_size,
                                  int             lane_id,
                                  elem_b        (&frag)[16])
    {
        const int col = tile_b_col + (lane_id % 16);
        #pragma unroll
        for(int e = 0; e < 16; ++e)
            frag[e] = tile_b_T_ptr[col * k_tile_size + ki + e];
    }

    // -- mma -------------------------------------------------------------------
    // Issue one v_wmma_f32_16x16x16_f16 instruction.
    //
    // The _Float16[16] fragment arrays share the same in-register layout as
    // v16half; a reinterpret_cast avoids any per-element copy.
    __device__ static void mma(Accumulator&       acc,
                               const elem_a     (&a_frag)[16],
                               const elem_b     (&b_frag)[16])
    {
#if defined(__gfx1100__) || defined(__gfx1101__) || defined(__gfx1102__) || \
    defined(__gfx1103__) || defined(__gfx1150__) || defined(__gfx1151__) || \
    defined(__gfx1152__) || defined(__gfx1153__)
        acc.regs = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(
            reinterpret_cast<const v16half&>(a_frag),
            reinterpret_cast<const v16half&>(b_frag),
            acc.regs);
#endif
    }

    // -- store_c ---------------------------------------------------------------
    // Scatter the 8 VGPR values to their global-memory positions.
    //
    // Inverse layout (lane L, VGPR index G in [0,7]):
    //   row = G * 2 + (L / 16)
    //   col = L mod 16
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
        for(int G = 0; G < 8; ++G)
        {
            const int i = G * 2 + (L / 16);
            const int r = out_row_base + i;
            const int c = out_col_base + j;
            if(r < m && c < n)
                C[r * n + c] = acc.regs[G];
        }
    }
};
// [Sphinx wmma rdna3 policy end]

// ===========================================================================
// Section 5: ScalarFMAPolicy -- portable FP32 fallback for non-gfx11 targets
// ===========================================================================

// [Sphinx scalar fma policy start]
template<int ThreadTileM_, int ThreadTileN_, int BlockDimX_>
struct ScalarFMAPolicy
{
    static constexpr int thread_tile_m   = ThreadTileM_;
    static constexpr int thread_tile_n   = ThreadTileN_;
    static constexpr int effective_lanes = 32;
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
// type) so that it works with both the FP16 RDNA3 policy and the FP32
// fallback policy.
// ===========================================================================

// [Sphinx gemm kernel start]
template<typename TilePolicy, typename ComputePolicy, typename ElemIn>
__global__ void
    __launch_bounds__(TilePolicy::block_tile_m / ComputePolicy::thread_tile_m
                          * TilePolicy::block_tile_n / ComputePolicy::thread_tile_n
                          * ComputePolicy::effective_lanes)
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
    const int lane_id     = tid % ComputePolicy::effective_lanes;
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
            typename ComputePolicy::elem_a a_frag[ComputePolicy::thread_tile_m];
            typename ComputePolicy::elem_b b_frag[ComputePolicy::thread_tile_n];

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
// Block tile: 32x32 (2x2 wavefronts, each computing a 16x16 WMMA tile).
// K-strip  :  16 (k_tile_size=16, k_step=16 → 1 WMMA call per tile).
//
// Thread grid:
//   Each block spans BLOCK_TILE_M rows and BLOCK_TILE_N cols of C.
//   Blocks per grid: ceil(M/32) x ceil(N/32).
//   Threads per block: (BLOCK_TILE_M/16) * (BLOCK_TILE_N/16) wavefronts * 32 lanes
//                    = 2 * 2 * 32 = 128.
// ===========================================================================

// [Sphinx wmma policy aliases start]
// TilePolicy for the FP16 WMMA path (transposed B in LDS).
using WmmaTilePolicy = SingleBufferTilePolicyF16<32, 32, 16>;
// [Sphinx wmma policy aliases end]

// [Sphinx wmma launch config start]
// Launch parameters:
//
// RDNA 3/3.5 (gfx11): 2x2 wavefronts of 32 lanes = 128 threads per block.
//   block_tile_m = 32, thread_tile_m = 16  → 2 wavefronts along M
//   block_tile_n = 32, thread_tile_n = 16  → 2 wavefronts along N
//
// Fallback (scalar): 8x8 thread tiles in a 32x32 block → 16 threads per
//   block, padded to one full wavefront of 32.
//
// The kernel to launch and its thread count are selected at runtime based
// on the device architecture string returned by hipGetDeviceProperties.
//
// Grid: one block per 32x32 output tile.
constexpr int BLOCK_TILE_M = 32;
constexpr int BLOCK_TILE_N = 32;
// [Sphinx wmma launch config end]

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
    const std::string arch(props.gcnArchName);
    const bool is_gfx11 = arch.rfind("gfx11", 0) == 0;
    const int  threads_per_block = is_gfx11 ? 128 : 32;
    std::cout << "Device       : " << props.name << "\n"
              << "GCN arch     : " << props.gcnArchName << "\n"
              << "Warp size    : " << props.warpSize << "\n"
              << "Matrix dims  : A(" << M << "x" << K << ") * B("
              << K << "x" << N << ")\n"
              << "Block tile   : " << BLOCK_TILE_M << "x" << BLOCK_TILE_N
              << "  K-strip: " << WmmaTilePolicy::k_tile_size << "\n"
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

    // [Sphinx wmma kernel launch start]
    // Grid: one block per 32x32 output tile.
    const dim3 grid((N + BLOCK_TILE_N - 1) / BLOCK_TILE_N,
                    (M + BLOCK_TILE_M - 1) / BLOCK_TILE_M);

    float       ms = 0.0f;
    const char* policy_label = nullptr;

    if(is_gfx11)
    {
        auto launch = [&]()
        {
            matrix_multiply_generic<WmmaTilePolicy,
                                    WmmaRdna3F16Policy,
                                    _Float16>
                <<<grid, dim3(128)>>>(d_A, d_B, d_C, M, N, K);
            HIP_CHECK(hipGetLastError());
        };
        ms           = time_kernel_ms(launch, WARMUP_RUNS, TIMING_RUNS);
        policy_label = "GemmKernel<WmmaTilePolicy, WmmaRdna3F16Policy> [gfx11]";
    }
    else
    {
        // Scalar FP32 fallback: upload A and B as FP32.
        std::vector<float> h_A_f32(M * K);
        for(int i = 0; i < M * K; ++i)
            h_A_f32[i] = static_cast<float>(h_A[i]);
        std::vector<float> h_B_f32(K * N, 0.0f);
        for(int i = 0; i < std::min(K, N); ++i)
            h_B_f32[i * N + i] = 1.0f;

        float* d_A_f32;
        float* d_B_f32;
        HIP_CHECK(hipMalloc(&d_A_f32, sizeof(float) * M * K));
        HIP_CHECK(hipMalloc(&d_B_f32, sizeof(float) * K * N));
        HIP_CHECK(hipMemcpy(d_A_f32, h_A_f32.data(), sizeof(float) * M * K,
                            hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(d_B_f32, h_B_f32.data(), sizeof(float) * K * N,
                            hipMemcpyHostToDevice));

        auto launch = [&]()
        {
            matrix_multiply_generic<SingleBufferTilePolicyF<32, 32, 8>,
                                    ScalarFMAPolicy<8, 8, 4>,
                                    float>
                <<<grid, dim3(32)>>>(d_A_f32, d_B_f32, d_C, M, N, K);
            HIP_CHECK(hipGetLastError());
        };
        ms           = time_kernel_ms(launch, WARMUP_RUNS, TIMING_RUNS);
        policy_label = "GemmKernel<SingleBufferTilePolicyF, ScalarFMAPolicy> [fallback]";
        HIP_CHECK(hipFree(d_A_f32));
        HIP_CHECK(hipFree(d_B_f32));
    }
    // [Sphinx wmma kernel launch end]

    HIP_CHECK(hipMemcpy(h_C.data(), d_C, sizeof(float) * M * N,
                        hipMemcpyDeviceToHost));
    const bool ok = verify_result(h_C, h_A, M, N, K);
    print_metrics(policy_label, ok, ms, sizeof(_Float16), M, N, K);

    HIP_CHECK(hipFree(d_A));
    HIP_CHECK(hipFree(d_B));
    HIP_CHECK(hipFree(d_C));

    return EXIT_SUCCESS;
}
