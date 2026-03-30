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

// ---------------------------------------------------------------------------
// matrix_multiply_generic.hip  –  Step 7: Stepanov-style generic GEMM
//
// This file demonstrates the Stepanov principle of "lifting" an algorithm into
// a fully generic form by separating *what* the algorithm does (policy-neutral
// orchestration) from *how* individual steps are performed (the policies).
//
// Two orthogonal policies parameterise the kernel:
//
//   TilePolicy    – controls data movement and LDS organisation
//                   (buffering strategy, vector width, LDS layout)
//
//   ComputePolicy – controls the arithmetic on a thread's output sub-tile
//                   (fragment types, load helpers, the FMA instruction itself)
//
// Both policies are validated at compile time using C++17 static_assert.
// A side-by-side C++20 transformation example illustrates the ergonomic
// improvement offered by `requires` clauses.
//
// Architecture dispatch
// ---------------------
// A compile-time dispatch block maps device ISA strings to concrete policies.
// MFMA (CDNA) and WMMA (RDNA3/4) policies are left as forward-declaration
// stubs — they will be filled in by architecture-specific follow-up PRs.
// For all other targets, ScalarFMAPolicy (portable scalar FP32 outer-product)
// is selected automatically.
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
// Section 1: TilePolicy interface and concept validation
// ===========================================================================
//
// A TilePolicy must expose:
//
//   static constexpr int num_buffers;         // 1 (single) or 2 (double)
//   static constexpr int block_tile_m;        // rows computed per block
//   static constexpr int block_tile_n;        // cols computed per block
//   static constexpr int k_tile_size;         // K-strip width
//
//   struct SharedStorage { … };               // LDS layout (trivially destructible)
//
//   // Called once before the main loop to load the first tile (double-buffer
//   // prologue) or do nothing (single-buffer).
//   __device__ static void prologue(SharedStorage& smem,
//                                   const float* A, const float* B,
//                                   int block_row_start, int block_col_start,
//                                   int m, int n, int k,
//                                   int tid, int num_threads);
//
//   // Load tile for iteration (t + 1) into the background buffer.
//   __device__ static void prefetch(SharedStorage& smem,
//                                   const float* A, const float* B,
//                                   int t_next,
//                                   int block_row_start, int block_col_start,
//                                   int m, int n, int k,
//                                   int tid, int num_threads);
//
//   // Issue synchronisation before compute begins for iteration t.
//   __device__ static void acquire(int t);
//
//   // Issue synchronisation after compute finishes for iteration t.
//   __device__ static void release(int t);
//
//   // Return the buffer index to use for compute during iteration t.
//   __device__ static int buf_idx(int t);
//
//   // Access tile_a / tile_b_T slices for a given buffer index.
//   __device__ static float* get_tile_a(SharedStorage& smem, int buf);
//   __device__ static float* get_tile_b_T(SharedStorage& smem, int buf);

// ---------------------------------------------------------------------------
// C++17 TilePolicy concept validation via static_assert
// ---------------------------------------------------------------------------
// [Sphinx tile policy traits start]
template<typename TilePolicy>
struct TilePolicyTraits
{
    // num_buffers must be 1 or 2
    static_assert(TilePolicy::num_buffers == 1 || TilePolicy::num_buffers == 2,
                  "TilePolicy::num_buffers must be 1 (single) or 2 (double-buffer)");

    // Tile dimensions must be positive multiples of k_tile_size
    static_assert(TilePolicy::block_tile_m > 0,
                  "TilePolicy::block_tile_m must be a positive integer");
    static_assert(TilePolicy::block_tile_n > 0,
                  "TilePolicy::block_tile_n must be a positive integer");
    static_assert(TilePolicy::k_tile_size > 0,
                  "TilePolicy::k_tile_size must be a positive integer");

    // SharedStorage must be trivially destructible (no LDS destructor call)
    static_assert(std::is_trivially_destructible_v<typename TilePolicy::SharedStorage>,
                  "TilePolicy::SharedStorage must be trivially destructible");

    // SharedStorage must actually occupy LDS
    static_assert(sizeof(typename TilePolicy::SharedStorage) > 0,
                  "TilePolicy::SharedStorage must have non-zero size");
};
// [Sphinx tile policy traits end]

// ---------------------------------------------------------------------------
// C++20 transformation example (informational – not compiled unless C++20)
// ---------------------------------------------------------------------------
//
// The C++17 static_assert approach above is entirely portable but has two
// drawbacks:
//
//  1. Trait structs must be explicitly instantiated – the compiler will not
//     check them unless something triggers instantiation.
//  2. Error messages appear at the point of TilePolicyTraits<P> instantiation,
//     which can be far from the call site.
//
// C++20 `requires` clauses solve both problems.  The equivalent constraint for
// the TilePolicy is:
//
// [Sphinx cpp20 concept start]
#if 0  // C++20 illustration – enable with -std=c++20
template<typename TilePolicy>
concept TilePolicyConcept = requires
{
    requires TilePolicy::num_buffers == 1 || TilePolicy::num_buffers == 2;
    requires TilePolicy::block_tile_m > 0;
    requires TilePolicy::block_tile_n > 0;
    requires TilePolicy::k_tile_size > 0;
    requires std::is_trivially_destructible_v<typename TilePolicy::SharedStorage>;
    requires sizeof(typename TilePolicy::SharedStorage) > 0;
};

// The kernel declaration then becomes self-documenting:
template<TilePolicyConcept TilePolicy, typename ComputePolicy>
__global__ void matrix_multiply_generic(/* … */);
//
// Constraint violations are reported at the point of the template
// instantiation with a clear "constraint not satisfied" diagnostic rather than
// a cryptic static_assert failure inside a trait struct.
#endif
// [Sphinx cpp20 concept end]

// ===========================================================================
// Section 2: ComputePolicy interface and concept validation
// ===========================================================================
//
// A ComputePolicy must expose:
//
//   static constexpr int thread_tile_m;    // output rows per thread
//   static constexpr int thread_tile_n;    // output cols per thread
//   static constexpr int effective_lanes;  // unique output lanes per wavefront
//                                          //   = warpSize for scalar/CDNA
//                                          //   = warpSize/2 for RDNA3 WMMA
//                                          //     (lanes L and L+16 share a fragment)
//   static constexpr int k_step;           // k-indices consumed per mma() call
//                                          //   = 1 for scalar FMA (one FMA per call)
//                                          //   > 1 for intrinsics that process
//                                          //     multiple k-indices per call
//                                          //   Must evenly divide TilePolicy::k_tile_size.
//
//   // Map a thread to the top-left corner of its output sub-tile within the
//   // block tile.  For scalar policies this is simply:
//   //     *thread_row = (tid / block_dim_x) * thread_tile_m
//   //     *thread_col = (tid % block_dim_x) * thread_tile_n
//   // For RDNA3 WMMA this is more complex (lane-mirroring aware).
//   __device__ static void thread_tile_offset(int tid,
//                                              int lane_id,
//                                              int* thread_row,
//                                              int* thread_col);
//
//   // Fragment type for a single element of A / B loaded from LDS.
//   using elem_a = float;   // (or __half, etc.)
//   using elem_b = float;
//
//   // Data-type scope of elem_a / elem_b
//   // ------------------------------------
//   // elem_a and elem_b are the types held in *registers* (the fragment
//   // loaded from LDS before the MMA instruction).  They are fully
//   // parameterised here and the kernel loop uses them correctly.
//   //
//   // The LDS tile type and the global memory pointer type are currently
//   // hardcoded to float (see cooperative_load_tile_a / _b_T and the kernel
//   // signature).  Two cases arise when introducing intrinsics:
//   //
//   //   Case 1 – FP32 in global memory, low-precision fragments
//   //     (e.g. global float → LDS float → __half fragment for MFMA)
//   //     Conversion happens inside load_a / load_b.  No changes to
//   //     TilePolicy or the kernel signature are required.
//   //
//   //   Case 2 – Low-precision in global memory
//   //     (e.g. global __half → LDS __half → __half fragment)
//   //     Requires adding an InputT template parameter to TilePolicy so
//   //     that SharedStorage can use InputT arrays and the cooperative load
//   //     helpers read InputT from global memory.  The kernel signature
//   //     must also change from `const float*` to `const InputT*`.
//   //
//   // This tutorial covers Case 1 only (FP32 throughout).  Case 2 is left
//   // as an extension for the architecture-specific follow-up sections.
//
//   // Accumulator array type.
//   struct Accumulator { float data[thread_tile_m][thread_tile_n]; };
//
//   // Zero-initialise the accumulator.
//   __device__ static void zero(Accumulator& acc);
//
//   // Load one A-fragment (column of the thread's sub-tile) from LDS.
//   // tile_a_row: first row of this thread's fragment within the A tile.
//   // lane_id: unused for scalar; required for WMMA.
//   // NOTE: tile_a_ptr points to float (LDS is always float in this file).
//   //       For Case 2 (low-precision global memory) this pointer type must
//   //       be templated on TilePolicy::input_t.
//   __device__ static void load_a(const float* tile_a_ptr,
//                                  int tile_a_row, int ki,
//                                  int k_tile_size,
//                                  int lane_id,
//                                  elem_a (&frag)[thread_tile_m]);
//
//   // Load one B-fragment (row of the thread's sub-tile, transposed) from LDS.
//   // Same float-LDS caveat as load_a above.
//   __device__ static void load_b(const float* tile_b_T_ptr,
//                                  int tile_b_col, int ki,
//                                  int k_tile_size,
//                                  int lane_id,
//                                  elem_b (&frag)[thread_tile_n]);
//
//   // Multiply-accumulate: acc += outer_product(a_frag, b_frag).
//   __device__ static void mma(Accumulator& acc,
//                               const elem_a (&a_frag)[thread_tile_m],
//                               const elem_b (&b_frag)[thread_tile_n]);
//
//   // Write the accumulator to global memory.
//   // lane_id is forwarded for RDNA3 (only the lower 16 lanes write).
//   __device__ static void store_c(const Accumulator& acc,
//                                   float* C,
//                                   int out_row_base, int out_col_base,
//                                   int m, int n,
//                                   int lane_id);

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

    // Accumulator must be trivially constructible (no runtime init in registers)
    static_assert(std::is_trivially_constructible_v<typename ComputePolicy::Accumulator>,
                  "ComputePolicy::Accumulator must be trivially constructible");

    // Fragment element types must have a well-defined size (covers both
    // scalar arithmetic types like float and vector types like half2).
    static_assert(sizeof(typename ComputePolicy::elem_a) > 0,
                  "ComputePolicy::elem_a must have non-zero size");
    static_assert(sizeof(typename ComputePolicy::elem_b) > 0,
                  "ComputePolicy::elem_b must have non-zero size");
};
// [Sphinx compute policy traits end]

// ===========================================================================
// Section 3: Concrete TilePolicy implementations
// ===========================================================================

// ---------------------------------------------------------------------------
// Helper: cooperative flat-strided load into LDS
// ---------------------------------------------------------------------------
// Every element of the dest buffer is loaded by exactly one thread.  The
// linear `tid` strides through the flat element index so that work is
// evenly distributed regardless of tile shape.
//
// This helper is shared by all TilePolicy implementations below.
// ---------------------------------------------------------------------------
__device__ inline void cooperative_load_tile_a(float*       dest,       // LDS tile_a slice
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

__device__ inline void cooperative_load_tile_b_T(float*       dest,       // LDS tile_b_T slice
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
    // tile_b_T is stored column-major: [BLOCK_TILE_N][K_TILE_SIZE]
    // i.e., dest[col_within_tile * k_tile_size + k_idx]
    const int tile_elems = block_tile_n * k_tile_size;
    for(int idx = tid; idx < tile_elems; idx += num_threads)
    {
        const int tile_col_idx = idx / k_tile_size; // column within block tile
        const int tile_k_idx   = idx % k_tile_size; // k-index within strip
        const int global_row   = t * k_tile_size + tile_k_idx;
        const int global_col   = block_col_start + tile_col_idx;
        dest[tile_col_idx * k_tile_size + tile_k_idx] =
            (global_row < k && global_col < n) ? B[global_row * n + global_col] : 0.0f;
    }
}

// ---------------------------------------------------------------------------
// SingleBufferTilePolicy: no prefetching, two __syncthreads per tile iteration
// ---------------------------------------------------------------------------
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
                                    int /*tid*/, int /*num_threads*/)
    {
        // No prologue needed for single-buffer; tile is loaded at the top of
        // each loop iteration in prefetch().
    }

    __device__ static void prefetch(SharedStorage& smem,
                                    const float* A, const float* B,
                                    int          t_next,
                                    int block_row_start, int block_col_start,
                                    int m, int n, int k,
                                    int tid, int num_threads)
    {
        // For single-buffer we load the *current* tile (t_next is actually t
        // from the caller's perspective — see kernel loop).
        // Rename for clarity: t_next here is the index we want to load.
        const int t = t_next;
        cooperative_load_tile_a(&smem.tile_a[0][0], A, block_row_start,
                                block_tile_m, t, k_tile_size, m, k, tid, num_threads);
        cooperative_load_tile_b_T(&smem.tile_b_T[0][0], B, block_col_start,
                                   block_tile_n, t, k_tile_size, k, n, tid, num_threads);
    }

    // After prefetch: wait for the load to complete before compute.
    __device__ static void acquire(int /*t*/) { __syncthreads(); }

    // After compute: wait for all threads to finish reading before the next
    // iteration overwrites the tile.
    __device__ static void release(int /*t*/) { __syncthreads(); }

    __device__ static int buf_idx(int /*t*/) { return 0; }

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

// ---------------------------------------------------------------------------
// SoftwareDoubleBufferTilePolicy: ping-pong LDS buffers to overlap load/compute
// ---------------------------------------------------------------------------
// [Sphinx double buffer policy start]
template<int BlockTileM_, int BlockTileN_, int KTileSize_>
struct SoftwareDoubleBufferTilePolicy
{
    static constexpr int num_buffers  = 2;
    static constexpr int block_tile_m = BlockTileM_;
    static constexpr int block_tile_n = BlockTileN_;
    static constexpr int k_tile_size  = KTileSize_;

    struct SharedStorage
    {
        float tile_a[2][BlockTileM_][KTileSize_];
        float tile_b_T[2][BlockTileN_][KTileSize_];
    };

    // Prologue: load tile 0 into buffer 0 and synchronise so compute can
    // begin immediately at the start of the first iteration.
    __device__ static void prologue(SharedStorage& smem,
                                    const float* A, const float* B,
                                    int block_row_start, int block_col_start,
                                    int m, int n, int k,
                                    int tid, int num_threads)
    {
        cooperative_load_tile_a(&smem.tile_a[0][0][0], A, block_row_start,
                                block_tile_m, 0, k_tile_size, m, k, tid, num_threads);
        cooperative_load_tile_b_T(&smem.tile_b_T[0][0][0], B, block_col_start,
                                   block_tile_n, 0, k_tile_size, k, n, tid, num_threads);
        __syncthreads(); // tile 0 visible to all threads before loop entry
    }

    // Prefetch tile t_next into the background buffer (1 - current) while
    // the current buffer is in use for compute.  No sync here — the sync that
    // ends the previous compute phase (release) also covers this load because
    // the background buffer is not touched until two iterations later.
    __device__ static void prefetch(SharedStorage& smem,
                                    const float* A, const float* B,
                                    int          t_next,
                                    int block_row_start, int block_col_start,
                                    int m, int n, int k,
                                    int tid, int num_threads)
    {
        if(t_next * k_tile_size < k) // guard: do not load beyond K
        {
            const int buf = t_next % 2; // background buffer index
            cooperative_load_tile_a(&smem.tile_a[buf][0][0], A, block_row_start,
                                    block_tile_m, t_next, k_tile_size, m, k, tid, num_threads);
            cooperative_load_tile_b_T(&smem.tile_b_T[buf][0][0], B, block_col_start,
                                       block_tile_n, t_next, k_tile_size, k, n, tid, num_threads);
        }
    }

    // acquire: the prologue already placed a barrier; nothing needed here.
    // The previous iteration's release() synced both the completed compute
    // and the in-flight prefetch into the background buffer.
    __device__ static void acquire(int /*t*/) {}

    // release: one __syncthreads covers two things:
    //   (a) all threads finished computing from the current buffer, and
    //   (b) all threads finished writing the prefetched tile into the
    //       background buffer (the load issued in prefetch() above).
    // Both must be visible before the next iteration swaps buffers.
    __device__ static void release(int /*t*/) { __syncthreads(); }

    __device__ static int buf_idx(int t) { return t % 2; }

    __device__ static float* get_tile_a(SharedStorage& smem, int buf)
    {
        return &smem.tile_a[buf][0][0];
    }
    __device__ static float* get_tile_b_T(SharedStorage& smem, int buf)
    {
        return &smem.tile_b_T[buf][0][0];
    }
};
// [Sphinx double buffer policy end]

// ===========================================================================
// Section 4: Concrete ComputePolicy implementations
// ===========================================================================

// ---------------------------------------------------------------------------
// ScalarFMAPolicy: portable scalar FP32 outer-product accumulation.
//
// Each thread computes a ThreadTileM x ThreadTileN sub-tile entirely in
// registers.  No architecture-specific intrinsics are used.
//
// thread_tile_offset: straightforward 2-D thread layout.
//   tx = tid % block_dim_x  → column tile index within block
//   ty = tid / block_dim_x  → row tile index within block
//   *thread_col = tx * thread_tile_n
//   *thread_row = ty * thread_tile_m
//
// effective_lanes = warp_size (all lanes contribute unique output elements).
//
// RDNA3 forward-compatibility note
// ---------------------------------
// The lane_id parameter in load_a, load_b, and store_c is not used here but
// is required by the interface so that WMMA policies (which need it for
// lane-mirroring) can be dropped in without changing the calling kernel.
// ---------------------------------------------------------------------------
// [Sphinx scalar fma policy start]
template<int ThreadTileM_, int ThreadTileN_, int BlockDimX_>
struct ScalarFMAPolicy
{
    static constexpr int thread_tile_m  = ThreadTileM_;
    static constexpr int thread_tile_n  = ThreadTileN_;
    // All lanes in a wavefront compute unique output elements (no mirroring).
    static constexpr int effective_lanes = 64; // AMD wavefront size
    static constexpr int k_step         = 1;  // one scalar FMA per ki iteration

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

    // Map thread id to the top-left corner of its output sub-tile.
    // lane_id is accepted but unused (interface forward-compatibility).
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

    // Load a column fragment of tile_a into registers.
    // tile_a is stored row-major: tile_a[row][ki] = ptr[row * k_tile_size + ki]
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

    // Load a row fragment of tile_b_T into registers.
    // tile_b_T is stored column-major: tile_b_T[col][ki] = ptr[col * k_tile_size + ki]
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

    // Outer-product FMA: acc += a_frag ⊗ b_frag
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

    // Write accumulator to global memory.
    // lane_id accepted but unused; all lanes write their unique sub-tile.
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

// ---------------------------------------------------------------------------
// DirectLoadTilePolicy: use __builtin_amdgcn_global_load_lds to bypass VGPRs
//
// On CDNA3 and CDNA4, the global_load_lds instruction transfers data from
// global memory directly into LDS without staging in vector registers.  This
// frees VGPRs for compute, potentially enabling higher occupancy.
//
// ISA-level operation (from the LLVM intrinsic description):
//   global_load_lds_dword is a wavefront-wide gather-to-LDS operation:
//
//   - VADDR (v[0:1]) is a vector register: each lane provides its OWN
//     global source address.  The global pointer need not be wave-uniform.
//   - M0 is a scalar register providing the LDS base address (wave-uniform).
//   - The LDS destination is implicitly offset by lane_id:
//       lane k writes to  M0 + offset + k * 4   (for size <= 4)
//       lane k writes to  M0 + offset + k * 16  (for size > 4)
//   - The offset is an immediate constant (same for all lanes).
//
//   A single dword instruction therefore transfers 64 floats (256 bytes)
//   from 64 (potentially different) global addresses into 64 contiguous
//   LDS locations.  No VGPR holds the loaded data at any point.
//
//   The VGPR savings compared to global_load_dword + ds_write come from
//   the data path: with the regular path, VGPRs hold the loaded values
//   between the global read and the LDS write.  In a VGPR-constrained
//   kernel those extra registers can be the difference between N and N+1
//   resident wavefronts per CU, improving latency hiding across the
//   kernel's entire execution.
//
// Intrinsic signature:
//   __builtin_amdgcn_global_load_lds(const void* src, void* dst,
//                                     int32_t size, int32_t offset,
//                                     int32_t aux);
//
//   src:    global-address-space pointer (address space 1).  Mapped to
//           VADDR — each lane can hold a different address.
//   dst:    LDS-address-space pointer (address space 3).  Mapped to M0
//           by the compiler.  Must be wave-uniform.  The hardware adds
//           lane_id * element_size implicitly.
//   size:   transfer width per lane in bytes; must be an immediate.
//           CDNA3: 1, 2, 4.  CDNA4: 1, 2, 4, 12, 16.
//   offset: byte offset added to both src and dst; must be an immediate.
//   aux:    cache policy bits (SC0, NT, SC1); must be an immediate.
//
// Loading strategy:
//   The tile is treated as a flat array of elements.  Elements are grouped
//   into chunks of 64 (one wavefront width).  Each wavefront handles one
//   chunk per iteration, round-robin across wavefronts:
//
//     M0 = &lds_dest[chunk_start]          (wave-uniform LDS base)
//     lane k: VADDR = &global_src[...]     (per-lane global address)
//     → hardware writes lane k's value to  M0 + k * 4
//
//   For BLOCK_TILE_M=128, K_TILE_SIZE=16: 2048 elements / 64 per instruction
//   = 32 chunks.  With 4 wavefronts: 8 instructions per wavefront.
//
// Constraint: the total number of tile elements must be a multiple of 64.
// ---------------------------------------------------------------------------
// [Sphinx direct load policy start]
template<int BlockTileM_, int BlockTileN_, int KTileSize_>
struct DirectLoadTilePolicy
{
    static constexpr int num_buffers  = 1;
    static constexpr int block_tile_m = BlockTileM_;
    static constexpr int block_tile_n = BlockTileN_;
    static constexpr int k_tile_size  = KTileSize_;

    static_assert((BlockTileM_ * KTileSize_) % 64 == 0,
                  "tile_a element count must be a multiple of 64 (wavefront width)");
    static_assert((BlockTileN_ * KTileSize_) % 64 == 0,
                  "tile_b_T element count must be a multiple of 64 (wavefront width)");

    struct SharedStorage
    {
        float tile_a[BlockTileM_][KTileSize_];
        float tile_b_T[BlockTileN_][KTileSize_];
    };

    // Helper: gather-load a flat tile from global memory to LDS.
    //
    // Elements are processed in chunks of 64 (one per wavefront-wide
    // instruction).  Each lane computes its own global source address;
    // the hardware writes lane k's value to M0 + k * sizeof(float).
    //
    // Parameters:
    //   lds_dest       – base of the flat LDS tile array
    //   global_src     – base of the global matrix
    //   num_rows/cols  – tile dimensions (LDS layout is [num_rows][num_cols])
    //   global_row/col – top-left corner of this tile in the global matrix
    //   global_stride  – row stride of the global matrix
    //   transpose      – if true, swap row/col in the global read pattern
    //                    (used for tile_b_T: LDS[col][ki], global B[ki][col])
    __device__ static void direct_load_flat(float*       lds_dest,
                                             const float* global_src,
                                             int          num_rows,
                                             int          num_cols,
                                             int          global_row_start,
                                             int          global_col_start,
                                             int          global_stride,
                                             bool         transpose,
                                             int          tid,
                                             int          num_threads)
    {
        constexpr int warp_size    = 64;
        const int     wave_id      = tid / warp_size;
        const int     lane         = tid % warp_size;
        const int     num_waves    = num_threads / warp_size;
        const int     total        = num_rows * num_cols;
        const int     total_chunks = total / warp_size;

        for(int chunk = wave_id; chunk < total_chunks; chunk += num_waves)
        {
            const int chunk_start = chunk * warp_size;

            // --- Per-lane global source address ---
            // Flat index of the element this lane handles.
            const int flat_idx = chunk_start + lane;
            const int lr       = flat_idx / num_cols; // row in LDS tile
            const int lc       = flat_idx % num_cols; // col in LDS tile

            // Map to global coordinates (swap row/col for transposed tiles).
            const int gr = global_row_start + (transpose ? lc : lr);
            const int gc = global_col_start + (transpose ? lr : lc);

            // Each lane computes its own global address (VADDR, per-lane).
            const float* src_lane = &global_src[gr * global_stride + gc];

            // --- Wave-uniform LDS destination ---
            // M0 points to the start of this 64-element chunk.
            // The hardware adds lane * sizeof(float) implicitly, so
            // lane k's value lands at lds_dest[chunk_start + k].
            float* dst_chunk = &lds_dest[chunk_start];

            __builtin_amdgcn_global_load_lds(
                src_lane,                       // per-lane global address
                static_cast<void*>(dst_chunk),  // wave-uniform LDS base (→ M0)
                4,                              // size: 4 bytes per lane
                0,                              // offset: immediate, 0
                0);                             // aux: wave scope, temporal
        }
    }

    __device__ static void prologue(SharedStorage& /*smem*/,
                                    const float* /*A*/, const float* /*B*/,
                                    int /*block_row_start*/, int /*block_col_start*/,
                                    int /*m*/, int /*n*/, int /*k*/,
                                    int /*tid*/, int /*num_threads*/)
    {
    }

    __device__ static void prefetch(SharedStorage& smem,
                                    const float*   A,
                                    const float*   B,
                                    int            t_next,
                                    int block_row_start, int block_col_start,
                                    int m, int n, int k,
                                    int tid, int num_threads)
    {
        const int t = t_next; // single-buffer: t_next == t

        // tile_a: row-major gather [block_tile_m × k_tile_size]
        // LDS flat layout matches global row-major order, so transpose=false.
        direct_load_flat(&smem.tile_a[0][0], A,
                         block_tile_m, k_tile_size,
                         block_row_start, t * k_tile_size,
                         k,
                         /*transpose=*/false,
                         tid, num_threads);

        // tile_b_T: transposed gather [block_tile_n × k_tile_size]
        //   LDS layout: tile_b_T[col_within_tile][k_idx]
        //   Global read: B[k_idx_global][col_global]
        //   transpose=true swaps lr↔lc in the global coordinate mapping.
        direct_load_flat(&smem.tile_b_T[0][0], B,
                         block_tile_n, k_tile_size,
                         t * k_tile_size, block_col_start,
                         n,
                         /*transpose=*/true,
                         tid, num_threads);
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
// [Sphinx direct load policy end]

// ---------------------------------------------------------------------------
// Architecture-specific policy stubs (filled by follow-up PRs)
// ---------------------------------------------------------------------------
// [Sphinx arch dispatch start]
//
// Selecting a ComputePolicy at compile time based on device ISA:
//
//   gfx90a / gfx940 / gfx941 / gfx942  →  MFMAPolicy<float,float>   (CDNA2/3)
//   gfx1100 / gfx1101                   →  WMMAPolicy<__half,float>  (RDNA3)
//   gfx1200 / gfx1201                   →  RDNA4WMMAPolicy (relaxed lane rules)
//   fallback (all others)               →  ScalarFMAPolicy
//
// Usage pattern (not yet compiled; shown for documentation):
//
// #if defined(__gfx90a__) || defined(__gfx940__) || defined(__gfx941__) || defined(__gfx942__)
//   using DefaultComputePolicy = MFMAPolicy</* … */>;
// #elif defined(__gfx1100__) || defined(__gfx1101__) || defined(__gfx1102__)
//   using DefaultComputePolicy = WMMAPolicy</* … */>;
// #elif defined(__gfx1200__) || defined(__gfx1201__)
//   using DefaultComputePolicy = RDNA4WMMAPolicy</* … */>;
// #else
//   using DefaultComputePolicy = ScalarFMAPolicy<8, 8, 16>;
// #endif
//
// The kernel template (GemmKernel) is identical for all policies; only the
// type alias changes.
// [Sphinx arch dispatch end]

// ===========================================================================
// Section 5: Generic kernel template
// ===========================================================================
// [Sphinx gemm kernel start]
template<typename TilePolicy, typename ComputePolicy>
__global__ void
    __launch_bounds__(TilePolicy::block_tile_m / ComputePolicy::thread_tile_m
                          * TilePolicy::block_tile_n / ComputePolicy::thread_tile_n)
    matrix_multiply_generic(const float* __restrict__ A,
                             const float* __restrict__ B,
                             float* __restrict__       C,
                             int                       m,
                             int                       n,
                             int                       k)
{
    // Instantiate TilePolicyTraits/ComputePolicyTraits to trigger static_assert
    // checks at the point where this kernel is instantiated.
    (void)TilePolicyTraits<TilePolicy>{};
    (void)ComputePolicyTraits<ComputePolicy>{};

    // [Sphinx gemm kernel smem start]
    __shared__ typename TilePolicy::SharedStorage smem;
    // [Sphinx gemm kernel smem end]

    const int tid         = threadIdx.y * blockDim.x + threadIdx.x;
    const int lane_id     = tid % 64; // wavefront lane index (AMD: 64-wide)
    const int num_threads = blockDim.x * blockDim.y;

    // Top-left corner of the block tile in global C
    const int block_row_start = blockIdx.y * TilePolicy::block_tile_m;
    const int block_col_start = blockIdx.x * TilePolicy::block_tile_n;

    // Thread's output sub-tile offset within the block tile
    int thread_row = 0, thread_col = 0;
    ComputePolicy::thread_tile_offset(tid, lane_id, &thread_row, &thread_col);

    // Initialise accumulator to zero
    typename ComputePolicy::Accumulator acc;
    ComputePolicy::zero(acc);

    const int num_tiles = (k + TilePolicy::k_tile_size - 1) / TilePolicy::k_tile_size;

    // [Sphinx gemm kernel prologue start]
    // Double-buffer prologue: load tile 0 for policies that support it.
    // For SingleBufferTilePolicy this is a no-op.
    TilePolicy::prologue(smem, A, B,
                         block_row_start, block_col_start,
                         m, n, k, tid, num_threads);
    // [Sphinx gemm kernel prologue end]

    // [Sphinx gemm kernel loop start]
    for(int t = 0; t < num_tiles; ++t)
    {
        // --- Prefetch next tile (or load current tile for single-buffer) ---
        // For SoftwareDoubleBufferTilePolicy: issues the load for tile (t+1)
        //   into the background buffer while we compute from the foreground.
        // For SingleBufferTilePolicy: loads tile t (t_next == t here).
        TilePolicy::prefetch(smem, A, B, t + (TilePolicy::num_buffers == 1 ? 0 : 1),
                             block_row_start, block_col_start,
                             m, n, k, tid, num_threads);

        // --- Acquire: synchronise before compute ---
        // Single-buffer:  __syncthreads() after load
        // Double-buffer:  no-op (previous release() already synced)
        TilePolicy::acquire(t);

        // --- Compute: outer-product accumulation from the active buffer ---
        const int buf = TilePolicy::buf_idx(t);
        const float* tile_a_ptr   = TilePolicy::get_tile_a(smem, buf);
        const float* tile_b_T_ptr = TilePolicy::get_tile_b_T(smem, buf);

        // k_step: number of k-indices consumed per mma() call.
        // ScalarFMAPolicy uses k_step=1 (one FMA per call); intrinsic-based
        // policies can use higher values (e.g. k_step=2 for dot-product ops).
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

        // --- Release: synchronise after compute ---
        // Single-buffer:  __syncthreads() before next iteration overwrites tile
        // Double-buffer:  __syncthreads() covers both compute-done and prefetch-done
        TilePolicy::release(t);
    }
    // [Sphinx gemm kernel loop end]

    // [Sphinx gemm kernel store start]
    // Write the accumulator to global memory.
    ComputePolicy::store_c(acc, C,
                            block_row_start + thread_row,
                            block_col_start + thread_col,
                            m, n, lane_id);
    // [Sphinx gemm kernel store end]
}
// [Sphinx gemm kernel end]

// ===========================================================================
// Section 6: Concrete policy instantiations for this example
// ===========================================================================

// TilePolicy parameters
constexpr int BLOCK_TILE_M  = 128;
constexpr int BLOCK_TILE_N  = 128;
constexpr int K_TILE_SIZE   = 16;

// ComputePolicy parameters
constexpr int THREAD_TILE_M = 8;
constexpr int THREAD_TILE_N = 8;
// Derived block dimensions
constexpr int BLOCK_DIM_X   = BLOCK_TILE_N / THREAD_TILE_N; // 16
constexpr int BLOCK_DIM_Y   = BLOCK_TILE_M / THREAD_TILE_M; // 16

// [Sphinx policy aliases start]
// Single-buffer variant (baseline – equivalent to register-tiled step 3)
using SingleBufPolicy = SingleBufferTilePolicy<BLOCK_TILE_M, BLOCK_TILE_N, K_TILE_SIZE>;

// Double-buffer variant (software ping-pong, same tile/compute parameters)
using DoubleBufPolicy = SoftwareDoubleBufferTilePolicy<BLOCK_TILE_M, BLOCK_TILE_N, K_TILE_SIZE>;

// Portable scalar outer-product compute policy
using ScalarPolicy = ScalarFMAPolicy<THREAD_TILE_M, THREAD_TILE_N, BLOCK_DIM_X>;

// Direct global→LDS tile policy (CDNA3/CDNA4, bypasses VGPRs during tile load)
using DirectLoadPolicy = DirectLoadTilePolicy<BLOCK_TILE_M, BLOCK_TILE_N, K_TILE_SIZE>;
// [Sphinx policy aliases end]

// ===========================================================================
// Section 7: Helpers (timing, verification, metrics)
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
// Section 8: main
// ===========================================================================
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
    // Host buffers: random A, identity B
    // ------------------------------------------------------------------
    std::vector<float> h_A(M * K);
    std::vector<float> h_B(K * N, 0.0f);
    std::vector<float> h_C(M * N, 0.0f);

    std::mt19937                          gen(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for(float& v : h_A)
        v = dist(gen);
    for(int i = 0; i < std::min(K, N); ++i)
        h_B[i * N + i] = 1.0f;

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
    // GemmKernel<SingleBufPolicy, ScalarPolicy>
    // ------------------------------------------------------------------
    {
        auto launch = [&]()
        {
            matrix_multiply_generic<SingleBufPolicy, ScalarPolicy>
                <<<grid, block>>>(d_A, d_B, d_C, M, N, K);
            HIP_CHECK(hipGetLastError());
        };

        const float ms = time_kernel_ms(launch, WARMUP_RUNS, TIMING_RUNS);
        HIP_CHECK(hipMemcpy(h_C.data(), d_C, sizeof(float) * M * N, hipMemcpyDeviceToHost));
        const bool ok = verify_result(h_C, h_A, M, N, K);
        print_metrics("GemmKernel<SingleBufPolicy, ScalarPolicy>", ok, ms, M, N, K);
    }

    // ------------------------------------------------------------------
    // GemmKernel<DoubleBufPolicy, ScalarPolicy>
    // ------------------------------------------------------------------
    {
        auto launch = [&]()
        {
            matrix_multiply_generic<DoubleBufPolicy, ScalarPolicy>
                <<<grid, block>>>(d_A, d_B, d_C, M, N, K);
            HIP_CHECK(hipGetLastError());
        };

        const float ms = time_kernel_ms(launch, WARMUP_RUNS, TIMING_RUNS);
        HIP_CHECK(hipMemcpy(h_C.data(), d_C, sizeof(float) * M * N, hipMemcpyDeviceToHost));
        const bool ok = verify_result(h_C, h_A, M, N, K);
        print_metrics("GemmKernel<DoubleBufPolicy, ScalarPolicy>", ok, ms, M, N, K);
    }

    // ------------------------------------------------------------------
    // GemmKernel<DirectLoadPolicy, ScalarPolicy>
    //
    // Same kernel template, same ComputePolicy, only the TilePolicy changes.
    // Tile loads use __builtin_amdgcn_global_load_lds to transfer data from
    // global memory directly to LDS without staging in VGPRs.
    //
    // Available on CDNA3 and CDNA4 only.  The arithmetic is identical to the
    // other variants (full FP32 scalar outer-product), so results match exactly.
    // ------------------------------------------------------------------
    {
        auto launch = [&]()
        {
            matrix_multiply_generic<DirectLoadPolicy, ScalarPolicy>
                <<<grid, block>>>(d_A, d_B, d_C, M, N, K);
            HIP_CHECK(hipGetLastError());
        };

        const float ms = time_kernel_ms(launch, WARMUP_RUNS, TIMING_RUNS);
        HIP_CHECK(hipMemcpy(h_C.data(), d_C, sizeof(float) * M * N, hipMemcpyDeviceToHost));
        const bool ok = verify_result(h_C, h_A, M, N, K);
        print_metrics("GemmKernel<DirectLoadPolicy, ScalarPolicy>", ok, ms, M, N, K);
    }

    // ------------------------------------------------------------------
    // Cleanup
    // ------------------------------------------------------------------
    HIP_CHECK(hipFree(d_A));
    HIP_CHECK(hipFree(d_B));
    HIP_CHECK(hipFree(d_C));

    return EXIT_SUCCESS;
}
