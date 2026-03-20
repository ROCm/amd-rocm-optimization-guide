.. meta::
  :description: Step-by-step optimization of a HIP matrix multiplication kernel
  :keywords: AMD, ROCm, HIP, matrix multiplication, LDS, register tiling, double buffering, vectorized loads, launch_bounds, occupancy, generic kernel, Stepanov

.. _matrix-multiply-optimization:

********************************************************************************
Optimizing matrix multiplication: a step-by-step guide
********************************************************************************

This tutorial walks through seven progressive optimization steps applied to a
general-purpose single-precision (FP32) matrix multiplication kernel
(GEMM: :math:`\pmb{C} = \pmb{A} \times \pmb{B}`).  Each step builds on the
previous one, introducing a specific technique and explaining how to measure its
effect with the ROCm performance analysis stack.

The complete source files for all steps are available at:

* :download:`Step 1 - Naïve <../../../tools/example_codes/matrix_multiply_naive.hip>`
* :download:`Step 2 - LDS tiling <../../../tools/example_codes/matrix_multiply_lds.hip>`
* :download:`Step 3 - Register tiling <../../../tools/example_codes/matrix_multiply_register_tiling.hip>`
* :download:`Step 4 - Double buffering <../../../tools/example_codes/matrix_multiply_double_buffer.hip>`
* :download:`Step 5 - Vectorized loads <../../../tools/example_codes/matrix_multiply_vectorized.hip>`
* :download:`Step 6 - Register pressure <../../../tools/example_codes/matrix_multiply_launch_bounds.hip>`
* :download:`Step 7 - Generic kernel <../../../tools/example_codes/matrix_multiply_generic.hip>`

.. note::

   All examples target :math:`4096 \times 4096` matrices in FP32 row-major
   layout and are validated against the identity-matrix test
   (:math:`B = I \implies C = A`).  They compile with
   ``amdclang++ -O3 -std=c++17`` and run on any ROCm-supported GPU architecture.

Prerequisites
=============

* ROCm installed and ``amdclang++`` available on ``PATH``.
* Familiarity with the HIP execution model (grids, blocks, warps) and its
  mapping to AMD GPU hardware (dispatches, work-groups, wavefronts).
* ``rocprofv3``, ROCm Compute Profiler (``rocprof-compute``) and ROCprof Compute
  Viewer (RCV) installed for performance analysis.

Background: the GEMM arithmetic intensity
==========================================

For an :math:`M \times K \times N` GEMM the arithmetic intensity—floating-point
operations per byte of DRAM traffic—is:

.. math::

   I = \frac{2 \cdot M \cdot N \cdot K}{\text{sizeof}(\text{float}) \cdot (M \cdot K + K \cdot N + M \cdot N)}

For :math:`M = N = K = 4096` this evaluates to roughly **4096 FLOPs/byte**, far
above the roofline crossover point of any current AMD GPU.  GEMM is therefore
**compute-bound** in principle—but only if data is supplied fast enough to keep
the compute units busy.  The naïve kernel falls well below the roofline because
it is *memory-bound in practice*: global memory latency stalls dominate.

The optimization steps that follow progressively close the gap between actual
and theoretical throughput by improving data reuse and instruction-level
efficiency.

Step 1: Naïve kernel
=====================

The naïve kernel assigns one thread per output element.  Each thread reads a
full row of :math:`\pmb{A}` and a full column of :math:`\pmb{B}` directly from
global memory.

.. literalinclude:: ../../../tools/example_codes/matrix_multiply_naive.hip
   :language: cpp
   :start-after: [Sphinx naive kernel start]
   :end-before: [Sphinx naive kernel end]

Launch configuration:

.. literalinclude:: ../../../tools/example_codes/matrix_multiply_naive.hip
   :language: cpp
   :start-after: [Sphinx naive launch config start]
   :end-before: [Sphinx naive launch config end]

**Compile and run:**

.. code-block:: bash

   hipcc -O3 -std=c++17 matrix_multiply_naive.hip -o mm_naive
   ./mm_naive

**Profile wall-clock time with rocprofv3:**

.. code-block:: bash

   rocprofv3 --hip-trace --output-format csv -- ./mm_naive

Examine the ``Kernel_Duration`` column for ``matrix_multiply_naive``.

**Profile memory traffic with rocprof-compute:**

.. code-block:: bash

   rocprof-compute profile --name naive -- ./mm_naive
   rocprof-compute analyze --path naive/

Focus on the **L2 Cache** panel and the ``TCP_TCC_READ_REQ_sum`` counter.  For
the naïve kernel you will see that the total bytes transferred from DRAM
significantly exceed the minimum required bandwidth (``sizeof(float) * (M*K +
K*N + M*N)``), confirming cache thrashing.

Step 2: LDS tiling
==================

The key insight is that every element of :math:`\pmb{A}` is used by :math:`N`
threads (one per output column) and every element of :math:`\pmb{B}` is used by
:math:`M` threads.  Caching a ``TILE_SIZE * TILE_SIZE`` strip of :math:`\pmb{A}`
and :math:`\pmb{B}` in Local Data Share (LDS) lets all ``TILE_SIZE²`` threads in
a block reuse that data without touching global memory again.

.. literalinclude:: ../../../tools/example_codes/matrix_multiply_lds.hip
   :language: cpp
   :start-after: [Sphinx LDS tile size start]
   :end-before: [Sphinx LDS tile size end]

**Shared memory allocation:**

.. literalinclude:: ../../../tools/example_codes/matrix_multiply_lds.hip
   :language: cpp
   :start-after: [Sphinx LDS shared memory start]
   :end-before: [Sphinx LDS shared memory end]

**Load phase (cooperative, one element per thread):**

.. literalinclude:: ../../../tools/example_codes/matrix_multiply_lds.hip
   :language: cpp
   :start-after: [Sphinx LDS load phase start]
   :end-before: [Sphinx LDS load phase end]

**Compute phase (inner product from LDS):**

.. literalinclude:: ../../../tools/example_codes/matrix_multiply_lds.hip
   :language: cpp
   :start-after: [Sphinx LDS compute phase start]
   :end-before: [Sphinx LDS compute phase end]

**LDS bank conflict analysis:**

AMD GPUs have 32 LDS banks interleaved at 4-byte granularity.  For
``TILE_SIZE = 16``:

* ``tile_a[ty][i]``: all threads in a wavefront access the same row at column
  ``i``—one broadcast, no conflict.
* ``tile_b[i][tx]``: consecutive ``tx`` values map to consecutive banks—no
  conflict.

.. note::

   If ``TILE_SIZE`` is increased to 32 (equal to the bank count), consecutive
   rows of ``tile_b`` alias to the same bank, causing 32-way conflicts during
   the compute phase.  Step 3 resolves this with a transposed layout.

**What to observe in rocprof-compute after this step:**

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Counter group
     - What to look for
   * - ``SQ`` / ``TCP``
     - Reduction in ``TCP_TCC_READ_REQ_sum`` proportional to ``TILE_SIZE``
   * - ``LDS``
     - Low ``LDS_BANK_CONFLICT`` count (confirm no conflicts for ``TILE_SIZE=16``)

Step 3: Register tiling
=======================

In the LDS kernel each thread computes exactly one output element, reading
``TILE_SIZE`` values from ``tile_a`` and ``TILE_SIZE`` values from ``tile_b``
for every K-strip.  If instead each thread computes a
``THREAD_TILE_M × THREAD_TILE_N`` sub-tile in registers, it amortises the LDS
load cost across ``THREAD_TILE_M × THREAD_TILE_N`` outputs.

The outer product of a length-``THREAD_TILE_M`` column fragment of A and a
length-``THREAD_TILE_N`` row fragment of B produces a full
``THREAD_TILE_M × THREAD_TILE_N`` block of C contributions using only
``THREAD_TILE_M + THREAD_TILE_N`` LDS reads instead of
``THREAD_TILE_M × THREAD_TILE_N``.

**Tile parameters:**

.. literalinclude:: ../../../tools/example_codes/matrix_multiply_register_tiling.hip
   :language: cpp
   :start-after: [Sphinx register tiling params start]
   :end-before: [Sphinx register tiling params end]

**Transposed tile_b layout:**

``tile_b`` is stored transposed in LDS as ``tile_b_T[BLOCK_TILE_N][K_TILE_SIZE]``
(column-major for B).  This change has two benefits:

1. Eliminates the bank-conflict risk that arises when ``TILE_SIZE`` equals the
   LDS bank count.
2. Produces stride-1 reads during the outer-product compute phase.

**LDS allocation:**

.. literalinclude:: ../../../tools/example_codes/matrix_multiply_register_tiling.hip
   :language: cpp
   :start-after: [Sphinx register tiling shared memory start]
   :end-before: [Sphinx register tiling shared memory end]

**Cooperative tile load:**

.. literalinclude:: ../../../tools/example_codes/matrix_multiply_register_tiling.hip
   :language: cpp
   :start-after: [Sphinx register tiling load phase start]
   :end-before: [Sphinx register tiling load phase end]

**Outer-product accumulation:**

.. literalinclude:: ../../../tools/example_codes/matrix_multiply_register_tiling.hip
   :language: cpp
   :start-after: [Sphinx register tiling compute phase start]
   :end-before: [Sphinx register tiling compute phase end]

**Write-back:**

.. literalinclude:: ../../../tools/example_codes/matrix_multiply_register_tiling.hip
   :language: cpp
   :start-after: [Sphinx register tiling store start]
   :end-before: [Sphinx register tiling store end]

**What to observe in rocprof-compute:**

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Counter group
     - What to look for
   * - ``VALU``
     - Increase in VALU utilization (more FMAs per LDS read)
   * - ``LDS``
     - Reduced LDS reads per output element (``THREAD_TILE_M + THREAD_TILE_N``
       instead of ``2 × THREAD_TILE_M × THREAD_TILE_N``)
   * - ``SQ``
     - Reduction in stall cycles (register reuse hides LDS latency)

Step 4: Double buffering
========================

Every iteration of the K-strip loop stalls at ``__syncthreads()`` waiting for
LDS tile loads to complete before compute can begin.  Software double buffering
hides this latency by maintaining two LDS buffer pairs (a *ping* and a *pong*)
and loading the next tile into the background buffer while the current buffer
is being consumed.

Both buffering strategies are hidden behind a ``TilePolicy`` interface so that
the kernel body is identical regardless of the chosen approach.

**Policy interface overview:**

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Method
     - Responsibility
   * - ``prologue``
     - Load tile 0 into buffer 0 and synchronise (double-buffer only; no-op for single)
   * - ``prefetch``
     - Issue the load for the next tile into the background buffer
   * - ``acquire``
     - Synchronise before compute (single-buffer: ``__syncthreads()``; double: no-op)
   * - ``release``
     - Synchronise after compute (both: ``__syncthreads()``)
   * - ``buf_idx``
     - Return which buffer to read for the current iteration

**Single-buffer policy** (baseline — same logic as Step 3):

.. literalinclude:: ../../../tools/example_codes/matrix_multiply_double_buffer.hip
   :language: cpp
   :start-after: [Sphinx single buffer policy start]
   :end-before: [Sphinx single buffer policy end]

**Software double-buffer policy:**

.. literalinclude:: ../../../tools/example_codes/matrix_multiply_double_buffer.hip
   :language: cpp
   :start-after: [Sphinx double buffer policy start]
   :end-before: [Sphinx double buffer policy end]

**Compile-time policy validation:**

.. literalinclude:: ../../../tools/example_codes/matrix_multiply_double_buffer.hip
   :language: cpp
   :start-after: [Sphinx tile policy static assert start]
   :end-before: [Sphinx tile policy static assert end]

**Unified kernel template:**

.. literalinclude:: ../../../tools/example_codes/matrix_multiply_double_buffer.hip
   :language: cpp
   :start-after: [Sphinx double buffer kernel start]
   :end-before: [Sphinx double buffer kernel end]

.. note::

   The LDS footprint doubles with software double buffering (two tile pairs
   instead of one).  For the parameters in this example
   (``BLOCK_TILE_M = BLOCK_TILE_N = 128``, ``K_TILE_SIZE = 16``):

   * Single-buffer LDS: 128 × 16 × 4 × 2 = 16 KiB
   * Double-buffer LDS: 16 KiB × 2 = 32 KiB

   This is within the 32–64 KiB LDS budget on all supported architectures, but
   leaves less headroom for occupancy.  Use the ROCprof Compute Viewer to verify
   occupancy does not drop when switching from single- to double-buffered policy.

**What to observe in rocprof-compute:**

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Counter group
     - What to look for
   * - ``SQ``
     - Reduction in ``SQ_WAIT_INST_LDS`` stall cycles (load latency hidden)
   * - ``TCP``
     - Similar total bandwidth to Step 3 (same number of DRAM fetches)

Step 5: Vectorized loads
========================

Global memory transactions are most efficient when each wavefront issues a
single 128-byte coalesced request.  For FP32 data (4 bytes per element),
reading 4 consecutive elements per thread (``float4``, 128-bit) saturates one
transaction per lane.

Two vector widths are shown alongside the scalar baseline:

.. list-table::
   :header-rows: 1
   :widths: 15 20 25 40

   * - Width
     - HIP type
     - Alignment
     - Transaction size per thread
   * - 1
     - ``float``
     - 4 bytes
     - 32-bit (scalar)
   * - 2
     - ``float2``
     - 8 bytes
     - 64-bit (DWORD2)
   * - 4
     - ``float4``
     - 16 bytes
     - 128-bit (DWORD4)

**Vector type helper:**

.. literalinclude:: ../../../tools/example_codes/matrix_multiply_vectorized.hip
   :language: cpp
   :start-after: [Sphinx vector type start]
   :end-before: [Sphinx vector type end]

**Vectorized load function:**

.. literalinclude:: ../../../tools/example_codes/matrix_multiply_vectorized.hip
   :language: cpp
   :start-after: [Sphinx vector load function start]
   :end-before: [Sphinx vector load function end]

**Alignment requirements:**

The ``reinterpret_cast`` in ``vectorized_load`` is only safe when the source
pointer is aligned to ``sizeof(Vec)`` bytes.  Two alignment guarantees apply:

1. ``hipMalloc`` returns a pointer aligned to at least 256 bytes, satisfying
   ``float4`` (16 bytes) for the base of any matrix.
2. Each row of A begins at offset ``row × K × sizeof(float)``.  For ``float4``
   loads the row length ``K`` must be a multiple of 4.  A compile-time
   ``static_assert`` enforces this for the tile parameters.

A runtime check is included in the example to catch misaligned user-supplied
pointers:

.. literalinclude:: ../../../tools/example_codes/matrix_multiply_vectorized.hip
   :language: cpp
   :start-after: [Sphinx alignment check start]
   :end-before: [Sphinx alignment check end]

.. note::

   Only the A-tile load is vectorized because its elements are contiguous in
   global memory (row-major, stride 1).  The B-tile is loaded column-by-column
   (stride N in global memory), which is not amenable to simple vector loads.
   Architecture-specific intrinsics (MFMA, WMMA), covered in follow-up sections,
   address this asymmetry.

**What to observe in rocprof-compute:**

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Counter group
     - What to look for
   * - ``TCP``
     - Reduction in ``TCP_TOTAL_CACHE_ACCESSES`` (fewer, wider transactions)
   * - ``SQ``
     - Reduction in issue stalls caused by scalar load throughput limits
   * - ``L2``
     - Effective bandwidth per request increases with vector width

Step 6: Register pressure and occupancy
========================================

GPU occupancy—the ratio of active wavefronts to the hardware maximum—is set by
the most constrained resource.  For register-tiled GEMM kernels that resource
is typically the **VGPR file**: each thread holds
``THREAD_TILE_M × THREAD_TILE_N`` accumulator registers plus fragment arrays,
and the compiler may allocate additional temporaries.

Higher register usage means fewer concurrent wavefronts per CU, which reduces
the GPU's ability to hide memory latency through wavefront switching.  Conversely,
forcibly reducing register usage can increase register spilling to scratch memory
(a slow VGPR-to-DRAM path), degrading performance.

Three kernel variants illustrate the tradeoff:

**No annotation (compiler decides freely):**

.. literalinclude:: ../../../tools/example_codes/matrix_multiply_launch_bounds.hip
   :language: cpp
   :start-after: [Sphinx no hint kernel start]
   :end-before: [Sphinx no hint kernel end]

**``__launch_bounds__``:**

.. literalinclude:: ../../../tools/example_codes/matrix_multiply_launch_bounds.hip
   :language: cpp
   :start-after: [Sphinx launch bounds kernel start]
   :end-before: [Sphinx launch bounds kernel end]

The two-argument form ``__launch_bounds__(max_threads, min_waves_per_eu)``
tells the compiler to allocate VGPRs such that at least ``min_waves_per_eu``
wavefronts can be resident per EU simultaneously, given
``max_threads / warpSize`` wavefronts per block.

**``[[clang::amdgpu_waves_per_eu]]``:**

.. literalinclude:: ../../../tools/example_codes/matrix_multiply_launch_bounds.hip
   :language: cpp
   :start-after: [Sphinx amdgpu waves per eu kernel start]
   :end-before: [Sphinx amdgpu waves per eu kernel end]

This Clang attribute directly instructs the backend to target a wavefront
occupancy in the range ``[min, max]`` per EU.

**Finding optimal values with ROCprof Compute Viewer:**

1. Profile all three kernel variants with ``rocprof-compute``:

   .. code-block:: bash

      rocprof-compute profile --name launch_bounds -- ./mm_launch_bounds
      rocprof-compute analyze --path launch_bounds/

2. Open the results in the ROCprof Compute Viewer.  In the **Kernel Statistics**
   panel, note the values of:

   * **VGPRs used** (vector register count allocated per thread)
   * **SGPRs used** (scalar register count)
   * **Scratch memory used** (>0 means VGPRs are spilling to DRAM—avoid this)
   * **Occupancy** (wavefronts per EU achieved at launch)

3. Calculate the theoretical maximum occupancy from the VGPR count using the
   formula for your architecture (available in the AMD ISA reference).
4. If the compiler allocated more VGPRs than necessary and occupancy is below
   the target, add ``__launch_bounds__`` with a ``min_waves_per_eu`` that
   reflects the desired occupancy.
5. Re-profile and re-check **Scratch memory used**—if it increases significantly,
   the compiler was forced to spill and the constraint is too aggressive.

.. note::

   The numeric values ``MIN_WAVES_PER_EU = 2`` and ``WAVES_PER_EU_MAX = 4``
   in the example are illustrative.  Optimal values depend on the target GPU
   and the exact kernel register usage shown in the ROCprof Compute Viewer.
   Always re-profile with ``rocprof-compute`` after applying the annotation to
   confirm that occupancy improves without introducing scratch-memory spilling.

**What to observe in the ROCprof Compute Viewer:**

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Panel / counter
     - What to look for
   * - Kernel Statistics → VGPRs / Scratch memory / Occupancy
     - Confirm VGPRs decrease and occupancy rises after adding the annotation;
       scratch memory must remain at zero
   * - ``SQ`` → ``SQ_WAVES``
     - Active wavefronts per CU (should increase with higher occupancy)
   * - ``SQ`` → ``SQ_WAIT_INST_LDS`` / ``SQ_WAIT_INST_VMEM``
     - Latency-hiding efficiency (stalls fall when more wavefronts are resident)

Step 7: Generic kernel (Stepanov-style)
=======================================

By this point the scalar kernel is already a strong general-purpose
implementation: it uses LDS tiling to exploit data reuse, register tiling to
maximise arithmetic intensity, software double buffering to hide load latency,
and vectorized loads to reduce memory transaction overhead.  For many workloads
this is sufficient.

Squeezing out the last few percent of throughput, however, requires
architecture-specific matrix-multiply instructions: MFMA on CDNA GPUs and WMMA
on RDNA3/4.  These instructions perform a small matrix multiply directly in
hardware and deliver substantially higher FLOP/s than an equivalent sequence of
scalar FMAs.

Steps 1–6 produced a well-optimised scalar GEMM kernel, but repeating the same
work for each architecture-specific instruction set—MFMA on CDNA, WMMA on RDNA3,
the relaxed WMMA variant on RDNA4—would mean maintaining several near-identical
copies of the kernel with only the inner computation swapped out.  Any future
improvement (a new tiling strategy, a wider vector load, a different buffering
depth) would have to be applied to every copy independently.

The goal of this step is to factor the kernel so that the **optimised
orchestration is written once** and architecture-specific intrinsics can be
**dropped in later as a policy**, touching no kernel code at all.

The insight from the preceding steps is that all the work decomposes into
exactly two independent concerns:

1. **Data movement** (``TilePolicy``) — tile shape, LDS layout, vector load
   width, and buffering strategy.  These optimisations are identical regardless
   of which instruction is used to compute the output; a ``float4`` load into a
   double-buffered LDS tile is just as beneficial whether the inner loop uses
   scalar FMAs or MFMA.

2. **Arithmetic** (``ComputePolicy``) — how a thread's register fragment is
   loaded from LDS and how the output accumulator is updated.  This is the only
   part that differs between scalar code and architecture-specific intrinsics.

Following Alexander Stepanov's principle of *lifting* an algorithm into its
most general form, these two responsibilities are encapsulated in two orthogonal
*policy classes*: ``TilePolicy`` and ``ComputePolicy``.  A single kernel
template ``matrix_multiply_generic<TilePolicy, ComputePolicy>`` orchestrates
the common control flow while delegating all architecture-specific details to
the policies.

The kernel presented here uses ``ScalarFMAPolicy`` as the ``ComputePolicy``.
It already incorporates all the optimisations from the preceding steps: LDS
tiling, register tiling, software double buffering, and vectorized loads.  When
an MFMA or WMMA ``ComputePolicy`` is provided in a follow-up section, the same
data-movement infrastructure and the same kernel orchestration are reused
unchanged—only the inner arithmetic changes.

Policy interfaces
-----------------

Both interfaces are documented in full in ``matrix_multiply_generic.hip``.
The key design decisions are:

**TilePolicy interface summary:**

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Requirement
     - Rationale
   * - ``num_buffers = 1 | 2``
     - Governs LDS layout (one or two buffer pairs) and sync strategy
   * - ``block_tile_m``, ``block_tile_n``, ``k_tile_size``
     - Tile shape needed by the kernel to compute grid dimensions
   * - ``SharedStorage`` (trivially destructible)
     - Placed in ``__shared__``; must not require a destructor call
   * - ``prologue``, ``prefetch``, ``acquire``, ``release``
     - Data movement hooks; see below

**ComputePolicy interface summary:**

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Requirement
     - Rationale
   * - ``thread_tile_m``, ``thread_tile_n``
     - Output sub-tile per thread; kernel derives block dimensions from these
   * - ``effective_lanes``
     - Unique output lanes per wavefront (64 for scalar/CDNA; 32 for RDNA3 WMMA
       due to lane-mirroring: lanes L and L+16 share the same fragment)
   * - ``thread_tile_offset(tid, lane_id, *row, *col)``
     - Architecture-aware thread→output mapping (lane_id needed for RDNA3)
   * - ``elem_a``, ``elem_b``
     - Element types of the register fragments (e.g. ``float``, ``__half``);
       the kernel loop is fully templated on these
   * - ``load_a``, ``load_b``, ``mma``, ``store_c``
     - Fragment load, multiply-accumulate, and write-back; all accept ``lane_id``
       for RDNA3 forward-compatibility (unused in scalar policy)

Data type scope: what the policies cover and what they don't
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``elem_a`` and ``elem_b`` parameterise the *register fragment* type and are
already fully wired through the kernel loop.  A future ``MFMAPolicy`` can
therefore set ``elem_a = __half`` and the conversion from LDS to fragment
happens inside ``load_a``—the kernel body is untouched.

However, two things are **not yet parameterised** and are hardcoded to
``float`` in this file:

* The **LDS storage type** — ``SharedStorage`` in every ``TilePolicy`` holds
  ``float`` arrays, and the cooperative load helpers write ``float`` into LDS.
* The **global memory pointer type** — the kernel signature takes
  ``const float* A``, ``const float* B``, ``float* C``.

This means two distinct cases arise when introducing intrinsics in follow-up
sections:

.. list-table::
   :header-rows: 1
   :widths: 20 30 50

   * - Case
     - Example
     - What needs to change
   * - **FP32 in memory, low-precision fragments**
     - Global ``float`` → LDS ``float`` → ``__half`` fragment for MFMA
     - Only ``ComputePolicy::load_a`` / ``load_b`` (conversion in registers).
       ``TilePolicy`` and the kernel signature are unchanged.
   * - **Low-precision in memory**
     - Global ``__half`` → LDS ``__half`` → ``__half`` fragment
     - ``TilePolicy`` needs an ``InputT`` template parameter so
       ``SharedStorage`` and the cooperative load helpers use ``InputT``
       instead of ``float``.  The kernel signature changes from
       ``const float*`` to ``const InputT*``.

This tutorial covers FP32 throughout and therefore only Case 1 is relevant
here.  Case 2 is left as an extension for the architecture-specific follow-up
sections.

RDNA3 lane-mirroring note
~~~~~~~~~~~~~~~~~~~~~~~~~

On RDNA3, each WMMA instruction operates on a 32-lane logical wavefront split
across 64 hardware lanes.  Lanes L and L+16 (0 ≤ L < 16) hold identical
copies of the same fragment element—only the lower 16 lanes write unique output
values.  The ``effective_lanes = 32`` constant and the ``lane_id`` parameter in
``store_c`` allow the ``ComputePolicy`` to apply this write-back rule without
any change to the calling kernel.  On RDNA4 this restriction is relaxed;
on RDNA2 WMMA is not available at all.

Compile-time validation
-----------------------

Both policy interfaces are validated with C++17 ``static_assert`` traits:

**TilePolicy traits:**

.. literalinclude:: ../../../tools/example_codes/matrix_multiply_generic.hip
   :language: cpp
   :start-after: [Sphinx tile policy traits start]
   :end-before: [Sphinx tile policy traits end]

**ComputePolicy traits:**

.. literalinclude:: ../../../tools/example_codes/matrix_multiply_generic.hip
   :language: cpp
   :start-after: [Sphinx compute policy traits start]
   :end-before: [Sphinx compute policy traits end]

C++17 vs C++20: concept syntax
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``static_assert`` approach above is portable to C++17 but requires the
traits struct to be instantiated explicitly, and error messages appear at the
trait instantiation site.  C++20 ``requires`` clauses provide a more ergonomic
alternative:

.. literalinclude:: ../../../tools/example_codes/matrix_multiply_generic.hip
   :language: cpp
   :start-after: [Sphinx cpp20 concept start]
   :end-before: [Sphinx cpp20 concept end]

With C++20, the constraint is expressed directly in the function signature and
violations are reported at the point of the invalid template instantiation with
a clear "constraint not satisfied" diagnostic.

Concrete policies
-----------------

**ScalarFMAPolicy** — portable scalar FP32 outer-product (no intrinsics):

.. literalinclude:: ../../../tools/example_codes/matrix_multiply_generic.hip
   :language: cpp
   :start-after: [Sphinx scalar fma policy start]
   :end-before: [Sphinx scalar fma policy end]

**SingleBufferTilePolicy** — single LDS buffer pair, equivalent to Step 3:

.. literalinclude:: ../../../tools/example_codes/matrix_multiply_generic.hip
   :language: cpp
   :start-after: [Sphinx single buffer policy start]
   :end-before: [Sphinx single buffer policy end]

**SoftwareDoubleBufferTilePolicy** — ping-pong LDS buffers, equivalent to Step 4:

.. literalinclude:: ../../../tools/example_codes/matrix_multiply_generic.hip
   :language: cpp
   :start-after: [Sphinx double buffer policy start]
   :end-before: [Sphinx double buffer policy end]

Generic kernel template
-----------------------

.. literalinclude:: ../../../tools/example_codes/matrix_multiply_generic.hip
   :language: cpp
   :start-after: [Sphinx gemm kernel start]
   :end-before: [Sphinx gemm kernel end]

Architecture dispatch
---------------------

A compile-time dispatch block selects the appropriate ``ComputePolicy`` based
on the target ISA.  The architecture-specific MFMA and WMMA policies are stubs
to be filled by follow-up architecture-specific sections:

.. literalinclude:: ../../../tools/example_codes/matrix_multiply_generic.hip
   :language: cpp
   :start-after: [Sphinx arch dispatch start]
   :end-before: [Sphinx arch dispatch end]

**Policy aliases used in this example:**

.. literalinclude:: ../../../tools/example_codes/matrix_multiply_generic.hip
   :language: cpp
   :start-after: [Sphinx policy aliases start]
   :end-before: [Sphinx policy aliases end]

**Compile and run:**

.. code-block:: bash

   hipcc -O3 -std=c++17 matrix_multiply_generic.hip -o mm_generic
   ./mm_generic

The program launches ``GemmKernel<SingleBufPolicy, ScalarPolicy>`` and
``GemmKernel<DoubleBufPolicy, ScalarPolicy>`` in sequence and prints effective
bandwidth and TFLOPS for each.  The two variants compile to distinct kernels
with different LDS footprints and synchronisation patterns, but identical
output.

**What to observe in rocprof-compute:**

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Counter group
     - What to look for
   * - ``SQ`` / ``TCP``
     - Confirm same DRAM traffic as Step 4 (same tile parameters)
   * - ``LDS``
     - Double-buffer LDS usage is 2× that of single-buffer
   * - ``SQ``
     - Lower ``SQ_WAIT_INST_LDS`` stalls for the double-buffer variant

Further reading
===============

* :doc:`Tiling and reuse: matrix multiplication <tiling-matrix-multiply>` —
  an accessible introduction to LDS tiling on which the early steps build.
* :ref:`rocprofv3 documentation <rocprofiler-sdk:using-rocprofv3>` — detailed
  guide to timeline and counter profiling.
* :doc:`ROCm Compute Profiler <rocprofiler-compute:index>` (``rocprof-compute``) —
  hardware-counter analysis and roofline modelling.
* AMD GPU architecture guides (ISA references) — VGPR budgets, LDS bank
  geometry, and wavefront scheduling details for each architecture family.
