.. meta::
  :description: Warp-level intrinsics for AMD ROCm GPU optimization
  :keywords: AMD, ROCm, HIP, warp, hardware intrinsics, DPP, warp-level operations

.. _warp_intrinsics:

********************************************************************************
Warp-level operations
********************************************************************************

Warp-level operations give you direct access to the hardware mechanisms that move and
aggregate data within a warp. The intrinsics in this topic work across both
CDNA (AMD Instinct) and RDNA (AMD Radeon) architectures, and cover three areas: lane operations, warp
reductions, and warp voting.

Lane operations
===============

Within a warp, lanes execute the same instruction simultaneously, but each holds
its own register values. Many algorithms require lanes to exchange or replicate
those values — to share a computed result, apply a cyclic shift, or reorganise
data before the next computation step. The intrinsics in this topic cover the
distinct ways of expressing that communication: reading from a named lane,
moving data in a cyclic pattern, applying a compile-time-fixed permutation, and
broadcasting within a fixed-size group. Each fills a different niche, and
choosing the right one affects both correctness and performance.

Broadcasting from a specific lane with ``readlane``
---------------------------------------------------

When a single lane holds a value that every other lane needs — a warp-wide
maximum, a shared configuration parameter, a count computed by one thread —
the most direct way to share it is ``__builtin_amdgcn_readlane``. Unlike a
shuffle, which requires every lane to participate with a relative offset,
``readlane`` reads from one named lane regardless of who is calling, making it
the natural choice when the source is fixed rather than relative. The example
here uses it to broadcast a warp maximum after a ``__shfl_down`` reduction,
which is a representative case: one lane holds the answer, and all others need
it.

``__builtin_amdgcn_readlane(val, lane)`` returns the value of ``val`` held by
the lane specified by ``lane``. Unlike ``readfirstlane``, which always reads
the first active lane, ``readlane`` accepts a runtime lane index. The lane
index must be uniform --- the same value across all lanes in the warp at the
point of the call. If the index is derived from per-lane data, use
``readfirstlane`` to promote it to a uniform value first. ``readfirstlane``
reads from the first active lane of the warp, which is a deterministic choice,
so every lane in the warp receives the same index value and the uniformity
requirement is satisfied.

.. literalinclude:: ../../tools/example_codes/warp_intrinsics.hip
   :language: cpp
   :linenos:
   :start-after: [Sphinx readlane start]
   :end-before: [Sphinx readlane end]

After a ``__shfl_down`` reduction, lane 0 holds the warp maximum. Passing the
literal ``0`` as the lane index satisfies the uniformity requirement and
broadcasts that value to every lane in the warp.

Warp rotation using ``mov_dpp`` and ``ds_bpermute``
---------------------------------------------------

Rotation shifts every lane's value one position around the warp, so lane
``i`` receives what lane ``i - 1`` held, and lane 0 wraps around to receive
what the last lane held. This cyclic movement is a building block for
algorithms that scan or pipeline values across lanes, such as prefix operations
or producer-consumer patterns within a warp. Data Parallel Primitives (DPP) is the hardware-native way to
express rotation on AMD GPUs, executing in a dedicated unit without consuming
the general instruction pipeline. On CDNA-based GPUs, a single ``wave_ror:1`` instruction
covers the entire wave64 atomically. On RDNA-based GPUs, where ``wave_ror`` is not
available on gfx11xx and later, the same result requires two steps: a
``row_ror:1`` rotates within each 16-lane row, and ``ds_bpermute`` supplies
the wrap-around value at the boundary of each row.

.. literalinclude:: ../../tools/example_codes/warp_intrinsics.hip
   :language: cpp
   :linenos:
   :start-after: [Sphinx warp rotate start]
   :end-before: [Sphinx warp rotate end]

``ds_bpermute`` uses byte addressing: to read from lane ``k``, pass ``k * 4``
as the index. Lane 0 of the first row reads from lane 31 (address 124), and
lane 0 of the second row reads from lane 15 (address 60). All other lanes keep
the ``row_ror`` result unchanged.

Warp rotation using ``__shfl``
------------------------------

The DPP rotation above is the preferred approach on AMD hardware, but the same
pattern can be expressed portably using ``__shfl``. This is useful when
targeting multiple GPU vendors or when the performance difference is not
significant for your workload. Each lane computes its source index as
``(lane - 1 + warpSize) % warpSize`` and because ``__shfl`` accepts any uniform
lane index, the modular arithmetic produces the wrap-around without the
architecture-specific workaround that the DPP path requires.

.. literalinclude:: ../../tools/example_codes/warp_intrinsics.hip
   :language: cpp
   :linenos:
   :start-after: [Sphinx warp rotate shfl start]
   :end-before: [Sphinx warp rotate shfl end]

The DPP and ``__shfl`` kernels produce identical output and can be verified against
the same CPU reference.

The two paths compile to different ISA on CDNA-based GPUs: the DPP path emits
``v_mov_b32_dpp wave_rol:1``, a pure register move that executes in the dedicated
DPP unit without touching memory, while the ``__shfl`` path emits
``ds_bpermute_b32``, which routes through the data-share unit. On RDNA-based
GPUs, where ``wave_ror`` is unavailable, the DPP path uses ``v_mov_b32_dpp
row_ror:1`` for the within-row lanes and ``ds_bpermute_b32`` for the cross-row
fixup; the ``__shfl`` path uses only ``ds_bpermute_b32``. The ISA output for
both can be inspected by compiling with ``-save-temps`` and examining the
generated ``.s`` file.

Lane swap using ``ds_swizzle``
------------------------------

While ``readlane`` and rotation address individual lanes or shift the whole
warp by one, some algorithms need a fixed, symmetric rearrangement of groups —
for example, swapping pairs of lanes, or exchanging two halves of a tile (a
fixed-size contiguous sub-group of lanes within a warp).
``__builtin_amdgcn_ds_swizzle`` is designed exactly for this: the permutation
is encoded entirely in a compile-time mask, so that the hardware can apply it in a
single instruction with no per-lane index computation. The mask specifies a
bitwise transformation of each lane index to its source, making it well-suited
to power-of-two group swaps that appear in butterfly networks, data
rearrangement before matrix operations, or paired-lane exchanges. For the full
signature, see ``__builtin_amdgcn_ds_swizzle`` in the :ref:`reference table
<warp_intrinsics>`.

The 16-bit mask encodes the permutation: bits [14:10] hold ``and_mask``, bits
[9:5] hold ``or_mask``, and bits [4:0] hold ``xor_mask``. The source lane is
computed as ``(lane & and_mask) | or_mask ^ xor_mask``. Setting
``and_mask = 0x1F`` and ``or_mask = 0``, and placing a single set bit in
``xor_mask``, swaps neighboring groups whose size is determined by the position
of that bit. The example demonstrates the swap pattern; the mask value and how
it determines group size are explained in the prose below.

.. literalinclude:: ../../tools/example_codes/warp_intrinsics.hip
   :language: cpp
   :linenos:
   :start-after: [Sphinx ds swizzle start]
   :end-before: [Sphinx ds swizzle end]

The mask ``0x101F`` sets ``xor_mask = 0x04``, which XORs each lane index with
4. Lanes 0--3 read from lanes 4--7 and vice versa, lanes 8--11 read from lanes
12--15, and so on across the warp. Shifting the set bit changes the group size:
``0x041F`` swaps neighboring pairs, ``0x081F`` swaps groups of two, ``0x201F``
swaps groups of eight, and ``0x401F`` swaps groups of sixteen.

Warp reductions
===============

A warp reduction combines a value held by each lane into a single scalar
result. This pattern appears constantly in GPU kernels — summing partial
products, finding a maximum across a tile, counting active threads — and the
efficiency of the reduction matters because it often sits in the critical path
of a kernel. The three implementations below all express the same butterfly
pattern, where lanes exchange values with increasingly distant partners until
one lane holds the aggregate. Still, they differ significantly in how directly they
map to hardware. The first two reduce floating-point values and show the
progression from portable shuffle-based code to hardware-native DPP. The third
uses an unsigned integer reduction to demonstrate ``wave_reduce_add_u32``, the
single-instruction form currently available in the compiler. Understanding all
three lets you choose the right level of abstraction for your target and
performance requirements.

Reduction using ``__shfl_down``
-------------------------------

``__shfl_down`` is the most readable starting point for warp reductions. It
reads from a lane a fixed offset ahead of the current lane, which maps directly
onto the butterfly pattern when called in a loop that halves the offset on each
step. The code structure closely mirrors the algorithm's logical structure,
making it straightforward to reason about and maintain. The kernel is
templated on ``WarpSize``, a compile-time constant passed at the call site,
which allows the compiler to fully unroll the reduction loop and makes the
code adapt to wave32 and wave64 without modification. This makes
``__shfl_down`` the right choice when clarity and
simplicity matter, or as a reference implementation against which a more
hardware-specific version can be verified.

.. literalinclude:: ../../tools/example_codes/warp_intrinsics.hip
   :language: cpp
   :linenos:
   :start-after: [Sphinx shfl down reduce start]
   :end-before: [Sphinx shfl down reduce end]

After the loop, lane 0 holds the sum of all lanes in the warp. The kernel uses
a two-phase structure: each warp reduces its slice independently, lane 0 writes
its partial result to shared memory, and the first warp reduces those partials
to the block result.

Reduction using ``mov_dpp``
---------------------------

When targeting supported AMD GPUs specifically, Data Parallel Primitives offer a
faster path. ``__builtin_amdgcn_mov_dpp`` moves data between lanes according
to a DPP control word, executing in a dedicated hardware unit rather than the
general instruction pipeline used by shuffle operations. The same butterfly
pattern as above can be expressed with DPP control words, and the result is a
reduction that makes better use of the hardware's data movement capabilities.
The trade-off is that DPP instructions expose architectural differences directly:
broadcast instructions are available on gfx9xx targets (CDNA), but not on
gfx10xx and later (RDNA, wave32), so the ``HAS_DPP_BROADCAST`` macro selects the
appropriate path at compile time.

.. literalinclude:: ../../tools/example_codes/warp_intrinsics.hip
   :language: cpp
   :linenos:
   :start-after: [Sphinx dpp reduce start]
   :end-before: [Sphinx dpp reduce end]

The first two steps use ``quad_perm`` control words, which exchange lanes within
groups of four. Steps three and four use ``row_ror`` rather than ``row_shr``:
rotation wraps lanes within the row boundary, whereas a shift leaves destination
registers undefined when the source is out of range. On RDNA, where DPP
broadcasts are unavailable, ``ds_swizzle`` with mask ``0x1e0`` replicates the
value held by lane 15 across lanes 16--31, combining the two rows of the wave32
warp in a single instruction. The control word constants used in the example
(``0xb1``, ``0x4e``, ``0x124``, and so on) are user-defined values derived from
the DPP control word encoding; they are not predefined by HIP or LLVM. For the
full DPP control word encoding and available patterns, refer to the Data Parallel
Primitives chapter of the
`CDNA3 ISA <https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/instruction-set-architectures/amd-instinct-mi300-cdna3-instruction-set-architecture.pdf>`_
or
`RDNA3 ISA <https://www.amd.com/content/dam/amd/en/documents/radeon-tech-docs/instruction-set-architectures/rdna3-shader-instruction-set-architecture-feb-2023_0.pdf>`_.

Reduction using ``wave_reduce_add_u32``
---------------------------------------

Both implementations above express the butterfly pattern explicitly in code,
requiring you to select control words, handle the architecture split, and manage
the reduction steps yourself. ``__builtin_amdgcn_wave_reduce_add_u32``
encapsulates all of that: you supply the value and a strategy hint, and the
intrinsic handles the reduction without you needing to implement or maintain the
underlying pattern. This represents the most hardware-specific end of the
spectrum — the implementation details are handled by the compiler and runtime
rather than by your code. Note that the wave reduce intrinsics currently cover
integer and bitwise operations; for float workloads, the DPP path above remains the recommended
approach.

.. literalinclude:: ../../tools/example_codes/warp_intrinsics.hip
   :language: cpp
   :linenos:
   :start-after: [Sphinx wave reduce start]
   :end-before: [Sphinx wave reduce end]

The second argument is a strategy hint: ``0`` lets the compiler choose, ``1``
requests the iterative strategy, and ``2`` requests the DPP-based strategy.

Warp voting
===========

Lane operations move data between lanes, and reductions aggregate it. Warp
voting answers a different question: which lanes satisfy a condition, and what
can the warp do with that information collectively? The ``ballot`` intrinsic
captures the answer as a bitmask — one bit per lane — which can then be
inspected, counted, or used to coordinate writes. The ``mbcnt`` intrinsic
builds on this by counting the number of lanes before the current one that have their bit
set, giving each qualifying lane a unique sequential index. Together, ballot and
mbcnt is the standard mechanism for stream compaction within a warp: filtering
a set of values to only those that pass a predicate, and writing them
contiguously to an output buffer without gaps or collisions, all without shared
memory or barriers.

Compaction index using ``ballot`` and ``mbcnt``
-----------------------------------------------

.. literalinclude:: ../../tools/example_codes/warp_intrinsics.hip
   :language: cpp
   :linenos:
   :start-after: [Sphinx ballot mbcnt start]
   :end-before: [Sphinx ballot mbcnt end]

``mbcnt_hi`` takes the upper 32 bits of the mask and the result of ``mbcnt_lo``
as its accumulator, so the final index is the count of passing lanes strictly
below the current lane across the full 64-bit mask. Lanes where the predicate
is false receive an index, too, but should not write to the output.

Reference
=========

Lane operations
---------------

.. list-table::
   :header-rows: 1
   :widths: 50 30 20

   * - Signature
     - Description
     - Supported architecture
   * - ``T __shfl(T val, int src_lane, int width=warpSize)``
     - Copy ``val`` from a specific lane
     - All
   * - ``T __shfl_up(T val, unsigned int offset, int width=warpSize)``
     - Copy ``val`` from a lane behind the current lane
     - All
   * - ``T __shfl_down(T val, unsigned int offset, int width=warpSize)``
     - Copy ``val`` from a lane ahead of the current lane
     - All
   * - ``T __shfl_xor(T val, int mask, int width=warpSize)``
     - Copy ``val`` from the lane at XOR of current lane index and ``mask``
     - All
   * - ``unsigned int __builtin_amdgcn_readfirstlane(unsigned int val)``
     - Read ``val`` from the first active lane
     - All
   * - ``unsigned int __builtin_amdgcn_readlane(unsigned int val, unsigned int lane)``
     - Read ``val`` from a specific lane by uniform runtime index
     - All
   * - ``unsigned int __builtin_amdgcn_writelane(unsigned int val, unsigned int lane, unsigned int old)``
     - Write ``val`` to ``lane``; all other lanes return ``old``
     - All
   * - ``int __builtin_amdgcn_mov_dpp(int val, int dpp_ctrl, int row_mask, int bank_mask, bool bound_ctrl)``
     - Move data between lanes using a DPP control word
     - All
   * - ``int __builtin_amdgcn_update_dpp(int old, int val, int dpp_ctrl, int row_mask, int bank_mask, bool bound_ctrl)``
     - DPP move with a fallback value for out-of-range lanes
     - All
   * - ``unsigned int __builtin_amdgcn_mov_dpp8(unsigned int val, unsigned int sel)``
     - DPP move using an 8-element compile-time permutation within groups of 8
     - gfx10xx, gfx11xx, gfx12xx
   * - ``int __builtin_amdgcn_ds_swizzle(int val, int mask)``
     - Apply a fixed bitmask permutation to lane indices
     - All
   * - ``int __builtin_amdgcn_ds_permute(int index, int val)``
     - Forward permutation: lane writes its value to the lane given by ``index``
     - All
   * - ``int __builtin_amdgcn_ds_bpermute(int index, int val)``
     - Backward permutation: lane reads from the lane given by ``index``
     - All
   * - ``int __builtin_amdgcn_permlane16(int old, int val, int lanesel_lo, int lanesel_hi, bool fi, bool bc)``
     - Permute within each 16-lane group using two 4-bit selectors
     - gfx10xx, gfx11xx, gfx12xx
   * - ``int __builtin_amdgcn_permlanex16(int old, int val, int lanesel_lo, int lanesel_hi, bool fi, bool bc)``
     - Same as ``permlane16`` but crosses the 16-lane boundary
     - gfx10xx, gfx11xx, gfx12xx
   * - ``int __builtin_amdgcn_permlane64(int val)``
     - Exchange data between the two 32-lane halves without a selector (wave64 only)
     - gfx11xx
   * - ``unsigned int __builtin_amdgcn_permlane16_var(unsigned int old, unsigned int val, unsigned int lanesel, bool fi, bool bc)``
     - ``permlane16`` with a runtime lane selector
     - gfx12xx
   * - ``unsigned int __builtin_amdgcn_permlanex16_var(unsigned int old, unsigned int val, unsigned int lanesel, bool fi, bool bc)``
     - ``permlanex16`` with a runtime lane selector
     - gfx12xx
   * - ``_Vector<2, unsigned int> __builtin_amdgcn_permlane16_swap(unsigned int src0, unsigned int src1, bool fi, bool bc)``
     - Swap odd and even 16-lane rows between two operands
     - gfx950
   * - ``_Vector<2, unsigned int> __builtin_amdgcn_permlane32_swap(unsigned int src0, unsigned int src1, bool fi, bool bc)``
     - Swap the upper and lower 32-lane halves between two operands
     - gfx950

Warp reductions
---------------

.. list-table::
   :header-rows: 1
   :widths: 50 30 20

   * - Signature
     - Description
     - Supported architecture
   * - ``unsigned int __builtin_amdgcn_wave_reduce_add_u32(unsigned int src, int strategy)``
     - Addition (u32)
     - All
   * - ``unsigned long __builtin_amdgcn_wave_reduce_add_u64(unsigned long src, int strategy)``
     - Addition (u64)
     - All
   * - ``unsigned int __builtin_amdgcn_wave_reduce_sub_u32(unsigned int src, int strategy)``
     - Subtraction (u32)
     - All
   * - ``unsigned long __builtin_amdgcn_wave_reduce_sub_u64(unsigned long src, int strategy)``
     - Subtraction (u64)
     - All
   * - ``int __builtin_amdgcn_wave_reduce_min_i32(int src, int strategy)``
     - Signed minimum (i32)
     - All
   * - ``unsigned int __builtin_amdgcn_wave_reduce_min_u32(unsigned int src, int strategy)``
     - Unsigned minimum (u32)
     - All
   * - ``long __builtin_amdgcn_wave_reduce_min_i64(long src, int strategy)``
     - Signed minimum (i64)
     - All
   * - ``unsigned long __builtin_amdgcn_wave_reduce_min_u64(unsigned long src, int strategy)``
     - Unsigned minimum (u64)
     - All
   * - ``int __builtin_amdgcn_wave_reduce_max_i32(int src, int strategy)``
     - Signed maximum (i32)
     - All
   * - ``unsigned int __builtin_amdgcn_wave_reduce_max_u32(unsigned int src, int strategy)``
     - Unsigned maximum (u32)
     - All
   * - ``long __builtin_amdgcn_wave_reduce_max_i64(long src, int strategy)``
     - Signed maximum (i64)
     - All
   * - ``unsigned long __builtin_amdgcn_wave_reduce_max_u64(unsigned long src, int strategy)``
     - Unsigned maximum (u64)
     - All
   * - ``int __builtin_amdgcn_wave_reduce_and_b32(int src, int strategy)``
     - Bitwise AND (b32)
     - All
   * - ``int __builtin_amdgcn_wave_reduce_and_b64(int src, int strategy)``
     - Bitwise AND (b64)
     - All
   * - ``int __builtin_amdgcn_wave_reduce_or_b32(int src, int strategy)``
     - Bitwise OR (b32)
     - All
   * - ``int __builtin_amdgcn_wave_reduce_or_b64(int src, int strategy)``
     - Bitwise OR (b64)
     - All
   * - ``int __builtin_amdgcn_wave_reduce_xor_b32(int src, int strategy)``
     - Bitwise XOR (b32)
     - All
   * - ``int __builtin_amdgcn_wave_reduce_xor_b64(int src, int strategy)``
     - Bitwise XOR (b64)
     - All

Warp voting
-----------

.. list-table::
   :header-rows: 1
   :widths: 50 30 20

   * - Signature
     - Description
     - Supported architecture
   * - ``unsigned long long __builtin_amdgcn_ballot_w64(bool pred)``
     - 64-bit mask of active lanes where ``pred`` is true
     - Supported when warp size is 64
   * - ``unsigned int __builtin_amdgcn_ballot_w32(bool pred)``
     - 32-bit mask of active lanes where ``pred`` is true
     - Supported when warp size is 32
   * - ``bool __builtin_amdgcn_inverse_ballot_w64(uint64_t mask)``
     - True if the current lane's bit is set in ``mask``
     - Supported when warp size is 64
   * - ``bool __builtin_amdgcn_inverse_ballot_w32(uint32_t mask)``
     - True if the current lane's bit is set in ``mask``
     - Supported when warp size is 32
   * - ``unsigned int __builtin_amdgcn_mbcnt_lo(unsigned int mask, unsigned int base)``
     - Count set bits in lower 32-bit half of ballot mask below current lane
     - All
   * - ``unsigned int __builtin_amdgcn_mbcnt_hi(unsigned int mask, unsigned int base)``
     - Count set bits in upper 32-bit half of ballot mask below current lane
     - All
   * - ``uint64_t __builtin_amdgcn_uicmp(unsigned int src0, unsigned int src1, int cond)`` *(deprecated)*
     - Compare two unsigned integers and return a ballot mask; use ``ballot_w64`` instead
     - All
   * - ``uint64_t __builtin_amdgcn_uicmpl(uint64_t src0, uint64_t src1, int cond)`` *(deprecated)*
     - Compare two 64-bit unsigned integers and return a ballot mask; use ``ballot_w64`` instead
     - All
   * - ``uint64_t __builtin_amdgcn_sicmp(int src0, int src1, int cond)`` *(deprecated)*
     - Compare two signed integers and return a ballot mask; use ``ballot_w64`` instead
     - All
   * - ``uint64_t __builtin_amdgcn_sicmpl(int64_t src0, int64_t src1, int cond)`` *(deprecated)*
     - Compare two 64-bit signed integers and return a ballot mask; use ``ballot_w64`` instead
     - All
   * - ``uint64_t __builtin_amdgcn_fcmp(double src0, double src1, int cond)`` *(deprecated)*
     - Compare two doubles and return a ballot mask; use ``ballot_w64`` instead
     - All
   * - ``uint64_t __builtin_amdgcn_fcmpf(float src0, float src1, int cond)`` *(deprecated)*
     - Compare two floats and return a ballot mask; use ``ballot_w64`` instead
     - All
   * - ``void __builtin_amdgcn_wave_barrier()``
     - Synchronize all lanes within the warp
     - All
   * - ``unsigned int __builtin_amdgcn_wave_id()``
     - Return the warp's index within the workgroup
     - All
