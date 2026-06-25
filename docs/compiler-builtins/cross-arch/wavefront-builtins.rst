.. meta::
   :description: Explore wavefront-level HIP builtins for AMD GPUs, including
      shuffle operations, cross-row permutations, wavefront reductions, and voting with CDNA and RDNA support.
   :keywords: AMD, ROCm, HIP, wavefront, Compiler builtins, DPP, wavefront-level operations

.. _wavefront_builtins:

********************************************************************************
Wavefront-level builtins for AMD GPUs
********************************************************************************

Wavefront-level operations give you direct access to the hardware mechanisms that move and
aggregate data within a wavefront. The builtins in this topic work across both
CDNA (AMD Instinct) and RDNA (AMD Radeon) architectures, and cover three areas: lane operations, wavefront
reductions, and wavefront voting.

Architecture availability
=========================

Most wavefront-level builtins are available on all AMD Instinct (CDNA) and AMD
Radeon (RDNA) architectures.  Some builtins --- particularly Data Parallel Primitives (DPP) DPP8,
cross-row permutations, and ballot width variants --- are limited to specific
generations.  For per-builtin availability, see the architecture tables in
the :ref:`reference pages <wavefront_builtin_reference>` below.

The complete source file is available for download:

* :download:`wavefront_builtins.hip <../../tools/example_codes/wavefront_builtins.hip>`

Lane operations
===============

Within a wavefront, lanes execute the same instruction simultaneously, but each holds
its own register values. Many algorithms require lanes to exchange or replicate
those values — to share a computed result, apply a cyclic shift, or reorganize
data before the next computation step. The builtins in this topic cover the
distinct ways of expressing that communication: reading from a named lane,
moving data in a cyclic pattern, applying a compile-time-fixed permutation, and
broadcasting within a fixed-size group. Each fills a different niche, and
choosing the right one affects both correctness and performance.

Broadcasting from a specific lane with ``readlane``
---------------------------------------------------

When a single lane holds a value that every other lane needs — a wavefront-wide
maximum, a shared configuration parameter, a count computed by one thread —
the most direct way to share it is ``__builtin_amdgcn_readlane``. Unlike a
shuffle, which requires every lane to participate with a relative offset,
``readlane`` reads from one named lane regardless of who is calling, making it
the natural choice when the source is fixed rather than relative. The example
here uses it to broadcast a wavefront maximum after a ``__shfl_down`` reduction,
which is a representative case: one lane holds the answer, and all others need
it.

``__builtin_amdgcn_readlane(val, lane)`` returns the value of ``val`` held by
the lane specified by ``lane``. Unlike ``readfirstlane``, which always reads
the first active lane, ``readlane`` accepts a runtime lane index. The lane
index must be uniform --- the same value across all lanes in the wavefront at the
point of the call. If the index is derived from per-lane data, use
``readfirstlane`` to promote it to a uniform value first. ``readfirstlane``
reads from the first active lane of the wavefront, which is a deterministic choice,
so every lane in the wavefront receives the same index value and the uniformity
requirement is satisfied.

.. literalinclude:: ../../tools/example_codes/wavefront_builtins.hip
   :language: cpp
   :linenos:
   :start-after: [Sphinx readlane start]
   :end-before: [Sphinx readlane end]

After a ``__shfl_down`` reduction, lane 0 holds the wavefront maximum. Passing the
literal ``0`` as the lane index satisfies the uniformity requirement and
broadcasts that value to every lane in the wavefront.

For the full signatures and parameter details, see
:ref:`readlane <shuffle-readlane>` and
:ref:`readfirstlane <shuffle-readfirstlane>` in the shuffle and lane access
reference.

Wavefront rotation using ``mov_dpp`` and ``ds_bpermute``
--------------------------------------------------------

Rotation shifts every lane's value one position around the wavefront, so lane
``i`` receives what lane ``i - 1`` held, and lane 0 wraps around to receive
what the last lane held. This cyclic movement is a building block for
algorithms that scan or pipeline values across lanes, such as prefix operations
or producer-consumer patterns within a wavefront. Data Parallel Primitives (DPP) is the hardware-native way to
express rotation on AMD GPUs, executing in a dedicated unit without consuming
the general instruction pipeline. On CDNA-based GPUs, a single ``wave_ror:1`` instruction
covers the entire wave64 atomically. On RDNA-based GPUs, where ``wave_ror`` is not
available on gfx11xx and later, the same result requires two steps: a
``row_ror:1`` rotates within each 16-lane row, and ``ds_bpermute`` supplies
the wrap-around value at the boundary of each row.

.. literalinclude:: ../../tools/example_codes/wavefront_builtins.hip
   :language: cpp
   :linenos:
   :start-after: [Sphinx wave rotate start]
   :end-before: [Sphinx wave rotate end]

``ds_bpermute`` uses byte addressing: to read from lane ``k``, pass ``k * 4``
as the index. Lane 0 of the first row reads from lane 31 (address 124), and
lane 0 of the second row reads from lane 15 (address 60). All other lanes keep
the ``row_ror`` result unchanged.

For the full signatures and parameter details, see
:ref:`mov_dpp <dpp-mov-dpp>` and :ref:`ds_bpermute <dpp-ds-bpermute>` in the
DPP and data-share permutation reference.

Wavefront rotation using ``__shfl``
------------------------------------

The DPP rotation above is the preferred approach on AMD hardware, but the same
pattern can be expressed using ``__shfl``. This is useful when the performance
difference is not significant for your workload. Each lane computes its source index as
``(lane - 1 + warpSize) % warpSize`` and because ``__shfl`` accepts any uniform
lane index, the modular arithmetic produces the wrap-around without the
architecture-specific workaround that the DPP path requires.

.. literalinclude:: ../../tools/example_codes/wavefront_builtins.hip
   :language: cpp
   :linenos:
   :start-after: [Sphinx wave rotate shfl start]
   :end-before: [Sphinx wave rotate shfl end]

The DPP and ``__shfl`` kernels produce identical output and can be verified against
the same CPU reference.

The two paths compile to different Instruction Set Architecture (ISA) on CDNA-based GPUs: the DPP path emits
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
wavefront by one, some algorithms need a fixed, symmetric rearrangement of groups —
for example, swapping pairs of lanes, or exchanging two halves of a tile (a
fixed-size contiguous sub-group of lanes within a wavefront).
``__builtin_amdgcn_ds_swizzle`` is designed exactly for this: the permutation
is encoded entirely in a compile-time mask, so that the hardware can apply it in a
single instruction with no per-lane index computation. The mask specifies a
bitwise transformation of each lane index to its source, making it well-suited
to power-of-two group swaps that appear in butterfly networks, data
rearrangement before matrix operations, or paired-lane exchanges. For the full
signature, see :ref:`ds_swizzle <dpp-ds-swizzle>` in the DPP and data-share
permutation reference.

The 16-bit mask encodes the permutation: bits [14:10] hold ``and_mask``, bits
[9:5] hold ``or_mask``, and bits [4:0] hold ``xor_mask``. The source lane is
computed as ``(lane & and_mask) | or_mask ^ xor_mask``. Setting
``and_mask = 0x1F`` and ``or_mask = 0``, and placing a single set bit in
``xor_mask``, swaps neighboring groups whose size is determined by the position
of that bit. The example demonstrates the swap pattern; the mask value and how
it determines group size are explained in the prose below.

.. literalinclude:: ../../tools/example_codes/wavefront_builtins.hip
   :language: cpp
   :linenos:
   :start-after: [Sphinx ds swizzle start]
   :end-before: [Sphinx ds swizzle end]

The mask ``0x101F`` sets ``xor_mask = 0x04``, which applies a bitwise XOR between each lane index and
4. Lanes 0--3 read from lanes 4--7 and vice versa, lanes 8--11 read from lanes
12--15, and so on across the wavefront. Shifting the set bit changes the group size:
``0x041F`` swaps neighboring pairs, ``0x081F`` swaps groups of two, ``0x201F``
swaps groups of eight, and ``0x401F`` swaps groups of sixteen.

For the full signature and parameter details, see
:ref:`ds_swizzle <dpp-ds-swizzle>` in the DPP and data-share permutation
reference.

Wavefront reductions
====================

A wavefront reduction combines a value held by each lane into a single scalar
result. This pattern appears constantly in GPU kernels — summing partial
products, finding a maximum across a tile, counting active threads — and the
efficiency of the reduction matters because it often sits in the critical path
of a kernel. The three implementations below all express the same butterfly
pattern, where lanes exchange values with increasingly distant partners until
one lane holds the aggregate. Still, they differ significantly in how directly they
map to hardware. The first two reduce floating-point values and show the
progression from shuffle-based code to hardware-native DPP. The third
uses an unsigned integer reduction to demonstrate ``wave_reduce_add_u32``, the
single-instruction form currently available in the compiler. Understanding all
three lets you choose the right level of abstraction for your target and
performance requirements.

Reduction using ``__shfl_down``
-------------------------------

``__shfl_down`` is the most readable starting point for wavefront reductions. It
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

.. literalinclude:: ../../tools/example_codes/wavefront_builtins.hip
   :language: cpp
   :linenos:
   :start-after: [Sphinx shfl down reduce start]
   :end-before: [Sphinx shfl down reduce end]

After the loop, lane 0 holds the sum of all lanes in the wavefront. The kernel uses
a two-phase structure: each wavefront reduces its slice independently, lane 0 writes
its partial result to shared memory, and the first wavefront reduces those partials
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

.. literalinclude:: ../../tools/example_codes/wavefront_builtins.hip
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
wavefront in a single instruction. The control word constants used in the example
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
builtin handles the reduction without you needing to implement or maintain the
underlying pattern. This represents the most hardware-specific end of the
spectrum — the implementation details are handled by the compiler and runtime
rather than by your code. Note that the wave reduce builtins currently cover
integer and bitwise operations; for float workloads, the DPP path above remains the recommended
approach.

.. literalinclude:: ../../tools/example_codes/wavefront_builtins.hip
   :language: cpp
   :linenos:
   :start-after: [Sphinx wave reduce start]
   :end-before: [Sphinx wave reduce end]

The second argument is a strategy hint: ``0`` lets the compiler choose, ``1``
requests the iterative strategy, and ``2`` requests the DPP-based strategy.

For the full signature and parameter details, see
:ref:`wave_reduce_add_u32 <wave-reduce-add-u32>` in the wavefront reduction
reference.

Wavefront voting
================

Lane operations move data between lanes, and reductions aggregate it. Wavefront
voting answers a different question: which lanes satisfy a condition, and what
can the wavefront do with that information collectively? The ``ballot`` builtin
captures the answer as a bitmask — one bit per lane — which can then be
inspected, counted, or used to coordinate writes. The ``mbcnt`` builtin
builds on this by counting the number of lanes before the current one that have their bit
set, giving each qualifying lane a unique sequential index. Together, ballot and
mbcnt is the standard mechanism for stream compaction within a wavefront: filtering
a set of values to only those that pass a predicate, and writing them
contiguously to an output buffer without gaps or collisions, all without shared
memory or barriers.

Compaction index using ``ballot`` and ``mbcnt``
-----------------------------------------------

.. literalinclude:: ../../tools/example_codes/wavefront_builtins.hip
   :language: cpp
   :linenos:
   :start-after: [Sphinx ballot mbcnt start]
   :end-before: [Sphinx ballot mbcnt end]

``mbcnt_hi`` takes the upper 32 bits of the mask and the result of ``mbcnt_lo``
as its accumulator, so the final index is the count of passing lanes strictly
below the current lane across the full 64-bit mask. Lanes where the predicate
is false receive an index, too, but should not write to the output.

For the full signatures and parameter details, see
:ref:`ballot_w64 <vote-ballot-w64>` and
:ref:`mbcnt_lo <vote-mbcnt-lo>` / :ref:`mbcnt_hi <vote-mbcnt-hi>` in the
wavefront voting reference.

**Compile and run:**

.. code-block:: bash

   amdclang++ -O3 -std=c++17 --offload-arch=gfx942 \
       wavefront_builtins.hip -o wavefront_builtins
   ./wavefront_builtins

.. note::

   The example above targets CDNA3 (``gfx942``, wave64).  Replace
   ``--offload-arch`` with the appropriate target for your GPU --- for example,
   ``gfx90a`` for CDNA2 or ``gfx1200`` for RDNA4 (wave32).  The ``__shfl*`` builtins work on all architectures; the DPP-specific code
   paths are guarded by architecture macros in the example file.

Naming convention
=================

Wavefront-level builtins come in two families:

**HIP wrappers** use the ``__shfl`` prefix:

.. code-block:: text

   __shfl[_up|_down|_xor](val, offset_or_lane, width)

The suffix indicates the direction or mode: ``_up`` reads from a lower lane,
``_down`` from a higher lane, ``_xor`` reads from a lane whose index is the bitwise XOR of
the current lane index and a mask, and no suffix reads from an absolute lane
index.

**Compiler builtins** use the ``__builtin_amdgcn_`` prefix and map
directly to ISA instructions:

.. code-block:: text

   __builtin_amdgcn_<operation>(args...)

Key operation names:

* ``readlane`` / ``readfirstlane`` / ``writelane`` -- named lane access
* ``mov_dpp`` / ``update_dpp`` / ``mov_dpp8`` -- Data Parallel Primitives
* ``ds_swizzle`` / ``ds_permute`` / ``ds_bpermute`` -- data-share permutations
* ``permlane16`` / ``permlanex16`` / ``permlane64`` -- cross-row permutations
* ``ballot_w64`` / ``ballot_w32`` -- wavefront voting
* ``mbcnt_lo`` / ``mbcnt_hi`` -- masked bit count (compaction index)
* ``wave_reduce_<op>_<type>`` -- single-instruction wavefront reductions, where
  ``<op>`` is ``add``, ``sub``, ``min``, ``max``, ``and``, ``or``, or ``xor``,
  and ``<type>`` is ``u32``, ``u64``, ``i32``, ``i64``, ``b32``, or ``b64``

Use the ``__shfl*`` family for standard shuffle operations.
Use the ``__builtin_amdgcn_*`` intrinsics when you need a specific hardware
feature (DPP, ``ds_swizzle``, wavefront voting) or when the ISA instruction gives
measurable performance benefit.

.. _wavefront_builtin_reference:

Wavefront builtin reference
============================

Each reference page documents the full signature, parameter details, and
architecture support for every builtin in that family.

.. list-table::
   :header-rows: 1
   :widths: auto

   * - Family
     - Description
   * - :doc:`Shuffle and lane access <wavefront-ref/shuffle-builtins>`
     - ``__shfl*`` wrappers and hardware ``readlane``,
       ``readfirstlane``, ``writelane``
   * - :doc:`DPP and data-share permutations <wavefront-ref/dpp-builtins>`
     - ``mov_dpp``, ``update_dpp``, ``mov_dpp8``, ``ds_swizzle``,
       ``ds_permute``, ``ds_bpermute``
   * - :doc:`Cross-row permutations <wavefront-ref/permlane-builtins>`
     - ``permlane16``, ``permlanex16``, ``permlane64``, and runtime/swap
       variants
   * - :doc:`Wavefront reductions <wavefront-ref/wave-reduce-builtins>`
     - ``wave_reduce_<op>_<type>`` for add, sub, min, max, and, or, xor
   * - :doc:`Wavefront voting and synchronization <wavefront-ref/vote-builtins>`
     - ``ballot``, ``inverse_ballot``, ``mbcnt``, ``wave_barrier``,
       ``wave_id``
