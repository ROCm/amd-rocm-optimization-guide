.. meta::
   :description: Reference for sparse matrix fused multiply-accumulate (SMFMAC) intrinsics on AMD Instinct accelerators, covering 4:2 structured sparsity with FP16, BF16, INT8, and FP8 data types on CDNA3 and CDNA4 GPUs.
   :keywords: AMD, ROCm, HIP, intrinsics, sparse, MFMA, SMFMAC, structured sparsity, matrix multiply, CDNA3, CDNA4, gfx942, gfx950, MI300, MI350

.. _sparse-mfma-intrinsics:

********************************************************************************
Sparse MFMA intrinsics
********************************************************************************

Sparse Matrix Fused Multiply-Accumulate (SMFMAC) intrinsics let you issue
hardware matrix multiply-accumulate operations that exploit 4:2 structured
sparsity directly from HIP device code on CDNA3 GPUs (``gfx942``,
MI300 series) and CDNA4 GPUs (``gfx950``, MI350 series).  Each SMFMAC
instruction multiplies a compressed
:math:`\pmb{A}` fragment by a dense :math:`\pmb{B}` fragment and accumulates
the result into a :math:`\pmb{D}` fragment, all within a single wavefront of
64 lanes.  Because the :math:`\pmb{A}` operand is stored in compressed form,
these intrinsics halve the storage and memory bandwidth required for
:math:`\pmb{A}` compared to their dense MFMA counterparts, while the hardware
uses a sparsity index to reconstruct the original element positions during
the multiply.

CDNA3 supports SMFMAC intrinsics with FP16, BF16, and INT8 inputs, plus
FP8 (E4M3) and BF8 (E5M2) variants in all four A×B type combinations.
CDNA4 adds doubled-K variants in all data types.

Architecture availability
=========================

The FP16, BF16, and INT8 SMFMAC variants at the base K depths are available
on ``gfx942``.  The FP8 and BF8 variants require ``gfx942``or later.
Doubled-K variants in all data types are available on ``gfx950``
only; these are noted in the individual intrinsic reference entries.

Naming convention
=================

All SMFMAC intrinsics follow the pattern:

.. code-block:: text

   __builtin_amdgcn_smfmac_<out_type>_<M>x<N>x<K>_<in_type_a>[_<in_type_b>]

``out_type``
    Accumulator element type (``f32`` or ``i32``).

``M``, ``N``, ``K``
    Tile dimensions in elements.  The instruction computes the contribution
    of a K-wide compressed panel of :math:`\pmb{A}` and a K-wide dense panel
    of :math:`\pmb{B}` to an :math:`M \times N` output tile.  Each
    instruction processes one K step; the caller loops over K to accumulate
    a full matrix product.

``in_type_a``
    Input element type of the :math:`\pmb{A}` matrix (``f16``, ``bf16``,
    ``i8``, ``fp8``, or ``bf8``).

``in_type_b`` (FP8/BF8 variants only)
    Input element type of the :math:`\pmb{B}` matrix.  Always present for
    FP8 and BF8 variants to disambiguate the four possible A×B type
    combinations (``fp8_fp8``, ``fp8_bf8``, ``bf8_fp8``, ``bf8_bf8``).
    Omitted for FP16, BF16, and INT8 variants where :math:`\pmb{A}` and
    :math:`\pmb{B}` always share the same type.

For example:

* ``__builtin_amdgcn_smfmac_f32_16x16x32_f16`` -- a :math:`16 \times 16`
  sparse MMA with :math:`K=32` that multiplies FP16 :math:`\pmb{A}` by
  FP16 :math:`\pmb{B}` and accumulates into FP32.
* ``__builtin_amdgcn_smfmac_f32_16x16x64_fp8_bf8`` -- a :math:`16 \times 16`
  sparse MMA with :math:`K=64` that multiplies FP8 :math:`\pmb{A}` by
  BF8 :math:`\pmb{B}` and accumulates into FP32.

Structured sparsity (4:2 pattern)
=================================

The SMFMAC instructions require the :math:`\pmb{A}` matrix to obey 4:2
structured sparsity: in every contiguous group of four elements along the K
dimension, exactly two are non-zero and the other two are zero.  This
constraint allows the :math:`\pmb{A}` operand to be stored in a compressed
representation that contains only the non-zero values, reducing the storage
to half the original K dimension.

Along with the compressed non-zero values, the hardware requires a sparsity
index that encodes which two of the four positions in each group hold the
non-zeros.  This index is passed as the ``idx`` argument to every SMFMAC
intrinsic.  The hardware uses it at execution time to align the compressed
:math:`\pmb{A}` elements against the correct rows of the :math:`\pmb{B}`
operand before accumulating the products.

Host-side compression is straightforward: walk the K dimension in groups of
four, extract the two non-zero positions, and pack them consecutively into
the output buffer.  The example kernels in this topic adopt a fixed pattern
that keeps positions 0 and 2 of every group, producing a compressed buffer
of half the original K length.

.. _smfmac-accumulator-layout:

Accumulator layout
==================

Every SMFMAC instruction computes a single independent :math:`M \times N`
output tile (block count = 1).  The accumulator (:math:`\pmb{C}` /
:math:`\pmb{D}`) layout across wavefront lanes and VGPRs is identical to
the 1-block layout of the corresponding dense MFMA tile shape.

.. note::

   On CDNA3, all four operands (:math:`\pmb{A}`, :math:`\pmb{B}`,
   :math:`\pmb{C}`, and :math:`\pmb{D}`) can reside in either accumulation
   VGPRs (accVGPRs) or standard architecture VGPRs (ArchVGPRs).  In HIP
   device code the compiler selects the appropriate register class
   automatically.  The layout tables and formulas below use *VGPR index* as
   a logical position label; the actual register class does not affect the
   layout.

The formulas in the subsections below use the following notation:

* :math:`i` -- zero-based row index within the tile, :math:`0 \le i < M`
* :math:`j` -- zero-based column index within the tile, :math:`0 \le j < N`
* **lane** -- wavefront lane that holds the element,
  :math:`0 \le \text{lane} < 64`
* **VGPR** -- zero-based index into that lane's accumulator register file

:math:`16 \times 16` layout
----------------------------

The :math:`16 \times 16` output tile occupies 4 VGPRs per lane (``v4float``
or ``v4int``).  The following diagrams show the VGPR index for each output
element; the two diagrams correspond to the two K depths available for the
:math:`16 \times 16` tile shape.

.. figure:: ../../data/hardware-intrinsics/cdna/sparse-mfma-intrinsics/smfmac-layout-16x16x32.svg
   :alt: :math:`16 \times 16`, K=32 SMFMAC accumulator layout -- VGPR index
         per output element, with lane groups colour-coded.
   :align: center
   :width: 70%

   :math:`16 \times 16`, **K=32 accumulator layout.**  Each cell shows the
   VGPR index that holds output element :math:`(i, j)`.  Rows 0---3 (teal,
   lanes 0---15), rows 4---7 (grey, lanes 16---31), rows 8---11 (teal,
   lanes 32---47), rows 12---15 (grey, lanes 48---63).  Column :math:`j`
   gives the lane offset within the group.

.. figure:: ../../data/hardware-intrinsics/cdna/sparse-mfma-intrinsics/smfmac-layout-16x16x64.svg
   :alt: :math:`16 \times 16`, K=64 SMFMAC accumulator layout -- VGPR index
         per output element, with lane groups colour-coded.
   :align: center
   :width: 70%

   :math:`16 \times 16`, **K=64 accumulator layout.**  The output layout is
   identical to K=32; only the :math:`\pmb{A}` and :math:`\pmb{B}` input
   fragment sizes differ.

Given output element :math:`(i, j)`:

.. math::

   \text{lane}    &= 16 \lfloor \frac{i}{4} \rfloor + j \\
   \text{VGPR}    &= i \bmod 4

Conversely, given lane :math:`L` and VGPR index :math:`G`:

.. math::

   i &= 4 \lfloor \frac{L}{16} \rfloor + (G \bmod 4) \\
   j &= L \bmod 16

The row-to-lane mapping:

.. list-table::
   :header-rows: 1
   :widths: auto

   * - Rows
     - Lanes
     - VGPRs
   * - 0---3
     - 0---15
     - 0---3
   * - 4---7
     - 16---31
     - 0---3
   * - 8---11
     - 32---47
     - 0---3
   * - 12---15
     - 48---63
     - 0---3

:math:`32 \times 32` layout
----------------------------

The :math:`32 \times 32` output tile occupies 16 VGPRs per lane
(``v16float`` or ``v16int``).  The following diagrams show the VGPR index
for each output element; the two diagrams correspond to the two K depths
available for the :math:`32 \times 32` tile shape.

.. figure:: ../../data/hardware-intrinsics/cdna/sparse-mfma-intrinsics/smfmac-layout-32x32x16.svg
   :alt: :math:`32 \times 32`, K=16 SMFMAC accumulator layout -- VGPR index
         per output element, with lane groups colour-coded.
   :align: center
   :width: 100%

   :math:`32 \times 32`, **K=16 accumulator layout.**  Each cell shows the
   VGPR index that holds output element :math:`(i, j)`.  Teal cells (rows
   where :math:`\lfloor i/4 \rfloor` is even) belong to lanes 0---31; grey
   cells to lanes 32---63.  Column :math:`j` gives the lane offset within
   the group.

.. figure:: ../../data/hardware-intrinsics/cdna/sparse-mfma-intrinsics/smfmac-layout-32x32x32.svg
   :alt: :math:`32 \times 32`, K=32 SMFMAC accumulator layout -- VGPR index
         per output element, with lane groups colour-coded.
   :align: center
   :width: 100%

   :math:`32 \times 32`, **K=32 accumulator layout.**  The output layout is
   identical to K=16; only the :math:`\pmb{A}` and :math:`\pmb{B}` input
   fragment sizes differ.

Given output element :math:`(i, j)`:

.. math::

   \text{lane}   &= \bigl(32 \cdot \lfloor \frac{i}{4} \rfloor\bigr) \bmod 64 + j \\
   \text{VGPR}   &= 4 \lfloor \frac{i}{8} \rfloor + (i \bmod 4)

Conversely, given lane :math:`L` and VGPR index :math:`G`:

.. math::

   i &= \bigl(8 \cdot \lfloor \frac{G}{4} \rfloor\bigr) \bmod 32
       + 4 \lfloor \frac{L}{32} \rfloor + (G \bmod 4) \\
   j &= L \bmod 32

The row-to-lane mapping:

.. list-table::
   :header-rows: 1
   :widths: auto

   * - Rows
     - Lanes
     - VGPRs
   * - 0---3
     - 0---31
     - 0---3
   * - 4---7
     - 32---63
     - 0---3
   * - 8---11
     - 0---31
     - 4---7
   * - 12---15
     - 32---63
     - 4---7
   * - 16---19
     - 0---31
     - 8---11
   * - 20---23
     - 32---63
     - 8---11
   * - 24---27
     - 0---31
     - 12---15
   * - 28---31
     - 32---63
     - 12---15

Register types used in this reference
=====================================

The signatures below use the following type aliases, which you can declare
with C++ attributes in any HIP translation unit:

.. code-block:: cpp

   using v4float  = float    [[clang::ext_vector_type(4)]];
   using v16float = float    [[clang::ext_vector_type(16)]];
   using v4half   = _Float16 [[clang::ext_vector_type(4)]];
   using v8half   = _Float16 [[clang::ext_vector_type(8)]];
   using v16half  = _Float16 [[clang::ext_vector_type(16)]];
   using v4short  = short    [[clang::ext_vector_type(4)]];   // BF16 storage
   using v8short  = short    [[clang::ext_vector_type(8)]];   // BF16 storage
   using v8bf16   = __bf16   [[clang::ext_vector_type(8)]];   // gfx950 BF16
   using v16bf16  = __bf16   [[clang::ext_vector_type(16)]];  // gfx950 BF16
   using v2int    = int      [[clang::ext_vector_type(2)]];
   using v4int    = int      [[clang::ext_vector_type(4)]];
   using v8int    = int      [[clang::ext_vector_type(8)]];
   using v16int   = int      [[clang::ext_vector_type(16)]];

Each type alias maps one-to-one to the corresponding LLVM vector type used
in the intrinsic definition.  The number in the name is the element count
per lane; the total VGPR count equals the element count multiplied by the
element size in 32-bit words.

.. _smfmac-common-parameters:

Common parameters
=================

Every SMFMAC intrinsic on this page shares the same trailing three
parameters.  ``cbsz`` and ``abid`` must be compile-time integer constants;
``idx`` is a runtime value.

.. list-table::
   :header-rows: 1
   :widths: auto

   * - Parameter
     - Type
     - Description
   * - ``idx``
     - int
     - Sparsity index.  Encodes which two of the four positions in each
       group of four elements along the K dimension hold the non-zeros in
       the compressed :math:`\pmb{A}` matrix.  The bit width of the index
       scales with K: a 2-bit field per group of four is needed, so K=16
       requires 8 bits, K=32 requires 16 bits, and K=64 requires 32 bits.
       This is a runtime value, not a compile-time constant.
   * - ``cbsz``
     - int
     - Control Broadcast Size modifier.  Controls broadcasting of
       :math:`\pmb{A}` data across lane groups.  Must be a compile-time
       constant.  Setting ``cbsz`` to ``0`` disables broadcasting (the
       default).
   * - ``abid``
     - int
     - :math:`\pmb{A}`-matrix Broadcast Identifier.  Used together with
       ``cbsz``; selects which lane group's :math:`\pmb{A}` data is
       broadcast.  Must be a compile-time constant.

.. note::

   SMFMAC intrinsics do not support the ``blgp`` (:math:`\pmb{B}`-matrix
   Lane Group Pattern) modifier available on dense MFMA instructions.

Example kernels
===============

The following example kernels demonstrate SMFMAC intrinsics in the context
of a tiled matrix multiplication.  Each kernel loads tiles of the compressed
:math:`\pmb{A}` matrix and the dense :math:`\pmb{B}` matrix into LDS,
then calls the SMFMAC intrinsic to replace the inner-product loop of a
conventional scalar kernel.

Dense LDS baseline
------------------

Before examining the sparse intrinsics, it is useful to establish a dense
tiled matrix multiplication as a baseline.  The following kernel tiles a
FP16 matrix multiply through LDS with a 32×32 CTA tile and a 32-element
K tile, accumulating into FP32.  Each warp computes a 16×16 sub-tile.

.. literalinclude:: ../../tools/example_codes/intrinsics_sparse_mfma.hip
   :language: cpp
   :linenos:
   :start-after: [Sphinx lds_baseline start]
   :end-before: [Sphinx lds_baseline end]

The inner loop loads :math:`\pmb{A}` and :math:`\pmb{B}` tiles into LDS,
synchronizes, then accumulates the products in scalar FP32 accumulators.
The sparse SMFMAC kernels below replace this scalar inner loop with a
single intrinsic call per K tile.

FP16 16×16 sparse kernel
-------------------------

This kernel uses ``__builtin_amdgcn_smfmac_f32_16x16x32_f16`` to compute
a :math:`16 \times 16` sparse matrix multiply-accumulate with :math:`K=32`
per instruction.  The compressed :math:`\pmb{A}` operand is loaded as a
``v4half`` vector, the dense :math:`\pmb{B}` as a ``v8half`` vector, and
the accumulator is a ``v4float`` vector.

.. literalinclude:: ../../tools/example_codes/intrinsics_sparse_mfma.hip
   :language: cpp
   :linenos:
   :start-after: [Sphinx smfmac_f32_16x16x32_f16 start]
   :end-before: [Sphinx smfmac_f32_16x16x32_f16 end]

Each lane loads its :math:`\pmb{A}` fragment from the compressed LDS tile,
where the K dimension is halved.  The :math:`\pmb{B}` fragment is loaded
from the full dense :math:`\pmb{B}` tile in LDS.  The intrinsic call
replaces the entire inner-product loop of the baseline kernel with a single
instruction.  The 4-element FP32 result vector maps to four rows of the
:math:`16 \times 16` output sub-tile, with the lane index selecting the
column.

FP16 32×32 sparse kernel
-------------------------

This kernel uses ``__builtin_amdgcn_smfmac_f32_32x32x16_f16`` to compute
a :math:`32 \times 32` sparse matrix multiply-accumulate with :math:`K=16`
per instruction.  The CTA tile grows to 64×64 to accommodate the larger
32×32 warp tiles.  The accumulator is a 16-element FP32 vector.

.. literalinclude:: ../../tools/example_codes/intrinsics_sparse_mfma.hip
   :language: cpp
   :linenos:
   :start-after: [Sphinx smfmac_f32_32x32x16_f16 start]
   :end-before: [Sphinx smfmac_f32_32x32x16_f16 end]

The lane-to-element mapping for the 16-element accumulator differs from
the :math:`16 \times 16` variant: each lane's 16 result elements are
distributed across four groups of four consecutive rows, with the group
stride determined by the lane's position within the wavefront.

FP8 16×16 sparse kernel
------------------------

This kernel uses ``__builtin_amdgcn_smfmac_f32_16x16x64_fp8_fp8`` to
compute a :math:`16 \times 16` sparse matrix multiply-accumulate with
:math:`K=64` per instruction using FP8 (E4M3) inputs.  Each 32-bit
register lane packs four FP8 bytes; the kernel constructs the packed
``v2int`` and ``v4int`` operands from individual bytes loaded from LDS.

.. literalinclude:: ../../tools/example_codes/intrinsics_sparse_mfma.hip
   :language: cpp
   :linenos:
   :start-after: [Sphinx smfmac_f32_16x16x64_fp8_fp8 start]
   :end-before: [Sphinx smfmac_f32_16x16x64_fp8_fp8 end]

The kernel packs four consecutive FP8 bytes into each 32-bit lane using
shifts and bitwise OR before passing the vectors to the intrinsic.  The
sparsity index is wider than the FP16 variant (``0x8888`` vs ``0x88``)
because the larger K dimension requires more index bits to cover all
element groups.

.. note::

   The FP8 encoding used by ``gfx942`` is FNUZ (Finite, No
   Unsigned Zero), while later architectures use the standard OCP (Open
   Compute Project) E4M3 encoding.  The example code selects the correct
   interpretation at runtime based on the device architecture.

FP8 16×16 sparse kernel (K=128, gfx950)
----------------------------------------

This kernel uses ``__builtin_amdgcn_smfmac_f32_16x16x128_fp8_fp8`` to
compute a :math:`16 \times 16` sparse matrix multiply-accumulate with
:math:`K=128` per instruction, doubling the K depth of the ``gfx942``
variant.  The compressed :math:`\pmb{A}` operand grows to a ``v4int``
vector and the dense :math:`\pmb{B}` to a ``v8int`` vector.  This
intrinsic is available on ``gfx950`` only.

.. literalinclude:: ../../tools/example_codes/intrinsics_sparse_mfma.hip
   :language: cpp
   :linenos:
   :start-after: [Sphinx smfmac_f32_16x16x128_fp8_fp8 start]
   :end-before: [Sphinx smfmac_f32_16x16x128_fp8_fp8 end]

The byte-packing pattern is the same as the K=64 variant, but each lane now
loads twice as many bytes.  The sparsity index expands to a full 32-bit
value (``0x88888888``) to cover the 128-element K dimension.

.. _smfmac-instruction-throughput:

Instruction throughput
======================

The cycle count below is the value used to compute theoretical peak
throughput: :math:`\text{peak throughput} =
\frac{\text{ops per instruction}}{\text{cycle count}} \times
\text{clock frequency}`.  All SMFMAC instructions support VALU
co-execution; the VALU co-execution cycle count gives the number of VALU
cycles available during the SMFMAC latency window.

.. list-table::
   :header-rows: 1
   :widths: auto

   * - Intrinsic
     - Ops
     - Cycle count
     - VALU co-execution cycles
   * - ``__builtin_amdgcn_smfmac_f32_16x16x32_f16``
     - 16384
     - 16
     - 8
   * - ``__builtin_amdgcn_smfmac_f32_32x32x16_f16``
     - 32768
     - 32
     - 24
   * - ``__builtin_amdgcn_smfmac_f32_16x16x32_bf16``
     - 16384
     - 16
     - 8
   * - ``__builtin_amdgcn_smfmac_f32_32x32x16_bf16``
     - 32768
     - 32
     - 24
   * - ``__builtin_amdgcn_smfmac_i32_16x16x64_i8``
     - 32768
     - 16
     - 8
   * - ``__builtin_amdgcn_smfmac_i32_32x32x32_i8``
     - 65536
     - 32
     - 24
   * - ``__builtin_amdgcn_smfmac_f32_16x16x64_fp8_fp8``
     - 32768
     - 16
     - 8
   * - ``__builtin_amdgcn_smfmac_f32_16x16x64_fp8_bf8``
     - 32768
     - 16
     - 8
   * - ``__builtin_amdgcn_smfmac_f32_16x16x64_bf8_fp8``
     - 32768
     - 16
     - 8
   * - ``__builtin_amdgcn_smfmac_f32_16x16x64_bf8_bf8``
     - 32768
     - 16
     - 8
   * - ``__builtin_amdgcn_smfmac_f32_32x32x32_fp8_fp8``
     - 65536
     - 32
     - 24
   * - ``__builtin_amdgcn_smfmac_f32_32x32x32_fp8_bf8``
     - 65536
     - 32
     - 24
   * - ``__builtin_amdgcn_smfmac_f32_32x32x32_bf8_fp8``
     - 65536
     - 32
     - 24
   * - ``__builtin_amdgcn_smfmac_f32_32x32x32_bf8_bf8``
     - 65536
     - 32
     - 24

.. _smfmac-intrinsic-reference:

Intrinsic reference
===================

FP32-accumulate intrinsics
--------------------------

These intrinsics accumulate into single-precision (FP32) output fragments.
They differ in the data type of the :math:`\pmb{A}` and :math:`\pmb{B}`
matrix inputs.

FP16 matrix inputs
^^^^^^^^^^^^^^^^^^

The following intrinsics accept compressed FP16 elements for :math:`\pmb{A}`
and dense FP16 elements for :math:`\pmb{B}`, accumulating into FP32 output
fragments.

.. include:: smfmac-ref/f32-16x16x32-f16.rst
.. include:: smfmac-ref/f32-32x32x16-f16.rst

The following doubled-K FP16 variants are available on ``gfx950`` only.

.. include:: smfmac-ref/f32-16x16x64-f16.rst
.. include:: smfmac-ref/f32-32x32x32-f16.rst

BF16 matrix inputs
^^^^^^^^^^^^^^^^^^

The following intrinsics accept compressed BF16 elements for :math:`\pmb{A}`
and dense BF16 elements for :math:`\pmb{B}`, accumulating into FP32 output
fragments.  BF16 values are stored as ``short`` in register.

.. include:: smfmac-ref/f32-16x16x32-bf16.rst
.. include:: smfmac-ref/f32-32x32x16-bf16.rst

The following doubled-K BF16 variants are available on ``gfx950`` only.
These use native ``__bf16`` register types rather than ``short`` storage.

.. include:: smfmac-ref/f32-16x16x64-bf16.rst
.. include:: smfmac-ref/f32-32x32x32-bf16.rst

FP8 and BF8 matrix inputs
^^^^^^^^^^^^^^^^^^^^^^^^^^

The following intrinsics accept compressed FP8 (E4M3) or BF8 (E5M2)
elements for :math:`\pmb{A}` and dense FP8 or BF8 elements for
:math:`\pmb{B}`, accumulating into FP32 output fragments.  All four
combinations of FP8 and BF8 input types are supported for each tile size.
These variants require ``gfx942`` or later.

.. include:: smfmac-ref/f32-16x16x64-fp8-fp8.rst
.. include:: smfmac-ref/f32-16x16x64-fp8-bf8.rst
.. include:: smfmac-ref/f32-16x16x64-bf8-fp8.rst
.. include:: smfmac-ref/f32-16x16x64-bf8-bf8.rst
.. include:: smfmac-ref/f32-32x32x32-fp8-fp8.rst
.. include:: smfmac-ref/f32-32x32x32-fp8-bf8.rst
.. include:: smfmac-ref/f32-32x32x32-bf8-fp8.rst
.. include:: smfmac-ref/f32-32x32x32-bf8-bf8.rst

The following doubled-K FP8 and BF8 variants are available on ``gfx950``
only.

.. include:: smfmac-ref/f32-16x16x128-fp8-fp8.rst
.. include:: smfmac-ref/f32-16x16x128-fp8-bf8.rst
.. include:: smfmac-ref/f32-16x16x128-bf8-fp8.rst
.. include:: smfmac-ref/f32-16x16x128-bf8-bf8.rst
.. include:: smfmac-ref/f32-32x32x64-fp8-fp8.rst
.. include:: smfmac-ref/f32-32x32x64-fp8-bf8.rst
.. include:: smfmac-ref/f32-32x32x64-bf8-fp8.rst
.. include:: smfmac-ref/f32-32x32x64-bf8-bf8.rst

INT32-accumulate intrinsics
---------------------------

These intrinsics accept four signed 8-bit integer elements per 32-bit
register lane for both :math:`\pmb{A}` and :math:`\pmb{B}` inputs, and
accumulate into INT32 output fragments.

INT8 matrix inputs
^^^^^^^^^^^^^^^^^^

.. include:: smfmac-ref/i32-16x16x64-i8.rst
.. include:: smfmac-ref/i32-32x32x32-i8.rst

The following doubled-K INT8 variants are available on ``gfx950`` only.

.. include:: smfmac-ref/i32-16x16x128-i8.rst
.. include:: smfmac-ref/i32-32x32x64-i8.rst
