.. meta::
   :description: Reference for CDNA4 (gfx950, MI350) dense MFMA intrinsics, covering all supported matrix multiply-accumulate variants, parameters, and output layouts.
   :keywords: CDNA4, gfx950, MI350, MFMA, matrix core, HIP intrinsics, FP32, FP16, BF16, FP64, FP8, BF8, FP6, FP4, INT8, accumulator, mfma_scale, __builtin_amdgcn_mfma

.. _cdna4-dense-mfma-intrinsics:

********************************************************************************
CDNA4 dense MFMA intrinsics
********************************************************************************

Matrix Fused Multiply-Add (MFMA) intrinsics let you issue hardware
matrix multiply-accumulate operations directly from HIP device code on
CDNA4 GPUs (``gfx950``, MI350 series). Each MFMA instruction multiplies a small
:math:`\pmb{A}` fragment by a small :math:`\pmb{B}` fragment and accumulates the
result into a :math:`\pmb{C}` fragment, all within a single wavefront of 64
lanes. The hardware delivers significantly higher throughput than an equivalent
sequence of scalar fused multiply-add (FMA) instructions.

CDNA4 retains FP32, FP16, BF16, INT8, FP64, FP8, and BF8 shapes from CDNA3
and adds wider-K single-block variants for FP16, BF16, and INT8, plus a new
class of scaled sub-byte intrinsics (``mfma_scale_*``) that support FP8, BF8,
FP6 (E2M3 and E3M2), and FP4 (E2M1) input formats with per-block scaling.
The eXtended Float32 (XF32) input format available on CDNA3 is not supported
on CDNA4. See the intrinsic reference below for the complete list of supported
shapes and formats.

Architecture availability
=========================

The intrinsics on this page target CDNA4 (``gfx950``, MI350 series) exclusively.
Equivalent intrinsics for other CDNA generations are documented on their own
reference pages:

* :ref:`cdna-dense-mfma-intrinsics` -- CDNA (``gfx908``, MI100 series)
* :ref:`cdna2-dense-mfma-intrinsics` -- CDNA2 (``gfx90a``, MI200 series)
* :ref:`cdna3-dense-mfma-intrinsics` -- CDNA3 (``gfx942``, MI300 series)

Naming convention
=================

All MFMA intrinsics follow the pattern:

.. code-block:: text

   __builtin_amdgcn_mfma_<out_type>_<M>x<N>x<K><in_type>

For FP8 and BF8 intrinsics, where the :math:`\pmb{A}` and :math:`\pmb{B}`
input types can differ, the pattern is:

.. code-block:: text

   __builtin_amdgcn_mfma_<out_type>_<M>x<N>x<K>_<typeA>_<typeB>

``out_type``
    Accumulator element type (``f32``, ``f64``, or ``i32``).

``M``, ``N``, ``K``
    Tile dimensions in elements.  The instruction computes the
    contribution of a K-wide panel of :math:`\pmb{A}` (:math:`M \times K`) and
    a K-wide panel of :math:`\pmb{B}` (:math:`K \times N`) to an
    :math:`M \times N` output tile.  Each instruction processes one K step;
    the caller loops over K to accumulate a full matrix product.

``in_type``
    Input element type (``f32``, ``f64``, ``f16``, ``bf16``, ``i8``,
    or ``xf32``).

``typeA``, ``typeB``
    For FP8 and BF8 intrinsics: the element type of the :math:`\pmb{A}` and
    :math:`\pmb{B}` matrices respectively.  Each is one of ``fp8`` (E4M3
    format) or ``bf8`` (E5M2 format).

.. note::

   CDNA4 renames several underlying instruction set architecture (ISA)
   instructions to include an explicit block count (for example,
   ``v_mfma_f32_32x32x1f32`` becomes ``v_mfma_f32_32x32x1_2b_f32`` in
   the CDNA4 ISA).  The HIP intrinsic
   names (``__builtin_amdgcn_mfma_*``) are unchanged; the rename is
   transparent to HIP device code.

.. _cdna4-dense-mfma-accumulator-layout:

Accumulator layout
==================

Each MFMA instruction computes one or more independent :math:`M \times N`
output tiles simultaneously across the 64 lanes of a wavefront.  The number
of independent tiles is called the *block count*.  It depends on how the
:math:`\pmb{A}` matrix rows are distributed across lane groups:

* **Scalar-input variants** (FP32 or FP64 :math:`\pmb{A}` and
  :math:`\pmb{B}`): each K position occupies a separate group of :math:`M`
  lanes, so
  :math:`\text{blocks} = \text{wavefront_size} / (M \times K)`.
* **Packed-input variants** (FP16, BF16, XF32, INT8, FP8, BF8
  :math:`\pmb{A}` and :math:`\pmb{B}`): all K positions are packed into the
  register bits of the *same* lane group, so
  :math:`\text{blocks} = \text{wavefront_size} / M` regardless of
  :math:`K`.

Each block is independent: the :math:`\pmb{A}`, :math:`\pmb{B}`, and
:math:`\pmb{C}`/:math:`\pmb{D}` operands of different blocks
occupy distinct lane and accVGPR positions and compute separate outer products.

The total number of accVGPRs per lane scales with the block count:

.. math::

   \text{accVGPRs per lane} = \text{blocks} \times \frac{M \times N}{\text{wavefront_size}}

.. note::

   On CDNA4, all four operands (:math:`\pmb{A}`, :math:`\pmb{B}`,
   :math:`\pmb{C}`, and :math:`\pmb{D}`) can reside in either accumulation
   VGPRs (accVGPRs) or standard architecture VGPRs (archVGPRs).  In HIP
   device code the compiler selects the appropriate register class
   automatically.

   Each of the 64 wavefront lanes maintains its own private accVGPR file.
   An accVGPR index always refers to a register within one specific lane's
   file; the same index in two different lanes denotes two distinct physical
   registers.  The layout tables and formulas in the following sections use
   *accVGPR index* as a logical position label; the actual register class
   (accVGPR or ArchVGPR) is chosen by the compiler and does not affect the
   layout.

The formulas in the subsections below use the following notation:

* :math:`i` -- zero-based row index within the tile, :math:`0 \le i < M`
* :math:`j` -- zero-based column index within the tile, :math:`0 \le j < N`
* :math:`b` -- block index, :math:`0 \le b < \text{blocks}`
* **lane** -- wavefront lane that holds the element,
  :math:`0 \le \text{lane} < 64`
* **accVGPR** -- zero-based index into that lane's *private* accumulator
  register file; the same index in two different lanes refers to two distinct
  physical registers

:math:`32 \times 32` layout
---------------------------

The :math:`32 \times 32` tile shape uses two blocks.  All 32 accVGPRs per
lane are active: accVGPRs 0--15 belong to block 0, accVGPRs 16--31 to block 1.

.. figure:: ../../data/hardware-intrinsics/cdna/mfma-intrinsics/mfma-layout-32x32.svg
   :alt: :math:`32 \times 32` MFMA accumulator layout -- accVGPR index
         per output element, with lane groups colour-coded.
   :align: center
   :width: 100%

   :math:`32 \times 32` **accumulator layout -- 2 blocks.**  Each cell shows the
   accVGPR index that holds output element :math:`(i, j)` of block 0; block 1
   adds 16.  Teal cells (rows where :math:`\lfloor i/4 \rfloor` is even) belong
   to lanes 0--31; grey cells to lanes 32--63.  Column :math:`j` gives the
   lane offset within the group.

Given output element :math:`(i, j)` in block :math:`b`:

.. math::

   \text{lane}   &= \bigl(32 \cdot \lfloor \frac{i}{4} \rfloor\bigr) \bmod 64 + j \\
   \text{accVGPR} &= 16 b + 4 \lfloor \frac{i}{8} \rfloor + (i \bmod 4)

Conversely, given a lane :math:`L` and accVGPR index :math:`G`:

.. math::

   i &= \bigl(8 \cdot \lfloor \frac{G}{4} \rfloor\bigr) \bmod 32
       + 4 \lfloor \frac{L}{32} \rfloor + (G \bmod 4) \\
   j &= L \bmod 32 \\
   b &= \lfloor \frac{G}{16} \rfloor

The row-to-lane mapping groups rows in bands of four.  Within block 0:

.. list-table::
   :header-rows: 1
   :widths: auto

   * - Rows
     - Lanes
     - accVGPRs
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

Block 1 uses the same lane pattern with accVGPRs 16--31.

Intrinsics with a **1-block** :math:`32 \times 32` result (``v16float`` /
``v16int``, 16 accVGPRs per lane) use only block 0 of the layout above.
These variants do not support the ``cbsz`` and ``abid`` modifiers.

:math:`16 \times 16` layout
---------------------------

The :math:`16 \times 16` tile shape uses four blocks.  The 16 accVGPRs per
lane are partitioned by block: accVGPRs 0---3 for block 0, 4---7 for block 1,
8---11 for block 2, and 12---15 for block 3.

.. figure:: ../../data/hardware-intrinsics/cdna/mfma-intrinsics/mfma-layout-16x16.svg
   :alt: :math:`16 \times 16` MFMA accumulator layout -- accVGPR index
         per output element, with lane groups colour-coded by block
         assignment.
   :align: center
   :width: 70%

   :math:`16 \times 16` **accumulator layout -- 4 blocks.**  Each cell shows
   the accVGPR index for block 0; block :math:`k` adds :math:`4k`.  Rows
   0---3 (teal, lanes 0---15), rows 4---7 (grey, lanes 16---31), rows 8---11
   (teal, lanes 32---47), rows 12---15 (grey, lanes 48---63).  Column
   :math:`j` gives the lane offset within the group.

Given output element :math:`(i, j)` in block :math:`b`:

.. math::

   \text{lane}    &= 16 \lfloor \frac{i}{4} \rfloor + j \\
   \text{accVGPR} &= 4 b + (i \bmod 4)

Conversely, given lane :math:`L` and accVGPR index :math:`G`:

.. math::

   i &= 4 \lfloor \frac{L}{16} \rfloor + (G \bmod 4) \\
   j &= L \bmod 16 \\
   b &= \lfloor \frac{G}{4} \rfloor

The row-to-lane mapping (identical for every block):

.. list-table::
   :header-rows: 1
   :widths: auto

   * - Rows
     - Lanes
   * - 0---3
     - 0---15
   * - 4---7
     - 16---31
   * - 8---11
     - 32---47
   * - 12---15
     - 48---63

Intrinsics with a **1-block** :math:`16 \times 16` result (``v4float`` /
``v4int``, 4 accVGPRs per lane) use only block 0 of the layout above.
These variants do not support the ``cbsz`` and ``abid`` modifiers.

:math:`4 \times 4` layout
-------------------------

The :math:`4 \times 4` tile shape uses 16 blocks.  A single wavefront
simultaneously computes 16 independent :math:`4 \times 4` outer products.
Each group of 4 consecutive lanes (lanes :math:`4b` through :math:`4b + 3`)
holds all 16 output elements of block :math:`b` across 4 accVGPRs.

.. figure:: ../../data/hardware-intrinsics/cdna/mfma-intrinsics/mfma-layout-4x4.svg
   :alt: :math:`4 \times 4` MFMA accumulator layout -- all 16 blocks shown as
         consecutive groups of 4 lanes across the full wavefront.
   :align: center
   :width: 100%

   :math:`4 \times 4` **accumulator layout -- 16 blocks.**  Columns are lanes
   0--63; each group of 4 consecutive lanes holds one complete
   :math:`4 \times 4` output tile in accVGPRs 0--3 (rows).  Teal groups are
   even-numbered blocks, grey groups odd-numbered blocks.

Given output element :math:`(i, j)` in block :math:`b`:

.. math::

   \text{lane}    &= 4 b + j \\
   \text{accVGPR} &= i

Conversely, given lane :math:`L` and accVGPR index :math:`G`:

.. math::

   i &= G \bmod 4 \\
   j &= L \bmod 4 \\
   b &= \lfloor \frac{L}{4} \rfloor

:math:`16 \times 16` FP64 layout
--------------------------------

The :math:`16 \times 16` FP64 tile shape uses one block.  Each lane holds
four FP64 output elements across 4 accVGPR pairs (8 physical accVGPRs, since
each FP64 value occupies two 32-bit registers).

.. note::

   FP64 accVGPR indices count *pairs* of physical registers.  accVGPR pair
   :math:`k` corresponds to physical registers ``v[2k+1:2k]``.

.. figure:: ../../data/hardware-intrinsics/cdna/mfma-intrinsics/mfma-layout-16x16-f64.svg
   :alt: :math:`16 \times 16` FP64 MFMA accumulator layout -- accVGPR pair index
         per output element, with lane groups colour-coded.
   :align: center
   :width: 70%

   :math:`16 \times 16` **FP64 accumulator layout -- 1 block.**  Each cell shows
   the accVGPR pair index :math:`k` that holds output element :math:`(i, j)`;
   physical registers are ``v[2k+1:2k]``.  Each group of 16 consecutive lanes
   (column offset :math:`j`) covers all four rows within one row-modulo-4 band.

Given output element :math:`(i, j)`:

.. math::

   \text{lane}            &= 16 \cdot (i \bmod 4) + j \\
   \text{accVGPR pair}\ k &= \lfloor \frac{i}{4} \rfloor

Conversely, given lane :math:`L` and accVGPR pair index :math:`k`:

.. math::

   i &= 4 k + \lfloor \frac{L}{16} \rfloor \\
   j &= L \bmod 16

The row-to-lane mapping.  Each group of 16 consecutive lanes covers one row
modulo 4; the accVGPR pair selects the row group:

.. list-table::
   :header-rows: 1
   :widths: auto

   * - Rows
     - Lanes
     - accVGPR pair (physical regs)
   * - 0, 4, 8, 12
     - 0---15
     - 0, 1, 2, 3 (``v[1:0]``, ``v[3:2]``, ``v[5:4]``, ``v[7:6]``)
   * - 1, 5, 9, 13
     - 16---31
     - 0, 1, 2, 3
   * - 2, 6, 10, 14
     - 32---47
     - 0, 1, 2, 3
   * - 3, 7, 11, 15
     - 48---63
     - 0, 1, 2, 3

:math:`4 \times 4` FP64 layout
------------------------------

The :math:`4 \times 4` FP64 tile shape uses four blocks.  Each lane holds one
FP64 output element in a single accVGPR pair (2 physical accVGPRs, ``v[1:0]``).

.. figure:: ../../data/hardware-intrinsics/cdna/mfma-intrinsics/mfma-layout-4x4-f64.svg
   :alt: :math:`4 \times 4` FP64 MFMA accumulator layout -- all 4 blocks shown
         across the full 64-lane wavefront, grouped into four 16-lane row groups.
   :align: center
   :width: 100%

   :math:`4 \times 4` **FP64 accumulator layout -- 4 blocks.**  Columns are lanes
   0--63; rows are matrix rows 0--3.  Within each 16-lane row group, four
   sub-groups of 4 lanes hold blocks 0--3.  Every cell holds one FP64 value in
   accVGPR pair 0 (``v[1:0]``).  Teal sub-groups are even-numbered blocks, grey
   sub-groups are odd-numbered blocks.

Given output element :math:`(i, j)` in block :math:`b`:

.. math::

   \text{lane} &= 16 i + 4 b + j

Conversely, given lane :math:`L`:

.. math::

   i &= \lfloor \frac{L}{16} \rfloor \\
   j &= L \bmod 4 \\
   b &= \lfloor \frac{L \bmod 16}{4} \rfloor

The row-to-lane mapping.  Lanes 0---15 hold all four blocks of row 0; the
pattern repeats for rows 1---3 at lane offsets of 16, 32, and 48:

.. list-table::
   :header-rows: 1
   :widths: auto

   * - Row
     - Block
     - Lanes
     - accVGPR pair
   * - 0
     - 0
     - 0---3
     - 0 (``v[1:0]``)
   * - 0
     - 1
     - 4---7
     - 0 (``v[1:0]``)
   * - 0
     - 2
     - 8---11
     - 0 (``v[1:0]``)
   * - 0
     - 3
     - 12---15
     - 0 (``v[1:0]``)
   * - 1
     - 0---3
     - 16---31
     - 0 (``v[1:0]``)
   * - 2
     - 0---3
     - 32---47
     - 0 (``v[1:0]``)
   * - 3
     - 0---3
     - 48---63
     - 0 (``v[1:0]``)

Register types used in this reference
=====================================

The signatures below use the following type aliases, which you can declare with
C++ attributes in any HIP translation unit:

.. code-block:: cpp

   using v4float   = float [[clang::ext_vector_type(4)]];
   using v16float  = float [[clang::ext_vector_type(16)]];
   using v32float  = float [[clang::ext_vector_type(32)]];
   using v4half    = _Float16 [[clang::ext_vector_type(4)]];
   using v8half    = _Float16 [[clang::ext_vector_type(8)]];
   using v4int     = int [[clang::ext_vector_type(4)]];
   using v6int     = int [[clang::ext_vector_type(6)]];        // FP6 scale inputs
   using v8int     = int [[clang::ext_vector_type(8)]];        // FP8/BF8 scale inputs
   using v16int    = int [[clang::ext_vector_type(16)]];
   using v32int    = int [[clang::ext_vector_type(32)]];
   using v2bfloat  = short [[clang::ext_vector_type(2)]];      // bf16 storage
   using v4bfloat  = short [[clang::ext_vector_type(4)]];      // bf16 storage
   using v8bfloat  = short [[clang::ext_vector_type(8)]];      // wider-K BF16
   using v4double  = double [[clang::ext_vector_type(4)]];

FP8 and BF8 operands passed to the standard (non-scale) MFMA intrinsics use a
64-bit integer (``long long``) that packs eight 8-bit values per lane. The
scaled sub-byte intrinsics (``mfma_scale_*``) use ``v4int``, ``v6int``, or
``v8int`` depending on the selected format; see the per-intrinsic reference for
details.

Each type alias maps one-to-one to the corresponding LLVM vector type used in
the intrinsic definition.  The number in the name is the element count per
lane; the total VGPR count equals the element count multiplied by the element
size in 32-bit words.

.. _cdna4-dense-mfma-common-parameters:

Common parameters
=================

See :doc:`mfma-common-parameters` for a complete description of the ``cbsz``,
``abid``, and ``blgp`` modifiers shared by all MFMA intrinsics.

Using MFMA intrinsics as a compute policy
=========================================

The matrix multiplication tutorial in
:ref:`matrix-multiply-optimization` uses a ``ComputePolicy`` type
parameter to separate the multiply-accumulate logic from the rest of the
kernel.  You can drop an MFMA-based policy into that framework without
changing the outer kernel.

The example below implements ``MfmaCdna4F16Policy`` using
``v_mfma_f32_16x16x32_f16`` -- a :math:`16 \times 16` FP16 intrinsic
available on CDNA4 (``gfx950``, MI350 series).  Each wavefront computes
a single :math:`16 \times 16` output tile; one block is active, giving 4
FP32 accVGPRs per lane.

The complete source file is available for download:

* :download:`matrix_multiply_cdna4_mfma.hip <../../tools/example_codes/matrix_multiply_cdna4_mfma.hip>`

.. rubric:: Policy constants

``v_mfma_f32_16x16x32_f16`` consumes 32 K-positions per call (K=32), so
``k_step = 32``.  Each lane provides eight FP16 values (``v8half``) for
:math:`\pmb{A}` and eight for :math:`\pmb{B}`; the 64 lanes partition
into four groups of 16, each group covering eight K-positions.  The
entire :math:`16 \times 16` tile belongs to one wavefront, so
``thread_tile_m = thread_tile_n = 16`` and ``effective_lanes = 64``.

.. rubric:: Accumulator layout

The intrinsic returns a ``v4float`` holding 4 FP32 values -- one for each
accVGPR.  The ``Accumulator`` struct wraps this directly.  The layout
formulas from the :ref:`cdna4-dense-mfma-accumulator-layout` section
translate ``(lane, accVGPR)`` back to :math:`(i, j)` coordinates during
the ``store_c()`` pass.

.. rubric:: Fragment loading

Because each lane provides eight FP16 values for K=32, each ``load_a``
call reads eight consecutive scalars.  Lane group :math:`g = \lfloor
\text{lane\_id} / 16 \rfloor` covers K-positions ``ki + 8g`` through
``ki + 8g + 7``; within the group, lane offset ``lane_id mod 16`` selects
one of the 16 :math:`\pmb{A}`-rows (or :math:`\pmb{B}`-columns).  The
``mma()`` method packs the eight values into a ``v8half`` before issuing
the instruction.

.. note::

   The all-modifier-zero constraint means ``cbsz``, ``abid``, and ``blgp``
   must all be passed as ``0``.  ``v_mfma_f32_16x16x32_f16`` is
   single-block only and does not support the ``cbsz``/``abid`` broadcast
   mechanism or the ``blgp`` lane-group pattern modifier.

.. literalinclude:: ../../tools/example_codes/matrix_multiply_cdna4_mfma.hip
   :language: cuda
   :start-after: [Sphinx mfma cdna4 policy start]
   :end-before: [Sphinx mfma cdna4 policy end]

.. rubric:: Instantiating the kernel

With ``MfmaCdna4F16Policy`` in hand, plug it into the generic kernel
alongside a ``TilePolicy`` whose ``block_tile_m`` and ``block_tile_n``
are multiples of 16 and whose ``k_tile_size`` is a multiple of
``k_step = 32``.  The policy aliases and launch configuration from the
example file are:

.. literalinclude:: ../../tools/example_codes/matrix_multiply_cdna4_mfma.hip
   :language: cuda
   :start-after: [Sphinx mfma policy aliases start]
   :end-before: [Sphinx mfma policy aliases end]

.. literalinclude:: ../../tools/example_codes/matrix_multiply_cdna4_mfma.hip
   :language: cuda
   :start-after: [Sphinx mfma launch config start]
   :end-before: [Sphinx mfma launch config end]

.. literalinclude:: ../../tools/example_codes/matrix_multiply_cdna4_mfma.hip
   :language: cuda
   :start-after: [Sphinx mfma kernel launch start]
   :end-before: [Sphinx mfma kernel launch end]

**Compile and run:**

.. code-block:: bash

   amdclang++ -O3 -std=c++17 --offload-arch=gfx950 \
       matrix_multiply_cdna4_mfma.hip -o mm_cdna4_mfma
   ./mm_cdna4_mfma

.. note::

   ``MfmaCdna4F16Policy`` requires a CDNA4 GPU (``gfx950``).
   Compile with ``--offload-arch=gfx950`` to select the correct
   architecture.  On other targets the ``#if defined(__gfx950__)`` guard
   selects ``ScalarFMAPolicy`` automatically, so the file compiles without
   modification.

.. _cdna4-dense-mfma-throughput:

Instruction throughput
======================

The cycle count below is the value used to compute theoretical peak
throughput: :math:`\text{peak throughput} =
\frac{\text{ops per instruction}}{\text{cycle count}} \times
\text{clock frequency}`.

.. list-table::
   :header-rows: 1
   :widths: auto

   * - Intrinsic
     - Ops
     - Cycle count
   * - ``__builtin_amdgcn_mfma_f32_32x32x1f32``
     - 4096
     - 64
   * - ``__builtin_amdgcn_mfma_f32_16x16x1f32``
     - 2048
     - 32
   * - ``__builtin_amdgcn_mfma_f32_4x4x1f32``
     - 512
     - 8
   * - ``__builtin_amdgcn_mfma_f32_32x32x2f32``
     - 4096
     - 64
   * - ``__builtin_amdgcn_mfma_f32_16x16x4f32``
     - 2048
     - 32
   * - ``__builtin_amdgcn_mfma_f32_32x32x4f16``
     - 16384
     - 64
   * - ``__builtin_amdgcn_mfma_f32_16x16x4f16``
     - 8192
     - 32
   * - ``__builtin_amdgcn_mfma_f32_4x4x4f16``
     - 2048
     - 8
   * - ``__builtin_amdgcn_mfma_f32_32x32x8f16``
     - 16384
     - 32
   * - ``__builtin_amdgcn_mfma_f32_16x16x16f16``
     - 8192
     - 16
   * - ``__builtin_amdgcn_mfma_f32_32x32x16_f16``
     - 32768
     - 32
   * - ``__builtin_amdgcn_mfma_f32_16x16x32_f16``
     - 16384
     - 16
   * - ``__builtin_amdgcn_mfma_f32_32x32x4bf16_1k``
     - 16384
     - 64
   * - ``__builtin_amdgcn_mfma_f32_16x16x4bf16_1k``
     - 8192
     - 32
   * - ``__builtin_amdgcn_mfma_f32_4x4x4bf16_1k``
     - 2048
     - 8
   * - ``__builtin_amdgcn_mfma_f32_32x32x8bf16_1k``
     - 16384
     - 32
   * - ``__builtin_amdgcn_mfma_f32_16x16x16bf16_1k``
     - 8192
     - 16
   * - ``__builtin_amdgcn_mfma_f32_32x32x16_bf16``
     - 32768
     - 32
   * - ``__builtin_amdgcn_mfma_f32_16x16x32_bf16``
     - 16384
     - 16
   * - ``__builtin_amdgcn_mfma_f32_32x32x16_fp8_fp8``
     - 32768
     - 32
   * - ``__builtin_amdgcn_mfma_f32_32x32x16_fp8_bf8``
     - 32768
     - 32
   * - ``__builtin_amdgcn_mfma_f32_32x32x16_bf8_fp8``
     - 32768
     - 32
   * - ``__builtin_amdgcn_mfma_f32_32x32x16_bf8_bf8``
     - 32768
     - 32
   * - ``__builtin_amdgcn_mfma_f32_16x16x32_fp8_fp8``
     - 16384
     - 16
   * - ``__builtin_amdgcn_mfma_f32_16x16x32_fp8_bf8``
     - 16384
     - 16
   * - ``__builtin_amdgcn_mfma_f32_16x16x32_bf8_fp8``
     - 16384
     - 16
   * - ``__builtin_amdgcn_mfma_f32_16x16x32_bf8_bf8``
     - 16384
     - 16
   * - ``__builtin_amdgcn_mfma_scale_f32_32x32x64_f8f6f4``
     - 131072
     - 32 (FP4/FP6), 64 (FP8/BF8)
   * - ``__builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4``
     - 65536
     - 16 (FP4/FP6), 32 (FP8/BF8)
   * - ``__builtin_amdgcn_mfma_f64_16x16x4f64``
     - 2048
     - 64
   * - ``__builtin_amdgcn_mfma_f64_4x4x4f64``
     - 512
     - 32
   * - ``__builtin_amdgcn_mfma_i32_32x32x4i8``
     - 16384
     - 64
   * - ``__builtin_amdgcn_mfma_i32_16x16x4i8``
     - 8192
     - 32
   * - ``__builtin_amdgcn_mfma_i32_4x4x4i8``
     - 2048
     - 8
   * - ``__builtin_amdgcn_mfma_i32_32x32x16_i8``
     - 32768
     - 32
   * - ``__builtin_amdgcn_mfma_i32_16x16x32_i8``
     - 16384
     - 16
   * - ``__builtin_amdgcn_mfma_i32_32x32x32_i8``
     - 65536
     - 32
   * - ``__builtin_amdgcn_mfma_i32_16x16x64_i8``
     - 32768
     - 16

.. seealso::

   :ref:`cdna4-mfma-lds-intrinsics` -- Transpose load intrinsics that load
   operands directly from LDS into the per-lane fragment layout expected by
   the intrinsics below, without a software shuffle step.

.. _cdna4-dense-mfma-intrinsic-reference:

Intrinsic reference
===================

FP32-accumulate intrinsics
--------------------------

These intrinsics accumulate into single-precision (FP32) output
fragments. They differ in the data type of the :math:`\pmb{A}` and
:math:`\pmb{B}` matrix inputs.

FP32 matrix inputs
^^^^^^^^^^^^^^^^^^

The following intrinsics accept one FP32 element per lane for both
:math:`\pmb{A}` and :math:`\pmb{B}` inputs and accumulate into FP32 output
fragments.

.. include:: mfma-ref/f32-32x32x1f32.rst
.. include:: mfma-ref/f32-16x16x1f32.rst
.. include:: mfma-ref/f32-4x4x1f32.rst
.. include:: mfma-ref/f32-32x32x2f32.rst
.. include:: mfma-ref/f32-16x16x4f32.rst

FP16 matrix inputs
^^^^^^^^^^^^^^^^^^

CDNA4 supports two FP16 intrinsic families that differ in K step and block
count.

The narrow-K multi-block variants accept four FP16 elements per lane packed into
a ``v4half`` register:

.. include:: mfma-ref/f32-32x32x4f16.rst
.. include:: mfma-ref/f32-16x16x4f16.rst
.. include:: mfma-ref/f32-4x4x4f16.rst
.. include:: mfma-ref/f32-32x32x8f16.rst
.. include:: mfma-ref/f32-16x16x16f16.rst

The wider-K single-block variants accept eight FP16 elements per lane packed
into a ``v8half`` register:

.. include:: mfma-ref/f32-32x32x16-f16.rst
.. include:: mfma-ref/f32-16x16x32-f16.rst

BF16 matrix inputs
^^^^^^^^^^^^^^^^^^

CDNA4 supports two BF16 intrinsic families that differ in K step and block
count.

The narrow-K ``_1k`` multi-block variants, inherited from CDNA3, accept four
BF16 elements per lane packed into a ``v4bfloat`` register:

.. include:: mfma-ref/f32-32x32x4bf16-1k.rst
.. include:: mfma-ref/f32-16x16x4bf16-1k.rst
.. include:: mfma-ref/f32-4x4x4bf16-1k.rst
.. include:: mfma-ref/f32-32x32x8bf16-1k.rst
.. include:: mfma-ref/f32-16x16x16bf16-1k.rst

The wider-K single-block variants accept eight BF16 elements per lane packed
into a ``v8bfloat`` register:

.. include:: mfma-ref/f32-32x32x16-bf16.rst
.. include:: mfma-ref/f32-16x16x32-bf16.rst

FP8 and BF8 matrix inputs
^^^^^^^^^^^^^^^^^^^^^^^^^

The following intrinsics accept eight 8-bit floating-point values per lane
packed into a ``long long`` register.  Two 8-bit formats are supported:

* **FP8** (``fp8``): E4M3 format (1 sign, 4 exponent, 3 mantissa bits).
* **BF8** (``bf8``): E5M2 format (1 sign, 5 exponent, 2 mantissa bits).

The :math:`\pmb{A}` and :math:`\pmb{B}` matrices can use different formats,
giving four combinations per tile shape.

.. include:: mfma-ref/f32-32x32x16fp8-fp8.rst
.. include:: mfma-ref/f32-32x32x16fp8-bf8.rst
.. include:: mfma-ref/f32-32x32x16bf8-fp8.rst
.. include:: mfma-ref/f32-32x32x16bf8-bf8.rst
.. include:: mfma-ref/f32-16x16x32fp8-fp8.rst
.. include:: mfma-ref/f32-16x16x32fp8-bf8.rst
.. include:: mfma-ref/f32-16x16x32bf8-fp8.rst
.. include:: mfma-ref/f32-16x16x32bf8-bf8.rst

FP64-accumulate intrinsics
--------------------------

CDNA4 retains native double-precision (FP64) matrix accumulation from
CDNA3.  These intrinsics accept one FP64 element per lane for both
:math:`\pmb{A}` and :math:`\pmb{B}` inputs and accumulate into FP64 output
fragments held in ``v4double`` registers.

.. note::

   On CDNA4, the ``blgp`` parameter of FP64 intrinsics is repurposed as a
   3-bit source negation modifier rather than a lane-group pattern selector.
   See the :ref:`cdna4-dense-mfma-common-parameters` section for details.

FP64 matrix inputs
^^^^^^^^^^^^^^^^^^

The following intrinsics accept one FP64 element per lane for both
:math:`\pmb{A}` and :math:`\pmb{B}` inputs and accumulate into FP64 output
fragments.

.. include:: mfma-ref/f64-16x16x4f64-cdna3.rst
.. include:: mfma-ref/f64-4x4x4f64-cdna3.rst

INT32-accumulate intrinsics
---------------------------

These intrinsics accumulate into signed 32-bit integer (INT32) output
fragments.

INT8 matrix inputs
^^^^^^^^^^^^^^^^^^

The following intrinsics accept four signed 8-bit integer elements per lane
packed into a single ``int`` register for both :math:`\pmb{A}` and
:math:`\pmb{B}` inputs.

.. include:: mfma-ref/i32-32x32x4i8.rst
.. include:: mfma-ref/i32-16x16x4i8.rst
.. include:: mfma-ref/i32-4x4x4i8.rst

The following wider-K variants accept eight signed 8-bit integer elements
per lane packed into a ``long long`` register:

.. include:: mfma-ref/i32-32x32x16i8.rst
.. include:: mfma-ref/i32-16x16x32i8.rst

The following CDNA4 wider-K single-block variants accept 16 signed 8-bit integer
elements per lane packed into a ``v4int`` (four 32-bit words) register:

.. include:: mfma-ref/i32-32x32x32-i8.rst
.. include:: mfma-ref/i32-16x16x64-i8.rst

Mixed-precision with scaling
----------------------------

CDNA4 introduces scaled sub-byte matrix intrinsics (``mfma_scale_*``) that
support FP8 (E4M3), BF8 (E5M2), FP6 (E2M3), FP6 (E3M2), and FP4 (E2M1)
input formats with per-block scaling.  Unlike the standard MFMA intrinsics,
the ``cbsz`` and ``blgp`` parameters select the **input format** for
:math:`\pmb{A}` and :math:`\pmb{B}` respectively rather than acting as
broadcast or lane-group-pattern modifiers.  The vector type of ``srcA`` and
``srcB`` must match the selected format: ``v8int`` for 8-bit formats, ``v6int``
for 6-bit formats, and ``v4int`` for 4-bit formats.

Each instruction also accepts a pair of per-block scale values loaded with
``v_mfma_ld_scale_b32``.  The 2-bit ``op_sel_a`` and ``op_sel_b`` immediates
select which quarter of the scale register applies to each output block.

FP8, FP6, and FP4 matrix inputs
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The following intrinsics accept scaled sub-byte matrix inputs with per-block
scaling.

.. include:: mfma-ref/f32-32x32x64-f8f6f4-scale.rst
.. include:: mfma-ref/f32-16x16x128-f8f6f4-scale.rst
