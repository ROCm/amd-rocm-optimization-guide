.. meta::
   :description: Use HIP arithmetic and packing intrinsics for AMD GPU optimization, including SAD, integer dot products, FP16 accumulation, and float-to-byte type conversion.
   :keywords: AMD, ROCm, HIP, hardware intrinsics, SAD, dot product, arithmetic intrinsics, packing, CDNA, RDNA

.. _arithmetic_packing_intrinsics:

*********************************************************
Arithmetic and packing intrinsics for AMD GPUs
*********************************************************

Arithmetic and packing intrinsics give you direct access to dedicated units for
the sum of absolute differences, integer dot products, FP16 (16-bit half-precision) dot products, and type
conversion and packing. For a complete listing of all intrinsics with their
signatures and supported architectures, see
:ref:`arithmetic_packing_intrinsic_reference`.

The complete source file is available for download:

* :download:`intrinsics_cross_arch_arithmetic.hip <../../tools/example_codes/intrinsics_cross_arch_arithmetic.hip>`

Architecture availability
=========================

Most arithmetic and packing intrinsics are available on all AMD Instinct (CDNA)
and AMD Radeon (RDNA) architectures.  Some intrinsics --- particularly
mixed-sign dot products, FP8/BF8 conversions, and stochastic rounding --- are
limited to specific generations.  For per-intrinsic availability, see the
architecture tables in the
:ref:`reference pages <arithmetic_packing_intrinsic_reference>` below.

Sum of absolute differences
===========================

The sum of absolute differences (SAD) family computes the L1 distance between
two byte vectors, producing a scalar accumulator that measures the similarity between
the two blocks of data. The fundamental use case is block matching in video encoding
and image processing. Each candidate reference block is compared against the
current block by accumulating the per-byte absolute differences, and the
candidate with the lowest SAD is the best match. The three intrinsics discussed below
cover single 4-byte SAD, dual-SAD packing, and four overlapping 8-byte SADs,
giving you progressively higher throughput as your data layout permits.

Masked SAD with ``msad_u8``
---------------------------

The ``__builtin_amdgcn_msad_u8`` intrinsic computes the SAD of four packed bytes,
excluding byte positions where the reference byte (``src1``) is zero from the
accumulation. This masking behavior is designed for padded
reference windows, where the encoder marks positions outside the frame boundary
with zero rather than valid pixel data. Accumulating those positions would
artificially inflate the SAD and bias the match; skipping them keeps the
comparison honest regardless of the padding layout.

.. literalinclude:: ../../tools/example_codes/intrinsics_cross_arch_arithmetic.hip
   :language: cpp
   :linenos:
   :start-after: [Sphinx msad start]
   :end-before: [Sphinx msad end]

The third argument is the initial accumulator value. Passing ``0u`` gives a
fresh SAD for each element; passing the result of a previous call accumulates
across multiple 32-bit values.

For the full signature and parameter details, see
:ref:`msad_u8 <sad-msad-u8>` in the SAD intrinsics reference.

Dual-SAD packing with ``sad_u8`` and ``sad_hi_u8``
--------------------------------------------------

``__builtin_amdgcn_sad_u8`` computes the SAD of four packed bytes and
accumulates the result into bits [15:0] of the destination 32-bit value.
``__builtin_amdgcn_sad_hi_u8`` computes the same byte SAD but shifts each
per-byte absolute difference left by 16 bits before accumulating, placing its
result in bits [31:16]. Because the two results occupy different halves of the
same 32-bit value, one call to each packs two independent 4-byte SADs into a
single 32-bit value with no additional packing instructions, storing two SAD
results in one register.

.. literalinclude:: ../../tools/example_codes/intrinsics_cross_arch_arithmetic.hip
   :language: cpp
   :linenos:
   :start-after: [Sphinx sad variants start]
   :end-before: [Sphinx sad variants end]

The low SAD accumulates from ``src0_lo`` against ``src1_lo``; the high SAD
accumulates from ``src0_hi`` against ``src1_hi`` into the same 32-bit output value.
The maximum value of a 4-byte SAD is 4 × 255 = 1020, which fits in 10 bits,
so neither half overflows into the other regardless of the input values.

For the full signatures and parameter details, see
:ref:`sad_u8 <sad-sad-u8>` and :ref:`sad_hi_u8 <sad-sad-hi-u8>` in the
SAD intrinsics reference.

Overlapping SAD with ``qsad_pk_u16_u8``
---------------------------------------

``__builtin_amdgcn_qsad_pk_u16_u8`` computes four overlapping 4-byte SADs from
a single 8-byte reference window in one instruction. The 8-byte source
(``src0``) is treated as a sliding window: the four SADs compare bytes [0:3],
[1:4], [2:5], and [3:6] of the window against the same 4-byte query (``src1``).
The four 16-bit results are packed into the returned 64-bit value, making this
the natural tool for a full search over a scan line where the reference window
advances one byte at a time.

.. literalinclude:: ../../tools/example_codes/intrinsics_cross_arch_arithmetic.hip
   :language: cpp
   :linenos:
   :start-after: [Sphinx qsad start]
   :end-before: [Sphinx qsad end]

Each of the four packed uint16 results can be extracted with a shift and mask:
``(result >> (i * 16)) & 0xffff`` gives the SAD for window offset ``i``. The
third argument is the initial accumulator for all four channels simultaneously.

For the full signature and parameter details, see
:ref:`qsad_pk_u16_u8 <sad-qsad-pk-u16-u8>` in the SAD intrinsics reference.

Integer dot products
====================

The dot product intrinsics compute the inner product of two short integer
vectors in a single instruction, accumulating the result into a wider integer.
Using the intrinsic guarantees the corresponding hardware instruction is
emitted regardless of compiler heuristics, and makes the intent explicit in
the source code.

Unsigned 4-element dot product with ``udot4``
---------------------------------------------

``__builtin_amdgcn_udot4`` computes the dot product of two vectors of four
unsigned bytes, accumulating into a 32-bit unsigned integer. The intrinsic
takes its inputs as packed 32-bit values. The kernel is responsible for packing
the individual bytes before the call. The fourth argument is a Boolean that
negates the result of the dot product before accumulation. Passing ``false``
gives a plain accumulate.

.. literalinclude:: ../../tools/example_codes/intrinsics_cross_arch_arithmetic.hip
   :language: cpp
   :linenos:
   :start-after: [Sphinx udot4 start]
   :end-before: [Sphinx udot4 end]

Each thread packs four consecutive bytes from each input array into a 32-bit value by
using shifts and bitwise OR, then calls the intrinsic with an accumulator of 0. The
packing is the only per-thread overhead; the dot product itself is one
instruction. For signed integer dot products, see :ref:`sdot4 <dot-integer-sdot4>` in the
integer dot product reference.

For the full signature and parameter details, see
:ref:`udot4 <dot-integer-udot4>` in the integer dot product reference.

FP16 dot product
================

The FP16 dot product intrinsic computes the inner product of two pairs of
half-precision values and accumulates into a single-precision float. The
primary motivation is throughput: FP16 multiply-add has higher theoretical
throughput than FP32 (32-bit single-precision) on hardware with dedicated mixed-precision units, and
accumulating into FP32 preserves the dynamic range needed for deep learning
and signal processing workloads.

FP16-to-FP32 accumulation with ``fdot2``
----------------------------------------

``__builtin_amdgcn_fdot2`` computes ``a[0] * b[0] + a[1] * b[1]`` in FP16
precision and accumulates the result into a FP32 accumulator. Inputs are
passed as ``half2`` two-element vectors. The fourth argument negates the dot
product before accumulation; passing ``false`` gives a plain accumulate.

.. literalinclude:: ../../tools/example_codes/intrinsics_cross_arch_arithmetic.hip
   :language: cpp
   :linenos:
   :start-after: [Sphinx fdot2 start]
   :end-before: [Sphinx fdot2 end]

The kernel casts two consecutive ``__fp16`` array elements to a ``half2``
vector via ``reinterpret_cast`` before the call. The intrinsic is most
valuable when the surrounding code is complex enough that the compiler does
not automatically generate the equivalent hardware instruction.

For the full signature and parameter details, see
:ref:`fdot2 <dot-float-fdot2>` in the floating-point dot product reference.

Conversion and packing
======================

The SAD and dot product intrinsics expect their inputs packed as byte fields
within 32-bit values. When your source data is stored as floats, converting and
packing manually would require several steps. ``__builtin_amdgcn_cvt_pk_u8_f32``
handles both in one instruction: it clamps the float to [0, 255], converts it
to an unsigned byte, and inserts it into one of the four byte positions of an
existing destination 32-bit value, leaving the other three bytes unchanged.

Byte packing with ``cvt_pk_u8_f32``
-----------------------------------

``__builtin_amdgcn_cvt_pk_u8_f32`` takes the source float, a compile-time byte
selector (0–3), and the current destination 32-bit value. Four calls, advancing the
selector from 0 to 3 and threading the output of each call into the next, fill
all four byte slots of the output 32-bit value in four instructions with no intermediate
registers.

.. literalinclude:: ../../tools/example_codes/intrinsics_cross_arch_arithmetic.hip
   :language: cpp
   :linenos:
   :start-after: [Sphinx cvt start]
   :end-before: [Sphinx cvt end]

The packed 32-bit value is directly usable as the ``src0`` argument to ``sad_u8`` or
as a component of the packed inputs to ``udot4``, making this intrinsic the
natural bridge between float pipelines and the byte-oriented arithmetic
intrinsics in this chapter.

For the full signature and parameter details, see
:ref:`cvt_pk_u8_f32 <cvt-pk-u8-f32>` in the conversion and packing reference.

**Compile and run:**

.. code-block:: bash

   amdclang++ -O3 -std=c++17 --offload-arch=gfx942 \
       intrinsics_cross_arch_arithmetic.hip -o arithmetic_intrinsics
   ./arithmetic_intrinsics

.. note::

   The example above targets CDNA3 (``gfx942``).  Replace ``--offload-arch``
   with the appropriate target for your GPU --- for example, ``gfx90a`` for
   CDNA2 or ``gfx1200`` for RDNA4.  All intrinsics used in the examples are
   available on every supported architecture.

Naming convention
=================

All hardware arithmetic and packing intrinsics use the prefix
``__builtin_amdgcn_`` followed by an operation name and type suffix:

.. code-block:: text

   __builtin_amdgcn_<operation>[_<qualifier>]_<type>

``operation``
    The core operation (``sad``, ``msad``, ``qsad``, ``udot``, ``sdot``,
    ``sudot``, ``fdot``, ``cvt``).

``qualifier`` (optional)
    A modifier that refines the operation.  ``hi`` places the result in the
    upper half (``sad_hi_u8``).  ``pk`` indicates a pack/unpack conversion
    (``cvt_pk_u8_f32``).  ``sr`` indicates stochastic rounding
    (``cvt_sr_fp8_f32``).

``type``
    The element type.  ``u8`` = unsigned 8-bit, ``u16`` = unsigned 16-bit,
    ``f32`` = FP32, ``f16`` = FP16, ``bf16`` = BF16, ``fp8`` = FP8 (E4M3),
    ``bf8`` = BF8 (E5M2), ``i4`` = 4-bit fixed-point.  When two types appear,
    the first is the input and the second is the output
    (``cvt_pk_u8_f32``: FP32 input, uint8 output).

The trailing digit on dot-product names indicates the number of element pairs
per call: ``udot4`` computes a 4-element dot product, ``udot8`` an 8-element
dot product, ``fdot2`` a 2-element dot product.

.. _arithmetic_packing_intrinsic_reference:

Arithmetic and packing intrinsic reference
==========================================

Each reference page documents the full signature, parameter details, and
architecture support for every intrinsic in that family.

.. list-table::
   :header-rows: 1
   :widths: auto

   * - Family
     - Description
   * - :doc:`Sum of absolute differences <arithmetic-ref/sad-intrinsics>`
     - L1 distance between packed byte vectors (``msad_u8``, ``sad_u8``,
       ``sad_hi_u8``, ``sad_u16``, ``qsad_pk_u16_u8``, ``mqsad_pk_u16_u8``,
       ``mqsad_u32_u8``)
   * - :doc:`Integer dot products <arithmetic-ref/dot-integer-intrinsics>`
     - Inner products of packed integer vectors (``udot4``, ``udot8``,
       ``sdot4``, ``sdot8``, ``sdot2``, ``udot2``, ``sudot4``, ``sudot8``)
   * - :doc:`Floating-point dot products <arithmetic-ref/dot-float-intrinsics>`
     - Inner products of packed FP16, BF16, FP8, and BF8 vectors (``fdot2``,
       ``fdot2_f16_f16``, ``fdot2_bf16_bf16``, ``fdot2_f32_bf16``,
       ``dot4_f32_fp8_*``, ``fdot2c_f32_bf16``)
   * - :doc:`Conversion and packing <arithmetic-ref/conversion-packing-intrinsics>`
     - Float-to-byte, byte-to-float, and pack/unpack conversions (``cvt_pk_*``,
       ``cvt_pkrtz``, ``cvt_pknorm_*``, ``cvt_sr_*``, ``cvt_f32_*``)
