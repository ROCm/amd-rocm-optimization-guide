.. meta::
  :description: Use HIP arithmetic and packing intrinsics for AMD GPU optimization, including SAD, integer dot products, FP16 accumulation, and float-to-byte type conversion.
  :keywords: AMD, ROCm, HIP, hardware intrinsics, SAD, dot product, arithmetic intrinsics, packing, CDNA, RDNA

.. _arithmetic_packing_intrinsics:

***********************************************
Arithmetic and packing intrinsics for AMD GPUs
***********************************************

Arithmetic and packing intrinsics give you direct access to dedicated units for
the sum of absolute differences, integer dot products, FP16 (16-bit half-precision) dot products, and type
conversion and packing. Most intrinsics noted in this topic are available across all AMD
Instinct (CDNA) and AMD Radeon (RDNA) architectures. Those available only on specific architectures are noted in the
Supported architecture column of the reference tables. For a complete listing of
all intrinsics with their signatures and supported architectures, see
:ref:`arithmetic_packing_intrinsic_reference`.

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
instruction. For signed integer dot products, see ``__builtin_amdgcn_sdot4``
in the reference table below.

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

.. _arithmetic_packing_intrinsic_reference:

Arithmetic and packing intrinsic reference
==========================================

The following tables list all arithmetic and packing intrinsics with their signatures,
descriptions, and supported architectures.

Sum of absolute differences
---------------------------

The following intrinsics compute the L1 distance between byte vectors.

.. list-table::
   :header-rows: 1
   :widths: 50 30 20

   * - Signature
     - Description
     - Supported architecture
   * - | ``unsigned int __builtin_amdgcn_msad_u8(``
       | ``unsigned int src0, unsigned int src1,``
       | ``unsigned int src2)``
     - SAD of four packed bytes, skipping positions where the ``src1`` byte is zero; accumulates into ``src2``
     - All
   * - | ``unsigned int __builtin_amdgcn_sad_u8(``
       | ``unsigned int src0, unsigned int src1,``
       | ``unsigned int src2)``
     - SAD of four packed bytes accumulated into bits [15:0] of ``src2``
     - All
   * - | ``unsigned int __builtin_amdgcn_sad_hi_u8(``
       | ``unsigned int src0, unsigned int src1,``
       | ``unsigned int src2)``
     - SAD of four packed bytes, each difference shifted left 16, accumulated into bits [31:16] of ``src2``
     - All
   * - | ``unsigned int __builtin_amdgcn_sad_u16(``
       | ``unsigned int src0, unsigned int src1,``
       | ``unsigned int src2)``
     - SAD of two packed uint16 values accumulated into ``src2``
     - All
   * - | ``uint64_t __builtin_amdgcn_qsad_pk_u16_u8(``
       | ``uint64_t src0, unsigned int src1,``
       | ``uint64_t src2)``
     - Four overlapping 4-byte SADs from an 8-byte window; results packed as four uint16 values
     - All
   * - | ``uint64_t __builtin_amdgcn_mqsad_pk_u16_u8(``
       | ``uint64_t src0, unsigned int src1,``
       | ``uint64_t src2)``
     - Masked variant of ``qsad_pk_u16_u8``; skips positions where the ``src1`` byte is zero
     - All
   * - | ``uint4 __builtin_amdgcn_mqsad_u32_u8(``
       | ``uint64_t src0, unsigned int src1,``
       | ``uint4 src2)``
     - Four overlapping 4-byte SADs returned as four uint32 values
     - All

Integer dot products
--------------------

The following intrinsics compute inner products of packed integer vectors in a single
instruction.

.. list-table::
   :header-rows: 1
   :widths: 50 30 20

   * - Signature
     - Description
     - Supported architecture
   * - | ``unsigned int __builtin_amdgcn_udot4(``
       | ``unsigned int src0, unsigned int src1,``
       | ``unsigned int src2, bool neg)``
     - Dot product of four unsigned byte pairs; accumulates into ``src2``; ``neg`` negates the dot product before accumulation
     - gfx9xx, gfx10xx, gfx11xx, gfx12xx
   * - | ``unsigned int __builtin_amdgcn_udot8(``
       | ``unsigned int src0, unsigned int src1,``
       | ``unsigned int src2, bool neg)``
     - Dot product of eight unsigned nibble pairs; accumulates into ``src2``
     - gfx9xx, gfx10xx, gfx11xx, gfx12xx
   * - | ``int __builtin_amdgcn_sdot4(``
       | ``int src0, int src1, int src2, bool neg)``
     - Dot product of four signed byte pairs; accumulates into ``src2``
     - gfx9xx
   * - | ``int __builtin_amdgcn_sdot8(``
       | ``int src0, int src1, int src2, bool neg)``
     - Dot product of eight signed nibble pairs; accumulates into ``src2``
     - gfx9xx
   * - | ``int __builtin_amdgcn_sdot2(``
       | ``short2 src0, short2 src1,``
       | ``int src2, bool neg)``
     - Dot product of two signed int16 pairs; accumulates into ``src2``
     - gfx9xx
   * - | ``unsigned int __builtin_amdgcn_udot2(``
       | ``ushort2 src0, ushort2 src1,``
       | ``unsigned int src2, bool neg)``
     - Dot product of two unsigned int16 pairs; accumulates into ``src2``
     - gfx9xx
   * - | ``int __builtin_amdgcn_sudot4(``
       | ``bool src0_neg, int src0,``
       | ``bool src1_neg, int src1,``
       | ``int src2, bool neg)``
     - Dot product of four byte pairs with per-operand sign control; ``src0_neg`` and ``src1_neg`` select signed or unsigned interpretation independently
     - gfx10xx, gfx11xx, gfx12xx
   * - | ``int __builtin_amdgcn_sudot8(``
       | ``bool src0_neg, int src0,``
       | ``bool src1_neg, int src1,``
       | ``int src2, bool neg)``
     - Dot product of eight nibble pairs with per-operand sign control
     - gfx10xx, gfx11xx, gfx12xx

Floating-point dot products
---------------------------

The following intrinsics compute inner products of floating-point vectors, accumulating
into FP16, BF16, or FP32.

.. list-table::
   :header-rows: 1
   :widths: 50 30 20

   * - Signature
     - Description
     - Supported architecture
   * - | ``float __builtin_amdgcn_fdot2(``
       | ``half2 src0, half2 src1,``
       | ``float src2, bool neg)``
     - Dot product of two FP16 pairs accumulated into FP32; ``neg`` negates the dot product before accumulation
     - gfx9xx, gfx10xx, gfx11xx, gfx12xx
   * - | ``__half __builtin_amdgcn_fdot2_f16_f16(``
       | ``half2 src0, half2 src1, __half src2)``
     - Dot product of two FP16 pairs accumulated into FP16
     - gfx11xx, gfx12xx
   * - | ``short __builtin_amdgcn_fdot2_bf16_bf16(``
       | ``short2 src0, short2 src1, short src2)``
     - Dot product of two BF16 pairs accumulated into BF16
     - gfx11xx, gfx12xx
   * - | ``float __builtin_amdgcn_fdot2_f32_bf16(``
       | ``short2 src0, short2 src1,``
       | ``float src2, bool neg)``
     - Dot product of two BF16 pairs accumulated into FP32; ``neg`` negates the dot product before accumulation
     - gfx11xx, gfx12xx
   * - | ``float __builtin_amdgcn_dot4_f32_fp8_fp8(``
       | ``unsigned int src0, unsigned int src1,``
       | ``float src2)``
     - Dot product of four FP8 (E4M3) pairs accumulated into FP32
     - gfx12xx
   * - | ``float __builtin_amdgcn_dot4_f32_bf8_bf8(``
       | ``unsigned int src0, unsigned int src1,``
       | ``float src2)``
     - Dot product of four BF8 (E5M2) pairs accumulated into FP32
     - gfx12xx
   * - | ``float __builtin_amdgcn_dot4_f32_fp8_bf8(``
       | ``unsigned int src0, unsigned int src1,``
       | ``float src2)``
     - Dot product of four pairs with FP8 (E4M3) ``src0`` and BF8 (E5M2) ``src1``, accumulated into FP32
     - gfx12xx
   * - | ``float __builtin_amdgcn_dot4_f32_bf8_fp8(``
       | ``unsigned int src0, unsigned int src1,``
       | ``float src2)``
     - Dot product of four pairs with BF8 (E5M2) ``src0`` and FP8 (E4M3) ``src1``, accumulated into FP32
     - gfx12xx
   * - | ``float __builtin_amdgcn_fdot2c_f32_bf16(``
       | ``__hip_bfloat162  src0,``
       | ``__hip_bfloat162  src1,``
       | ``float src2, bool neg)``
     - Dot product of two BF16 pairs accumulated into FP32; ``neg`` negates the dot product before accumulation
     - gfx950

Conversion and packing
----------------------

The following intrinsics convert between floating-point and integer types and pack
multiple values into a single 32-bit value, bridging float pipelines with the
byte-oriented arithmetic intrinsics above.

.. list-table::
   :header-rows: 1
   :widths: 50 30 20

   * - Signature
     - Description
     - Supported architecture
   * - | ``unsigned int __builtin_amdgcn_cvt_pk_u8_f32(``
       | ``float src, unsigned int byte_sel,``
       | ``unsigned int dst)``
     - Clamp ``src`` to [0, 255], convert to uint8, and insert at byte position ``byte_sel`` of ``dst``
     - All
   * - | ``__half2 __builtin_amdgcn_cvt_pkrtz(``
       | ``float src0, float src1)``
     - Convert two FP32 values to FP16 using round-toward-zero and pack into a two-element vector
     - All
   * - | ``short2 __builtin_amdgcn_cvt_pk_i16(``
       | ``int src0, int src1)``
     - Pack two int32 values into a two-element int16 vector
     - All
   * - | ``ushort2 __builtin_amdgcn_cvt_pk_u16(``
       | ``unsigned int src0, unsigned int src1)``
     - Pack two uint32 values into a two-element uint16 vector
     - All
   * - | ``float __builtin_amdgcn_cvt_off_f32_i4(``
       | ``int src)``
     - Convert bits [3:0] of ``src`` as a 4-bit fixed-point fraction to FP32; maps nibble ``n`` to ``n / 16.0``
     - All
   * - | ``short2 __builtin_amdgcn_cvt_pknorm_i16(``
       | ``float src0, float src1)``
     - Convert two FP32 values to normalized int16 and pack into a two-element vector
     - gfx10xx, gfx11xx, gfx12xx
   * - | ``ushort2 __builtin_amdgcn_cvt_pknorm_u16(``
       | ``float src0, float src1)``
     - Convert two FP32 values to normalized uint16 and pack into a two-element vector
     - gfx10xx, gfx11xx, gfx12xx
   * - | ``int __builtin_amdgcn_cvt_pk_fp8_f32(``
       | ``float src0, float src1, int dst, bool sel)``
     - Convert two FP32 values to FP8 (E4M3) and pack into the upper or lower 16 bits of ``dst`` depending on ``sel``
     - gfx940, gfx12xx
   * - | ``int __builtin_amdgcn_cvt_pk_bf8_f32(``
       | ``float src0, float src1, int dst, bool sel)``
     - Convert two FP32 values to BF8 (E5M2) and pack into the upper or lower 16 bits of ``dst`` depending on ``sel``
     - gfx940, gfx12xx
   * - | ``float2 __builtin_amdgcn_cvt_pk_f32_fp8(``
       | ``int src, bool sel)``
     - Unpack two FP8 (E4M3) bytes from the upper or lower 16 bits of ``src`` to two FP32 values
     - gfx940, gfx12xx
   * - | ``float2 __builtin_amdgcn_cvt_pk_f32_bf8(``
       | ``int src, bool sel)``
     - Unpack two BF8 (E5M2) bytes from the upper or lower 16 bits of ``src`` to two FP32 values
     - gfx940, gfx12xx
   * - | ``float __builtin_amdgcn_cvt_f32_fp8(``
       | ``int src, int byte_sel)``
     - Unpack one FP8 (E4M3) byte at position ``byte_sel`` from ``src`` to FP32
     - gfx940, gfx12xx
   * - | ``float __builtin_amdgcn_cvt_f32_bf8(``
       | ``int src, int byte_sel)``
     - Unpack one BF8 (E5M2) byte at position ``byte_sel`` from ``src`` to FP32
     - gfx940, gfx12xx
   * - | ``int __builtin_amdgcn_cvt_sr_fp8_f32(``
       | ``float src, int rand,``
       | ``int dst, int byte_sel)``
     - Convert FP32 to FP8 (E4M3) with stochastic rounding using ``rand`` and insert at byte position ``byte_sel`` of ``dst``
     - gfx940, gfx12xx
   * - | ``int __builtin_amdgcn_cvt_sr_bf8_f32(``
       | ``float src, int rand,``
       | ``int dst, int byte_sel)``
     - Convert FP32 to BF8 (E5M2) with stochastic rounding using ``rand`` and insert at byte position ``byte_sel`` of ``dst``
     - gfx940, gfx12xx
   * - | ``__hip_bfloat162  __builtin_amdgcn_cvt_sr_bf16_f32(``
       | ``__hip_bfloat162  src0, float src1,``
       | ``unsigned int rand, bool sel)``
     - Convert FP32 to BF16 with stochastic rounding using ``rand`` and insert into the lane of ``src0`` selected by ``sel``
     - gfx950
   * - | ``half2 __builtin_amdgcn_cvt_sr_f16_f32(``
       | ``half2 src0, float src1,``
       | ``unsigned int rand, bool sel)``
     - Convert FP32 to FP16 with stochastic rounding using ``rand`` and insert into the lane of ``src0`` selected by ``sel``
     - gfx950
