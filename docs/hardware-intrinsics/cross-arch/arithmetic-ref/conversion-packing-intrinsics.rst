.. meta::
   :description: Reference for AMD GPU conversion and packing intrinsics,
      including cvt_pk, cvt_pkrtz, cvt_pknorm, and cvt_sr for FP8/BF8 with architecture availability.
   :keywords: AMD, ROCm, HIP, conversion, packing, cvt_pk, cvt_sr, FP8, BF8, stochastic rounding, intrinsics, CDNA, RDNA

.. _conversion-packing-intrinsics:

********************************************************************************
Conversion and packing intrinsics
********************************************************************************

The conversion and packing intrinsics convert between floating-point and
integer types and pack multiple values into a single 32-bit register.  They
bridge float pipelines with the byte-oriented arithmetic intrinsics (SAD, dot
products) by converting FP32 results to packed bytes, or unpacking reduced-
precision inputs to FP32 for further computation.

Common parameters
=================

Several parameters appear across multiple conversion intrinsics:

``byte_sel`` (byte position selector)
   An integer in the range 0--3 that selects which byte of a 32-bit
   destination to overwrite.  Byte 0 is the least significant.  Used by
   ``cvt_pk_u8_f32`` and the single-element FP8/BF8 conversion intrinsics
   (``cvt_sr_fp8_f32``, ``cvt_sr_bf8_f32``).

``sel`` (half selector)
   A boolean that selects the upper or lower 16-bit half of the destination.
   When ``false``, the lower 16 bits are written; when ``true``, the upper
   16 bits.  Used by the pack-two variants (``cvt_pk_fp8_f32``,
   ``cvt_pk_bf8_f32``) and the stochastic-rounding BF16/FP16 variants
   (``cvt_sr_bf16_f32``, ``cvt_sr_f16_f32``).

``dst`` / ``src0`` (destination register for in-place insertion)
   The current value of the destination register.  The selected byte or
   half is overwritten; the remaining bits are preserved.  This allows
   multiple calls to fill a packed register incrementally without
   extra move instructions.

``rand`` (stochastic rounding seed)
   A 32-bit random value used for stochastic rounding.  The intrinsic
   uses this value to randomly round up or down to the nearest representable
   value in the target format, providing an unbiased estimator that improves
   convergence in training workloads compared to deterministic rounding.
   Used by ``cvt_sr_fp8_f32``, ``cvt_sr_bf8_f32``, ``cvt_sr_bf16_f32``,
   and ``cvt_sr_f16_f32``.

Architecture availability
=========================

The following table summarizes architecture support for each intrinsic.

.. list-table::
   :header-rows: 1
   :widths: auto

   * - Intrinsic
     - CDNA
     - CDNA2
     - CDNA3
     - CDNA4
     - RDNA2
     - RDNA3
     - RDNA3.5
     - RDNA4
   * - ``cvt_pk_u8_f32``, ``cvt_pkrtz``, ``cvt_pk_i16``, ``cvt_pk_u16``,
       ``cvt_off_f32_i4``
     - Yes
     - Yes
     - Yes
     - Yes
     - Yes
     - Yes
     - Yes
     - Yes
   * - ``cvt_pknorm_i16``, ``cvt_pknorm_u16``
     - No
     - No
     - No
     - No
     - Yes
     - Yes
     - Yes
     - Yes
   * - ``cvt_pk_fp8_f32``, ``cvt_pk_bf8_f32``, ``cvt_pk_f32_fp8``,
       ``cvt_pk_f32_bf8``, ``cvt_f32_fp8``, ``cvt_f32_bf8``,
       ``cvt_sr_fp8_f32``, ``cvt_sr_bf8_f32``
     - No
     - No
     - Yes
     - Yes
     - No
     - No
     - No
     - Yes
   * - ``cvt_sr_bf16_f32``, ``cvt_sr_f16_f32``
     - No
     - No
     - No
     - Yes
     - No
     - No
     - No
     - No

Intrinsic reference
===================

Each intrinsic's full signature, parameters, and return value are documented
below.

FP32 to packed integer
----------------------

These intrinsics convert FP32 values to packed unsigned byte representations.

.. _cvt-pk-u8-f32:

``__builtin_amdgcn_cvt_pk_u8_f32``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Signature and parameters for this intrinsic.

.. code-block:: cpp

   unsigned int __builtin_amdgcn_cvt_pk_u8_f32(
       float        src,
       unsigned int byte_sel,
       unsigned int dst);

Clamps a float to [0, 255], converts it to an unsigned byte, and inserts it at
one of the four byte positions of an existing 32-bit value, leaving the other
three bytes unchanged.  Four calls, advancing ``byte_sel`` from 0 to 3 and
threading the output of each call into the next, fill all four byte slots in
four instructions with no intermediate registers.

.. list-table::
   :header-rows: 1
   :widths: auto

   * - Parameter
     - Type
     - Description
   * - ``src``
     - float
     - Source float value.  Clamped to [0, 255] before conversion.
   * - ``byte_sel``
     - unsigned int
     - Byte position selector (0--3).  Selects which byte of ``dst`` to
       overwrite.
   * - ``dst``
     - unsigned int
     - Current destination value.  The selected byte is overwritten; the
       other three bytes are preserved.

**Returns** ``unsigned int`` -- ``dst`` with byte position ``byte_sel``
replaced by the clamped and converted ``src`` value.

FP32 to packed FP16
-------------------

These intrinsics convert pairs of FP32 values to packed FP16 vectors.

.. _cvt-pkrtz:

``__builtin_amdgcn_cvt_pkrtz``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Signature and parameters for this intrinsic.

.. code-block:: cpp

   __half2 __builtin_amdgcn_cvt_pkrtz(
       float src0,
       float src1);

Converts two FP32 values to FP16 using round-toward-zero and packs them into
a two-element vector.  Round-toward-zero is faster than round-to-nearest on
some hardware and is acceptable when the conversion is followed by further
reduced-precision computation.

.. list-table::
   :header-rows: 1
   :widths: auto

   * - Parameter
     - Type
     - Description
   * - ``src0``
     - float
     - First FP32 value to convert (placed in element 0).
   * - ``src1``
     - float
     - Second FP32 value to convert (placed in element 1).

**Returns** ``__half2`` -- two-element FP16 vector containing the converted
values.

Integer packing
---------------

These intrinsics pack pairs of 32-bit integer values into 16-bit vector types.

.. _cvt-pk-i16:

``__builtin_amdgcn_cvt_pk_i16``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Signature and parameters for this intrinsic.

.. code-block:: cpp

   short2 __builtin_amdgcn_cvt_pk_i16(
       int src0,
       int src1);

Packs two int32 values into a two-element int16 vector by taking the lower
16 bits of each input.

.. list-table::
   :header-rows: 1
   :widths: auto

   * - Parameter
     - Type
     - Description
   * - ``src0``
     - int
     - First int32 value (lower 16 bits used).
   * - ``src1``
     - int
     - Second int32 value (lower 16 bits used).

**Returns** ``short2`` -- two-element int16 vector.

.. _cvt-pk-u16:

``__builtin_amdgcn_cvt_pk_u16``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Signature and parameters for this intrinsic.

.. code-block:: cpp

   ushort2 __builtin_amdgcn_cvt_pk_u16(
       unsigned int src0,
       unsigned int src1);

Packs two uint32 values into a two-element uint16 vector by taking the lower
16 bits of each input.

.. list-table::
   :header-rows: 1
   :widths: auto

   * - Parameter
     - Type
     - Description
   * - ``src0``
     - unsigned int
     - First uint32 value (lower 16 bits used).
   * - ``src1``
     - unsigned int
     - Second uint32 value (lower 16 bits used).

**Returns** ``ushort2`` -- two-element uint16 vector.

Fixed-point conversion
----------------------

These intrinsics convert between fixed-point nibble values and FP32.

.. _cvt-off-f32-i4:

``__builtin_amdgcn_cvt_off_f32_i4``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Signature and parameters for this intrinsic.

.. code-block:: cpp

   float __builtin_amdgcn_cvt_off_f32_i4(
       int src);

Converts bits [3:0] of ``src`` as a 4-bit fixed-point fraction to FP32,
mapping nibble value ``n`` to ``n / 16.0``.  Used internally by some
dithering and noise-generation patterns.

.. list-table::
   :header-rows: 1
   :widths: auto

   * - Parameter
     - Type
     - Description
   * - ``src``
     - int
     - Source value; only bits [3:0] are used.

**Returns** ``float`` -- the 4-bit nibble interpreted as a fixed-point
fraction in the range [0, 15/16].

Normalized packing
------------------

These intrinsics convert FP32 values to normalized 16-bit integers and pack them into vectors.

.. _cvt-pknorm-i16:

``__builtin_amdgcn_cvt_pknorm_i16``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Signature and parameters for this intrinsic.

.. code-block:: cpp

   short2 __builtin_amdgcn_cvt_pknorm_i16(
       float src0,
       float src1);

Converts two FP32 values to normalized int16 and packs them into a
two-element vector.  The FP32 inputs are clamped to [-1.0, 1.0] and scaled
to the int16 range [-32768, 32767].

.. list-table::
   :header-rows: 1
   :widths: auto

   * - Parameter
     - Type
     - Description
   * - ``src0``
     - float
     - First FP32 value to convert.
   * - ``src1``
     - float
     - Second FP32 value to convert.

**Returns** ``short2`` -- two-element normalized int16 vector.

.. _cvt-pknorm-u16:

``__builtin_amdgcn_cvt_pknorm_u16``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Signature and parameters for this intrinsic.

.. code-block:: cpp

   ushort2 __builtin_amdgcn_cvt_pknorm_u16(
       float src0,
       float src1);

Converts two FP32 values to normalized uint16 and packs them into a
two-element vector.  The FP32 inputs are clamped to [0.0, 1.0] and scaled
to the uint16 range [0, 65535].

.. list-table::
   :header-rows: 1
   :widths: auto

   * - Parameter
     - Type
     - Description
   * - ``src0``
     - float
     - First FP32 value to convert.
   * - ``src1``
     - float
     - Second FP32 value to convert.

**Returns** ``ushort2`` -- two-element normalized uint16 vector.

FP8 and BF8 pack and unpack
---------------------------

These intrinsics convert between FP32 and 8-bit floating-point formats (FP8 E4M3 and BF8 E5M2).

.. _cvt-pk-fp8-f32:

``__builtin_amdgcn_cvt_pk_fp8_f32``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Signature and parameters for this intrinsic.

.. code-block:: cpp

   int __builtin_amdgcn_cvt_pk_fp8_f32(
       float src0,
       float src1,
       int   dst,
       bool  sel);

Converts two FP32 values to FP8 (E4M3) and packs them into the upper or
lower 16 bits of ``dst`` depending on ``sel``.

.. list-table::
   :header-rows: 1
   :widths: auto

   * - Parameter
     - Type
     - Description
   * - ``src0``
     - float
     - First FP32 value to convert.
   * - ``src1``
     - float
     - Second FP32 value to convert.
   * - ``dst``
     - int
     - Current destination value.  The unselected 16 bits are preserved.
   * - ``sel``
     - bool
     - When ``false``, writes to the lower 16 bits; when ``true``, the upper
       16 bits.

**Returns** ``int`` -- ``dst`` with the selected half replaced by the two
packed FP8 values.

.. _cvt-pk-bf8-f32:

``__builtin_amdgcn_cvt_pk_bf8_f32``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Signature and parameters for this intrinsic.

.. code-block:: cpp

   int __builtin_amdgcn_cvt_pk_bf8_f32(
       float src0,
       float src1,
       int   dst,
       bool  sel);

Converts two FP32 values to BF8 (E5M2) and packs them into the upper or
lower 16 bits of ``dst`` depending on ``sel``.

.. list-table::
   :header-rows: 1
   :widths: auto

   * - Parameter
     - Type
     - Description
   * - ``src0``
     - float
     - First FP32 value to convert.
   * - ``src1``
     - float
     - Second FP32 value to convert.
   * - ``dst``
     - int
     - Current destination value.  The unselected 16 bits are preserved.
   * - ``sel``
     - bool
     - When ``false``, writes to the lower 16 bits; when ``true``, the upper
       16 bits.

**Returns** ``int`` -- ``dst`` with the selected half replaced by the two
packed BF8 values.

.. _cvt-pk-f32-fp8:

``__builtin_amdgcn_cvt_pk_f32_fp8``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Signature and parameters for this intrinsic.

.. code-block:: cpp

   float2 __builtin_amdgcn_cvt_pk_f32_fp8(
       int  src,
       bool sel);

Unpacks two FP8 (E4M3) bytes from the upper or lower 16 bits of ``src``
and converts them to two FP32 values.

.. list-table::
   :header-rows: 1
   :widths: auto

   * - Parameter
     - Type
     - Description
   * - ``src``
     - int
     - Packed source containing four FP8 values.
   * - ``sel``
     - bool
     - When ``false``, reads from the lower 16 bits; when ``true``, the upper
       16 bits.

**Returns** ``float2`` -- two FP32 values converted from the selected FP8
pair.

.. _cvt-pk-f32-bf8:

``__builtin_amdgcn_cvt_pk_f32_bf8``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Signature and parameters for this intrinsic.

.. code-block:: cpp

   float2 __builtin_amdgcn_cvt_pk_f32_bf8(
       int  src,
       bool sel);

Unpacks two BF8 (E5M2) bytes from the upper or lower 16 bits of ``src``
and converts them to two FP32 values.

.. list-table::
   :header-rows: 1
   :widths: auto

   * - Parameter
     - Type
     - Description
   * - ``src``
     - int
     - Packed source containing four BF8 values.
   * - ``sel``
     - bool
     - When ``false``, reads from the lower 16 bits; when ``true``, the upper
       16 bits.

**Returns** ``float2`` -- two FP32 values converted from the selected BF8
pair.

.. _cvt-f32-fp8:

``__builtin_amdgcn_cvt_f32_fp8``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Signature and parameters for this intrinsic.

.. code-block:: cpp

   float __builtin_amdgcn_cvt_f32_fp8(
       int src,
       int byte_sel);

Unpacks one FP8 (E4M3) byte at position ``byte_sel`` from ``src`` and
converts it to FP32.

.. list-table::
   :header-rows: 1
   :widths: auto

   * - Parameter
     - Type
     - Description
   * - ``src``
     - int
     - Packed source containing four FP8 values.
   * - ``byte_sel``
     - int
     - Byte position selector (0--3).

**Returns** ``float`` -- the selected FP8 value converted to FP32.

.. _cvt-f32-bf8:

``__builtin_amdgcn_cvt_f32_bf8``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Signature and parameters for this intrinsic.

.. code-block:: cpp

   float __builtin_amdgcn_cvt_f32_bf8(
       int src,
       int byte_sel);

Unpacks one BF8 (E5M2) byte at position ``byte_sel`` from ``src`` and
converts it to FP32.

.. list-table::
   :header-rows: 1
   :widths: auto

   * - Parameter
     - Type
     - Description
   * - ``src``
     - int
     - Packed source containing four BF8 values.
   * - ``byte_sel``
     - int
     - Byte position selector (0--3).

**Returns** ``float`` -- the selected BF8 value converted to FP32.

Stochastic rounding conversions
-------------------------------

These intrinsics convert FP32 values to reduced-precision formats using stochastic rounding.

.. _cvt-sr-fp8-f32:

``__builtin_amdgcn_cvt_sr_fp8_f32``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Signature and parameters for this intrinsic.

.. code-block:: cpp

   int __builtin_amdgcn_cvt_sr_fp8_f32(
       float src,
       int   rand,
       int   dst,
       int   byte_sel);

Converts FP32 to FP8 (E4M3) with stochastic rounding and inserts the result
at byte position ``byte_sel`` of ``dst``.

.. list-table::
   :header-rows: 1
   :widths: auto

   * - Parameter
     - Type
     - Description
   * - ``src``
     - float
     - FP32 value to convert.
   * - ``rand``
     - int
     - Random seed for stochastic rounding.
   * - ``dst``
     - int
     - Current destination value.  The selected byte is overwritten.
   * - ``byte_sel``
     - int
     - Byte position selector (0--3).

**Returns** ``int`` -- ``dst`` with byte position ``byte_sel`` replaced by
the FP8 value produced by stochastic rounding.

.. _cvt-sr-bf8-f32:

``__builtin_amdgcn_cvt_sr_bf8_f32``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Signature and parameters for this intrinsic.

.. code-block:: cpp

   int __builtin_amdgcn_cvt_sr_bf8_f32(
       float src,
       int   rand,
       int   dst,
       int   byte_sel);

Converts FP32 to BF8 (E5M2) with stochastic rounding and inserts the result
at byte position ``byte_sel`` of ``dst``.

.. list-table::
   :header-rows: 1
   :widths: auto

   * - Parameter
     - Type
     - Description
   * - ``src``
     - float
     - FP32 value to convert.
   * - ``rand``
     - int
     - Random seed for stochastic rounding.
   * - ``dst``
     - int
     - Current destination value.  The selected byte is overwritten.
   * - ``byte_sel``
     - int
     - Byte position selector (0--3).

**Returns** ``int`` -- ``dst`` with byte position ``byte_sel`` replaced by
the BF8 value produced by stochastic rounding.

.. _cvt-sr-bf16-f32:

``__builtin_amdgcn_cvt_sr_bf16_f32``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Signature and parameters for this intrinsic.

.. code-block:: cpp

   __hip_bfloat162 __builtin_amdgcn_cvt_sr_bf16_f32(
       __hip_bfloat162 src0,
       float           src1,
       unsigned int    rand,
       bool            sel);

Converts FP32 to BF16 with stochastic rounding and inserts the result into
the lane of ``src0`` selected by ``sel``.

.. list-table::
   :header-rows: 1
   :widths: auto

   * - Parameter
     - Type
     - Description
   * - ``src0``
     - __hip_bfloat162
     - Current two-element BF16 vector.  The unselected lane is preserved.
   * - ``src1``
     - float
     - FP32 value to convert.
   * - ``rand``
     - unsigned int
     - Random seed for stochastic rounding.
   * - ``sel``
     - bool
     - When ``false``, writes to element 0; when ``true``, element 1.

**Returns** ``__hip_bfloat162`` -- ``src0`` with the selected element replaced
by the BF16 value produced by stochastic rounding.

.. _cvt-sr-f16-f32:

``__builtin_amdgcn_cvt_sr_f16_f32``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Signature and parameters for this intrinsic.

.. code-block:: cpp

   half2 __builtin_amdgcn_cvt_sr_f16_f32(
       half2        src0,
       float        src1,
       unsigned int rand,
       bool         sel);

Converts FP32 to FP16 with stochastic rounding and inserts the result into
the lane of ``src0`` selected by ``sel``.

.. list-table::
   :header-rows: 1
   :widths: auto

   * - Parameter
     - Type
     - Description
   * - ``src0``
     - half2
     - Current two-element FP16 vector.  The unselected lane is preserved.
   * - ``src1``
     - float
     - FP32 value to convert.
   * - ``rand``
     - unsigned int
     - Random seed for stochastic rounding.
   * - ``sel``
     - bool
     - When ``false``, writes to element 0; when ``true``, element 1.

**Returns** ``half2`` -- ``src0`` with the selected element replaced by the
FP16 value produced by stochastic rounding.
