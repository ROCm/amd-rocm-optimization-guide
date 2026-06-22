.. meta::
   :description: Reference for AMD GPU floating-point dot product builtins,
      including fdot2 for FP16, BF16, FP8, and BF8 types with usage and architecture availability.
   :keywords: AMD, ROCm, HIP, dot product, floating-point, fdot2, FP16, BF16, FP8, BF8, builtins, CDNA, RDNA

.. _dot-float-builtins:

********************************************************************************
Floating-point dot product builtins
********************************************************************************

The floating-point dot product builtins compute inner products of packed
floating-point vectors, accumulating into FP16, BF16, or FP32.  They provide
higher throughput than scalar multiply-add sequences on hardware with dedicated
mixed-precision units, and are used in deep learning inference, signal
processing, and any workload that benefits from reduced-precision computation
with full-precision accumulation.

Common parameters
=================

Most floating-point dot product builtins share the following parameters:

``src0``, ``src1`` (packed input vectors)
   Packed floating-point vectors.  The element type depends on the variant:
   ``half2`` for FP16, ``short2`` for BF16 (storage type), or ``unsigned int``
   for FP8/BF8 (four packed 8-bit floats).

``src2`` (accumulator)
   Initial accumulator value.  The dot product result is added to (or
   subtracted from, if ``neg`` is present and true) this value.  The
   accumulator type matches the output type: ``float`` for FP32-accumulate
   variants, ``__half`` for FP16-accumulate, ``short`` for BF16-accumulate.

``neg`` (negate before accumulation)
   When ``true``, negates the dot product result before adding to ``src2``.
   Pass ``false`` for a plain accumulate.  Present on ``fdot2``,
   ``fdot2_f32_bf16``, and ``fdot2c_f32_bf16``.  Not present on the
   same-type-accumulate variants (``fdot2_f16_f16``, ``fdot2_bf16_bf16``)
   or the FP8/BF8 variants.

Architecture availability
=========================

The following table summarizes architecture support for each builtin.

.. list-table::
   :header-rows: 1
   :widths: auto

   * - Builtin
     - CDNA
     - CDNA2
     - CDNA3
     - CDNA4
     - RDNA2
     - RDNA3
     - RDNA3.5
     - RDNA4
   * - ``fdot2``
     - Yes
     - Yes
     - Yes
     - Yes
     - Yes
     - Yes
     - Yes
     - Yes
   * - ``fdot2_f16_f16``, ``fdot2_bf16_bf16``
     - No
     - No
     - No
     - No
     - No
     - Yes
     - Yes
     - Yes
   * - ``fdot2_f32_bf16``
     - No
     - No
     - No
     - No
     - No
     - Yes
     - Yes
     - Yes
   * - ``dot4_f32_fp8_fp8``, ``dot4_f32_bf8_bf8``, ``dot4_f32_fp8_bf8``,
       ``dot4_f32_bf8_fp8``
     - No
     - No
     - No
     - No
     - No
     - No
     - No
     - Yes
   * - ``fdot2c_f32_bf16``
     - No
     - No
     - No
     - Yes
     - No
     - No
     - No
     - No

Builtin reference
===================

Each builtin's full signature, parameters, and return value are documented
below.

.. _dot-float-fdot2:

``__builtin_amdgcn_fdot2``
--------------------------

Signature and parameters for this builtin.

.. code-block:: cpp

   float __builtin_amdgcn_fdot2(
       half2 src0,
       half2 src1,
       float src2,
       bool  neg);

Computes the dot product of two pairs of FP16 values and accumulates the
result into a single-precision FP32 accumulator.  The FP16 multiply-add has
higher throughput than FP32 on hardware with dedicated mixed-precision units,
while accumulating into FP32 preserves the dynamic range needed for deep
learning and signal processing workloads.

.. list-table::
   :header-rows: 1
   :widths: auto

   * - Parameter
     - Type
     - Description
   * - ``src0``
     - half2
     - Two-element FP16 vector: the first input.
   * - ``src1``
     - half2
     - Two-element FP16 vector: the second input.
   * - ``src2``
     - float
     - FP32 accumulator value.
   * - ``neg``
     - bool
     - When ``true``, negates the dot product before accumulation.

**Returns** ``float`` -- updated accumulator
(:math:`\text{src2} + (-1)^{\text{neg}} (\text{src0}[0] \times \text{src1}[0] + \text{src0}[1] \times \text{src1}[1])`).

.. _dot-float-fdot2-f16-f16:

``__builtin_amdgcn_fdot2_f16_f16``
----------------------------------

Signature and parameters for this builtin.

.. code-block:: cpp

   __half __builtin_amdgcn_fdot2_f16_f16(
       half2  src0,
       half2  src1,
       __half src2);

Dot product of two FP16 pairs accumulated into FP16.  Both computation and
accumulation use FP16 precision, avoiding the cost of FP32 accumulation when
the reduced dynamic range is acceptable.

.. list-table::
   :header-rows: 1
   :widths: auto

   * - Parameter
     - Type
     - Description
   * - ``src0``
     - half2
     - Two-element FP16 vector: the first input.
   * - ``src1``
     - half2
     - Two-element FP16 vector: the second input.
   * - ``src2``
     - __half
     - FP16 accumulator value.

**Returns** ``__half`` -- updated FP16 accumulator.

.. _dot-float-fdot2-bf16-bf16:

``__builtin_amdgcn_fdot2_bf16_bf16``
------------------------------------

Signature and parameters for this builtin.

.. code-block:: cpp

   short __builtin_amdgcn_fdot2_bf16_bf16(
       short2 src0,
       short2 src1,
       short  src2);

Dot product of two BF16 pairs accumulated into BF16.  Uses ``short``/``short2``
as the storage type for BF16 values.

.. list-table::
   :header-rows: 1
   :widths: auto

   * - Parameter
     - Type
     - Description
   * - ``src0``
     - short2
     - Two-element BF16 vector (storage type): the first input.
   * - ``src1``
     - short2
     - Two-element BF16 vector (storage type): the second input.
   * - ``src2``
     - short
     - BF16 accumulator value (storage type).

**Returns** ``short`` -- updated BF16 accumulator (storage type).

.. _dot-float-fdot2-f32-bf16:

``__builtin_amdgcn_fdot2_f32_bf16``
-----------------------------------

Signature and parameters for this builtin.

.. code-block:: cpp

   float __builtin_amdgcn_fdot2_f32_bf16(
       short2 src0,
       short2 src1,
       float  src2,
       bool   neg);

Dot product of two BF16 pairs accumulated into FP32.  Combines the throughput
advantage of BF16 inputs with the dynamic range of FP32 accumulation.

.. list-table::
   :header-rows: 1
   :widths: auto

   * - Parameter
     - Type
     - Description
   * - ``src0``
     - short2
     - Two-element BF16 vector (storage type): the first input.
   * - ``src1``
     - short2
     - Two-element BF16 vector (storage type): the second input.
   * - ``src2``
     - float
     - FP32 accumulator value.
   * - ``neg``
     - bool
     - When ``true``, negates the dot product before accumulation.

**Returns** ``float`` -- updated FP32 accumulator.

.. _dot-float-dot4-f32-fp8-fp8:

``__builtin_amdgcn_dot4_f32_fp8_fp8``
-------------------------------------

Signature and parameters for this builtin.

.. code-block:: cpp

   float __builtin_amdgcn_dot4_f32_fp8_fp8(
       unsigned int src0,
       unsigned int src1,
       float        src2);

Dot product of four FP8 (E4M3) pairs accumulated into FP32.  Each 32-bit
input packs four 8-bit E4M3 floating-point values.

.. list-table::
   :header-rows: 1
   :widths: auto

   * - Parameter
     - Type
     - Description
   * - ``src0``
     - unsigned int
     - Four packed FP8 (E4M3) values: the first input vector.
   * - ``src1``
     - unsigned int
     - Four packed FP8 (E4M3) values: the second input vector.
   * - ``src2``
     - float
     - FP32 accumulator value.

**Returns** ``float`` -- updated FP32 accumulator.

.. _dot-float-dot4-f32-bf8-bf8:

``__builtin_amdgcn_dot4_f32_bf8_bf8``
-------------------------------------

Signature and parameters for this builtin.

.. code-block:: cpp

   float __builtin_amdgcn_dot4_f32_bf8_bf8(
       unsigned int src0,
       unsigned int src1,
       float        src2);

Dot product of four BF8 (E5M2) pairs accumulated into FP32.

.. list-table::
   :header-rows: 1
   :widths: auto

   * - Parameter
     - Type
     - Description
   * - ``src0``
     - unsigned int
     - Four packed BF8 (E5M2) values: the first input vector.
   * - ``src1``
     - unsigned int
     - Four packed BF8 (E5M2) values: the second input vector.
   * - ``src2``
     - float
     - FP32 accumulator value.

**Returns** ``float`` -- updated FP32 accumulator.

.. _dot-float-dot4-f32-fp8-bf8:

``__builtin_amdgcn_dot4_f32_fp8_bf8``
-------------------------------------

Signature and parameters for this builtin.

.. code-block:: cpp

   float __builtin_amdgcn_dot4_f32_fp8_bf8(
       unsigned int src0,
       unsigned int src1,
       float        src2);

Dot product of four pairs with FP8 (E4M3) ``src0`` and BF8 (E5M2) ``src1``,
accumulated into FP32.

.. list-table::
   :header-rows: 1
   :widths: auto

   * - Parameter
     - Type
     - Description
   * - ``src0``
     - unsigned int
     - Four packed FP8 (E4M3) values: the first input vector.
   * - ``src1``
     - unsigned int
     - Four packed BF8 (E5M2) values: the second input vector.
   * - ``src2``
     - float
     - FP32 accumulator value.

**Returns** ``float`` -- updated FP32 accumulator.

.. _dot-float-dot4-f32-bf8-fp8:

``__builtin_amdgcn_dot4_f32_bf8_fp8``
-------------------------------------

Signature and parameters for this builtin.

.. code-block:: cpp

   float __builtin_amdgcn_dot4_f32_bf8_fp8(
       unsigned int src0,
       unsigned int src1,
       float        src2);

Dot product of four pairs with BF8 (E5M2) ``src0`` and FP8 (E4M3) ``src1``,
accumulated into FP32.

.. list-table::
   :header-rows: 1
   :widths: auto

   * - Parameter
     - Type
     - Description
   * - ``src0``
     - unsigned int
     - Four packed BF8 (E5M2) values: the first input vector.
   * - ``src1``
     - unsigned int
     - Four packed FP8 (E4M3) values: the second input vector.
   * - ``src2``
     - float
     - FP32 accumulator value.

**Returns** ``float`` -- updated FP32 accumulator.

.. _dot-float-fdot2c-f32-bf16:

``__builtin_amdgcn_fdot2c_f32_bf16``
------------------------------------

Signature and parameters for this builtin.

.. code-block:: cpp

   float __builtin_amdgcn_fdot2c_f32_bf16(
       __hip_bfloat162 src0,
       __hip_bfloat162 src1,
       float           src2,
       bool            neg);

Dot product of two BF16 pairs accumulated into FP32, using the HIP
``__hip_bfloat162`` type for inputs.  This variant uses the native BF16 type
rather than the ``short2`` storage type used by ``fdot2_f32_bf16``.

.. list-table::
   :header-rows: 1
   :widths: auto

   * - Parameter
     - Type
     - Description
   * - ``src0``
     - __hip_bfloat162
     - Two-element BF16 vector: the first input.
   * - ``src1``
     - __hip_bfloat162
     - Two-element BF16 vector: the second input.
   * - ``src2``
     - float
     - FP32 accumulator value.
   * - ``neg``
     - bool
     - When ``true``, negates the dot product before accumulation.

**Returns** ``float`` -- updated FP32 accumulator.
