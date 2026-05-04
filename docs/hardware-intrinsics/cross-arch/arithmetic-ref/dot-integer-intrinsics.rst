.. meta::
   :description: Reference for AMD GPU integer dot product intrinsics for INT8
      and INT4 data types, including udot4, sdot4, udot8, sdot8, and architecture availability.
   :keywords: AMD, ROCm, HIP, dot product, integer, udot4, sdot4, INT8, INT4, intrinsics, CDNA, RDNA

.. _dot-integer-intrinsics:

********************************************************************************
Integer dot product intrinsics
********************************************************************************

The integer dot product intrinsics compute inner products of packed integer
vectors in a single instruction, accumulating into a 32-bit result.  They are
the foundation for INT8 and INT4 quantized inference, where weights and
activations are packed as four bytes or eight nibbles per register.

Common parameters
=================

All integer dot product intrinsics share the following parameters:

``src0``, ``src1`` (packed input vectors)
   Packed integer vectors.  The element width and sign interpretation depend on the
   variant: ``udot4``/``sdot4`` pack four 8-bit elements, ``udot8``/``sdot8``
   pack eight 4-bit elements, and ``sdot2``/``udot2`` pack two 16-bit elements.
   The caller is responsible for packing individual values into the 32-bit
   register before the call.

``src2`` (accumulator)
   Initial accumulator value.  The dot product result is added to (or
   subtracted from, if ``neg`` is true) this value.  Pass ``0`` for a fresh
   dot product; pass the result of a previous call to chain multiple
   dot products across a K-dimension loop.

``neg`` (negate before accumulation)
   When ``true``, negates the dot product result before adding to ``src2``.
   Pass ``false`` for a plain accumulate.  Present on all variants.

The ``sudot4``/``sudot8`` variants add per-operand sign control:

``src0_neg``, ``src1_neg`` (sign selectors)
   When ``true``, the corresponding operand is treated as signed; when
   ``false``, it is treated as unsigned.  This allows mixed-sign dot products
   (e.g., signed weights times unsigned activations) without separate
   intrinsics for every sign combination.

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
   * - ``udot4``, ``udot8``
     - Yes
     - Yes
     - Yes
     - Yes
     - Yes
     - Yes
     - Yes
     - Yes
   * - ``sdot4``, ``sdot8``, ``sdot2``, ``udot2``
     - Yes
     - Yes
     - Yes
     - Yes
     - No
     - No
     - No
     - No
   * - ``sudot4``, ``sudot8``
     - No
     - No
     - No
     - No
     - Yes
     - Yes
     - Yes
     - Yes

Intrinsic reference
===================

Each intrinsic's full signature, parameters, and return value are documented
below.

.. _dot-integer-udot4:

``__builtin_amdgcn_udot4``
--------------------------

Signature and parameters for this intrinsic.

.. code-block:: cpp

   unsigned int __builtin_amdgcn_udot4(
       unsigned int src0,
       unsigned int src1,
       unsigned int src2,
       bool         neg);

Computes the dot product of two vectors of four unsigned bytes, accumulating
into a 32-bit unsigned integer.

.. list-table::
   :header-rows: 1
   :widths: auto

   * - Parameter
     - Type
     - Description
   * - ``src0``
     - unsigned int
     - Four packed unsigned bytes: the first input vector.
   * - ``src1``
     - unsigned int
     - Four packed unsigned bytes: the second input vector.
   * - ``src2``
     - unsigned int
     - Accumulator value.
   * - ``neg``
     - bool
     - When ``true``, negates the dot product before accumulation.

**Returns** ``unsigned int`` -- updated accumulator
(:math:`\text{src2} + (-1)^{\text{neg}} \sum_{k=0}^{3} \text{src0}[k] \times \text{src1}[k]`).

.. _dot-integer-udot8:

``__builtin_amdgcn_udot8``
--------------------------

Signature and parameters for this intrinsic.

.. code-block:: cpp

   unsigned int __builtin_amdgcn_udot8(
       unsigned int src0,
       unsigned int src1,
       unsigned int src2,
       bool         neg);

Dot product of eight unsigned nibble (4-bit) pairs, accumulating into a 32-bit
unsigned integer.

.. list-table::
   :header-rows: 1
   :widths: auto

   * - Parameter
     - Type
     - Description
   * - ``src0``
     - unsigned int
     - Eight packed unsigned 4-bit values: the first input vector.
   * - ``src1``
     - unsigned int
     - Eight packed unsigned 4-bit values: the second input vector.
   * - ``src2``
     - unsigned int
     - Accumulator value.
   * - ``neg``
     - bool
     - When ``true``, negates the dot product before accumulation.

**Returns** ``unsigned int`` -- updated accumulator
(:math:`\text{src2} + (-1)^{\text{neg}} \sum_{k=0}^{7} \text{src0}[k] \times \text{src1}[k]`).

.. _dot-integer-sdot4:

``__builtin_amdgcn_sdot4``
--------------------------

Signature and parameters for this intrinsic.

.. code-block:: cpp

   int __builtin_amdgcn_sdot4(
       int  src0,
       int  src1,
       int  src2,
       bool neg);

Dot product of four signed byte pairs, accumulating into a 32-bit signed
integer.

.. list-table::
   :header-rows: 1
   :widths: auto

   * - Parameter
     - Type
     - Description
   * - ``src0``
     - int
     - Four packed signed bytes: the first input vector.
   * - ``src1``
     - int
     - Four packed signed bytes: the second input vector.
   * - ``src2``
     - int
     - Accumulator value.
   * - ``neg``
     - bool
     - When ``true``, negates the dot product before accumulation.

**Returns** ``int`` -- updated accumulator.

.. _dot-integer-sdot8:

``__builtin_amdgcn_sdot8``
--------------------------

Signature and parameters for this intrinsic.

.. code-block:: cpp

   int __builtin_amdgcn_sdot8(
       int  src0,
       int  src1,
       int  src2,
       bool neg);

Dot product of eight signed nibble (4-bit) pairs, accumulating into a 32-bit
signed integer.

.. list-table::
   :header-rows: 1
   :widths: auto

   * - Parameter
     - Type
     - Description
   * - ``src0``
     - int
     - Eight packed signed 4-bit values: the first input vector.
   * - ``src1``
     - int
     - Eight packed signed 4-bit values: the second input vector.
   * - ``src2``
     - int
     - Accumulator value.
   * - ``neg``
     - bool
     - When ``true``, negates the dot product before accumulation.

**Returns** ``int`` -- updated accumulator.

.. _dot-integer-sdot2:

``__builtin_amdgcn_sdot2``
--------------------------

Signature and parameters for this intrinsic.

.. code-block:: cpp

   int __builtin_amdgcn_sdot2(
       short2 src0,
       short2 src1,
       int    src2,
       bool   neg);

Dot product of two signed 16-bit pairs, accumulating into a 32-bit signed
integer.

.. list-table::
   :header-rows: 1
   :widths: auto

   * - Parameter
     - Type
     - Description
   * - ``src0``
     - short2
     - Two-element signed int16 vector: the first input.
   * - ``src1``
     - short2
     - Two-element signed int16 vector: the second input.
   * - ``src2``
     - int
     - Accumulator value.
   * - ``neg``
     - bool
     - When ``true``, negates the dot product before accumulation.

**Returns** ``int`` -- updated accumulator.

.. _dot-integer-udot2:

``__builtin_amdgcn_udot2``
--------------------------

Signature and parameters for this intrinsic.

.. code-block:: cpp

   unsigned int __builtin_amdgcn_udot2(
       ushort2      src0,
       ushort2      src1,
       unsigned int src2,
       bool         neg);

Dot product of two unsigned 16-bit pairs, accumulating into a 32-bit unsigned
integer.

.. list-table::
   :header-rows: 1
   :widths: auto

   * - Parameter
     - Type
     - Description
   * - ``src0``
     - ushort2
     - Two-element unsigned int16 vector: the first input.
   * - ``src1``
     - ushort2
     - Two-element unsigned int16 vector: the second input.
   * - ``src2``
     - unsigned int
     - Accumulator value.
   * - ``neg``
     - bool
     - When ``true``, negates the dot product before accumulation.

**Returns** ``unsigned int`` -- updated accumulator.

.. _dot-integer-sudot4:

``__builtin_amdgcn_sudot4``
---------------------------

Signature and parameters for this intrinsic.

.. code-block:: cpp

   int __builtin_amdgcn_sudot4(
       bool src0_neg,
       int  src0,
       bool src1_neg,
       int  src1,
       int  src2,
       bool neg);

Dot product of four byte pairs with per-operand sign control.  ``src0_neg``
and ``src1_neg`` independently select whether each operand is interpreted as
signed or unsigned, enabling mixed-sign dot products without separate
intrinsics.

.. list-table::
   :header-rows: 1
   :widths: auto

   * - Parameter
     - Type
     - Description
   * - ``src0_neg``
     - bool
     - When ``true``, ``src0`` bytes are treated as signed (int8); when
       ``false``, unsigned (uint8).
   * - ``src0``
     - int
     - Four packed bytes: the first input vector.
   * - ``src1_neg``
     - bool
     - When ``true``, ``src1`` bytes are treated as signed; when ``false``,
       unsigned.
   * - ``src1``
     - int
     - Four packed bytes: the second input vector.
   * - ``src2``
     - int
     - Accumulator value.
   * - ``neg``
     - bool
     - When ``true``, negates the dot product before accumulation.

**Returns** ``int`` -- updated accumulator.

.. _dot-integer-sudot8:

``__builtin_amdgcn_sudot8``
---------------------------

Signature and parameters for this intrinsic.

.. code-block:: cpp

   int __builtin_amdgcn_sudot8(
       bool src0_neg,
       int  src0,
       bool src1_neg,
       int  src1,
       int  src2,
       bool neg);

Dot product of eight nibble pairs with per-operand sign control.  Same
mixed-sign semantics as ``sudot4`` but operating on 4-bit elements.

.. list-table::
   :header-rows: 1
   :widths: auto

   * - Parameter
     - Type
     - Description
   * - ``src0_neg``
     - bool
     - When ``true``, ``src0`` nibbles are treated as signed; when ``false``,
       unsigned.
   * - ``src0``
     - int
     - Eight packed 4-bit values: the first input vector.
   * - ``src1_neg``
     - bool
     - When ``true``, ``src1`` nibbles are treated as signed; when ``false``,
       unsigned.
   * - ``src1``
     - int
     - Eight packed 4-bit values: the second input vector.
   * - ``src2``
     - int
     - Accumulator value.
   * - ``neg``
     - bool
     - When ``true``, negates the dot product before accumulation.

**Returns** ``int`` -- updated accumulator.
