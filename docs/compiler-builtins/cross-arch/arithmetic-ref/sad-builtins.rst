.. meta::
   :description: Reference for AMD GPU sum of absolute differences builtins,
      including msad_u8, sad_u8, sad_hi_u8, and qsad with usage and architecture availability.
   :keywords: AMD, ROCm, HIP, SAD, sum of absolute differences, msad_u8, sad_u8, builtins, CDNA, RDNA

.. _sad-builtins:

********************************************************************************
Sum of absolute differences builtins
********************************************************************************

The SAD builtins compute the L1 distance between packed byte vectors in a
single instruction.  They are used in motion estimation, template matching,
and image comparison kernels where many small absolute differences must be
accumulated quickly.

Common parameters
=================

All SAD builtins share the same three-operand pattern:

``src0`` (reference block)
   Four packed unsigned bytes (or eight bytes for the ``qsad`` and ``mqsad``
   variants).  This is the data being compared against a query.

``src1`` (query block)
   Four packed unsigned bytes representing the query.  For ``msad_u8`` and
   ``mqsad_pk_u16_u8``, byte positions where ``src1`` is zero are excluded
   from the accumulation --- a masking behavior designed for padded reference
   windows where positions outside the frame boundary are marked with zero.

``src2`` (accumulator)
   Initial accumulator value.  Pass ``0`` for a fresh SAD.  Pass the result of
   a previous call to chain multiple SAD operations into a single running total
   without additional add instructions.

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
   * - ``msad_u8``, ``sad_u8``, ``sad_hi_u8``, ``sad_u16``
     - Yes
     - Yes
     - Yes
     - Yes
     - Yes
     - Yes
     - Yes
     - Yes
   * - ``qsad_pk_u16_u8``, ``mqsad_pk_u16_u8``, ``mqsad_u32_u8``
     - Yes
     - Yes
     - Yes
     - Yes
     - Yes
     - Yes
     - Yes
     - Yes

Builtin reference
===================

Each builtin's full signature, parameters, and return value are documented
below.

.. _sad-msad-u8:

``__builtin_amdgcn_msad_u8``
----------------------------

Signature and parameters for this builtin.

.. code-block:: cpp

   unsigned int __builtin_amdgcn_msad_u8(
       unsigned int src0,
       unsigned int src1,
       unsigned int src2);

Computes the sum of absolute differences of four packed unsigned bytes,
skipping byte positions where the corresponding ``src1`` byte is zero.  This
masking behavior is designed for padded reference windows, where positions
outside the frame boundary are marked with zero rather than valid pixel data.

.. list-table::
   :header-rows: 1
   :widths: auto

   * - Parameter
     - Type
     - Description
   * - ``src0``
     - unsigned int
     - Four packed unsigned bytes: the source data to compare.
   * - ``src1``
     - unsigned int
     - Four packed unsigned bytes: the reference data.  Byte positions where
       ``src1`` is zero are excluded from the accumulation.
   * - ``src2``
     - unsigned int
     - Initial accumulator value.  Pass ``0u`` for a fresh SAD; pass the
       result of a previous call to accumulate across multiple 32-bit values.

**Returns** ``unsigned int`` -- accumulated SAD
(:math:`\text{src2} + \sum_{k=0}^{3} |\text{src0}[k] - \text{src1}[k]|`
for positions where ``src1[k] != 0``).

.. _sad-sad-u8:

``__builtin_amdgcn_sad_u8``
---------------------------

Signature and parameters for this builtin.

.. code-block:: cpp

   unsigned int __builtin_amdgcn_sad_u8(
       unsigned int src0,
       unsigned int src1,
       unsigned int src2);

Computes the sum of absolute differences of four packed unsigned bytes and
accumulates the result into bits [15:0] of the destination.  Pair with
``sad_hi_u8`` to pack two independent 4-byte SADs into a single 32-bit value.

.. list-table::
   :header-rows: 1
   :widths: auto

   * - Parameter
     - Type
     - Description
   * - ``src0``
     - unsigned int
     - Four packed unsigned bytes: the source data.
   * - ``src1``
     - unsigned int
     - Four packed unsigned bytes: the reference data.
   * - ``src2``
     - unsigned int
     - Initial accumulator value (bits [15:0] are used).

**Returns** ``unsigned int`` -- accumulated SAD in bits [15:0]
(:math:`\text{src2}[15{:}0] + \sum_{k=0}^{3} |\text{src0}[k] - \text{src1}[k]|`).

.. _sad-sad-hi-u8:

``__builtin_amdgcn_sad_hi_u8``
------------------------------

Signature and parameters for this builtin.

.. code-block:: cpp

   unsigned int __builtin_amdgcn_sad_hi_u8(
       unsigned int src0,
       unsigned int src1,
       unsigned int src2);

Computes the sum of absolute differences of four packed unsigned bytes, shifts
each per-byte absolute difference left by 16 bits before accumulating, and
places the result in bits [31:16] of the destination.  Pair with ``sad_u8``
to pack two independent 4-byte SADs into a single 32-bit value without
additional packing instructions.

.. list-table::
   :header-rows: 1
   :widths: auto

   * - Parameter
     - Type
     - Description
   * - ``src0``
     - unsigned int
     - Four packed unsigned bytes: the source data.
   * - ``src1``
     - unsigned int
     - Four packed unsigned bytes: the reference data.
   * - ``src2``
     - unsigned int
     - Initial accumulator value (bits [31:16] are used).

**Returns** ``unsigned int`` -- accumulated SAD in bits [31:16].

.. _sad-sad-u16:

``__builtin_amdgcn_sad_u16``
----------------------------

Signature and parameters for this builtin.

.. code-block:: cpp

   unsigned int __builtin_amdgcn_sad_u16(
       unsigned int src0,
       unsigned int src1,
       unsigned int src2);

Computes the sum of absolute differences of two packed unsigned 16-bit values
and accumulates the result into ``src2``.

.. list-table::
   :header-rows: 1
   :widths: auto

   * - Parameter
     - Type
     - Description
   * - ``src0``
     - unsigned int
     - Two packed unsigned 16-bit values: the source data.
   * - ``src1``
     - unsigned int
     - Two packed unsigned 16-bit values: the reference data.
   * - ``src2``
     - unsigned int
     - Initial accumulator value.

**Returns** ``unsigned int`` -- accumulated SAD
(:math:`\text{src2} + \sum_{k=0}^{1} |\text{src0}[k] - \text{src1}[k]|`).

.. _sad-qsad-pk-u16-u8:

``__builtin_amdgcn_qsad_pk_u16_u8``
-----------------------------------

Signature and parameters for this builtin.

.. code-block:: cpp

   uint64_t __builtin_amdgcn_qsad_pk_u16_u8(
       uint64_t     src0,
       unsigned int src1,
       uint64_t     src2);

Computes four overlapping 4-byte SADs from a single 8-byte reference window in
one instruction.  The 8-byte source (``src0``) is treated as a sliding window:
the four SADs compare bytes [0:3], [1:4], [2:5], and [3:6] against the same
4-byte query (``src1``).  The four 16-bit results are packed into the returned
64-bit value.

.. list-table::
   :header-rows: 1
   :widths: auto

   * - Parameter
     - Type
     - Description
   * - ``src0``
     - uint64_t
     - Eight packed unsigned bytes: the sliding reference window.
   * - ``src1``
     - unsigned int
     - Four packed unsigned bytes: the query to compare against each window
       position.
   * - ``src2``
     - uint64_t
     - Initial accumulator for all four channels simultaneously.

**Returns** ``uint64_t`` -- four 16-bit SAD results packed into a 64-bit value.
Extract individual results with ``(result >> (i * 16)) & 0xffff`` for window
offset ``i``.

.. _sad-mqsad-pk-u16-u8:

``__builtin_amdgcn_mqsad_pk_u16_u8``
------------------------------------

Signature and parameters for this builtin.

.. code-block:: cpp

   uint64_t __builtin_amdgcn_mqsad_pk_u16_u8(
       uint64_t     src0,
       unsigned int src1,
       uint64_t     src2);

Masked variant of ``qsad_pk_u16_u8``.  Computes four overlapping 4-byte SADs
from an 8-byte sliding window, but skips byte positions where the corresponding
``src1`` byte is zero.  Provides the same sliding-window behavior as
``qsad_pk_u16_u8`` with the same masking semantics as ``msad_u8``.

.. list-table::
   :header-rows: 1
   :widths: auto

   * - Parameter
     - Type
     - Description
   * - ``src0``
     - uint64_t
     - Eight packed unsigned bytes: the sliding reference window.
   * - ``src1``
     - unsigned int
     - Four packed unsigned bytes: the query.  Byte positions where ``src1``
       is zero are excluded from the accumulation.
   * - ``src2``
     - uint64_t
     - Initial accumulator for all four channels simultaneously.

**Returns** ``uint64_t`` -- four 16-bit masked SAD results packed into a
64-bit value.

.. _sad-mqsad-u32-u8:

``__builtin_amdgcn_mqsad_u32_u8``
---------------------------------

Signature and parameters for this builtin.

.. code-block:: cpp

   uint4 __builtin_amdgcn_mqsad_u32_u8(
       uint64_t     src0,
       unsigned int src1,
       uint4        src2);

Computes four overlapping 4-byte SADs from an 8-byte sliding window, returning
four full 32-bit results instead of the 16-bit packed format of the
``qsad``/``mqsad`` variants.  Use this when the accumulated SAD values may
exceed the 16-bit range (65535).

.. list-table::
   :header-rows: 1
   :widths: auto

   * - Parameter
     - Type
     - Description
   * - ``src0``
     - uint64_t
     - Eight packed unsigned bytes: the sliding reference window.
   * - ``src1``
     - unsigned int
     - Four packed unsigned bytes: the query.
   * - ``src2``
     - uint4
     - Four-element uint32 accumulator vector.

**Returns** ``uint4`` -- four 32-bit SAD results.
