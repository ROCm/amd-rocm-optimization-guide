.. meta::
   :description: Reference for AMD GPU Data Parallel Primitives (DPP) and data-share permutation intrinsics, including mov_dpp, ds_swizzle, ds_permute, and architecture availability.
   :keywords: AMD, ROCm, HIP, DPP, data parallel primitives, mov_dpp, ds_swizzle, ds_permute, ds_bpermute, intrinsics, CDNA, RDNA

.. _dpp-intrinsics:

********************************************************************************
DPP and data-share permutation intrinsics
********************************************************************************

Data Parallel Primitives (DPP) move data between lanes using a hardware
control word that encodes fixed permutation patterns (quad permute, row shift,
row rotate, row broadcast, etc.).  DPP instructions execute in a dedicated
hardware unit, offering lower latency than general-purpose shuffle operations.
The data-share permutation intrinsics (``ds_swizzle``, ``ds_permute``,
``ds_bpermute``) provide complementary patterns that operate through the Local Data Share (LDS)
unit without reading or writing shared memory.

Common parameters
=================

**DPP parameters** (used by ``mov_dpp`` and ``update_dpp``):

``dpp_ctrl`` (control word)
   Encodes the permutation pattern.  User-defined value derived from the DPP
   control word encoding in the Instruction Set Architecture (ISA) specification.  Common patterns include
   ``quad_perm`` (arbitrary permutation within groups of 4), ``row_shr``
   (shift right within a row of 16), ``row_ror`` (rotate right within a row),
   ``row_bcast`` (broadcast from one lane to the row, CDNA/gfx9xx only),
   and ``wave_shr`` and ``wave_ror`` (shift and rotate across the full wave).  See
   the *Data Parallel Primitives* chapter of the
   `CDNA3 ISA <https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/instruction-set-architectures/amd-instinct-mi300-cdna3-instruction-set-architecture.pdf>`_
   or
   `RDNA3 ISA <https://www.amd.com/content/dam/amd/en/documents/radeon-tech-docs/instruction-set-architectures/rdna3-shader-instruction-set-architecture-feb-2023_0.pdf>`_
   for the full encoding.

``row_mask`` (row enable mask)
   4-bit mask selecting which rows of 16 lanes participate.  Each bit
   enables one row: bit 0 = lanes 0--15, bit 1 = lanes 16--31, bit 2 =
   lanes 32--47, bit 3 = lanes 48--63.  Disabled rows return their own value
   unchanged.  Pass ``0xf`` to enable all rows.

``bank_mask`` (bank enable mask)
   4-bit mask selecting which banks of 4 lanes participate within each row.
   Each bit enables one bank within the 16-lane row.  Disabled banks return
   their own value unchanged.  Pass ``0xf`` to enable all banks.

``bound_ctrl`` (boundary control)
   Controls the value for out-of-range source lanes.  When ``true``,
   out-of-range lanes return zero.  When ``false``, out-of-range lanes return
   their own value (identity behavior).

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
   * - ``mov_dpp``, ``update_dpp``
     - Yes
     - Yes
     - Yes
     - Yes
     - Yes
     - Yes
     - Yes
     - Yes
   * - ``mov_dpp8``
     - No
     - No
     - No
     - No
     - Yes
     - Yes
     - Yes
     - Yes
   * - ``ds_swizzle``, ``ds_permute``, ``ds_bpermute``
     - Yes
     - Yes
     - Yes
     - Yes
     - Yes
     - Yes
     - Yes
     - Yes

Intrinsic reference
===================

Each intrinsic's full signature, parameters, and return value are documented
below.

DPP move operations
-------------------

These intrinsics move data between lanes using DPP control words that encode fixed permutation patterns.

.. _dpp-mov-dpp:

``__builtin_amdgcn_mov_dpp``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Signature and parameters for this intrinsic.

.. code-block:: cpp

   int __builtin_amdgcn_mov_dpp(
       int  val,
       int  dpp_ctrl,
       int  row_mask,
       int  bank_mask,
       bool bound_ctrl);

Moves data between lanes using a DPP control word.  The control word encodes
the permutation pattern; the row and bank masks filter which lanes participate.

.. list-table::
   :header-rows: 1
   :widths: auto

   * - Parameter
     - Type
     - Description
   * - ``val``
     - int
     - The value each lane provides.  The DPP control word determines which
       lane's value each destination lane receives.
   * - ``dpp_ctrl``
     - int
     - DPP control word encoding the permutation pattern.
   * - ``row_mask``
     - int
     - 4-bit row enable mask.  Pass ``0xf`` for all rows.
   * - ``bank_mask``
     - int
     - 4-bit bank enable mask.  Pass ``0xf`` for all banks.
   * - ``bound_ctrl``
     - bool
     - When ``true``, out-of-range lanes return zero; when ``false``, they
       return their own ``val``.

**Returns** ``int`` -- the value from the source lane determined by
``dpp_ctrl``, or zero / identity for out-of-range lanes depending on
``bound_ctrl``.

.. _dpp-update-dpp:

``__builtin_amdgcn_update_dpp``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Signature and parameters for this intrinsic.

.. code-block:: cpp

   int __builtin_amdgcn_update_dpp(
       int  old,
       int  val,
       int  dpp_ctrl,
       int  row_mask,
       int  bank_mask,
       bool bound_ctrl);

DPP move with a fallback value for out-of-range or masked-off lanes.  Lanes
that are disabled by ``row_mask``/``bank_mask`` or whose source is
out-of-range receive ``old`` instead of zero or their own value.

.. list-table::
   :header-rows: 1
   :widths: auto

   * - Parameter
     - Type
     - Description
   * - ``old``
     - int
     - Fallback value for disabled or out-of-range lanes.
   * - ``val``
     - int
     - The value each lane provides for the DPP operation.
   * - ``dpp_ctrl``
     - int
     - DPP control word encoding the permutation pattern.
   * - ``row_mask``
     - int
     - 4-bit row enable mask.  Pass ``0xf`` for all rows.
   * - ``bank_mask``
     - int
     - 4-bit bank enable mask.  Pass ``0xf`` for all banks.
   * - ``bound_ctrl``
     - bool
     - When ``true``, out-of-range lanes return zero; when ``false``, they
       return ``old``.

**Returns** ``int`` -- the DPP result for participating lanes, or ``old``
for disabled/out-of-range lanes.

.. _dpp-mov-dpp8:

``__builtin_amdgcn_mov_dpp8``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Signature and parameters for this intrinsic.

.. code-block:: cpp

   unsigned int __builtin_amdgcn_mov_dpp8(
       unsigned int val,
       unsigned int sel);

DPP move using an 8-element compile-time permutation within groups of 8
lanes.  The ``sel`` parameter encodes a 3-bit lane index for each of the 8
positions, packed into a 24-bit value.

.. list-table::
   :header-rows: 1
   :widths: auto

   * - Parameter
     - Type
     - Description
   * - ``val``
     - unsigned int
     - The value each lane provides.
   * - ``sel``
     - unsigned int
     - Compile-time permutation selector.  Each group of 3 bits encodes the
       source lane index (0--7) within the 8-lane group for the corresponding
       destination lane.

**Returns** ``unsigned int`` -- the value from the source lane within the
8-lane group as determined by ``sel``.

Data-share permutations
-----------------------

These intrinsics permute data between lanes through the LDS unit without reading or writing shared memory.

.. _dpp-ds-swizzle:

``__builtin_amdgcn_ds_swizzle``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Signature and parameters for this intrinsic.

.. code-block:: cpp

   int __builtin_amdgcn_ds_swizzle(
       int val,
       int mask);

Applies a fixed bitmask permutation to lane indices, returning the value held
by the computed source lane.  The permutation is encoded entirely in a
compile-time mask, so the hardware applies it in a single instruction with no
per-lane index computation.

.. list-table::
   :header-rows: 1
   :widths: auto

   * - Parameter
     - Type
     - Description
   * - ``val``
     - int
     - The value each lane provides.
   * - ``mask``
     - int
     - 16-bit permutation mask.  Bits [14:10] = ``and_mask``, bits [9:5] =
       ``or_mask``, bits [4:0] = ``xor_mask``.  The source lane for each lane
       ``L`` is: ``(L & and_mask) | or_mask ^ xor_mask``.

**Returns** ``int`` -- the value from the source lane computed by the
bitmask permutation.

.. _dpp-ds-permute:

``__builtin_amdgcn_ds_permute``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Signature and parameters for this intrinsic.

.. code-block:: cpp

   int __builtin_amdgcn_ds_permute(
       int index,
       int val);

Forward permutation: each lane writes its value to the lane given by
``index``.  Uses byte addressing: to write to lane ``k``, pass ``k * 4`` as
the index.

.. list-table::
   :header-rows: 1
   :widths: auto

   * - Parameter
     - Type
     - Description
   * - ``index``
     - int
     - Byte address of the destination lane.  Lane ``k`` is addressed by
       passing ``k * 4``.
   * - ``val``
     - int
     - The value to send to the destination lane.

**Returns** ``int`` -- the value received from whichever lane targeted the
current lane.

.. _dpp-ds-bpermute:

``__builtin_amdgcn_ds_bpermute``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Signature and parameters for this intrinsic.

.. code-block:: cpp

   int __builtin_amdgcn_ds_bpermute(
       int index,
       int val);

Backward permutation: each lane reads the value from the lane specified by
``index``.  Uses byte addressing: to read from lane ``k``, pass ``k * 4`` as
the index.

.. list-table::
   :header-rows: 1
   :widths: auto

   * - Parameter
     - Type
     - Description
   * - ``index``
     - int
     - Byte address of the source lane.  Lane ``k``'s value is accessed by
       passing ``k * 4``.  Each lane can specify a different source.
   * - ``val``
     - int
     - The value each lane provides for others to read.

**Returns** ``int`` -- the value of ``val`` from the lane specified by
``index / 4``.
