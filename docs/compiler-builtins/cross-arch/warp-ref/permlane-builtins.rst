.. meta::
   :description: Reference for AMD GPU cross-row permutation builtins,
      including permlane16, permlanex16, and permlane64 with usage details and architecture availability.
   :keywords: AMD, ROCm, HIP, permlane16, permlanex16, permlane64, cross-row permutation, warp, builtins, CDNA, RDNA

.. _permlane-builtins:

********************************************************************************
Cross-row permutation builtins
********************************************************************************

The permlane builtins move data across 16-lane row boundaries, complementing
the Data Parallel Primitives (DPP) instructions which operate within rows.  They enable cross-row
communication patterns (such as combining results from different rows in a
reduction, or redistributing data across the full wavefront) without going
through shared memory.

Common parameters
=================

``old`` (fallback value)
   The value returned by lanes that are inactive or not targeted by the
   permutation.  Present on all permlane variants except ``permlane64``.

``val`` (source value)
   The per-lane value to permute.

``fi`` (fetch-inactive)
   When ``true``, inactive lanes are included in the permutation (their
   last-written value is used as the source).  When ``false``, inactive lanes
   are excluded and the corresponding destination receives ``old``.

``bc`` (bound-control / broadcast)
   When ``true``, the lane selector wraps at the row boundary, effectively
   broadcasting.  When ``false``, out-of-range selectors produce ``old``.

``lanesel_lo``, ``lanesel_hi`` (lane selectors)
   Two 4-bit values that select which lane within each 16-lane group provides
   the source data for the lower and upper halves of the warp respectively.
   Only present on the compile-time selector variants (``permlane16``,
   ``permlanex16``).

``lanesel`` (runtime lane selector)
   A per-lane selector value replacing the compile-time ``lanesel_lo``/
   ``lanesel_hi`` pair.  Only present on the ``_var`` variants.

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
   * - ``permlane16``, ``permlanex16``
     - No
     - No
     - No
     - No
     - Yes
     - Yes
     - Yes
     - Yes
   * - ``permlane64``
     - No
     - No
     - No
     - No
     - No
     - Yes
     - Yes
     - No
   * - ``permlane16_var``, ``permlanex16_var``
     - No
     - No
     - No
     - No
     - No
     - No
     - No
     - Yes
   * - ``permlane16_swap``, ``permlane32_swap``
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

.. _permlane-permlane16:

``__builtin_amdgcn_permlane16``
-------------------------------

Signature and parameters for this builtin.

.. code-block:: cpp

   int __builtin_amdgcn_permlane16(
       int  old,
       int  val,
       int  lanesel_lo,
       int  lanesel_hi,
       bool fi,
       bool bc);

Permutes within each 16-lane group using two 4-bit selectors.  Each lane
in the lower 16-lane group reads from the lane at ``lanesel_lo``; each lane
in the upper group reads from ``lanesel_hi``.

.. list-table::
   :header-rows: 1
   :widths: auto

   * - Parameter
     - Type
     - Description
   * - ``old``
     - int
     - Fallback value for inactive or out-of-range lanes.
   * - ``val``
     - int
     - Per-lane value to permute.
   * - ``lanesel_lo``
     - int
     - 4-bit source lane selector for the lower 16-lane group.
   * - ``lanesel_hi``
     - int
     - 4-bit source lane selector for the upper 16-lane group.
   * - ``fi``
     - bool
     - Fetch-inactive: include inactive lanes as sources.
   * - ``bc``
     - bool
     - Bound-control: wrap at row boundary.

**Returns** ``int`` -- permuted value.

.. _permlane-permlanex16:

``__builtin_amdgcn_permlanex16``
--------------------------------

Signature and parameters for this builtin.

.. code-block:: cpp

   int __builtin_amdgcn_permlanex16(
       int  old,
       int  val,
       int  lanesel_lo,
       int  lanesel_hi,
       bool fi,
       bool bc);

Same as ``permlane16`` but crosses the 16-lane boundary: each lane reads
from the *other* 16-lane group.

.. list-table::
   :header-rows: 1
   :widths: auto

   * - Parameter
     - Type
     - Description
   * - ``old``
     - int
     - Fallback value for inactive or out-of-range lanes.
   * - ``val``
     - int
     - Per-lane value to permute.
   * - ``lanesel_lo``
     - int
     - 4-bit source lane selector for the lower 16-lane group.
   * - ``lanesel_hi``
     - int
     - 4-bit source lane selector for the upper 16-lane group.
   * - ``fi``
     - bool
     - Fetch-inactive: include inactive lanes as sources.
   * - ``bc``
     - bool
     - Bound-control: wrap at row boundary.

**Returns** ``int`` -- permuted value from the opposite 16-lane group.

.. _permlane-permlane64:

``__builtin_amdgcn_permlane64``
-------------------------------

Signature and parameters for this builtin.

.. code-block:: cpp

   int __builtin_amdgcn_permlane64(
       int val);

Exchanges data between the two 32-lane halves of a wave64 without a
selector.  Lane ``k`` in the lower half reads from lane ``k + 32`` in the
upper half, and vice versa.

.. list-table::
   :header-rows: 1
   :widths: auto

   * - Parameter
     - Type
     - Description
   * - ``val``
     - int
     - Per-lane value to exchange.

**Returns** ``int`` -- the value from the corresponding lane in the other
32-lane half.

.. _permlane-permlane16-var:

``__builtin_amdgcn_permlane16_var``
-----------------------------------

Signature and parameters for this builtin.

.. code-block:: cpp

   unsigned int __builtin_amdgcn_permlane16_var(
       unsigned int old,
       unsigned int val,
       unsigned int lanesel,
       bool         fi,
       bool         bc);

``permlane16`` with a runtime lane selector.  Each lane independently selects
its source lane via ``lanesel``, rather than using a single compile-time
selector for the entire group.

.. list-table::
   :header-rows: 1
   :widths: auto

   * - Parameter
     - Type
     - Description
   * - ``old``
     - unsigned int
     - Fallback value for inactive or out-of-range lanes.
   * - ``val``
     - unsigned int
     - Per-lane value to permute.
   * - ``lanesel``
     - unsigned int
     - Per-lane runtime source lane selector.
   * - ``fi``
     - bool
     - Fetch-inactive: include inactive lanes as sources.
   * - ``bc``
     - bool
     - Bound-control: wrap at row boundary.

**Returns** ``unsigned int`` -- permuted value.

.. _permlane-permlanex16-var:

``__builtin_amdgcn_permlanex16_var``
------------------------------------

Signature and parameters for this builtin.

.. code-block:: cpp

   unsigned int __builtin_amdgcn_permlanex16_var(
       unsigned int old,
       unsigned int val,
       unsigned int lanesel,
       bool         fi,
       bool         bc);

``permlanex16`` with a runtime lane selector.  Same cross-row behavior as
``permlanex16`` but with per-lane source selection.

.. list-table::
   :header-rows: 1
   :widths: auto

   * - Parameter
     - Type
     - Description
   * - ``old``
     - unsigned int
     - Fallback value for inactive or out-of-range lanes.
   * - ``val``
     - unsigned int
     - Per-lane value to permute.
   * - ``lanesel``
     - unsigned int
     - Per-lane runtime source lane selector.
   * - ``fi``
     - bool
     - Fetch-inactive: include inactive lanes as sources.
   * - ``bc``
     - bool
     - Bound-control: wrap at row boundary.

**Returns** ``unsigned int`` -- permuted value from the opposite 16-lane
group.

.. _permlane-permlane16-swap:

``__builtin_amdgcn_permlane16_swap``
------------------------------------

Signature and parameters for this builtin.

.. code-block:: cpp

   _Vector<2, unsigned int> __builtin_amdgcn_permlane16_swap(
       unsigned int src0,
       unsigned int src1,
       bool         fi,
       bool         bc);

Swaps odd and even 16-lane rows between two operands.  Returns a two-element
vector where element 0 is the new value for ``src0`` and element 1 is the new
value for ``src1``.

.. list-table::
   :header-rows: 1
   :widths: auto

   * - Parameter
     - Type
     - Description
   * - ``src0``
     - unsigned int
     - First operand.
   * - ``src1``
     - unsigned int
     - Second operand.
   * - ``fi``
     - bool
     - Fetch-inactive: include inactive lanes as sources.
   * - ``bc``
     - bool
     - Bound-control: wrap at row boundary.

**Returns** ``_Vector<2, unsigned int>`` -- two-element vector with the
swapped values.

.. _permlane-permlane32-swap:

``__builtin_amdgcn_permlane32_swap``
------------------------------------

Signature and parameters for this builtin.

.. code-block:: cpp

   _Vector<2, unsigned int> __builtin_amdgcn_permlane32_swap(
       unsigned int src0,
       unsigned int src1,
       bool         fi,
       bool         bc);

Swaps the upper and lower 32-lane halves between two operands.

.. list-table::
   :header-rows: 1
   :widths: auto

   * - Parameter
     - Type
     - Description
   * - ``src0``
     - unsigned int
     - First operand.
   * - ``src1``
     - unsigned int
     - Second operand.
   * - ``fi``
     - bool
     - Fetch-inactive: include inactive lanes as sources.
   * - ``bc``
     - bool
     - Bound-control: wrap at row boundary.

**Returns** ``_Vector<2, unsigned int>`` -- two-element vector with the
swapped values.
