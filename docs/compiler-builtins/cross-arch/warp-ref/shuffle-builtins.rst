.. meta::
   :description: Reference for AMD GPU shuffle and lane access builtins,
      including __shfl, readlane, readfirstlane, and writelane with usage and architecture availability.
   :keywords: AMD, ROCm, HIP, shuffle, readlane, readfirstlane, writelane, warp, lane access, builtins, CDNA, RDNA

.. _shuffle-builtins:

********************************************************************************
Shuffle and lane access builtins
********************************************************************************

Shuffle builtins copy a value from one lane to another within a warp.  The
``__shfl*`` family provides a higher-level interface, while the
``__builtin_amdgcn_*`` Compiler builtins map directly to AMD Instruction Set Architecture (ISA)
instructions and offer lower-level control.

Common parameters
=================

``val`` (source value)
   The value each lane provides.  One lane's value is selected and returned
   to the requesting lane(s) based on the shuffle mode.

``width`` (sub-warp partition size)
   Divides the warp into independent partitions of ``width`` lanes.  Shuffles
   only move data within a partition, not across partitions.  Must be a power
   of two.  Defaults to ``warpSize`` (the full warp).

``offset`` / ``src_lane`` (source selector)
   Identifies which lane to read from.  The meaning depends on the shuffle
   variant: ``__shfl`` reads from an absolute lane index, ``__shfl_up`` and
   ``__shfl_down`` read from a relative offset, and ``__shfl_xor`` computes
   the source lane by applying a bitwise XOR between the current lane index and a mask.

``lane`` (Compiler builtins)
   A uniform runtime lane index.  Must hold the same value across all active
   lanes at the point of the call.

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
   * - ``__shfl``, ``__shfl_up``, ``__shfl_down``, ``__shfl_xor``
     - Yes
     - Yes
     - Yes
     - Yes
     - Yes
     - Yes
     - Yes
     - Yes
   * - ``readfirstlane``, ``readlane``, ``writelane``
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

HIP shuffles
------------

These builtins provide a higher-level interface for copying values between lanes within a warp.

.. _shuffle-shfl:

``__shfl``
^^^^^^^^^^

Signature and parameters for this builtin.

.. code-block:: cpp

   T __shfl(T val, int src_lane, int width = warpSize);

Copies ``val`` from an absolute lane index within the current partition.

.. list-table::
   :header-rows: 1
   :widths: auto

   * - Parameter
     - Type
     - Description
   * - ``val``
     - T
     - Value to read from the source lane.
   * - ``src_lane``
     - int
     - Absolute lane index within the partition.  If ``src_lane >= width``,
       behavior is undefined.
   * - ``width``
     - int
     - Partition size (power of two).  Defaults to ``warpSize``.

**Returns** ``T`` -- ``val`` from lane ``src_lane`` within the partition.

.. _shuffle-shfl-up:

``__shfl_up``
^^^^^^^^^^^^^

Signature and parameters for this builtin.

.. code-block:: cpp

   T __shfl_up(T val, unsigned int offset, int width = warpSize);

Copies ``val`` from a lane ``offset`` positions behind the current lane within
the partition.  If the source lane would be negative (below the partition
start), the current lane's own value is returned.

.. list-table::
   :header-rows: 1
   :widths: auto

   * - Parameter
     - Type
     - Description
   * - ``val``
     - T
     - Value to read.
   * - ``offset``
     - unsigned int
     - Number of lanes behind the current lane to read from.
   * - ``width``
     - int
     - Partition size.  Defaults to ``warpSize``.

**Returns** ``T`` -- ``val`` from lane ``current_lane - offset``, or the
current lane's ``val`` if the source is out of range.

.. _shuffle-shfl-down:

``__shfl_down``
^^^^^^^^^^^^^^^

Signature and parameters for this builtin.

.. code-block:: cpp

   T __shfl_down(T val, unsigned int offset, int width = warpSize);

Copies ``val`` from a lane ``offset`` positions ahead of the current lane
within the partition.  If the source lane would be beyond the partition end,
the current lane's own value is returned.

.. list-table::
   :header-rows: 1
   :widths: auto

   * - Parameter
     - Type
     - Description
   * - ``val``
     - T
     - Value to read.
   * - ``offset``
     - unsigned int
     - Number of lanes ahead of the current lane to read from.
   * - ``width``
     - int
     - Partition size.  Defaults to ``warpSize``.

**Returns** ``T`` -- ``val`` from lane ``current_lane + offset``, or the
current lane's ``val`` if the source is out of range.

.. _shuffle-shfl-xor:

``__shfl_xor``
^^^^^^^^^^^^^^

Signature and parameters for this builtin.

.. code-block:: cpp

   T __shfl_xor(T val, int mask, int width = warpSize);

Copies ``val`` from the lane whose index is the bitwise XOR of the current lane index and
``mask``.  This is the building block for butterfly reductions.

.. list-table::
   :header-rows: 1
   :widths: auto

   * - Parameter
     - Type
     - Description
   * - ``val``
     - T
     - Value to read.
   * - ``mask``
     - int
     - XOR mask applied to the current lane index to compute the source lane.
   * - ``width``
     - int
     - Partition size.  Defaults to ``warpSize``.

**Returns** ``T`` -- ``val`` from lane ``current_lane ^ mask``.

Hardware lane access
--------------------

These builtins map directly to AMD ISA instructions for reading and writing individual lane values.

.. _shuffle-readfirstlane:

``__builtin_amdgcn_readfirstlane``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Signature and parameters for this builtin.

.. code-block:: cpp

   unsigned int __builtin_amdgcn_readfirstlane(
       unsigned int val);

Returns the value of ``val`` held by the first active lane of the warp.  The
result is uniform (the same value across all lanes), which makes this
builtin useful for promoting a per-lane value to a uniform value for use as
an index argument to ``readlane`` or as a scalar operand.

.. list-table::
   :header-rows: 1
   :widths: auto

   * - Parameter
     - Type
     - Description
   * - ``val``
     - unsigned int
     - The value to read from the first active lane.

**Returns** ``unsigned int`` -- the value of ``val`` in the first active lane,
broadcast to all lanes.

.. _shuffle-readlane:

``__builtin_amdgcn_readlane``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Signature and parameters for this builtin.

.. code-block:: cpp

   unsigned int __builtin_amdgcn_readlane(
       unsigned int val,
       unsigned int lane);

Returns the value of ``val`` held by the specified lane.  Unlike
``readfirstlane``, which always reads the first active lane, ``readlane``
accepts a runtime lane index.  The lane index must be uniform --- the same
value across all lanes in the warp at the point of the call.

.. list-table::
   :header-rows: 1
   :widths: auto

   * - Parameter
     - Type
     - Description
   * - ``val``
     - unsigned int
     - The value to read from the specified lane.
   * - ``lane``
     - unsigned int
     - Uniform runtime lane index.  Must hold the same value across all
       active lanes.  If derived from per-lane data, use ``readfirstlane``
       to promote it to a uniform value first.

**Returns** ``unsigned int`` -- the value of ``val`` in the specified lane,
broadcast to all lanes.

.. _shuffle-writelane:

``__builtin_amdgcn_writelane``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Signature and parameters for this builtin.

.. code-block:: cpp

   unsigned int __builtin_amdgcn_writelane(
       unsigned int val,
       unsigned int lane,
       unsigned int old);

Writes ``val`` to a single lane; all other lanes return ``old``.  This is the
inverse of ``readlane``: instead of broadcasting one lane's value to all lanes,
it injects a value into one specific lane while leaving the rest unchanged.

.. list-table::
   :header-rows: 1
   :widths: auto

   * - Parameter
     - Type
     - Description
   * - ``val``
     - unsigned int
     - The value to write into the target lane.
   * - ``lane``
     - unsigned int
     - Uniform runtime lane index of the target lane.
   * - ``old``
     - unsigned int
     - The value returned by all non-target lanes.

**Returns** ``unsigned int`` -- ``val`` in the target lane, ``old`` in all
other lanes.
