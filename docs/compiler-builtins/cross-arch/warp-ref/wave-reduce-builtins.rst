.. meta::
   :description: Reference for AMD GPU single-instruction warp reduction builtins, including wave_reduce_add, wave_reduce_min, wave_reduce_max, and architecture availability.
   :keywords: AMD, ROCm, HIP, wave reduce, warp reduction, wave_reduce_add, wave_reduce_min, wave_reduce_max, builtins, CDNA, RDNA

.. _wave-reduce-builtins:

********************************************************************************
Warp reduction builtins
********************************************************************************

The warp reduction builtins compute a single scalar result from per-lane
values across all active lanes in a single builtin call.  They encapsulate
the butterfly reduction pattern --- exchanging values with increasingly
distant partners --- without requiring the caller to manage Data Parallel Primitives (DPP) control words
or architecture-specific code paths.

Common parameters
=================

All warp reduction builtins share the same two-parameter interface:

``src`` (per-lane input)
   The value contributed by each lane.  The type (``unsigned int``,
   ``unsigned long``, ``int``, or ``long``) matches the builtin's type
   suffix.

``strategy`` (reduction strategy hint)
   Controls how the compiler implements the reduction:

   - ``0`` -- let the compiler choose the best strategy for the target.
   - ``1`` -- request the iterative (loop-based) strategy.
   - ``2`` -- request the DPP-based strategy.

   The compiler might ignore the hint if the requested strategy is not
   available on the target architecture.  Use ``0`` unless profiling shows
   a measurable difference.

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
   * - All ``wave_reduce_*`` variants
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

Addition
--------

These builtins compute the sum of per-lane values across all active lanes.

.. _wave-reduce-add-u32:

``__builtin_amdgcn_wave_reduce_add_u32``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Signature and parameters for this builtin.

.. code-block:: cpp

   unsigned int __builtin_amdgcn_wave_reduce_add_u32(
       unsigned int src,
       int          strategy);

Computes the sum of ``src`` across all active lanes.

**Returns** ``unsigned int`` -- the sum, broadcast to every lane.

.. _wave-reduce-add-u64:

``__builtin_amdgcn_wave_reduce_add_u64``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Signature and parameters for this builtin.

.. code-block:: cpp

   unsigned long __builtin_amdgcn_wave_reduce_add_u64(
       unsigned long src,
       int           strategy);

64-bit unsigned addition across all active lanes.

**Returns** ``unsigned long`` -- the sum, broadcast to every lane.

Subtraction
-----------

These builtins compute a subtraction reduction across all active lanes.

.. _wave-reduce-sub-u32:

``__builtin_amdgcn_wave_reduce_sub_u32``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Signature and parameters for this builtin.

.. code-block:: cpp

   unsigned int __builtin_amdgcn_wave_reduce_sub_u32(
       unsigned int src,
       int          strategy);

Unsigned 32-bit subtraction reduction across all active lanes.

**Returns** ``unsigned int`` -- the result, broadcast to every lane.

.. _wave-reduce-sub-u64:

``__builtin_amdgcn_wave_reduce_sub_u64``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Signature and parameters for this builtin.

.. code-block:: cpp

   unsigned long __builtin_amdgcn_wave_reduce_sub_u64(
       unsigned long src,
       int           strategy);

Unsigned 64-bit subtraction reduction across all active lanes.

**Returns** ``unsigned long`` -- the result, broadcast to every lane.

Minimum
-------

These builtins find the minimum value across all active lanes.

.. _wave-reduce-min-i32:

``__builtin_amdgcn_wave_reduce_min_i32``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Signature and parameters for this builtin.

.. code-block:: cpp

   int __builtin_amdgcn_wave_reduce_min_i32(
       int src,
       int strategy);

Signed 32-bit minimum across all active lanes.

**Returns** ``int`` -- the minimum value, broadcast to every lane.

.. _wave-reduce-min-u32:

``__builtin_amdgcn_wave_reduce_min_u32``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Signature and parameters for this builtin.

.. code-block:: cpp

   unsigned int __builtin_amdgcn_wave_reduce_min_u32(
       unsigned int src,
       int          strategy);

Unsigned 32-bit minimum across all active lanes.

**Returns** ``unsigned int`` -- the minimum value, broadcast to every lane.

.. _wave-reduce-min-i64:

``__builtin_amdgcn_wave_reduce_min_i64``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Signature and parameters for this builtin.

.. code-block:: cpp

   long __builtin_amdgcn_wave_reduce_min_i64(
       long src,
       int  strategy);

Signed 64-bit minimum across all active lanes.

**Returns** ``long`` -- the minimum value, broadcast to every lane.

.. _wave-reduce-min-u64:

``__builtin_amdgcn_wave_reduce_min_u64``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Signature and parameters for this builtin.

.. code-block:: cpp

   unsigned long __builtin_amdgcn_wave_reduce_min_u64(
       unsigned long src,
       int           strategy);

Unsigned 64-bit minimum across all active lanes.

**Returns** ``unsigned long`` -- the minimum value, broadcast to every lane.

Maximum
-------

These builtins find the maximum value across all active lanes.

.. _wave-reduce-max-i32:

``__builtin_amdgcn_wave_reduce_max_i32``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Signature and parameters for this builtin.

.. code-block:: cpp

   int __builtin_amdgcn_wave_reduce_max_i32(
       int src,
       int strategy);

Signed 32-bit maximum across all active lanes.

**Returns** ``int`` -- the maximum value, broadcast to every lane.

.. _wave-reduce-max-u32:

``__builtin_amdgcn_wave_reduce_max_u32``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Signature and parameters for this builtin.

.. code-block:: cpp

   unsigned int __builtin_amdgcn_wave_reduce_max_u32(
       unsigned int src,
       int          strategy);

Unsigned 32-bit maximum across all active lanes.

**Returns** ``unsigned int`` -- the maximum value, broadcast to every lane.

.. _wave-reduce-max-i64:

``__builtin_amdgcn_wave_reduce_max_i64``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Signature and parameters for this builtin.

.. code-block:: cpp

   long __builtin_amdgcn_wave_reduce_max_i64(
       long src,
       int  strategy);

Signed 64-bit maximum across all active lanes.

**Returns** ``long`` -- the maximum value, broadcast to every lane.

.. _wave-reduce-max-u64:

``__builtin_amdgcn_wave_reduce_max_u64``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Signature and parameters for this builtin.

.. code-block:: cpp

   unsigned long __builtin_amdgcn_wave_reduce_max_u64(
       unsigned long src,
       int           strategy);

Unsigned 64-bit maximum across all active lanes.

**Returns** ``unsigned long`` -- the maximum value, broadcast to every lane.

Bitwise AND
-----------

These builtins compute the bitwise AND of all active lanes.

.. _wave-reduce-and-b32:

``__builtin_amdgcn_wave_reduce_and_b32``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Signature and parameters for this builtin.

.. code-block:: cpp

   int __builtin_amdgcn_wave_reduce_and_b32(
       int src,
       int strategy);

Bitwise AND of all active lanes (32-bit).

**Returns** ``int`` -- the AND of all lane values, broadcast to every lane.

.. _wave-reduce-and-b64:

``__builtin_amdgcn_wave_reduce_and_b64``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Signature and parameters for this builtin.

.. code-block:: cpp

   int __builtin_amdgcn_wave_reduce_and_b64(
       int src,
       int strategy);

Bitwise AND of all active lanes (64-bit).

**Returns** ``int`` -- the AND of all lane values, broadcast to every lane.

Bitwise OR
----------

These builtins compute the bitwise OR of all active lanes.

.. _wave-reduce-or-b32:

``__builtin_amdgcn_wave_reduce_or_b32``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Signature and parameters for this builtin.

.. code-block:: cpp

   int __builtin_amdgcn_wave_reduce_or_b32(
       int src,
       int strategy);

Bitwise OR of all active lanes (32-bit).

**Returns** ``int`` -- the OR of all lane values, broadcast to every lane.

.. _wave-reduce-or-b64:

``__builtin_amdgcn_wave_reduce_or_b64``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Signature and parameters for this builtin.

.. code-block:: cpp

   int __builtin_amdgcn_wave_reduce_or_b64(
       int src,
       int strategy);

Bitwise OR of all active lanes (64-bit).

**Returns** ``int`` -- the OR of all lane values, broadcast to every lane.

Bitwise XOR
-----------

These builtins compute the bitwise XOR of all active lanes.

.. _wave-reduce-xor-b32:

``__builtin_amdgcn_wave_reduce_xor_b32``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Signature and parameters for this builtin.

.. code-block:: cpp

   int __builtin_amdgcn_wave_reduce_xor_b32(
       int src,
       int strategy);

Bitwise XOR of all active lanes (32-bit).

**Returns** ``int`` -- the XOR of all lane values, broadcast to every lane.

.. _wave-reduce-xor-b64:

``__builtin_amdgcn_wave_reduce_xor_b64``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Signature and parameters for this builtin.

.. code-block:: cpp

   int __builtin_amdgcn_wave_reduce_xor_b64(
       int src,
       int strategy);

Bitwise XOR of all active lanes (64-bit).

**Returns** ``int`` -- the XOR of all lane values, broadcast to every lane.
