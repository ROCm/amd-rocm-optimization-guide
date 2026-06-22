.. meta::
   :description: Reference for AMD GPU warp voting and synchronization builtins,
      including ballot, inverse_ballot, mbcnt, and wave_barrier with architecture support.
   :keywords: AMD, ROCm, HIP, ballot, mbcnt, warp voting, wave_barrier, synchronization, builtins, CDNA, RDNA

.. _vote-builtins:

********************************************************************************
Warp voting and synchronization builtins
********************************************************************************

Warp voting builtins answer the question "which lanes satisfy a condition?"
by converting per-lane predicates into bitmasks, and vice versa.  The
``ballot`` builtin captures a predicate as a bitmask; ``mbcnt`` counts the
number of set bits below the current lane to produce a compaction index.
Together, ballot and mbcnt form the standard mechanism for stream compaction
within a warp: filtering values, computing unique sequential indices, and
writing contiguously to an output buffer without shared memory or barriers.

Common parameters
=================

``pred`` (per-lane predicate)
   A boolean value evaluated independently by each lane.  Lanes where
   ``pred`` is true have their corresponding bit set in the ballot mask.

``mask`` (ballot mask half)
   A 32-bit value representing half of a 64-bit ballot mask.  ``mbcnt_lo``
   takes the lower 32 bits; ``mbcnt_hi`` takes the upper 32 bits.

``base`` (initial count)
   An initial value added to the popcount result.  For ``mbcnt_lo``, pass
   ``0``.  For ``mbcnt_hi``, pass the result of ``mbcnt_lo`` to accumulate
   the count across both halves of the 64-bit mask.

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
   * - ``ballot_w64``, ``inverse_ballot_w64``
     - Yes
     - Yes
     - Yes
     - Yes
     - No
     - No
     - No
     - No
   * - ``ballot_w32``, ``inverse_ballot_w32``
     - No
     - No
     - No
     - No
     - Yes
     - Yes
     - Yes
     - Yes
   * - ``mbcnt_lo``, ``mbcnt_hi``
     - Yes
     - Yes
     - Yes
     - Yes
     - Yes
     - Yes
     - Yes
     - Yes
   * - ``wave_barrier``, ``wave_id``
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

Ballot
------

These builtins convert per-lane predicates into bitmasks and vice versa.

.. _vote-ballot-w64:

``__builtin_amdgcn_ballot_w64``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Signature and parameters for this builtin.

.. code-block:: cpp

   unsigned long long __builtin_amdgcn_ballot_w64(
       bool pred);

Returns a 64-bit bitmask in which bit ``k`` is set if and only if lane ``k``
is active and its ``pred`` argument is true.  This is the wave64 variant;
use ``ballot_w32`` on wave32 targets.

.. list-table::
   :header-rows: 1
   :widths: auto

   * - Parameter
     - Type
     - Description
   * - ``pred``
     - bool
     - Per-lane predicate.

**Returns** ``unsigned long long`` -- 64-bit mask of active lanes where
``pred`` is true.

.. _vote-ballot-w32:

``__builtin_amdgcn_ballot_w32``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Signature and parameters for this builtin.

.. code-block:: cpp

   unsigned int __builtin_amdgcn_ballot_w32(
       bool pred);

Returns a 32-bit bitmask in which bit ``k`` is set if and only if lane ``k``
is active and its ``pred`` argument is true.  This is the wave32 variant.

.. list-table::
   :header-rows: 1
   :widths: auto

   * - Parameter
     - Type
     - Description
   * - ``pred``
     - bool
     - Per-lane predicate.

**Returns** ``unsigned int`` -- 32-bit mask of active lanes where ``pred``
is true.

.. _vote-inverse-ballot-w64:

``__builtin_amdgcn_inverse_ballot_w64``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Signature and parameters for this builtin.

.. code-block:: cpp

   bool __builtin_amdgcn_inverse_ballot_w64(
       uint64_t mask);

Returns ``true`` if the current lane's bit is set in ``mask``.  The inverse of
``ballot_w64``: converts a bitmask back to a per-lane predicate.

.. list-table::
   :header-rows: 1
   :widths: auto

   * - Parameter
     - Type
     - Description
   * - ``mask``
     - uint64_t
     - 64-bit lane mask.

**Returns** ``bool`` -- ``true`` if the current lane's bit is set in
``mask``.

.. _vote-inverse-ballot-w32:

``__builtin_amdgcn_inverse_ballot_w32``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Signature and parameters for this builtin.

.. code-block:: cpp

   bool __builtin_amdgcn_inverse_ballot_w32(
       uint32_t mask);

Returns ``true`` if the current lane's bit is set in ``mask``.  The inverse of
``ballot_w32``.

.. list-table::
   :header-rows: 1
   :widths: auto

   * - Parameter
     - Type
     - Description
   * - ``mask``
     - uint32_t
     - 32-bit lane mask.

**Returns** ``bool`` -- ``true`` if the current lane's bit is set in
``mask``.

Masked bit count
----------------

These builtins count set bits in a ballot mask below the current lane to produce compaction indices.

.. _vote-mbcnt-lo:

``__builtin_amdgcn_mbcnt_lo``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Signature and parameters for this builtin.

.. code-block:: cpp

   unsigned int __builtin_amdgcn_mbcnt_lo(
       unsigned int mask,
       unsigned int base);

Counts the number of set bits in the lower 32 bits of a ballot mask that
correspond to lanes below the current lane.  Used together with ``mbcnt_hi``
to compute a compaction index across the full 64-bit ballot mask.

.. list-table::
   :header-rows: 1
   :widths: auto

   * - Parameter
     - Type
     - Description
   * - ``mask``
     - unsigned int
     - Lower 32 bits of the ballot mask.
   * - ``base``
     - unsigned int
     - Initial count value.  Pass ``0`` when used as the first step.

**Returns** ``unsigned int`` -- ``base`` plus the number of set bits in
``mask`` below the current lane index.

.. _vote-mbcnt-hi:

``__builtin_amdgcn_mbcnt_hi``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Signature and parameters for this builtin.

.. code-block:: cpp

   unsigned int __builtin_amdgcn_mbcnt_hi(
       unsigned int mask,
       unsigned int base);

Counts the number of set bits in the upper 32 bits of a ballot mask that
correspond to lanes below the current lane.  Pass the result of ``mbcnt_lo``
as ``base`` to compute the full compaction index across a 64-bit ballot mask.

.. list-table::
   :header-rows: 1
   :widths: auto

   * - Parameter
     - Type
     - Description
   * - ``mask``
     - unsigned int
     - Upper 32 bits of the ballot mask.
   * - ``base``
     - unsigned int
     - Result of ``mbcnt_lo`` (count from the lower 32-bit half).

**Returns** ``unsigned int`` -- ``base`` plus the number of set bits in
``mask`` below the current lane index, giving the total compaction index
across the full 64-bit mask.

Synchronization
---------------

These builtins provide warp-level synchronization and identification.

.. _vote-wave-barrier:

``__builtin_amdgcn_wave_barrier``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Signature and parameters for this builtin.

.. code-block:: cpp

   void __builtin_amdgcn_wave_barrier();

Synchronizes all lanes within the warp.  Guarantees that all lanes have
reached this point before any lane proceeds past it.  This is a compiler
scheduling barrier, not a memory fence.

**Returns** ``void``

.. _vote-wave-id:

``__builtin_amdgcn_wave_id``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Signature and parameters for this builtin.

.. code-block:: cpp

   unsigned int __builtin_amdgcn_wave_id();

Returns the warp's index within the workgroup.

**Returns** ``unsigned int`` -- the warp index.
