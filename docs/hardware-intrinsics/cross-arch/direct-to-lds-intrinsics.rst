.. meta::
   :description: Reference for AMD GPU intrinsics that transfer data from global or buffer memory directly into LDS without staging in VGPRs. Covers CDNA, CDNA2, CDNA3, CDNA4, and RDNA2.
   :keywords: AMD, ROCm, HIP, intrinsics, LDS, global_load_lds, load_to_lds, buffer_load_lds, VGPR, CDNA, RDNA2, direct-to-LDS

.. _direct-to-lds-intrinsics:

********************************************************************************
Global-to-LDS intrinsics
********************************************************************************

AMD GPUs provide a family of intrinsics that load data from global or buffer
memory directly into Local Data Share (LDS) without staging the values in
Vector General-Purpose Registers (VGPRs).  Because the loaded data never
occupies a VGPR, these intrinsics reduce register pressure and can enable
higher occupancy in VGPR-constrained kernels.

All intrinsics in this family are wavefront-wide operations: each lane
provides its own source address, and the hardware writes lane *k*'s value to
the LDS base address plus an implicit per-lane stride.

Architecture availability
=========================

The following table summarizes which intrinsics are available on each
architecture.

+-------------------------------------------------+------+-------+-------+--------+-------+-------+---------+-------+
| Intrinsic                                       | CDNA | CDNA2 | CDNA3 | CDNA 4 | RDNA2 | RDNA3 | RDNA3.5 | RDNA4 |
+=================================================+======+=======+=======+========+=======+=======+=========+=======|
| ``__builtin_amdgcn_global_load_lds``            | No   | No    | Yes   | Yes    | No    | No    | No      | No    |
+-------------------------------------------------+------+-------+-------+--------+-------+-------+---------+-------+
| ``__builtin_amdgcn_load_to_lds``                | No   | No    | Yes   | Yes    | No    | No    | No      | No    |
+-------------------------------------------------+------+-------+-------+--------+-------+-------+---------+-------+
| ``__builtin_amdgcn_raw_ptr_buffer_load_lds``    | No   | No    | Yes   | Yes    | Yes   | No    | No      | No    |
+-------------------------------------------------+------+-------+-------+--------+-------+-------+---------+-------+
| ``__builtin_amdgcn_struct_ptr_buffer_load_lds`` | No   | No    | Yes   | Yes    | Yes   | No    | No      | No    |
+-------------------------------------------------+------+-------+-------+--------+-------+-------+---------+-------+

Flat-addressed intrinsics
=========================

These intrinsics take a pointer directly as the source address.

``__builtin_amdgcn_global_load_lds``
-------------------------------------

.. code-block:: cuda

   void __builtin_amdgcn_global_load_lds(const void* src,
                                          __shared__ void* dst_base,
                                          std::int32_t size,
                                          std::int32_t offset,
                                          std::int32_t aux);

Wavefront-wide gather from global memory to LDS, bypassing VGPRs.  Each lane
loads ``size`` bytes from its own global address; the hardware writes lane
*k*'s value to ``dst_base + offset + k * stride`` where ``stride = 4`` for
``size <= 4`` and (on CDNA4 GPUs) ``16`` for ``size > 4``.

+--------------+---------------------------------------------------------------+
| Parameter    | Description                                                   |
+==============+===============================================================+
| ``src``      | Global address space pointer. Each lane can hold a different  |
|              | address.                                                      |
+--------------+---------------------------------------------------------------+
| ``dst_base`` | LDS address space base pointer. Must be wave-uniform. The     |
|              | pointer is implicitly offset by ``lane_id * stride``.         |
|              | ``stride`` is ``4`` for ``size <= 4`` and (on CDNA4 GPUs)     |
|              | ``16`` for ``size > 4``.                                      |
+--------------+---------------------------------------------------------------+
| ``size``     | Transfer size per lane in bytes. Must be a compile-time       |
|              | constant. CDNA3 supports 1, 2, and 4.  CDNA4 additionally     |
|              | supports 12 and 16.                                           |
+--------------+---------------------------------------------------------------+
| ``offset``   | Signed byte offset applied to **both** ``src`` and            |
|              | ``dst_base``.  Must be a compile-time constant. Encoded as a  |
|              | 13-bit signed immediate (valid range: ``-4096`` to ``4095``). |
+--------------+---------------------------------------------------------------+
| ``aux``      | Cache policy bits.  Must be a compile-time constant.  See     |
|              | :ref:`direct-to-lds-cache-policy`.                            |
+--------------+---------------------------------------------------------------+

``__builtin_amdgcn_load_to_lds``
---------------------------------

.. code-block:: cuda

   void __builtin_amdgcn_load_to_lds(const void* src,
                                      __shared__ void* dst_base,
                                      std::int32_t size,
                                      std::int32_t offset,
                                      std::int32_t aux);

Wavefront-wide gather from generic, global, or buffer memory to LDS, bypassing
VGPRs.  Each lane loads ``size`` bytes from its own address; the hardware
writes lane *k*'s value to ``dst_base + offset + k * stride`` where
``stride = 4`` for ``size <= 4`` and (on CDNA4 GPUs) ``16`` for ``size > 4``.

+--------------+---------------------------------------------------------------+
| Parameter    | Description                                                   |
+==============|===============================================================|
| ``src``      | Generic, global, or buffer address pointer. Each lane can     |
|              | hold a different address.                                     |
+--------------+---------------------------------------------------------------+
| ``dst_base`` | LDS address space base pointer. Must be wave-uniform. The     |
|              | pointer is implicitly offset by ``lane_id * stride``.         |
|              | ``stride`` is ``4`` for ``size <= 4`` and (on CDNA4 GPUs)     |
|              | ``16`` for ``size > 4``.                                      |
+--------------+---------------------------------------------------------------+
| ``size``     | Transfer size per lane in bytes. Must be a compile-time       |
|              | constant. CDNA3 supports 1, 2, and 4.  CDNA4 additionally     |
|              | supports 12 and 16.                                           |
+--------------+---------------------------------------------------------------+
| ``offset``   | Signed byte offset applied to **both** ``src`` and            |
|              | ``dst_base``.  Must be a compile-time constant. Encoded as a  |
|              | 13-bit signed immediate (valid range: ``-4096`` to ``4095``). |
+--------------+---------------------------------------------------------------+
| ``aux``      | Cache policy bits.  Must be a compile-time constant.  See     |
|              | :ref:`direct-to-lds-cache-policy`.                            |
+--------------+---------------------------------------------------------------+

Buffer-addressed intrinsics
============================

These intrinsics take a buffer resource descriptor as the source, created with
``__builtin_amdgcn_make_buffer_rsrc``.  The three offset parameters (per-lane,
wave-uniform, and compile-time) are combined to form the final source address.

.. note::

   The buffer-addressed intrinsics are available on CDNA, CDNA2, CDNA3, CDNA4,
   and RDNA2.  They are **not supported** on RDNA3, RDNA3.5, or RDNA4.

``__builtin_amdgcn_raw_ptr_buffer_load_lds``
---------------------------------------------

.. code-block:: cuda

   void __builtin_amdgcn_raw_ptr_buffer_load_lds(
       __amdgpu_buffer_rsrc_t src,
       __shared__ void*       dst_base,
       std::int32_t           size,
       std::int32_t           voffset,
       std::int32_t           soffset,
       std::int32_t           offset,
       std::int32_t           aux);

Buffer-addressed direct-to-LDS load.  The source is described by a buffer
resource descriptor created with ``__builtin_amdgcn_make_buffer_rsrc``.

For each lane in a wavefront, the **load address** (global source) is:

.. code-block:: text

   src_addr = src_base + soffset + offset + voffset + stride * lane_id

``src_base`` and ``stride`` come from the buffer resource descriptor.  The
``lane_id`` term is only active if the corresponding flag was set when creating
the resource descriptor; otherwise it evaluates to zero.

For each lane in a wavefront, the **store address** (LDS destination) is:

.. code-block:: text

   dst_addr = dst_base + offset + lane_id * 4    // size <= 4
   dst_addr = dst_base + offset + lane_id * 16   // size > 4 (CDNA4 only)

The ``lane_id`` term in the LDS address is always active.

+--------------+---------------------------------------------------------------+
| Parameter    | Description                                                   |
+==============+===============================================================+
| ``src``      | Buffer resource descriptor (``__amdgpu_buffer_rsrc_t``).      |
|              | Created with ``__builtin_amdgcn_make_buffer_rsrc``.  The      |
|              | descriptor wraps a base pointer, stride, extent, and flags.   |
+--------------+---------------------------------------------------------------+
| ``dst_base`` | LDS address space base pointer. Must be wave-uniform. The     |
|              | hardware adds ``lane_id * stride`` implicitly.                |
+--------------+---------------------------------------------------------------+
| ``size``     | Transfer size per lane in bytes.  Must be a compile-time      |
|              | constant. CDNA, CDNA2, CDNA3, and RDNA2 support 1, 2, and 4.  |
|              | CDNA4 additionally supports 12 and 16.                        |
+--------------+---------------------------------------------------------------+
| ``voffset``  | Per-lane byte offset into the buffer. Each lane can provide a |
|              | different value, enabling gather-style reads from the buffer. |
+--------------+---------------------------------------------------------------+
| ``soffset``  | Wave-uniform byte offset into the buffer.                     |
+--------------+---------------------------------------------------------------+
| ``offset``   | Unsigned compile-time byte offset.  Encoded as a 12-bit       |
|              | unsigned immediate (valid range: ``0`` to ``4095``).          |
+--------------+---------------------------------------------------------------+
| ``aux``      | Cache policy bits (compile-time constant).  Bit 3 enables     |
|              | buffer swizzling on all supported architectures.  The         |
|              | remaining bits differ by architecture. See                    |
|              | :ref:`direct-to-lds-cache-policy`.                            |
+--------------+---------------------------------------------------------------+

``__builtin_amdgcn_struct_ptr_buffer_load_lds``
------------------------------------------------

.. code-block:: cuda

   void __builtin_amdgcn_struct_ptr_buffer_load_lds(
       __amdgpu_buffer_rsrc_t src,
       __shared__ void*       dst_base,
       std::int32_t           size,
       std::int32_t           vindex,
       std::int32_t           voffset,
       std::int32_t           soffset,
       std::int32_t           offset,
       std::int32_t           aux);

Structured-buffer direct-to-LDS load.  The source is described by a buffer
resource descriptor created with ``__builtin_amdgcn_make_buffer_rsrc``.
The ``vindex`` parameter provides a per-lane index that is combined with
``lane_id`` in the source address calculation, enabling structured access
patterns over the buffer.

For each lane in a wavefront, the **load address** (global source) is:

.. code-block:: text

   src_addr = src_base + soffset + offset + voffset + stride * (vindex + lane_id)

``src_base`` and ``stride`` come from the buffer resource descriptor.  The
``lane_id`` term is only active if the corresponding flag was set when creating
the resource descriptor; otherwise it evaluates to zero.

For each lane in a wavefront, the **store address** (LDS destination) is:

.. code-block:: text

   dst_addr = dst_base + offset + lane_id * 4    // size <= 4
   dst_addr = dst_base + offset + lane_id * 16   // size > 4 (CDNA4 only)

The ``lane_id`` term in the LDS address is always active.

+--------------+---------------------------------------------------------------+
| Parameter    | Description                                                   |
+==============+===============================================================+
| ``src``      | Buffer resource descriptor (``__amdgpu_buffer_rsrc_t``).      |
|              | Created with ``__builtin_amdgcn_make_buffer_rsrc``.  The      |
|              | descriptor wraps a base pointer, stride, extent, and flags.   |
+--------------+---------------------------------------------------------------+
| ``dst_base`` | LDS address space base pointer. Must be wave-uniform. The     |
|              | hardware adds ``lane_id * stride`` implicitly.                |
+--------------+---------------------------------------------------------------+
| ``size``     | Transfer size per lane in bytes.  Must be a compile-time      |
|              | constant. CDNA, CDNA2, CDNA3, and RDNA2 support 1, 2, and 4.  |
|              | CDNA4 additionally supports 12 and 16.                        |
+--------------+---------------------------------------------------------------+
| ``vindex``   | Per-lane index value.                                         |
+--------------+---------------------------------------------------------------+
| ``voffset``  | Per-lane byte offset into the buffer. Each lane can provide a |
|              | different value, enabling gather-style reads from the buffer. |
+--------------+---------------------------------------------------------------+
| ``soffset``  | Wave-uniform byte offset into the buffer.                     |
+--------------+---------------------------------------------------------------+
| ``offset``   | Unsigned compile-time byte offset.  Encoded as a 12-bit       |
|              | unsigned immediate (valid range: ``0`` to ``4095``).          |
+--------------+---------------------------------------------------------------+
| ``aux``      | Cache policy bits (compile-time constant).  Bit 3 enables     |
|              | buffer swizzling on all supported architectures.  The         |
|              | remaining bits differ by architecture. See                    |
|              | :ref:`direct-to-lds-cache-policy`.                            |
+--------------+---------------------------------------------------------------+

.. _direct-to-lds-cache-policy:

Cache policy (``aux`` parameter)
=================================

All direct-to-LDS intrinsics accept an ``aux`` parameter whose individual bits
control cache scope and temporal reuse hints.  The bit layout differs by
architecture generation.  For the buffer-addressed intrinsics, bit 3 enables
buffer swizzling on all supported architectures.

In a single-GPU kernel, wave scope with temporal reuse (``aux = 0``) is the
typical choice.

.. _direct-to-lds-cache-policy-cdna3-cdna4:

CDNA3 and CDNA4
-----------------

Three control bits are relevant:

* **SC0** (bit 0) and **SC1** (bit 4): together define the coherence scope.
* **NT** (bit 1, non-temporal): ``0`` = expect temporal reuse, ``1`` = do not
  expect temporal reuse.

.. list-table::
   :header-rows: 1
   :widths: 10 8 8 8 22 22 22

   * - Scope
     - SC1
     - SC0
     - NT
     - L1 cache behavior
     - L2 cache behavior
     - Last-level cache behavior
   * - Wave
     - 0
     - 0
     - 0
     - Hit LRU
     - Hit LRU
     - Hit LRU
   * - Wave
     - 0
     - 0
     - 1
     - Miss Evict
     - Hit Stream
     - Hit Evict
   * - Group
     - 0
     - 1
     - 0
     - Hit LRU (Miss LRU with ``tgsplit``)
     - Hit LRU
     - Hit Evict (Hit LRU with ``tgsplit``)
   * - Group
     - 0
     - 1
     - 1
     - Miss Evict
     - Hit Stream
     - Hit Evict
   * - Device
     - 1
     - 0
     - 0
     - Miss Evict
     - Hit LRU (1 L2) or coherent cache bypass (>1 L2)
     - Hit LRU
   * - Device
     - 1
     - 0
     - 1
     - Miss Evict
     - Hit Stream (1 L2) or coherent cache bypass (>1 L2)
     - Hit Evict
   * - System
     - 1
     - 1
     - 0
     - Miss Evict
     - Coherent cache bypass
     - Hit LRU
   * - System
     - 1
     - 1
     - 1
     - Miss Evict
     - Coherent cache bypass
     - Hit Evict

.. _direct-to-lds-cache-policy-cdna-cdna2:

CDNA and CDNA2
---------------

One control bit is relevant:

* **GLC** (bit 0): controls L1 cache behavior.

.. list-table::
   :header-rows: 1
   :widths: 10 45 45

   * - GLC
     - L1 cache behavior
     - Notes
   * - 0
     - The load can read data from the L1 cache.
     -
   * - 1
     - The load intentionally misses the L1 cache and reads from L2.  If a
       matching line exists in L1, it is invalidated and L2 is re-read.
     - Depending on alignment, L2 might not be re-read for every lane in the
       same wavefront.  If the address is aligned correctly, the first lane
       brings in the line from L2 (or beyond), and all other lanes in the
       wavefront read from the same L1 cache line.

.. _direct-to-lds-cache-policy-rdna2:

RDNA2
------

Three control bits are relevant:

* **GLC** (bit 0): controls L0 cache behavior (same semantics as CDNA/CDNA2
  GLC, but applied to L0 instead of L1).
* **SLC** (bit 1) and **DLC** (bit 2): control L2 and L1 cache behavior.

GLC behavior:

.. list-table::
   :header-rows: 1
   :widths: 10 45 45

   * - GLC
     - L0 cache behavior
     - Notes
   * - 0
     - The load can read data from the L0 cache.
     -
   * - 1
     - The load intentionally misses the L0 cache and reads from L2.  If a
       matching line exists in L0, it is invalidated and L2 is re-read.
     - Depending on alignment, L2 might not be re-read for every lane in the
       same wavefront.  If the address is aligned correctly, the first lane
       brings in the line from L2 (or beyond), and all other lanes in the
       wavefront read from the same L0 cache line.

SLC and DLC behavior:

.. list-table::
   :header-rows: 1
   :widths: 10 10 30 30

   * - SLC
     - DLC
     - L2 cache
     - L1 cache
   * - 0
     - 0
     - LRU
     - Hit LRU (reads can hit on previous data)
   * - 0
     - 1
     - LRU
     - Miss Evict (reads miss)
   * - 1
     - 0
     - Stream
     - Hit LRU
   * - 1
     - 1
     - Hit No Allocate
     - Miss Evict
