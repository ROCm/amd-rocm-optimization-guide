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

.. list-table::
   :header-rows: 1
   :widths: auto

   * - Parameter
     - Description
   * - ``src``
     - Pointer to global address space. Each lane can hold a different address.
   * - ``dst_base``
     - Pointer to LDS address space base. Must be wave-uniform. The pointer is
       implicitly offset by ``lane_id * stride``. ``stride`` is ``4`` for
       ``size <= 4`` and (on CDNA4 GPUs) ``16`` for ``size > 4``.
   * - ``size``
     - Transfer size per lane in bytes. Must be a compile-time constant. CDNA3
       supports 1, 2, and 4. CDNA4 additionally supports 12 and 16.
   * - ``offset``
     - Signed byte offset applied to **both** ``src`` and ``dst_base``. Must be
       a compile-time constant. Encoded as a 13-bit signed immediate (valid
       range: ``-4096`` to ``4095``).
   * - ``aux``
     - Cache policy bits. Must be a compile-time constant. See
       :ref:`direct-to-lds-cache-policy`.

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

.. list-table::
   :header-rows: 1
   :widths: auto

   * - Parameter
     - Description
   * - ``src``
     - Generic, global, or buffer address pointer. Each lane can hold a
       different address.
   * - ``dst_base``
     - Pointer to LDS address space base. Must be wave-uniform. The pointer is
       implicitly offset by ``lane_id * stride``. ``stride`` is ``4`` for
       ``size <= 4`` and (on CDNA4 GPUs) ``16`` for ``size > 4``.
   * - ``size``
     - Transfer size per lane in bytes. Must be a compile-time constant. CDNA3
       supports 1, 2, and 4. CDNA4 additionally supports 12 and 16.
   * - ``offset``
     - Signed byte offset applied to **both** ``src`` and ``dst_base``. Must be
       a compile-time constant. Encoded as a 13-bit signed immediate (valid
       range: ``-4096`` to ``4095``).
   * - ``aux``
     - Cache policy bits. Must be a compile-time constant. See
       :ref:`direct-to-lds-cache-policy`.

Buffer-addressed intrinsics
============================

These intrinsics take a buffer resource descriptor as the source, created with
``__builtin_amdgcn_make_buffer_rsrc``.  The three offset parameters (per-lane,
wave-uniform, and compile-time) are combined to form the final source address.

.. note::

   The buffer-addressed intrinsics are available on CDNA, CDNA2, CDNA3, CDNA4,
   and RDNA2.  They're not supported on RDNA3, RDNA3.5, or RDNA4.

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

For each lane in a wavefront, the **load address** (global source) is:

.. code-block:: text

   src_addr = src_base + soffset + offset + voffset + stride * lane_id

``src_base`` and ``stride`` come from the buffer resource descriptor.  The
``lane_id`` term is active only if the corresponding flag was set when creating
the resource descriptor. Otherwise, it evaluates to zero.

For each lane in a wavefront, the **store address** (LDS destination) is:

.. code-block:: text

   dst_addr = dst_base + offset + lane_id * 4    // size <= 4
   dst_addr = dst_base + offset + lane_id * 16   // size > 4 (CDNA4 only)

The ``lane_id`` term in the LDS address is always active.

.. list-table::
   :header-rows: 1
   :widths: auto

   * - Parameter
     - Description
   * - ``src``
     - Buffer resource descriptor (``__amdgpu_buffer_rsrc_t``). Created with
       ``__builtin_amdgcn_make_buffer_rsrc``. The descriptor wraps a base
       pointer, stride, extent, and flags.
   * - ``dst_base``
     - Pointer to LDS address space base. Must be wave-uniform. The hardware
       adds ``lane_id * stride`` implicitly.
   * - ``size``
     - Transfer size per lane in bytes. Must be a compile-time constant. CDNA,
       CDNA2, CDNA3, and RDNA2 support 1, 2, and 4. CDNA4 additionally
       supports 12 and 16.
   * - ``voffset``
     - Per-lane byte offset into the buffer. Each lane can provide a different
       value, enabling gather-style reads from the buffer.
   * - ``soffset``
     - Wave-uniform byte offset into the buffer.
   * - ``offset``
     - Unsigned compile-time byte offset. Encoded as a 12-bit unsigned
       immediate (valid range: ``0`` to ``4095``).
   * - ``aux``
     - Cache policy bits (compile-time constant). Bit 3 enables buffer
       swizzling on all supported architectures. The remaining bits differ by
       architecture. See :ref:`direct-to-lds-cache-policy`.

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
``lane_id`` term is active only if the corresponding flag was set when creating
the resource descriptor. Otherwise, it evaluates to zero.

For each lane in a wavefront, the **store address** (LDS destination) is:

.. code-block:: text

   dst_addr = dst_base + offset + lane_id * 4    // size <= 4
   dst_addr = dst_base + offset + lane_id * 16   // size > 4 (CDNA4 only)

The ``lane_id`` term in the LDS address is always active.

.. list-table::
   :header-rows: 1
   :widths: auto

   * - Parameter
     - Description
   * - ``src``
     - Buffer resource descriptor (``__amdgpu_buffer_rsrc_t``). Created with
       ``__builtin_amdgcn_make_buffer_rsrc``. The descriptor wraps a base
       pointer, stride, extent, and flags.
   * - ``dst_base``
     - Pointer to LDS address space base. Must be wave-uniform. The hardware
       adds ``lane_id * stride`` implicitly.
   * - ``size``
     - Transfer size per lane in bytes. Must be a compile-time constant. CDNA,
       CDNA2, CDNA3, and RDNA2 support 1, 2, and 4. CDNA4 additionally
       supports 12 and 16.
   * - ``vindex``
     - Per-lane index value.
   * - ``voffset``
     - Per-lane byte offset into the buffer. Each lane can provide a different
       value, enabling gather-style reads from the buffer.
   * - ``soffset``
     - Wave-uniform byte offset into the buffer.
   * - ``offset``
     - Unsigned compile-time byte offset. Encoded as a 12-bit unsigned
       immediate (valid range: ``0`` to ``4095``).
   * - ``aux``
     - Cache policy bits (compile-time constant). Bit 3 enables buffer
       swizzling on all supported architectures. The remaining bits differ by
       architecture. See :ref:`direct-to-lds-cache-policy`.

Architecture availability
=========================

The following table summarizes which intrinsics are available for each
ROCm-supported GPU architecture. For the list of supported GPU models, see
:ref:`rocm:system_requirements`.

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
   * - ``__builtin_amdgcn_global_load_lds``
     - No
     - No
     - Yes
     - Yes
     - No
     - No
     - No
     - No
   * - ``__builtin_amdgcn_load_to_lds``
     - No
     - No
     - Yes
     - Yes
     - No
     - No
     - No
     - No
   * - ``__builtin_amdgcn_raw_ptr_buffer_load_lds``
     - No
     - No
     - Yes
     - Yes
     - Yes
     - No
     - No
     - No
   * - ``__builtin_amdgcn_struct_ptr_buffer_load_lds``
     - No
     - No
     - Yes
     - Yes
     - Yes
     - No
     - No
     - No

.. _direct-to-lds-cache-policy:

Cache policy (``aux`` parameter)
=================================

All direct-to-LDS intrinsics accept an ``aux`` parameter whose individual bits
control cache scope and temporal reuse hints.  The bit layout differs by
architecture generation.  For the buffer-addressed intrinsics, bit 3 enables
buffer swizzling on all supported architectures.

For background on AMD GPU cache hierarchy and coherence scopes, see the
`CDNA3 Instruction Set Architecture <https://gpuopen.com/amd-cdna3-white-paper/>`_
white paper and the ISA reference guides published on
`GPUOpen <https://gpuopen.com/>`_.

In a single-GPU kernel, wave scope with temporal reuse (``aux = 0``) is the
typical choice.

.. _direct-to-lds-cache-policy-cdna3-cdna4:

CDNA3 and CDNA4
-----------------

Three control bits are relevant:

* **SC0** (Scope Control 0, bit 0) and **SC1** (Scope Control 1, bit 4):
  together define the coherence scope.
* **NT** (Non-Temporal, bit 1): ``0`` = expect temporal reuse, ``1`` = do not
  expect temporal reuse.

.. list-table::
   :header-rows: 1
   :widths: auto

   * - SC1
     - SC0
     - NT
     - L1 cache behavior
     - L2 cache behavior
     - Last-level cache behavior
   * - 0
     - 0
     - 0
     - Hit LRU
     - Hit LRU
     - Hit LRU
   * - 0
     - 0
     - 1
     - Miss Evict
     - Hit Stream
     - Hit Evict
   * - 0
     - 1
     - 0
     - Hit LRU / Miss LRU [#t]_
     - Hit LRU
     - Hit Evict / Hit LRU [#t]_
   * - 0
     - 1
     - 1
     - Miss Evict
     - Hit Stream
     - Hit Evict
   * - 1
     - 0
     - 0
     - Miss Evict
     - Hit LRU (1 L2) or coherent cache bypass (>1 L2)
     - Hit LRU
   * - 1
     - 0
     - 1
     - Miss Evict
     - Hit Stream (1 L2) or coherent cache bypass (>1 L2)
     - Hit Evict
   * - 1
     - 1
     - 0
     - Miss Evict
     - Coherent cache bypass
     - Hit LRU
   * - 1
     - 1
     - 1
     - Miss Evict
     - Coherent cache bypass
     - Hit Evict

.. [#t] When compiled with ``tgsplit+``.

The following terms describe cache line behavior in the table above:

* **Hit LRU** -- cache the data and replace the least recently used line when
  the cache is full. This is the normal temporal caching behavior.
* **Miss Evict** -- bypass the cache level; fetch data directly from the next
  level.
* **Hit Stream** -- on a cache hit, serve the data and evict the line; on a
  cache miss, fetch the data into the cache, use it once, then discard it.
* **Hit Evict** -- on a cache hit, immediately discard the line after serving
  the request.
* **Coherent cache bypass** -- when reading a cache line: if the line is
  modified, write it back to memory and reissue the read; if not modified,
  discard it and fetch a new line.

.. _direct-to-lds-cache-policy-cdna-cdna2:

CDNA and CDNA2
---------------

One control bit is relevant:

* **GLC** (Global Level Coherent, bit 0): controls the first-level cache (L1).

.. list-table::
   :header-rows: 1
   :widths: auto

   * - GLC
     - L1 cache behavior
     - Notes
   * - 0
     - The load can read data from the L1 cache.
     -
   * - 1
     - The load intentionally misses the L1 cache and reads from L2. If a
       matching line exists in L1, it is invalidated and L2 is re-read.
     - Depending on alignment, L2 might not be re-read for every lane in the
       same wavefront. If the address is aligned correctly, the first lane
       brings in the line from L2 (or beyond), and all other lanes in the
       wavefront read from the same L1 cache line.

.. _direct-to-lds-cache-policy-rdna2:

RDNA2
------

Three control bits are relevant:

* **GLC** (Global Level Coherent, bit 0): controls the first-level cache (L0).
* **SLC** (System Level Coherent, bit 1) and **DLC** (Device Level Coherent,
  bit 2): together control the behavior of L1 and L2.

GLC behavior:

.. list-table::
   :header-rows: 1
   :widths: auto

   * - GLC
     - L0 cache behavior
     - Notes
   * - 0
     - The load can read data from the L0 cache.
     -
   * - 1
     - The load intentionally misses the L0 cache and reads from L2. If a
       matching line exists in L0, it is invalidated and L2 is re-read.
     - Depending on alignment, L2 might not be re-read for every lane in the
       same wavefront. If the address is aligned correctly, the first lane
       brings in the line from L2 (or beyond), and all other lanes in the
       wavefront read from the same L0 cache line.

SLC and DLC behavior:

.. list-table::
   :header-rows: 1
   :widths: auto

   * - SLC
     - DLC
     - L2 cache
     - L1 cache
   * - 0
     - 0
     - Hit LRU
     - Hit LRU
   * - 0
     - 1
     - Hit LRU
     - Miss Evict
   * - 1
     - 0
     - Hit Stream
     - Hit LRU
   * - 1
     - 1
     - Hit No Allocate
     - Miss Evict

The following terms describe cache line behavior in the table above:

* **Hit LRU** -- cache the data and replace the least recently used line when
  the cache is full. This is the normal temporal caching behavior.
* **Miss Evict** -- bypass the cache level; fetch data directly from the next
  level.
* **Hit Stream** -- on a cache hit, serve the data and evict the line; on a
  cache miss, fetch the data into the cache, use it once, then discard it.
* **Hit No Allocate** -- on a cache hit, serve the data and evict the line;
  on a cache miss, fetch the data directly from the next memory level without
  allocating a cache line.

.. tip::

   When choosing between *Hit Stream* and *Hit No Allocate*, consider the
   spatial locality of the access pattern. For contiguous or regularly-strided
   reads, prefer *Hit Stream*: even though each cache line is discarded after
   one use, a single memory fetch serves multiple lanes that share the same
   line, so the fetch cost is amortized. For scattered or random reads, prefer
   *Hit No Allocate*: each fetch is unlikely to benefit more than one lane, so
   skipping cache allocation avoids polluting L2 with lines that will never be
   reused.