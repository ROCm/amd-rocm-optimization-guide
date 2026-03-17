.. meta::
  :description: HIP histogram optimization tutorial
  :keywords: AMD, ROCm, HIP, histogram, atomics, shared memory, local memory, LDS, vectorized loads, tutorial

.. _histogram:

*************************************************************
Histogram
*************************************************************

Histogram is a fundamental operation that counts how often each value (or range
of values) appears in an input dataset. It appears throughout GPU workloads
including image processing, radix sort, database operations, and machine
learning. The challenge on GPUs is that the output location of each write is
determined by the input value at runtime, meaning multiple threads may attempt
to update the same output bin simultaneously. Managing this concurrent access
efficiently is the central optimization problem.

This tutorial walks through a series of HIP kernels for computing a 256-bin
histogram over a large array of unsigned integers. Each version addresses a
specific performance bottleneck. Starting from a naive kernel that issues one
global atomic per thread, each step moves contention to faster memory and
increases the number of elements each thread processes.

Histogram fundamentals
======================

A histogram maps each element of an input array to one of ``num_bins`` output
counters, called bins, and increments that counter. Formally, for an input
sequence :math:`x_1, \ldots, x_N` and a bin mapping function :math:`f`, the
count for bin :math:`b` is:

.. math::

   H[b] = \sum_{i=1}^{N} \delta\bigl(b - \lfloor f(x_i) \rfloor\bigr)

where :math:`\delta(\cdot)` is 1 when its argument is zero and 0 otherwise.
For a simple integer input, :math:`f(x_i) = x_i \bmod B` maps each value to
one of :math:`B` bins by remainder.

The algorithm consists of three steps:

1. Read each input element.
2. Determine its bin.
3. Increment that bin's counter.

On a CPU this is straightforward. On a GPU, thousands of threads execute these
steps simultaneously, and many may land on the same bin at the same time.

Race conditions
===============

A **race condition** occurs when two or more threads attempt to
read-modify-write the same memory location concurrently. Consider two threads
that both want to increment ``histogram[bin]``:

.. code-block:: cpp

   histogram[bin] = histogram[bin] + 1;

If both threads read the value before either writes back, they both compute the
same result and one increment is silently discarded. The final count is lower
than it should be. Because GPU threads execute asynchronously across many
compute units, this interleaving can happen unpredictably, producing different
results on different runs.

Race conditions on the bin counters are inevitable when the input is large
enough that multiple threads map to the same bin. The solution is to make each
increment atomic.

Atomic operations
=================

An **atomic operation** executes a read-modify-write sequence as an
indivisible unit. No other thread can observe a partially completed operation
or interleave its own update between the read and write. From the hardware's
perspective, the memory arbitration unit locks the relevant cache line,
performs the update, and releases the lock. All competing threads observe
results as if the operations occurred in a single sequential order.

HIP provides a set of atomic primitives for both global and shared memory:

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Operation
     - Description
   * - ``atomicAdd``
     - Adds a value to a memory location and returns the old value.
   * - ``atomicSub``
     - Subtracts a value from a memory location and returns the old value.
   * - ``atomicExch``
     - Exchanges a register value with a memory location.
   * - ``atomicCAS``
     - Compares a memory location to an expected value and, if equal, replaces
       it with a new value. The fundamental building block for custom atomic
       operations.
   * - ``atomicMax`` / ``atomicMin``
     - Updates a memory location to the maximum or minimum of its current value
       and a given value.
   * - ``atomicInc`` / ``atomicDec``
     - Atomically increments or decrements a counter, wrapping at a boundary.

Atomic operations can target shared memory (block scope), global memory
(device scope), or system memory, depending on hardware support. For more
information, see the
`GPU atomics operations reference <https://rocm.docs.amd.com/en/latest/reference/gpu-atomics-operation.html>`_.

Naive kernel
============

The naive kernel issues one global ``atomicAdd`` per thread:

.. literalinclude:: ../../tools/example_codes/histogram.hip
   :language: cpp
   :linenos:
   :start-after: [Sphinx histogram naive kernel start]
   :end-before: [Sphinx histogram naive kernel end]

This is correct — the atomic eliminates the race condition — but slow. Global
memory atomics must travel through the full memory hierarchy. When many threads
target the same bin, the hardware serializes their updates: only one proceeds
at a time while the others stall. This is called **atomic contention**, and it
limits throughput proportionally to how many threads compete for the same
address.

Two factors make contention worse in practice:

- **Hot bins:** When the input distribution is skewed, a small number of bins
  receive a disproportionate share of increments. Every thread targeting a hot
  bin serializes against every other.

- **Warp serialization:** Within a warp, if multiple lanes map to the same bin,
  the hardware issues their atomic operations one at a time, stalling the whole
  warp until each completes.

For uniformly distributed data, average contention per bin is low, but each
atomic still pays full global memory latency — hundreds of cycles per
operation. This is the fundamental bottleneck in the naive kernel.

Local memory histogram
======================

Atomic operations in shared memory (the Local Data Share, or LDS, on AMD
GPUs) are an order of magnitude faster than global memory atomics because the
LDS is on-chip and directly connected to the compute units. The local memory
kernel exploits this by giving each block its own private histogram in LDS,
accumulating into it with fast local atomics, and only issuing one global
atomic per bin at the end of the block.

This two-phase approach has three steps:

1. Initialize the per-block histogram in LDS to zero.
2. Each thread atomically increments the appropriate LDS bin for its input
   element.
3. Each thread merges the LDS histogram into the global histogram using global
   atomics.

The global merge in step 3 still uses atomic operations, but it issues at most
``num_bins`` global atomics per block rather than one per thread. For a block
of 256 threads and 256 bins, this reduces global atomic traffic by a factor of
256.

.. literalinclude:: ../../tools/example_codes/histogram.hip
   :language: cpp
   :linenos:
   :start-after: [Sphinx histogram local kernel start]
   :end-before: [Sphinx histogram local kernel end]

The two ``__syncthreads()`` calls are necessary. The first ensures that all
threads have zeroed the LDS histogram before any thread begins accumulating.
The second ensures that all accumulations are complete before any thread reads
the LDS histogram for the global merge.

.. note::

   The LDS histogram occupies ``num_bins * sizeof(unsigned int)`` bytes. For
   256 bins this is 1 KB, well within the 64 KB of LDS available per Compute
   Unit on CDNA GPUs and per Work Group Processor on RDNA GPUs.

Vectorized loads
================

All kernels so far issue one 32-bit load per thread. The GPU memory system
can issue 128-bit loads at the same cost, so replacing four scalar loads with
a single ``uint4`` read quadruples the data moved per instruction. Each thread
loads four consecutive elements, increments four LDS bins, and then
participates in the same merge step as before.

This also reduces the number of blocks launched for the same input size,
which reduces the total number of global atomic merge operations.

.. literalinclude:: ../../tools/example_codes/histogram.hip
   :language: cpp
   :linenos:
   :start-after: [Sphinx histogram multi kernel start]
   :end-before: [Sphinx histogram multi kernel end]

The fast path uses a ``uint4`` reinterpret cast to load four elements in a
single 128-bit instruction. The slow path handles the tail of the input when
fewer than four elements per thread remain.

.. note::

   The ``uint4`` reinterpret cast requires the input pointer to be 16-byte
   aligned. Allocations from ``hipMalloc`` satisfy this requirement.

Best practices
==============

Use atomic operations only where necessary
------------------------------------------

Atomic instructions serialize access to a memory location and reduce SIMT
efficiency. Restrict atomic usage to code paths where data races cannot be
eliminated through algorithmic restructuring, such as the bin merge step.

Minimize contention
-------------------

High contention on a single address or small set of addresses leads to
serialization. The local memory approach reduces global atomic contention
significantly, but LDS atomics for hot bins can still serialize within a
block. If the input distribution is highly skewed, consider splitting each
block's LDS into per-warp sub-histograms and reducing them before the global
merge.

Validate correctness against a CPU reference
--------------------------------------------

Confirm GPU kernel results against a single-threaded CPU baseline. Atomic
operations prevent race conditions but do not protect against logic errors in
bin assignment. Testing with a known reference catches both categories of bug.

Profile atomic throughput
-------------------------

GPU performance is sensitive to atomic contention, memory access patterns, and
occupancy. Use
`rocprofv3 <https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/latest/>`_
or
`ROCm Compute Profiler <https://rocm.docs.amd.com/projects/rocprofiler-compute/en/latest/>`_
to examine warp stalls, memory-coalescing behavior, and atomic throughput
bottlenecks before tuning.

For production use, `rocPRIM <https://rocm.docs.amd.com/projects/rocPRIM/en/latest/index.html>`_
provides highly optimized histogram primitives that handle edge cases and apply
architecture-specific tuning automatically. The kernels in this tutorial are
intended to build intuition for the optimization principles those primitives
apply internally.
