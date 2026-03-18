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
histogram over a large array of unsigned integers. Starting from a naive kernel
that issues one global atomic per input element, each step reduces global atomic
traffic: first by moving accumulation into shared memory, then by having each
thread process more elements so fewer blocks — and therefore fewer merge
operations — are needed.

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

A race condition occurs when two or more threads attempt to read-modify-write
the same memory location concurrently. Consider two threads that both want to
increment ``histogram[bin]``:

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

An atomic operation executes a read-modify-write sequence as an indivisible
unit. No other thread can observe a partially completed operation or interleave
its own update between the read and write. From the hardware's perspective,
the memory arbitration unit locks the relevant cache line, performs the update,
and releases the lock. All competing threads observe results as if the
operations occurred in a single sequential order.

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

Global memory atomics must travel through the full memory hierarchy. When many
threads target the same bin, the hardware serializes their updates: only one
proceeds at a time while the others stall. This is called **atomic contention**,
and it limits throughput proportionally to how many threads compete for the same
address.

Two factors make contention worse in practice:

- **Hot bins:** When the input distribution is skewed, a small number of bins
  receive a disproportionate share of increments. Every thread targeting a hot
  bin serializes against every other.

- **Warp serialization:** Within a warp, if multiple lanes map to the same bin,
  the hardware issues their atomic operations one at a time, stalling the whole
  warp until each completes.

The example code uses a skewed input — every fourth element is fixed to bin 1
— to reflect a realistic distribution where one bin is significantly busier
than the rest.

Local memory histogram
======================

Atomic operations in shared memory (the Local Data Share, or LDS, on AMD
GPUs) are an order of magnitude faster than global memory atomics because the
LDS is on-chip and directly connected to the compute units. The local memory
kernel reduces global atomic traffic by two independent means: moving
accumulation into LDS, and having each thread process multiple elements so
fewer blocks are launched overall.

The kernel has three phases:

1. Initialize the per-block histogram in LDS to zero.
2. Each thread processes ``ITEMS_PER_THREAD`` input elements (default: 16),
   atomically incrementing the appropriate LDS bin for each.
3. One thread per bin merges the LDS histogram into the global histogram with a
   single global atomic.

.. literalinclude:: ../../tools/example_codes/histogram.hip
   :language: cpp
   :linenos:
   :start-after: [Sphinx histogram local kernel start]
   :end-before: [Sphinx histogram local kernel end]

The inner loop uses the stride ``i * blockDim.x``, so consecutive threads in a
warp always read consecutive memory addresses in each iteration — the access
pattern remains coalesced throughout. Because ``ITEMS_PER_THREAD`` is a
compile-time constant, ``#pragma unroll`` allows the compiler to eliminate the
loop counter and branch overhead.

With ``block_size = 256`` and ``ITEMS_PER_THREAD = 16``, each block covers
4,096 input elements. For a 16 M-element input this launches 4,096 blocks,
compared to 65,536 for the naive kernel. The global merge step issues at most
``num_bins`` atomics per block — 256 per block × 4,096 blocks = ~1 M global
atomics total, versus 16 M for the naive kernel.

.. note::

   The LDS histogram occupies ``num_bins * sizeof(unsigned int)`` bytes. For
   256 bins this is 1 KB, well within the 64 KB of LDS available per Compute
   Unit on CDNA GPUs and per Work Group Processor on RDNA GPUs.

Vectorized loads
================

The GPU memory system can issue 128-bit loads at the same cost as a 32-bit
load. Replacing four scalar reads with a single ``uint4`` instruction
quadruples the data fetched per instruction and reduces load-instruction
pressure. The vectorized kernel applies this to the local-memory approach with
four elements per thread:

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

With only four elements per thread, this kernel launches four times more blocks
than the local kernel at its default ``ITEMS_PER_THREAD = 16``, and therefore
issues four times more global atomic merge operations. On this workload, where
the bottleneck is global atomic traffic rather than load bandwidth, the local
kernel is faster despite its narrower loads. The results on an RDNA3 GPU
illustrate this:

.. list-table::
   :header-rows: 1
   :widths: 40 30 30

   * - Kernel
     - Items per thread
     - Time (ms)
   * - Naive (global atomics)
     - 1
     - 10.1
   * - Local memory
     - 16
     - 7.7
   * - Vectorized loads (``uint4``)
     - 4
     - 8.3

Widening the load is most effective when the kernel is bottlenecked on memory
bandwidth. Here the bottleneck is atomic serialization in the global merge, so
the primary lever is reducing block count — which ``ITEMS_PER_THREAD`` controls
directly. Compiling with a larger value (``-DITEMS_PER_THREAD=32``, for
example) further reduces the block count and merge traffic, at the cost of
higher register pressure.

For production use, `rocPRIM <https://rocm.docs.amd.com/projects/rocPRIM/en/latest/index.html>`_
provides highly optimized histogram primitives that handle edge cases and apply
architecture-specific tuning automatically. The kernels in this tutorial are
intended to build intuition for the optimization principles those primitives
apply internally.
