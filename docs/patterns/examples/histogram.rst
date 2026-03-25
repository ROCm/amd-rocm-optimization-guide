.. meta::
  :description:  Learn how to optimize GPU histogram computation in HIP using shared memory atomics, LDS accumulation, and vectorized loads on AMD CDNA and RDNA GPUs.
  :keywords: AMD, ROCm, HIP, histogram, atomics, shared memory, LDS, vectorized loads, tutorial

.. _histogram:

*************************************************************
Histogram
*************************************************************

Histogram is an operation that counts how often each value (or range of
values) appears in an input dataset. It appears throughout GPU workloads,
including image processing, radix sort, database operations, and machine
learning. The challenge on GPUs is that the output location of each write is
determined by the input value at runtime, meaning multiple threads can attempt
to update the same output bin simultaneously. Efficiently managing this concurrent access
is the central optimization problem.

This tutorial walks through a series of HIP kernels for computing a 256-bin
histogram over a large array of unsigned integers. Starting from a naive kernel
that issues one global atomic per input element, each step reduces global atomic
traffic: first by moving accumulation into shared memory, then by having each
thread process more elements so fewer blocks are launched. The final approach
eliminates global atomics at the merge step entirely by having each block write
its local histogram to a private slice of a temporary buffer, then summing those
slices in a separate reduction kernel.

Histogram fundamentals
======================

A histogram maps each element of an input array to one of ``num_bins`` output
counters, called bins, and increments that counter. Formally, for an input
sequence :math:`x_1, \ldots, x_N` and a bin mapping function :math:`f`, the
count for bin :math:`b` is:

.. math::

   H[b] = \sum_{i=1}^{N} \delta\bigl(b - \lfloor f(x_i) \rfloor\bigr)

where :math:`\delta( )` is 1 when its argument is zero and 0 otherwise.
For a simple integer input, :math:`f(x_i) = x_i \bmod B` maps each value to
one of :math:`B` bins by remainder.

The algorithm consists of three steps:

1. Read each input element.
2. Determine its bin.
3. Increment that bin's counter.

On a CPU, this is straightforward. On a GPU, thousands of threads execute these
steps simultaneously, and many might land in the same bin at the same time.

Race conditions
===============

A race condition occurs when two or more threads concurrently attempt to read-modify-write
the same memory location. Consider two threads that both want to
increment ``histogram[bin]``:

.. code-block:: cpp

   histogram[bin] = histogram[bin] + 1;

If both threads read the value before either writes back, they both compute the
same result, and the second write overwrites the first, so one increment is
lost. This is a natural consequence of parallel execution: with many threads
running concurrently across compute units, some will read the same value before
any of them writes back.

When multiple threads map to the same bin, this kind of overlap is expected.
Atomic operations, covered in the next section, are the standard solution.

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
   * - ``atomicMax``/``atomicMin``
     - Updates a memory location to the maximum or minimum of its current value
       and a given value.
   * - ``atomicInc``/``atomicDec``
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

Global memory atomics must traverse the full memory hierarchy. When many
threads target the same bin, the hardware serializes their updates. That is, only one can
proceed at a time while the others stall. This is called **atomic contention**,
and it limits throughput in proportion to the number of threads competing for the same
address.

Two factors make contention worse in practice:

- **Hot bins:** When the input distribution is skewed, a small number of bins
  receive a disproportionate share of increments. Every thread targeting a hot
  bin serializes against every other thread.

- **Warp serialization:** Within a warp, if multiple lanes map to the same bin,
  the hardware issues their atomic operations one at a time, stalling the whole
  warp until each completes.

The example code uses a skewed input where every fourth element is fixed to
bin 1, to reflect a realistic distribution in which one bin is significantly
busier than the others.

Shared memory histogram
=======================

Atomic operations in shared memory (the Local Data Share, or LDS, on AMD
GPUs) are an order of magnitude faster than global memory atomics because the
LDS is on-chip and directly connected to the compute units. The shared memory
kernel reduces global atomic traffic in two independent ways: moving
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
   :start-after: [Sphinx histogram shared kernel start]
   :end-before: [Sphinx histogram shared kernel end]

The inner loop uses the stride ``i * blockDim.x``, so consecutive threads in a
warp always read consecutive memory addresses in each iteration, so the access
pattern remains coalesced throughout. Because ``ITEMS_PER_THREAD`` is a
compile-time constant, ``#pragma unroll`` allows the compiler to eliminate the
loop counter and branch overhead.

With ``block_size = 256`` and ``ITEMS_PER_THREAD = 16``, each block covers
4,096 input elements. For a 16 M-element input, this launches 4,096 blocks,
compared to 65,536 for the naive kernel. The global merge step issues at most
``num_bins`` atomics per block: 256 per block × 4,096 blocks = ~1 M global
atomics total, versus 16 M for the naive kernel.

.. note::

   The LDS histogram occupies ``num_bins * sizeof(unsigned int)`` bytes. For
   256 bins, this is 1 KB, well within the 64 KB of LDS available per Compute
   Unit on CDNA GPUs and per Work Group Processor on RDNA GPUs.

Partial histograms
==================

The shared memory kernel still issues up to ``num_bins`` global atomics per
block during the merge phase. For 4,096 blocks and 256 bins, that is roughly
one million global atomic operations.

A partial histogram avoids this by deferring the merge entirely. Instead of
each block atomically adding its local counts into a single shared output array,
each block writes its ``num_bins`` counts to its own reserved slice of a
temporary ``partial_histogram`` buffer — using plain stores, with no
contention. The global output is then computed in a second kernel that sums
across all the per-block slices for each bin. Because the two passes are
separated, neither requires global atomics: the first pass uses conflict-free
stores, and the second pass is a straightforward parallel reduction (see
:ref:`reduction`).

The first kernel is identical to the shared memory kernel except for the merge
step, which becomes a plain store rather than a global atomic:

.. literalinclude:: ../../tools/example_codes/histogram.hip
   :language: cpp
   :linenos:
   :start-after: [Sphinx histogram partial kernel start]
   :end-before: [Sphinx histogram partial kernel end]

The ``partial_histogram`` array has ``num_blocks * num_bins`` elements, laid
out as ``partial_histogram[blockIdx.x * num_bins + bin]``. Writing a full row
of ``num_bins`` consecutive values per block keeps the stores coalesced.

The second kernel assigns one block to each bin. Threads within each block
accumulate their share of the partial results with a stride loop, then reduce
to a single bin count using a shared memory tree reduction:

.. literalinclude:: ../../tools/example_codes/histogram.hip
   :language: cpp
   :linenos:
   :start-after: [Sphinx histogram reduce kernel start]
   :end-before: [Sphinx histogram reduce kernel end]

The kernel launches ``num_bins`` blocks of ``block_size`` threads each. Each
thread accumulates every ``blockDim.x``-th partial block into a register
accumulator, writes the result to shared memory, and participates in the tree
reduction. Thread 0 of each block writes the final bin count. The access
pattern ``partial_histogram[i * num_bins + bin]`` strides by ``num_bins``
between iterations, so consecutive threads in a warp read consecutive addresses
so the reads are coalesced.

.. note::

   The ``partial_histogram`` buffer requires ``num_blocks * num_bins *
   sizeof(unsigned int)`` bytes of device memory. With ``ITEMS_PER_THREAD =
   16``, ``block_size = 256``, and a 16 M-element input, this is 4,096 blocks
   × 256 bins × 4 bytes = 4 MB.

The results on an AMD Radeon (RDNA3-based) GPU show how ``ITEMS_PER_THREAD`` affects each kernel.
Relative performance is measured as kernel time divided by naive kernel time
from the same run, so lower values indicate faster execution.

.. list-table::
   :header-rows: 1
   :widths: 25 25 25 25

   * - Items per thread
     - Naive
     - Shared memory
     - Partial + reduce
   * - 1
     - 1.00
     - 2.57
     - 3.27
   * - 2
     - 1.00
     - 1.60
     - 1.95
   * - 4
     - 1.00
     - 1.10
     - 1.29
   * - 8
     - 1.00
     - 0.87
     - 0.97
   * - 16
     - 1.00
     - 0.77
     - 0.83
   * - 32
     - 1.00
     - 0.72
     - 0.76
   * - 64
     - 1.00
     - 0.73
     - 0.75

At low ``ITEMS_PER_THREAD`` values, each block covers only a small portion of
the input, so many blocks are launched and the global atomic merge in the shared
memory kernel, or the partial histogram buffer and second kernel launch in the
two-pass approach, dominates runtime. Performance improves steadily up to
``ITEMS_PER_THREAD = 32``, where the block count is low enough that merge
overhead is no longer the bottleneck. Above 32, the benefit plateaus because
the kernel becomes compute-bound within each block rather than overhead-bound.

The two-pass approach consistently runs slightly slower than the shared memory
kernel at the same ``ITEMS_PER_THREAD`` approach. Eliminating the global atomics at
the merge step saves some cost. Still, the additional kernel launch, the larger
temporary buffer, and the tree reduction in the second kernel together exceed
that saving on this workload.

For production use, `rocPRIM <https://rocm.docs.amd.com/projects/rocPRIM/en/latest/index.html>`_
provides highly optimized histogram primitives that handle edge cases and automatically apply
architecture-specific tuning. The kernels in this tutorial are
intended to build intuition for the optimization principles, those primitives
apply internally.
