.. meta::
   :description: Introduction to AMD ROCm Compiler builtins and specialized GPU instructions, including MFMA matrix cores, WMMA, and dot product operations for CDNA and RDNA GPUs.
   :keywords: AMD, ROCm, HIP, Compiler builtins, GPU optimization, specialized instructions, CDNA, RDNA, MFMA, WMMA, dot product, wave-level operations

.. _compiler_builtins:

********************************************************************************
Introduction to Compiler builtins
********************************************************************************

In the previous chapters, you learned optimization patterns using standard HIP
programming constructs. You implemented reduction algorithms, histogram
computations, and matrix multiplications using shared memory, efficient memory
access patterns, and workgroup-level parallelism. These patterns teach
fundamental optimization principles that apply across GPU architectures.

This chapter introduces Compiler builtins that can significantly accelerate
the patterns you have already learned. Compiler builtins provide direct
access to specialized GPU units and instructions that compilers cannot
automatically generate. Think of this chapter as your hardware acceleration
toolkit, organized by architecture for quick reference when optimizing
specific patterns.

Why Compiler builtins matter
==============================

Modern compilers are sophisticated, but they cannot always use the full
range of hardware capabilities available on AMD GPUs. The compiler rarely
auto-vectorizes code to use dedicated dot product units, even when the pattern
is evident. Wave-level reduction hardware requires explicit builtin calls
rather than relying on compiler optimization. Matrix multiply-accumulate units
require direct invocation to achieve peak performance.

Builtins bridge this gap by giving you explicit control over specialized
operations. When used appropriately, they can provide orders-of-magnitude
performance improvements for compute-bound workloads.

How this chapter is organized
=============================

This chapter organizes builtins by their availability across AMD GPU
architectures. Cross-architecture builtins work on both CDNA and RDNA
architectures, providing arithmetic and packing operations (dot products, SAD,
type conversion), warp-level operations (shuffle, DPP, reductions, permlane,
vote), and direct-to-Local Data Share (LDS) memory transfers. CDNA-specific builtins cover
matrix operations (MFMA) with both dense and sparse variants. RDNA-specific
builtins cover wave-matrix multiply-accumulate (WMMA) operations, including
sparse variants on RDNA4.

Some builtins require specific hardware features beyond basic architecture
support, which are noted with feature flags where applicable.

When to use builtins
======================

Compiler builtins are most beneficial when:

**Compiler optimization is insufficient**
  The compiler cannot automatically vectorize your code to use specialized
  units, even when the pattern is obvious.

**Performance-critical loops dominate runtime**
  Operations in tight loops that account for most execution time can benefit
  significantly from hardware acceleration.

**Algorithmic patterns map to hardware features**
  Operations like reductions, matrix multiplications, and histograms have
  dedicated hardware support that builtins expose.

**You are targeting a specific GPU architecture**
  When you know your target architecture (CDNA or RDNA), you can use its
  specialized capabilities to maximize performance.

However, builtins introduce trade-offs that you should weigh against the
performance benefits:

**Reduced portability across architectures**
  Builtins tie your code to specific GPU architectures or architecture
  families. Code using MFMA builtins will not compile for RDNA targets, and
  WMMA builtins will not compile for CDNA targets. If your application must
  run across multiple architectures, you need separate code paths or an
  abstraction layer.

**Increased maintenance burden**
  Builtin interfaces can change between architecture generations. New GPU
  families may introduce different builtins for the same operation, requiring
  you to update and test multiple code paths. Standard HIP code, by contrast,
  benefits from compiler improvements automatically.

**Lower code readability**
  Builtin calls are harder to read and reason about than equivalent
  high-level code. Operations expressed as
  ``__builtin_amdgcn_mfma_f32_32x32x8f16`` are less self-documenting than a
  straightforward nested loop. This increases the cost of bringing new team
  members up to speed and reviewing changes.

**Harder debugging and validation**
  Debugging builtin-heavy code requires architecture-specific knowledge of
  register layouts, lane assignments, and hardware execution semantics.
  Standard debugging techniques such as printf-based inspection become more
  difficult when data is distributed across wavefront lanes in
  hardware-defined patterns.

**Premature optimization risk**
  Using builtins before profiling can waste development effort on code paths
  that are not performance bottlenecks. Always profile first to confirm that a
  particular operation is the dominant cost, then apply builtins to that
  specific bottleneck rather than rewriting an entire application.

Start with standard HIP constructs and the optimization patterns in the
preceding chapters. Introduce builtins selectively where profiling shows
clear opportunities, and isolate builtin code behind abstraction boundaries
to limit the impact on portability and maintenance.

Connection to optimization patterns
===================================

The builtins in this chapter directly accelerate the patterns you learned
previously. Wave reduction builtins replace manual shuffle-based reductions
with single-instruction operations. Direct-to-LDS builtins bypass register
staging to reduce register pressure in memory-bound kernels. Matrix
multiply-accumulate builtins leverage dedicated matrix cores to achieve
orders-of-magnitude higher throughput than scalar operations.

Each builtin section includes notes on which patterns benefit from its use,
helping you identify optimization opportunities in your code.

What you will learn
===================

Working through the Compiler builtins sections will help you:

* Identify which builtins are available on your target GPU architecture
* Apply dot product acceleration and wave-level operations
* Leverage matrix operations (MFMA on CDNA, WMMA on RDNA) for compute-bound
  applications
* Use direct-to-LDS memory transfers to reduce register pressure
* Apply architecture-specific optimizations for CDNA or RDNA GPUs
* Integrate builtins into the optimization patterns you learned previously

Prerequisites
=============

Before working through this chapter, you should be familiar with:

* The optimization patterns covered in previous chapters (reduction, histogram,
  matrix multiplication)
* HIP programming fundamentals, including kernel launch, memory hierarchy, and
  workgroup organization
* GPU architecture concepts such as waves, wavefronts, and LDS
* Profiling and performance analysis tools to measure optimization benefits
