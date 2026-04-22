.. meta::
   :description: Explore AMD ROCm hardware intrinsics and specialized GPU instructions, including MFMA matrix cores, WMMA, and dot product operations for CDNA and RDNA GPUs.
   :keywords: AMD, ROCm, HIP, hardware intrinsics, GPU optimization, specialized instructions, CDNA, RDNA, MFMA, WMMA, dot product, wave-level operations

.. _hardware_intrinsics:

********************************************************************************
Hardware intrinsics and specialized instructions
********************************************************************************

In the previous chapters, you learned optimization patterns using standard HIP
programming constructs. You implemented reduction algorithms, histogram
computations, and matrix multiplications using shared memory, efficient memory
access patterns, and workgroup-level parallelism. These patterns teach
fundamental optimization principles that apply across GPU architectures.

This chapter introduces hardware intrinsics that can significantly accelerate
the patterns you have already learned. Hardware intrinsics provide direct
access to specialized GPU units and instructions that compilers cannot
automatically generate. Think of this chapter as your hardware acceleration
toolkit, organized by architecture for quick reference when optimizing
specific patterns.

Why hardware intrinsics matter
==============================

Modern compilers are sophisticated, but they cannot always use the full
range of hardware capabilities available on AMD GPUs. The compiler rarely
auto-vectorizes code to use dedicated dot product units, even when the pattern
is evident. Wave-level reduction hardware requires explicit intrinsic calls
rather than relying on compiler optimization. Matrix multiply-accumulate units
require direct invocation to achieve peak performance.

Intrinsics bridge this gap by giving you explicit control over specialized
operations. When used appropriately, they can provide orders-of-magnitude
performance improvements for compute-bound workloads.

How this chapter is organized
=============================

This chapter organizes intrinsics by their availability across AMD GPU
architectures. Cross-architecture intrinsics work on both CDNA and RDNA
architectures, providing arithmetic and packing operations (dot products, SAD,
type conversion), warp-level operations (shuffle, DPP, reductions, permlane,
vote), and direct-to-Local Data Share (LDS) memory transfers. CDNA-specific intrinsics cover
matrix operations (MFMA) with both dense and sparse variants. RDNA-specific
intrinsics cover wave-matrix multiply-accumulate (WMMA) operations, including
sparse variants on RDNA4.

Some intrinsics require specific hardware features beyond basic architecture
support, which are noted with feature flags where applicable.

When to use intrinsics
======================

Hardware intrinsics are most beneficial when:

**Compiler optimization is insufficient**
  The compiler cannot automatically vectorize your code to use specialized
  units, even when the pattern is obvious.

**Performance-critical loops dominate runtime**
  Operations in tight loops that account for most execution time can benefit
  significantly from hardware acceleration.

**Algorithmic patterns map to hardware features**
  Operations like reductions, matrix multiplications, and histograms have
  dedicated hardware support that intrinsics expose.

**You are targeting a specific GPU architecture**
  When you know your target architecture (CDNA or RDNA), you can use its
  specialized capabilities to maximize performance.

Connection to optimization patterns
===================================

The intrinsics in this chapter directly accelerate the patterns you learned
previously. Wave reduction intrinsics replace manual shuffle-based reductions
with single-instruction operations. Direct-to-LDS intrinsics bypass register
staging to reduce register pressure in memory-bound kernels. Matrix
multiply-accumulate intrinsics leverage dedicated matrix cores to achieve
orders-of-magnitude higher throughput than scalar operations.

Each intrinsic section includes notes on which patterns benefit from its use,
helping you identify optimization opportunities in your code.

What you will learn
===================

Working through the hardware intrinsics sections will help you:

* Identify which intrinsics are available on your target GPU architecture
* Apply dot product acceleration and wave-level operations
* Leverage matrix operations (MFMA on CDNA, WMMA on RDNA) for compute-bound
  applications
* Use direct-to-LDS memory transfers to reduce register pressure
* Apply architecture-specific optimizations for CDNA or RDNA GPUs
* Integrate intrinsics into the optimization patterns you learned previously

Prerequisites
=============

Before working through this chapter, you should be familiar with:

* The optimization patterns covered in previous chapters (reduction, histogram,
  matrix multiplication)
* HIP programming fundamentals, including kernel launch, memory hierarchy, and
  workgroup organization
* GPU architecture concepts such as waves, wavefronts, and LDS
* Profiling and performance analysis tools to measure optimization benefits
