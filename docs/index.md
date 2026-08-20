---
myst:
  html_meta:
    "description": "AMD ROCm optimization guide"
    "keywords": "HIP, ROCm, AMD ROCm optimization guide, performance optimization, GPU tuning"
---

<!-- markdownlint-disable MD036 -->

# AMD ROCm Optimization Guide

The AMD ROCm Optimization Guide focuses on performance optimization techniques for
AMD GPUs using the {doc}`HIP programming language <hip:index>`. This guide provides comprehensive
patterns and best practices for maximizing GPU performance, covering essential
topics such as parallel workload optimization, reduction operations, memory coalescing,
and multi-GPU programming.

**Getting started**

* {doc}`./conceptual/introduction`
* {doc}`./conceptual/introduction/understand`
* {doc}`./conceptual/introduction/guidelines`
* {doc}`./how-to/multi-gpu-programming`

**Performance optimization patterns**

* {doc}`./patterns/introduction`
* {doc}`./patterns/examples/reduction`
* {doc}`./patterns/examples/histogram`
* {doc}`./patterns/examples/matrix-multiply-optimization`

**Compiler builtins**

* {doc}`./compiler-builtins/introduction`
* {doc}`./compiler-builtins/cross-arch/arithmetic-builtins`
  * {doc}`./compiler-builtins/cross-arch/arithmetic-ref/sad-builtins`
  * {doc}`./compiler-builtins/cross-arch/arithmetic-ref/dot-integer-builtins`
  * {doc}`./compiler-builtins/cross-arch/arithmetic-ref/dot-float-builtins`
  * {doc}`./compiler-builtins/cross-arch/arithmetic-ref/conversion-packing-builtins`
* {doc}`./compiler-builtins/cross-arch/direct-to-lds-builtins`
* {doc}`./compiler-builtins/cross-arch/wavefront-builtins`
  * {doc}`./compiler-builtins/cross-arch/wavefront-ref/shuffle-builtins`
  * {doc}`./compiler-builtins/cross-arch/wavefront-ref/dpp-builtins`
  * {doc}`./compiler-builtins/cross-arch/wavefront-ref/permlane-builtins`
  * {doc}`./compiler-builtins/cross-arch/wavefront-ref/wave-reduce-builtins`
  * {doc}`./compiler-builtins/cross-arch/wavefront-ref/vote-builtins`
* {doc}`./compiler-builtins/cdna/mfma-builtins`
  * {doc}`./compiler-builtins/cdna/dense-mfma-builtins`
    * {doc}`./compiler-builtins/cdna/cdna-dense-mfma-builtins`
    * {doc}`./compiler-builtins/cdna/cdna2-dense-mfma-builtins`
    * {doc}`./compiler-builtins/cdna/cdna3-dense-mfma-builtins`
    * {doc}`./compiler-builtins/cdna/cdna4-dense-mfma-builtins`
  * {doc}`./compiler-builtins/cdna/sparse-mfma-builtins`
    * {doc}`./compiler-builtins/cdna/cdna-sparse-mfma-builtins`
    * {doc}`./compiler-builtins/cdna/cdna3-sparse-mfma-builtins`
    * {doc}`./compiler-builtins/cdna/cdna4-sparse-mfma-builtins`
  * {doc}`./compiler-builtins/cdna/cdna4-mfma-lds-builtins`
  * {doc}`./compiler-builtins/cdna/mfma-common-parameters`
* {doc}`./compiler-builtins/rdna/wmma-builtins`
  * {doc}`./compiler-builtins/rdna/dense-wmma-builtins`
    * {doc}`./compiler-builtins/rdna/rdna3-dense-wmma-builtins`
    * {doc}`./compiler-builtins/rdna/rdna4-dense-wmma-builtins`
  * {doc}`./compiler-builtins/rdna/rdna4-sparse-wmma-builtins`

Known issues are listed and can be reported on the [AMD ROCm Optimization Guide GitHub repository](https://github.com/ROCm/amd-rocm-optimization-guide/issues).

To contribute to the documentation, see {doc}`Contributing to ROCm docs <rocm:contribute/contributing>` for contribution guidelines.

You can find licensing information on the [Licensing](https://rocm.docs.amd.com/en/latest/about/license.html) page.
