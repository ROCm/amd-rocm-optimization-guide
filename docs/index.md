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

**Hardware intrinsics**

* {doc}`./hardware-intrinsics/introduction`
* {doc}`./hardware-intrinsics/cross-arch/arithmetic-intrinsics`
  * {doc}`./hardware-intrinsics/cross-arch/arithmetic-ref/sad-intrinsics`
  * {doc}`./hardware-intrinsics/cross-arch/arithmetic-ref/dot-integer-intrinsics`
  * {doc}`./hardware-intrinsics/cross-arch/arithmetic-ref/dot-float-intrinsics`
  * {doc}`./hardware-intrinsics/cross-arch/arithmetic-ref/conversion-packing-intrinsics`
* {doc}`./hardware-intrinsics/cross-arch/direct-to-lds-intrinsics`
* {doc}`./hardware-intrinsics/cross-arch/warp-intrinsics`
  * {doc}`./hardware-intrinsics/cross-arch/warp-ref/shuffle-intrinsics`
  * {doc}`./hardware-intrinsics/cross-arch/warp-ref/dpp-intrinsics`
  * {doc}`./hardware-intrinsics/cross-arch/warp-ref/permlane-intrinsics`
  * {doc}`./hardware-intrinsics/cross-arch/warp-ref/wave-reduce-intrinsics`
  * {doc}`./hardware-intrinsics/cross-arch/warp-ref/vote-intrinsics`
* {doc}`./hardware-intrinsics/cdna/mfma-intrinsics`
  * {doc}`./hardware-intrinsics/cdna/dense-mfma-intrinsics`
    * {doc}`./hardware-intrinsics/cdna/cdna-dense-mfma-intrinsics`
    * {doc}`./hardware-intrinsics/cdna/cdna2-dense-mfma-intrinsics`
    * {doc}`./hardware-intrinsics/cdna/cdna3-dense-mfma-intrinsics`
    * {doc}`./hardware-intrinsics/cdna/cdna4-dense-mfma-intrinsics`
  * {doc}`./hardware-intrinsics/cdna/sparse-mfma-intrinsics`
    * {doc}`./hardware-intrinsics/cdna/cdna-sparse-mfma-intrinsics`
    * {doc}`./hardware-intrinsics/cdna/cdna3-sparse-mfma-intrinsics`
    * {doc}`./hardware-intrinsics/cdna/cdna4-sparse-mfma-intrinsics`
  * {doc}`./hardware-intrinsics/cdna/cdna4-mfma-lds-intrinsics`
  * {doc}`./hardware-intrinsics/cdna/mfma-common-parameters`
* {doc}`./hardware-intrinsics/rdna/wmma-intrinsics`
  * {doc}`./hardware-intrinsics/rdna/dense-wmma-intrinsics`
    * {doc}`./hardware-intrinsics/rdna/rdna3-dense-wmma-intrinsics`
    * {doc}`./hardware-intrinsics/rdna/rdna4-dense-wmma-intrinsics`
  * {doc}`./hardware-intrinsics/rdna/rdna4-sparse-wmma-intrinsics`

Known issues are listed and can be reported on the [AMD ROCm Optimization Guide GitHub repository](https://github.com/ROCm/amd-rocm-optimization-guide/issues).

To contribute to the documentation, see {doc}`Contributing to ROCm docs <rocm:contribute/contributing>` for contribution guidelines.

You can find licensing information on the [Licensing](https://rocm.docs.amd.com/en/latest/about/license.html) page.
