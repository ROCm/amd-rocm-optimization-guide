---
myst:
  html_meta:
    "description": "AMD ROCm optimization guide"
    "keywords": "HIP, ROCm, AMD ROCm optimization guide, performance optimization, GPU tuning"
---

<!-- markdownlint-disable MD036 -->

# AMD ROCm Optimization Guide

The AMD ROCm Optimization Guide focuses on performance optimization techniques for AMD GPUs using the HIP programming language. This guide provides comprehensive tutorials and best practices for maximizing GPU performance, covering essential topics such as parallel workload optimization, reduction operations, memory coalescing, and multi-GPU programming.

**Performance optimization techniques**

* {doc}`./how-to/performance_optimization`
* {doc}`./tutorial/hip-performance-optimization`

  * {doc}`./tutorial/hip-performance-optimization/highly-parallel-image-gamma-correction`
  * {doc}`./tutorial/hip-performance-optimization/fixed-size-kernels-image-gamma-correction`
  * {doc}`./tutorial/hip-performance-optimization/reduction`
  * {doc}`./tutorial/hip-performance-optimization/tiling-matrix-multiply`
  * {doc}`./tutorial/hip-performance-optimization/tiling-matrix-transpose`

* {doc}`./how-to/multi-gpu_programming`

For additional ROCm documentation, see the [ROCm documentation portal](https://rocm.docs.amd.com).

To contribute to the documentation, see {doc}`Contributing to ROCm docs <rocm:contribute/contributing>` for contribution guidelines.
