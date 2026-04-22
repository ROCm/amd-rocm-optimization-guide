.. meta::
   :description: Explore WMMA and SWMMAC matrix intrinsics on AMD RDNA GPUs,
      covering dense and sparse wave-matrix multiply-accumulate variants across RDNA generations.
   :keywords: AMD, ROCm, HIP, RDNA, WMMA, SWMMAC, matrix cores, intrinsics,
      wave-matrix multiply-accumulate, dense WMMA, sparse SWMMAC

.. _wmma-intrinsics:

********************************************************************************
WMMA intrinsics
********************************************************************************

Wave-Matrix Multiply-Accumulate (WMMA) intrinsics let you issue hardware
matrix multiply-accumulate operations directly from HIP device code.
This reference covers both dense and sparse WMMA variants on AMD RDNA
architectures (AMD Radeon GPUs).  Coverage is organized first by operand
density -- dense or sparse -- and then by RDNA generation.

* :doc:`Dense WMMA intrinsics <dense-wmma-intrinsics>`

  * :doc:`RDNA3 dense WMMA intrinsics <rdna3-dense-wmma-intrinsics>`
  * :doc:`RDNA4 dense WMMA intrinsics <rdna4-dense-wmma-intrinsics>`

* Sparse WMMA intrinsics

  * :doc:`RDNA4 sparse WMMA intrinsics <rdna4-sparse-wmma-intrinsics>`
