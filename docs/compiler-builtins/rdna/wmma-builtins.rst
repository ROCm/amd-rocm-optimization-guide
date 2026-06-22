.. meta::
   :description: Explore WMMA and SWMMAC matrix builtins on AMD RDNA GPUs,
      covering dense and sparse wave-matrix multiply-accumulate variants across RDNA generations.
   :keywords: AMD, ROCm, HIP, RDNA, WMMA, SWMMAC, matrix cores, builtins,
      wave-matrix multiply-accumulate, dense WMMA, sparse SWMMAC

.. _wmma-builtins:

********************************************************************************
WMMA builtins
********************************************************************************

Wave-Matrix Multiply-Accumulate (WMMA) builtins let you issue hardware
matrix multiply-accumulate operations directly from HIP device code.
This reference covers both dense and sparse WMMA variants on AMD RDNA
architectures (AMD Radeon GPUs).  Coverage is organized first by operand
density -- dense or sparse -- and then by RDNA generation.

* :doc:`Dense WMMA builtins <dense-wmma-builtins>`

  * :doc:`RDNA3 dense WMMA builtins <rdna3-dense-wmma-builtins>`
  * :doc:`RDNA4 dense WMMA builtins <rdna4-dense-wmma-builtins>`

* Sparse WMMA builtins

  * :doc:`RDNA4 sparse WMMA builtins <rdna4-sparse-wmma-builtins>`
