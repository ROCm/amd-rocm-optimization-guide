.. meta::
   :description: Reference documentation for dense WMMA intrinsics on AMD RDNA
      GPUs, covering RDNA3/3.5 and RDNA4 generations.
   :keywords: AMD, ROCm, HIP, RDNA, WMMA, dense WMMA, wave-matrix, intrinsics,
      matrix multiply-accumulate

.. _dense-wmma-intrinsics:

********************************************************************************
Dense WMMA intrinsics
********************************************************************************

Dense Wave-Matrix Multiply-Accumulate (WMMA) intrinsics multiply full A and B
fragments and accumulate the result into a C fragment.  All operands are fully
populated with data.  Select a generation below for the corresponding intrinsic
reference.

* :doc:`RDNA3 WMMA intrinsics <rdna3-wmma-intrinsics>`
* :doc:`RDNA4 dense WMMA intrinsics <rdna4-dense-wmma-intrinsics>`
