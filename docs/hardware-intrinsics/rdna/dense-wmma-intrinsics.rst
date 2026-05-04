.. meta::
   :description: Reference for dense WMMA wave-matrix multiply-accumulate
      intrinsics on AMD RDNA GPUs, with supported data types and output layouts for RDNA3, RDNA3.5, and RDNA4.
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

* :doc:`RDNA3 dense WMMA intrinsics <rdna3-dense-wmma-intrinsics>`
* :doc:`RDNA4 dense WMMA intrinsics <rdna4-dense-wmma-intrinsics>`
