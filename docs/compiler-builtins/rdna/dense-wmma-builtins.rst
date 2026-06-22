.. meta::
   :description: Reference for dense WMMA wave-matrix multiply-accumulate
      builtins on AMD RDNA GPUs, with supported data types and output layouts for RDNA3, RDNA3.5, and RDNA4.
   :keywords: AMD, ROCm, HIP, RDNA, WMMA, dense WMMA, wave-matrix, builtins,
      matrix multiply-accumulate

.. _dense-wmma-builtins:

********************************************************************************
Dense WMMA builtins
********************************************************************************

Dense Wave-Matrix Multiply-Accumulate (WMMA) builtins multiply full A and B
fragments and accumulate the result into a C fragment.  All operands are fully
populated with data.  Select a generation below for the corresponding builtin
reference.

* :doc:`RDNA3 dense WMMA builtins <rdna3-dense-wmma-builtins>`
* :doc:`RDNA4 dense WMMA builtins <rdna4-dense-wmma-builtins>`
