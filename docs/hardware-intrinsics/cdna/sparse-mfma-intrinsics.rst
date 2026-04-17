.. meta::
   :description: Reference documentation for sparse MFMA intrinsics on AMD CDNA
      GPUs, covering all CDNA generations.
   :keywords: AMD, ROCm, HIP, CDNA, MFMA, sparse MFMA, matrix cores, intrinsics,
      matrix multiply-accumulate, structured sparsity

.. _sparse-mfma-intrinsics:

********************************************************************************
Sparse MFMA intrinsics
********************************************************************************

Sparse Matrix Fused Multiply-Add (MFMA) intrinsics exploit structured sparsity
in the A operand to deliver higher effective throughput on suitable workloads.
Coverage for sparse variants is added in later CDNA generation branches. For
trailing parameter documentation, see
:doc:`Common MFMA parameters <mfma-common-parameters>`.
