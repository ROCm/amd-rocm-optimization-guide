.. meta::
   :description: Reference documentation for MFMA intrinsics on AMD CDNA GPUs,
      covering dense and sparse variants across all CDNA generations.
   :keywords: AMD, ROCm, HIP, CDNA, MFMA, matrix cores, intrinsics,
      matrix multiply-accumulate, dense MFMA, sparse MFMA

.. _mfma-intrinsics:

********************************************************************************
MFMA intrinsics
********************************************************************************

Matrix Fused Multiply-Add (MFMA) intrinsics let you issue hardware
matrix multiply-accumulate operations directly from HIP device code.
This reference covers both dense and sparse MFMA variants on AMD CDNA
architectures (AMD Instinct GPUs). Coverage is organized first by operand
density -- dense or sparse -- and then by CDNA generation.

* :doc:`Dense MFMA intrinsics <dense-mfma-intrinsics>`

  * :doc:`CDNA MFMA intrinsics <cdna-mfma-intrinsics>`
  * :doc:`CDNA2 MFMA intrinsics <cdna2-mfma-intrinsics>`
  * :doc:`CDNA3 dense MFMA intrinsics <cdna3-dense-mfma-intrinsics>`

* :doc:`Sparse MFMA intrinsics <sparse-mfma-intrinsics>`
* :doc:`Common MFMA parameters <mfma-common-parameters>`
