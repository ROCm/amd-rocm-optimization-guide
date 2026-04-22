.. meta::
   :description: Reference for MFMA matrix multiply-accumulate intrinsics on AMD CDNA GPUs, covering dense and sparse variants across CDNA, CDNA2, CDNA3, and CDNA4 generations.
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
density -- dense or sparse -- and then by CDNA generation. The CDNA4 MFMA LDS
transpose load intrinsics are covered on a separate page.

* :doc:`Dense MFMA intrinsics <dense-mfma-intrinsics>`

  * :doc:`CDNA dense MFMA intrinsics <cdna-dense-mfma-intrinsics>`
  * :doc:`CDNA2 dense MFMA intrinsics <cdna2-dense-mfma-intrinsics>`
  * :doc:`CDNA3 dense MFMA intrinsics <cdna3-dense-mfma-intrinsics>`
  * :doc:`CDNA4 dense MFMA intrinsics <cdna4-dense-mfma-intrinsics>`

* :doc:`Sparse MFMA intrinsics <sparse-mfma-intrinsics>`

  * :doc:`CDNA and CDNA2 sparse MFMA intrinsics <cdna-sparse-mfma-intrinsics>`
  * :doc:`CDNA3 sparse MFMA intrinsics <cdna3-sparse-mfma-intrinsics>`
  * :doc:`CDNA4 sparse MFMA intrinsics <cdna4-sparse-mfma-intrinsics>`

* :doc:`CDNA4 MFMA LDS transpose load intrinsics <cdna4-mfma-lds-intrinsics>`
* :doc:`Common MFMA parameters <mfma-common-parameters>`
