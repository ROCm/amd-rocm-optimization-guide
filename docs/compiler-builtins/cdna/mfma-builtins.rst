.. meta::
   :description: Reference for MFMA matrix multiply-accumulate builtins on AMD CDNA GPUs, covering dense and sparse variants across CDNA, CDNA2, CDNA3, and CDNA4 generations.
   :keywords: AMD, ROCm, HIP, CDNA, MFMA, matrix cores, builtins,
      matrix multiply-accumulate, dense MFMA, sparse MFMA

.. _mfma-builtins:

********************************************************************************
MFMA builtins
********************************************************************************

Matrix Fused Multiply-Add (MFMA) builtins let you issue hardware
matrix multiply-accumulate operations directly from HIP device code.
This reference covers both dense and sparse MFMA variants on AMD CDNA
architectures (AMD Instinct GPUs). Coverage is organized first by operand
density -- dense or sparse -- and then by CDNA generation. The CDNA4 MFMA LDS
transpose load builtins are covered on a separate page.

* :doc:`Dense MFMA builtins <dense-mfma-builtins>`

  * :doc:`CDNA dense MFMA builtins <cdna-dense-mfma-builtins>`
  * :doc:`CDNA2 dense MFMA builtins <cdna2-dense-mfma-builtins>`
  * :doc:`CDNA3 dense MFMA builtins <cdna3-dense-mfma-builtins>`
  * :doc:`CDNA4 dense MFMA builtins <cdna4-dense-mfma-builtins>`

* :doc:`Sparse MFMA builtins <sparse-mfma-builtins>`

  * :doc:`CDNA and CDNA2 sparse MFMA builtins <cdna-sparse-mfma-builtins>`
  * :doc:`CDNA3 sparse MFMA builtins <cdna3-sparse-mfma-builtins>`
  * :doc:`CDNA4 sparse MFMA builtins <cdna4-sparse-mfma-builtins>`

* :doc:`CDNA4 MFMA LDS transpose load builtins <cdna4-mfma-lds-builtins>`
* :doc:`Common MFMA parameters <mfma-common-parameters>`
