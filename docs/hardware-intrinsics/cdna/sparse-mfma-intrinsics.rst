.. meta::
   :description: Reference documentation for sparse MFMA (SMFMAC) intrinsics on
      AMD CDNA GPUs, covering all CDNA generations with 4:2 structured sparsity.
   :keywords: AMD, ROCm, HIP, CDNA, MFMA, sparse MFMA, SMFMAC, matrix cores,
      intrinsics, structured sparsity, matrix multiply-accumulate

.. _sparse-mfma-intrinsics:

********************************************************************************
Sparse MFMA intrinsics
********************************************************************************

Sparse Matrix Fused Multiply-Accumulate (SMFMAC) intrinsics multiply a
compressed :math:`\pmb{A}` tile by a dense :math:`\pmb{B}` tile and
accumulate the result into a :math:`\pmb{D}` tile, exploiting 4:2 structured
sparsity to halve the storage and bandwidth required for :math:`\pmb{A}`.
Select a CDNA generation below for the corresponding intrinsic reference.
For trailing parameter documentation, see
:doc:`Common MFMA parameters <mfma-common-parameters>`.

* :doc:`CDNA and CDNA2 sparse MFMA intrinsics <cdna-sparse-mfma-intrinsics>`
* :doc:`CDNA3 sparse MFMA intrinsics <cdna3-sparse-mfma-intrinsics>`
* :doc:`CDNA4 sparse MFMA intrinsics <cdna4-sparse-mfma-intrinsics>`
