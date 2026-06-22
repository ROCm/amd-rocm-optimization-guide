.. meta::
   :description: Reference for sparse MFMA (SMFMAC) builtins on AMD CDNA GPUs, covering 4:2 structured sparsity variants across CDNA, CDNA2, CDNA3, and CDNA4 generations.
   :keywords: AMD, ROCm, HIP, CDNA, MFMA, sparse MFMA, SMFMAC, matrix cores,
      builtins, structured sparsity, matrix multiply-accumulate

.. _sparse-mfma-builtins:

********************************************************************************
Sparse MFMA builtins
********************************************************************************

Sparse Matrix Fused Multiply-Accumulate (SMFMAC) builtins multiply a
compressed :math:`\pmb{A}` tile by a dense :math:`\pmb{B}` tile and
accumulate the result into a :math:`\pmb{C}` tile, exploiting 4:2 structured
sparsity to halve the storage and bandwidth required for :math:`\pmb{A}`.
Select a CDNA generation below for the corresponding builtin reference.
For trailing parameter documentation, see
:doc:`Common MFMA parameters <mfma-common-parameters>`.

* :doc:`CDNA and CDNA2 sparse MFMA builtins <cdna-sparse-mfma-builtins>`
* :doc:`CDNA3 sparse MFMA builtins <cdna3-sparse-mfma-builtins>`
* :doc:`CDNA4 sparse MFMA builtins <cdna4-sparse-mfma-builtins>`
