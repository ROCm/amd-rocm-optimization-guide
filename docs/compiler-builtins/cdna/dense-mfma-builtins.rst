.. meta::
   :description: Reference for dense MFMA matrix multiply-accumulate builtins on AMD CDNA GPUs, covering supported data types and output layouts across all CDNA generations.
   :keywords: AMD, ROCm, HIP, CDNA, MFMA, dense MFMA, matrix cores, builtins,
      matrix multiply-accumulate

.. _dense-mfma-builtins:

********************************************************************************
Dense MFMA builtins
********************************************************************************

Dense Matrix Fused Multiply-Add (MFMA) builtins multiply full A and B tiles
and accumulate the result into a C tile. All three operands are fully populated
with data. Select a CDNA generation below for the corresponding builtin
reference. For trailing parameter documentation, see
:doc:`Common MFMA parameters <mfma-common-parameters>`.

* :doc:`CDNA dense MFMA builtins <cdna-dense-mfma-builtins>`
* :doc:`CDNA2 dense MFMA builtins <cdna2-dense-mfma-builtins>`
* :doc:`CDNA3 dense MFMA builtins <cdna3-dense-mfma-builtins>`
* :doc:`CDNA4 dense MFMA builtins <cdna4-dense-mfma-builtins>`
