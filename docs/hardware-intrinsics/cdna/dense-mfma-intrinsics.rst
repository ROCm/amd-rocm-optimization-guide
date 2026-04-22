.. meta::
   :description: Reference for dense MFMA matrix multiply-accumulate intrinsics on AMD CDNA GPUs, covering supported data types and output layouts across all CDNA generations.
   :keywords: AMD, ROCm, HIP, CDNA, MFMA, dense MFMA, matrix cores, intrinsics,
      matrix multiply-accumulate

.. _dense-mfma-intrinsics:

********************************************************************************
Dense MFMA intrinsics
********************************************************************************

Dense Matrix Fused Multiply-Add (MFMA) intrinsics multiply full A and B tiles
and accumulate the result into a C tile. All three operands are fully populated
with data. Select a CDNA generation below for the corresponding intrinsic
reference. For trailing parameter documentation, see
:doc:`Common MFMA parameters <mfma-common-parameters>`.

* :doc:`CDNA dense MFMA intrinsics <cdna-dense-mfma-intrinsics>`
* :doc:`CDNA2 dense MFMA intrinsics <cdna2-dense-mfma-intrinsics>`
* :doc:`CDNA3 dense MFMA intrinsics <cdna3-dense-mfma-intrinsics>`
* :doc:`CDNA4 dense MFMA intrinsics <cdna4-dense-mfma-intrinsics>`
