.. meta::
   :description: Reference documentation for dense MFMA intrinsics on AMD CDNA
      GPUs, covering all CDNA generations.
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

* :doc:`CDNA MFMA intrinsics <cdna-mfma-intrinsics>`
* :doc:`CDNA2 MFMA intrinsics <cdna2-mfma-intrinsics>`
* :doc:`CDNA3 dense MFMA intrinsics <cdna3-dense-mfma-intrinsics>`
