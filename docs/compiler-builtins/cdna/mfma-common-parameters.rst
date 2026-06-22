.. meta::
   :description: Learn about common trailing parameters shared by all CDNA MFMA builtins, including cbsz broadcast control, abid bank selection, and blgp lane group pattern.
   :keywords: AMD, ROCm, HIP, MFMA, cbsz, abid, blgp, broadcast, lane group, matrix cores, CDNA

.. _cdna-mfma-common-parameters:

********************************************************************************
Common MFMA parameters
********************************************************************************

Every MFMA builtin across all CDNA generations shares trailing integer
parameters that must be compile-time integer constants. Some parameters apply to
all MFMA variants; others are specific to dense or sparse forms.

Shared parameters
=================

The following parameters are present on every MFMA builtin.

.. list-table::
   :header-rows: 1
   :widths: auto

   * - Parameter
     - Type
     - Description
   * - ``cbsz``
     - int
     - Control Broadcast Size modifier. Supported by MFMA builtins
       operating on multiple input blocks for :math:`\pmb{A}`. Legal
       range 0..4, but must not exceed
       :math:`\log_{2}(\text{input blocks})`. Setting ``cbsz`` informs
       the instruction to broadcast values of one chosen input block to
       :math:`2^{cbsz} - 1` neighboring blocks in :math:`\pmb{A}`. The
       input block is chosen by setting ``abid``. For example, for a
       16-block :math:`\pmb{A}` matrix, setting ``cbsz`` to ``1``
       results in blocks 0 and 1 receiving the same input values, blocks
       2 and 3 receiving the same input values, blocks 4 and 5 receiving
       the same input values, etc. Setting ``cbsz`` to ``0`` results in
       no broadcast.
   * - ``abid``
     - int
     - :math:`\pmb{A}`-matrix Broadcast Identifier. Supported by MFMA builtins
       operating on multiple input blocks for :math:`\pmb{A}`. Used
       together with ``cbsz``; indicates which input block is selected
       for broadcast to neighboring blocks in :math:`\pmb{A}`. For
       example, for a 16-block :math:`\pmb{A}` matrix, setting ``cbsz``
       to ``2`` and ``abid`` to ``1`` broadcasts block 1's values to
       blocks 0, 2, and 3; block 5's values to blocks 4, 6, and 7; etc.

Dense MFMA parameters
=====================

The following parameter applies to dense MFMA builtins only.

.. list-table::
   :header-rows: 1
   :widths: auto

   * - Parameter
     - Type
     - Description
   * - ``blgp``
     - int
     - :math:`\pmb{B}`-matrix Lane Group Pattern modifier. Allows a
       constrained set of swizzling operations on :math:`\pmb{B}` data
       between lanes. Supported values:

       * ``0``: No swizzling; normal matrix layout for :math:`\pmb{B}`.
       * ``1``: Data from lanes 0---31 is broadcast into lanes 32---63.
       * ``2``: Data from lanes 32---63 is broadcast into lanes 0---31.
       * ``3``: Data from all lanes is rotated down by 16 positions:
         lane 0's data goes to lane 48, lane 16's data goes to lane 0,
         etc.
       * ``4``: Data from lanes 0---15 is broadcast into lanes 16---31,
         32---47, and 48---63.
       * ``5``: Data from lanes 16---31 is broadcast into lanes 0---15,
         32---47, and 48---63.
       * ``6``: Data from lanes 32---47 is broadcast into lanes 0---15,
         16---31, and 48---63.
       * ``7``: Data from lanes 48---63 is broadcast into lanes 0---15,
         16---31, and 32---47.

       .. note::

          **CDNA3 FP64 exception:** for FP64 MFMA builtins on CDNA3
          (``gfx942``), ``blgp`` is repurposed as a 3-bit NEG modifier.
          Each bit negates the corresponding source operand: bit 0 negates
          :math:`\pmb{A}`, bit 1 negates :math:`\pmb{B}`, and bit 2 negates
          :math:`\pmb{C}`.  Lane-group pattern swizzling is not available for
          FP64 on CDNA3.

.. _smfmac-common-parameters:

Sparse MFMA parameters
======================

Sparse MFMA (SMFMAC) builtins share the ``cbsz`` and ``abid`` parameters
described above with the same semantics as their dense counterparts.  In
addition, SMFMAC builtins accept the following parameter:

.. list-table::
   :header-rows: 1
   :widths: auto

   * - Parameter
     - Type
     - Description
   * - ``idx``
     - int
     - Sparsity index.  Encodes which two of the four positions in each
       group of four elements along the K dimension hold the non-zeros in
       the compressed :math:`\pmb{A}` matrix.  The bit width of the index
       scales with K: a 2-bit field per group of four is needed, so K=16
       requires 8 bits, K=32 requires 16 bits, and K=64 requires 32 bits.
       This is a runtime value, not a compile-time constant.

.. note::

   SMFMAC builtins do not support the ``blgp`` (:math:`\pmb{B}`-matrix
   Lane Group Pattern) modifier available on dense MFMA instructions.
