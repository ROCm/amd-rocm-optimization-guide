.. RST include fragment -- do not add to toctree.
   Intrinsic: __builtin_amdgcn_wmma_f16_16x16x16_f16_tied_w32
   Include via: .. include:: wmma-ref/f16-16x16x16f16-tied-rdna3.rst

``__builtin_amdgcn_wmma_f16_16x16x16_f16_tied_w32``
"""""""""""""""""""""""""""""""""""""""""""""""""""

Signature and parameters for this builtin.

.. code-block:: cpp

   v16half __builtin_amdgcn_wmma_f16_16x16x16_f16_tied_w32(
       v16half srcA,
       v16half srcB,
       v16half srcC,
       bool    opsel);

Register-tied variant of
``__builtin_amdgcn_wmma_f16_16x16x16_f16_w32``.  The compiler
constrains ``srcC`` and the return value to the same physical register,
guaranteeing that the non-selected half of the accumulator (the half not
written by this ``opsel`` setting) is preserved in place.

Use this variant when you chain two WMMA calls with opposite ``opsel``
values into the same ``v16half`` accumulator so that both halves remain
live without an extra register copy.

Parameters, types, and return value are identical to the non-tied variant
above.
