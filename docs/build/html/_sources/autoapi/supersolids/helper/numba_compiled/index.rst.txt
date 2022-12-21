:py:mod:`supersolids.helper.numba_compiled`
===========================================

.. py:module:: supersolids.helper.numba_compiled

.. autoapi-nested-parse::

   Function speeded up with numba



Module Contents
---------------


Functions
~~~~~~~~~

.. autoapisummary::

   supersolids.helper.numba_compiled.get_H_pot_exponent_terms_jit
   supersolids.helper.numba_compiled.get_H_pot_jit
   supersolids.helper.numba_compiled.f_lam
   supersolids.helper.numba_compiled.eta_dVdna_jit
   supersolids.helper.numba_compiled.eta_dVdnb_jit



Attributes
~~~~~~~~~~

.. autoapisummary::

   supersolids.helper.numba_compiled.cc
   supersolids.helper.numba_compiled.verbose


.. py:data:: cc
   

   

.. py:data:: verbose
   :annotation: = True

   

.. py:function:: get_H_pot_exponent_terms_jit(V_val, a_dd_factor, a_s_factor, dipol_term, contact_interaction, mu_lhy)


.. py:function:: get_H_pot_jit(U, dt, terms, split_step = 0.5)


.. py:function:: f_lam(A, lam, eta_aa, eta_bb, eta_ab)


.. py:function:: eta_dVdna_jit(A, lam, eta_aa, eta_bb, eta_ab)


.. py:function:: eta_dVdnb_jit(A, lam, eta_aa, eta_bb, eta_ab)


