:py:mod:`supersolids.helper.numbas`
===================================

.. py:module:: supersolids.helper.numbas

.. autoapi-nested-parse::

   Function speeded up with numba



Module Contents
---------------


Functions
~~~~~~~~~

.. autoapisummary::

   supersolids.helper.numbas.get_H_pot_exponent_terms_jit
   supersolids.helper.numbas.get_H_pot_jit
   supersolids.helper.numbas.f_lam
   supersolids.helper.numbas.eta_dVdna_jit
   supersolids.helper.numbas.eta_dVdnb_jit
   supersolids.helper.numbas.get_density_jit
   supersolids.helper.numbas.get_H_pot_exponent_terms_jit
   supersolids.helper.numbas.get_H_pot_jit
   supersolids.helper.numbas.f_lam
   supersolids.helper.numbas.eta_dVdna_jit
   supersolids.helper.numbas.eta_dVdnb_jit



.. py:function:: get_H_pot_exponent_terms_jit(V_val, a_dd_factor, a_s_factor, dipol_term, contact_interaction, mu_lhy)


.. py:function:: get_H_pot_jit(U, dt, terms, split_step = 0.5)


.. py:function:: f_lam(A, lam, eta_aa, eta_bb, eta_ab)


.. py:function:: eta_dVdna_jit(A, lam, eta_aa, eta_bb, eta_ab)


.. py:function:: eta_dVdnb_jit(A, lam, eta_aa, eta_bb, eta_ab)


.. py:function:: get_density_jit(func_val, p)

   Calculates :math:`|\psi|^2` for 1D, 2D or 3D (depending on self.dim).

   :param func_val: Array of function values to get p-norm for.

   :return: :math:`|\psi|^2`



.. py:function:: get_H_pot_exponent_terms_jit(V_val, a_dd_factor, a_s_factor, dipol_term, contact_interaction, mu_lhy)


.. py:function:: get_H_pot_jit(U, dt, terms, split_step = 0.5)


.. py:function:: f_lam(A, lam, eta_aa, eta_bb, eta_ab)


.. py:function:: eta_dVdna_jit(A, lam, eta_aa, eta_bb, eta_ab)


.. py:function:: eta_dVdnb_jit(A, lam, eta_aa, eta_bb, eta_ab)


