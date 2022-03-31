:py:mod:`supersolids.SchroedingerMixture`
=========================================

.. py:module:: supersolids.SchroedingerMixture

.. autoapi-nested-parse::

   Numerical solver for non-linear time-dependent Schrodinger equation (eGPE) for dipolar mixtures.




Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   supersolids.SchroedingerMixture.SchroedingerMixture



Functions
~~~~~~~~~

.. autoapisummary::

   supersolids.SchroedingerMixture.smaller_slice
   supersolids.SchroedingerMixture.get_A_density_total
   supersolids.SchroedingerMixture.get_mu_lhy_integrated_list



Attributes
~~~~~~~~~~

.. autoapisummary::

   supersolids.SchroedingerMixture.numba_used


.. py:data:: numba_used
   

   

.. py:function:: smaller_slice(val0, val1)


.. py:function:: get_A_density_total(density_list)


.. py:function:: get_mu_lhy_integrated_list(func_list)


.. py:class:: SchroedingerMixture(MyBox, Res, max_timesteps, dt, N_list, m_list, a_s_array, a_dd_array, t = 0.0, a_s_factor = 4.0 * np.pi, a_dd_factor = 3.0, nA_max = 100, dt_func = None, w_x = 2.0 * np.pi * 33.0, w_y = 2.0 * np.pi * 80.0, w_z = 2.0 * np.pi * 167.0, imag_time = True, mu_arr = None, E = 1.0, V = functions.v_harmonic_3d, V_interaction = None, psi_0_list = [functions.psi_gauss_3d], psi_0_noise_list = [functions.noise_mesh], psi_sol_list = [functions.thomas_fermi_3d], mu_sol_list = [functions.mu_3d], input_path = Path('~/supersolids/results').expanduser())

   Bases: :py:obj:`supersolids.Schroedinger.Schroedinger`

   Implements a numerical solution of the dimensionless time-dependent
   non-linear Schroedinger equation for an arbitrary potential:

   .. math::

      i \partial_t \psi = [&-\frac{1}{2} \nabla ^2
                             + \frac{1}{2} (x^2 + (y \alpha_y)^2 + (z \alpha_z)^2) \\
                            &+ g |\psi|^2  + g_{qf} |\psi|^3 + U_{dd}] \psi \\

   With :math:`U_{dd} = \mathcal{F}^{-1}(\mathcal{F}(H_{pot} \psi) \epsilon_{dd} g (3 (k_z / k)^2 - 1))`

   The split operator method with the Trotter-Suzuki approximation
   for the commutator relation (:math:`H = H_{pot} + H_{kin}`) is used.
   Hence the accuracy is proportional to :math:`dt^4`
   The approximation is needed because of the Baker-Campell-Hausdorff formula.

   m is the atomic mass
   :math:`C_{d d}=\mu_{0} \mu^{2}` sets the strength of the dipolar interaction
   with :math:`\mu=9.93 \mu_{\mathrm{B}}` the magnetic dipole moment of
   :math:`^{162}\mathrm{Dy}`.

   We use dipolar units, obtained from the characteristic dipolar length
   :math:`r_{0}= \frac{m C_{d d}}{4 \pi \hbar^{2}}  = 387.672168  a_0`
   and the dipolar scale of energy :math:`\epsilon_{0} = \frac{\hbar^{2}}{m r_{0}^{2}}`


   .. py:method:: func_energy(self, u)

      (V_+)**5/2 + (V_-)**5/2



   .. py:method:: func_f_symb(self, u, func, eta_a, eta_b)


   .. py:method:: mu_lhy_integrand(self, u, eta_dVdn)


   .. py:method:: eta_dVdna(self, lam, eta_aa, eta_bb, eta_ab)


   .. py:method:: eta_dVdnb(self, lam, eta_aa, eta_bb, eta_ab)


   .. py:method:: get_mu_lhy_list(self, density_list)


   .. py:method:: get_energy_lhy(self, density_list)


   .. py:method:: save_psi_val(self, input_path, filename_steps, steps_format, frame)


   .. py:method:: use_summary(self, summary_name = None)


   .. py:method:: load_summary(self, input_path, steps_format, frame, summary_name = 'SchroedingerMixtureSummary_')


   .. py:method:: load_mu(self, filename_mu_a = 'interpolator_mu_a.pkl', filename_mu_b = 'interpolator_mu_b.pkl')


   .. py:method:: load_lhy(self, filename_lhy = 'interpolator_lhy_energy.pkl')


   .. py:method:: energy_density_interaction(self, density_list, U_dd_list)


   .. py:method:: get_E(self)


   .. py:method:: energy(self, density_list, U_dd_list, mu_lhy_list)

      Input psi_1, psi_2 need to be normalized.
      density1 and density2 need to be build by the normalized psi_1, psi_2.



   .. py:method:: get_density_list(self, jit=True)


   .. py:method:: get_center_of_mass(self, Mx0 = None, Mx1 = None, My0 = None, My1 = None, Mz0 = None, Mz1 = None)

      Calculates the center of mass of the System.



   .. py:method:: get_parity(self, axis = 2, Mx0 = None, Mx1 = None, My0 = None, My1 = None, Mz0 = None, Mz1 = None)


   .. py:method:: distmat(self, a, index)


   .. py:method:: get_contrast_old(self, axis = 2, Mx0 = None, Mx1 = None, My0 = None, My1 = None, Mz0 = None, Mz1 = None)


   .. py:method:: get_contrast_old_smart(self, axis = 2, Mx0 = None, Mx1 = None, My0 = None, My1 = None, Mz0 = None, Mz1 = None)


   .. py:method:: on_edge(self, indices, Mx0 = None, Mx1 = None, My0 = None, My1 = None, Mz0 = None, Mz1 = None)


   .. py:method:: get_contrast(self, number_of_peaks, prob_min_start, prob_step = 0.01, prob_min_edge = 0.015, region_threshold = 100)


   .. py:method:: sum_along(self, func_val, axis, l_0 = None)


   .. py:method:: get_U_dd_list(self, density_list)


   .. py:method:: get_H_pot(self, terms, split_step = 0.5)


   .. py:method:: get_H_pot_exponent_terms(self, dipol_term, contact_interaction, mu_lhy)


   .. py:method:: split_operator_pot(self, split_step = 0.5, jit = True)


   .. py:method:: split_operator_kin(self)


   .. py:method:: normalize_psi_val(self)


   .. py:method:: time_step(self)

      Evolves System according Schrödinger Equations by using the
      split operator method with the Trotter-Suzuki approximation.




