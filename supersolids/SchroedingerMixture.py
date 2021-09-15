#!/usr/bin/env python

# author: Daniel Scheiermann
# email: daniel.scheiermann@itp.uni-hannover.de
# license: MIT
# Please feel free to use and modify this, but keep the above information.

"""
Numerical solver for non-linear time-dependent Schrodinger equation (eGPE) for dipolar mixtures.


"""

import pickle
import sys
from typing import Callable, Union, Optional
from pathlib import Path

import dill
import numpy as np
from scipy.constants import mu_0

from supersolids.Schroedinger import Schroedinger
from supersolids.SchroedingerMixtureSummary import SchroedingerMixtureSummary
from supersolids.helper import functions, constants


class SchroedingerMixture(Schroedinger):
    """
    Implements a numerical solution of the dimensionless time-dependent
    non-linear Schroedinger equation for an arbitrary potential:

    .. math::

       i \\partial_t \psi = [&-\\frac{1}{2} \\nabla ^2
                              + \\frac{1}{2} (x^2 + (y \\alpha_y)^2 + (z \\alpha_z)^2) \\\\
                             &+ g |\psi|^2  + g_{qf} |\psi|^3 + U_{dd}] \psi \\\\

    With :math:`U_{dd} = \\mathcal{F}^{-1}(\\mathcal{F}(H_{pot} \psi) \epsilon_{dd} g (3 (k_z / k)^2 - 1))`

    The split operator method with the Trotter-Suzuki approximation
    for the commutator relation (:math:`H = H_{pot} + H_{kin}`) is used.
    Hence the accuracy is proportional to :math:`dt^4`
    The approximation is needed because of the Baker-Campell-Hausdorff formula.

    m is the atomic mass
    :math:`C_{d d}=\mu_{0} \mu^{2}` sets the strength of the dipolar interaction
    with :math:`\mu=9.93 \mu_{\mathrm{B}}` the magnetic dipole moment of
    :math:`^{162}\mathrm{Dy}`.

    We use dipolar units, obtained from the characteristic dipolar length
    :math:`r_{0}= \\frac{m C_{d d}}{4 \pi \hbar^{2}}  = 387.672168  a_0`
    and the dipolar scale of energy :math:`\epsilon_{0} = \\frac{\hbar^{2}}{m r_{0}^{2}}`

    """

    def __init__(self,
                 System: Schroedinger,
                 a_11_bohr: float = 95.0,
                 a_12_bohr: float = 95.0,
                 a_22_bohr: float = 95.0,
                 N2: float = 0.5,
                 m1: float = 1.0,
                 m2: float = 1.0,
                 mu_1: float = 9.93,
                 mu_2: float = 9.93,
                 psi_0_noise: np.ndarray = functions.noise_mesh,
                 psi2_0: Callable = functions.psi_gauss_3d,
                 psi2_0_noise: np.ndarray = functions.noise_mesh,
                 mu_sol: Optional[Callable] = functions.mu_3d,
                 input_path: Path = Path("~/supersolids/results").expanduser(),
                 ) -> None:
        super().__init__(N=System.N,
                         MyBox=System.Box,
                         Res=System.Res,
                         max_timesteps=System.max_timesteps,
                         dt=System.dt,
                         dt_func=System.dt,
                         g=System.g,
                         g_qf=System.g_qf,
                         w_x=System.w_x,
                         w_y=System.w_y,
                         w_z=System.w_z,
                         a_s=System.a_s,
                         e_dd=System.e_dd,
                         imag_time=System.imag_time,
                         mu=System.mu,
                         E=System.E,
                         psi_0=System.psi,
                         V=System.V,
                         V_interaction=System.V_interaction,
                         psi_sol=System.psi_sol,
                         mu_sol=mu_sol,
                         psi_0_noise=psi_0_noise,
                         )

        # TODO: needed but destroys back compatiblity to old npz
        #       (fork needed, accumulate those to do it in one go)
        # self.name: str = "SchroedingerMixtureSummary_"

        self.input_path: Path = input_path
        self.psi2_0: Callable = psi2_0
        self.psi2_0_noise: Optional[np.ndarray] = psi2_0_noise

        if self.dim == 1:
            if psi2_0_noise is None:
                self.psi2_val: np.ndarray = self.psi2_0(self.x)
            else:
                self.psi2_val = psi2_0_noise * self.psi2_0(self.x)
        elif self.dim == 2:
            if psi2_0_noise is None:
                self.psi2_val = self.psi2_0(self.pos)
            else:
                self.psi2_val = psi2_0_noise * self.psi2_0(self.pos)
        elif self.dim == 3:
            if psi2_0_noise is None:
                self.psi2_val = self.psi2_0(self.x_mesh, self.y_mesh, self.z_mesh)
            else:
                self.psi2_val = psi2_0_noise * self.psi2_0(self.x_mesh, self.y_mesh, self.z_mesh)
        else:
            sys.exit("Spatial dimension over 3. This is not implemented.")

        self.mu_a, self.mu_b = self.load_mu()
        self.lhy = self.load_lhy()

        self.m1 = m1
        self.m2 = m2
        self.N2 = N2

        self.mu_DEFAULT = 9.93  # BY_DEFAULT !! do not change this values
        self.mu_1 = mu_1
        self.mu_2 = mu_2
        # g, g_qf, e_dd, a_s_l_ho_ratio = functions.get_parameters(
        #     N=args.N, m=args.m, a_s=args.a_s, a_dd=args.a_dd, w_x=args.w_x)
        r_0 = (161.9 * constants.u_in_kg * mu_0 * (self.mu_DEFAULT * constants.mu_bohr) ** 2.0
               / (4.0 * np.pi * constants.hbar ** 2.0)
               ) / constants.a_0
        print(f"r_0: {r_0}")
        # r_0 = 387.4

        self.H2_kin = self.H_kin * np.exp((1 / self.m2))
        self.V2_val = self.V_val
        self.V2_k_val = self.V_k_val

        self.a_11_bohr = a_11_bohr
        self.a_12_bohr = a_12_bohr
        self.a_22_bohr = a_22_bohr

        self.mixture_variables(r_0=r_0)

    def mixture_variables(self, r_0):
        a_11 = self.a_11_bohr / r_0
        a_22 = self.a_22_bohr / r_0
        a_12 = self.a_12_bohr / r_0  # variable
        self.g_11 = 4.0 * np.pi * a_11
        self.g_22 = 4.0 * np.pi * a_22
        self.g_12 = 4.0 * np.pi * a_12  # mass is the same

    def use_summary(self, summary_name: Optional[str] = None):
        if summary_name is None:
            summary_name = "SchroedingerMixtureSummary_"

        Summary: SchroedingerMixtureSummary = SchroedingerMixtureSummary(self)

        return Summary, summary_name

    def load_summary(self, input_path, steps_format, frame,
                     summary_name: Optional[str] = None):
        if summary_name is None:
            summary_name = "SchroedingerMixtureSummary_"

        try:
            # load SchroedingerSummary
            system_summary_path = Path(input_path, summary_name + steps_format % frame + ".pkl")
            with open(system_summary_path, "rb") as f:
                SystemSummary: SchroedingerMixtureSummary = dill.load(file=f)
                SystemSummary.copy_to(self)
        except Exception:
            print(f"{system_summary_path} not found.")

        return self

    def dEps_dPsi(self, density1, density2, U_dd1, U_dd2):
        lmu_a = (self.g_11 * density1
                 + self.g_12 * density2
                 + (self.mu_1 / self.mu_DEFAULT) ** 2.0 * U_dd1
                 + (self.mu_1 / self.mu_DEFAULT) * (self.mu_2 / self.mu_DEFAULT) * U_dd2
                 )
        lmu_b = (self.g_22 * density2
                 + self.g_12 * density1
                 + (self.mu_2 / self.mu_DEFAULT) ** 2.0 * U_dd2
                 + (self.mu_1 / self.mu_DEFAULT) * (self.mu_2 / self.mu_DEFAULT) * U_dd1
                 )

        lmu_a += self.mu_a(density1, density2, grid=False)
        lmu_b += self.mu_b(density1, density2, grid=False)

        return lmu_a, lmu_b

    def load_mu(self,
                filename_mu_a="interpolator_mu_a.pkl",
                filename_mu_b="interpolator_mu_b.pkl"):
        print("Loading mu from interpolating pickles ... ")
        with open(Path(self.input_path, filename_mu_a), "rb") as f:
            mu_a = pickle.load(f)
        with open(Path(self.input_path, filename_mu_b), "rb") as f:
            mu_b = pickle.load(f)
        print("mu pickles sucessfully loaded!")

        return mu_a, mu_b

    def load_lhy(self, filename_lhy="interpolator_lhy_energy.pkl"):
        print("Loading lhy from interpolating pickles ... ")
        with open(Path(self.input_path, filename_lhy), "rb") as f:
            lhy_energy = pickle.load(f)
        print("lhy pickle sucessfully loaded!")

        return lhy_energy

    def energy_density_interaction(self, density1, density2, U_dd1, U_dd2):
        dV = self.volume_element(fourier_space=False)

        en_mf = (self.g_12 * density1 * density2
                 + (0.5 * self.g_11 * density1 ** 2.0).astype(np.complex128)
                 + (0.5 * self.g_22 * density2 ** 2.0).astype(np.complex128)
                 )
        en_mf += ((self.mu_1 / self.mu_DEFAULT) ** 2.0 * 0.5 * U_dd1 * density1
                  + (self.mu_2 / self.mu_DEFAULT) ** 2.0 * 0.5 * U_dd2 * density2
                  + (self.mu_1 / self.mu_DEFAULT) * (
                          self.mu_2 / self.mu_DEFAULT) * U_dd1 * density2
                  )
        E_lhy = self.lhy(density1, density2, grid=False)

        return np.array([dV * np.sum(en_mf.real), dV * np.sum(E_lhy)])

    def energy(self, psi_1, psi_2, density1, density2, U_dd1, U_dd2):
        """
        Input psi_1, psi_2 need to be normalized.
        density1 and density2 need to be build by the normalized psi_1, psi_2.

        """
        dV = self.volume_element(fourier_space=False)
        p_ext = np.array([self.sum_dV(density1 * self.V_val, dV=dV),
                          self.sum_dV(density2 * self.V2_val, dV=dV)])
        p_int = self.energy_density_interaction(density1, density2, U_dd1, U_dd2)

        E_U_dd = (1 / np.sqrt(2.0 * np.pi) ** 3.0) * self.sum_dV(
            self.V_k_val * np.abs(np.fft.fftn(psi_2)) ** 2.0, fourier_space=True)

        psi_val_k = np.fft.fftn(self.psi_val)
        psi_norm_k: float = self.get_norm(psi_val_k, fourier_space=True)
        psi_val_k = psi_val_k / np.sqrt(psi_norm_k)

        # E_kin = self.get_norm(0.5 * self.k_squared * psi_val_k, fourier_space=True)
        k_en = np.array(
            [dV * np.sum((np.conj(psi_1) * np.fft.ifftn(psi_val_k * 0.5 * self.k_squared)).real),
             dV * np.sum(
                 (np.conj(psi_2) * np.fft.ifftn(psi_val_k * 0.5 * self.k_squared / self.m2)).real)
             ])
        return np.concatenate((k_en, p_ext, p_int), axis=None)

    def save_psi_val(self, input_path, filename_steps, steps_format, frame):
        super().save_psi_val(input_path, filename_steps, steps_format, frame)
        with open(Path(input_path, "2-" + filename_steps + steps_format % frame + ".npz"),
                  "wb") as g:
            np.savez_compressed(g, psi2_val=self.psi2_val)

    def time_step(self) -> None:
        """
        Evolves System according Schrödinger Equations by using the
        split operator method with the Trotter-Suzuki approximation.

        """
        # adjust dt, to get the time accuracy when needed
        # self.dt = self.dt_func(self.t, self.dt)

        # Calculate the interaction by applying it to the psi_2 in k-space
        # (transform back and forth)
        density1: np.ndarray = self.get_density(func=self.psi_val, p=2.0)
        density2: np.ndarray = self.get_density(func=self.psi2_val, p=2.0)

        # get the effective potentials
        U_dd1: np.ndarray = np.fft.ifftn(self.V_k_val * np.fft.fftn(density1))
        U_dd2: np.ndarray = np.fft.ifftn(self.V_k_val * np.fft.fftn(density2))

        # update H_pot before use
        pot_1, pot_2 = self.dEps_dPsi(density1, density1, U_dd1, U_dd2)
        H_pot_1: np.ndarray = np.exp(self.U
                                     * (0.5 * self.dt)
                                     * (self.V_val
                                        + pot_1)
                                     )
        H_pot_2: np.ndarray = np.exp(self.U
                                     * (0.5 * self.dt)
                                     * (self.V2_val
                                        + pot_2)
                                     )

        # multiply element-wise the (1D, 2D or 3D) arrays with each other
        self.psi_val = H_pot_1 * self.psi_val
        self.psi2_val = H_pot_2 * self.psi2_val

        self.psi_val = np.fft.fftn(self.psi_val)
        self.psi2_val = np.fft.fftn(self.psi2_val)
        # H_kin is just dependent on U and the grid-points, which are constants,
        # so it does not need to be recalculated
        # multiply element-wise the (1D, 2D or 3D) array (H_kin) with psi_val
        # (1D, 2D or 3D)
        self.psi_val = self.H_kin * self.psi_val
        self.psi2_val = self.H2_kin * self.psi2_val
        self.psi_val = np.fft.ifftn(self.psi_val)
        self.psi2_val = np.fft.ifftn(self.psi2_val)

        # update H_pot, psi_2, U_dd before use
        density1: np.ndarray = self.get_density(func=self.psi_val, p=2.0)
        density2: np.ndarray = self.get_density(func=self.psi2_val, p=2.0)

        # update H_pot before use
        pot_1, pot_2 = self.dEps_dPsi(density1, density1, U_dd1, U_dd2)
        H_pot_1: np.ndarray = np.exp(self.U
                                     * (0.5 * self.dt)
                                     * (self.V_val
                                        + pot_1)
                                     )
        H_pot_2: np.ndarray = np.exp(self.U
                                     * (0.5 * self.dt)
                                     * (self.V2_val
                                        + pot_2)
                                     )

        # multiply element-wise the (1D, 2D or 3D) arrays with each other
        self.psi_val = H_pot_1 * self.psi_val
        self.psi2_val = H_pot_2 * self.psi2_val

        self.t = self.t + self.dt

        # for self.imag_time=False, renormalization should be preserved,
        # but we play safe here (regardless of speedup)
        # if self.imag_time:
        psi_norm_after_evolution: float = self.trapez_integral(np.abs(self.psi_val) ** 2.0)
        self.psi_val = self.psi_val / np.sqrt(psi_norm_after_evolution)
        psi2_norm_after_evolution: float = self.trapez_integral(np.abs(self.psi2_val) ** 2.0)
        self.psi2_val = self.psi2_val / np.sqrt(psi2_norm_after_evolution)

        # update for energy calculation
        density1: np.ndarray = self.get_density(func=self.psi_val, p=2.0)
        density2: np.ndarray = self.get_density(func=self.psi2_val, p=2.0)
        U_dd1: np.ndarray = np.fft.ifftn(self.V_k_val * np.fft.fftn(density1))
        U_dd2: np.ndarray = np.fft.ifftn(self.V_k_val * np.fft.fftn(density2))

        # use normalized inputs for energy
        E_parts = self.energy(self.psi_val, self.psi2_val, density1, density1, U_dd1, U_dd2)
        self.E = np.sum(E_parts)

        self.mu = - np.log(psi_norm_after_evolution) / (2.0 * self.dt)
