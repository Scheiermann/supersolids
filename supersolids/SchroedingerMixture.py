#!/usr/bin/env python

# author: Daniel Scheiermann
# email: daniel.scheiermann@itp.uni-hannover.de
# license: MIT
# Please feel free to use and modify this, but keep the above information.

"""
Numerical solver for non-linear time-dependent Schrodinger equation (eGPE) for dipolar mixtures.


"""
import dill
import functools
import numpy as np
import pickle
import sys
from pathlib import Path
from typing import Optional, Callable, Union, List

from scipy.constants import mu_0
from scipy.interpolate import interpolate
from scipy.integrate import quad_vec

from supersolids.Schroedinger import Schroedinger
from supersolids.SchroedingerMixtureSummary import SchroedingerMixtureSummary
from supersolids.helper import functions, constants
from supersolids.helper.Box import Box
from supersolids.helper.Resolution import Resolution


"""
from sympy import diff, sqrt, re
from sympy import symbols
from sympy import lambdify

def eta_V_symb(lam, aa, bb, ab, a, b):
    return (aa * a + bb * b + lam * sqrt((aa * a - bb * b) ** 2.0 + 4.0 * ab ** 2.0 * a * b)) ** 2.5


def func_V_symb():
    lam, aa, bb, ab, a, b = symbols('lam aa bb ab a b')
    eta_V_symb_s = eta_V_symb(lam, aa, bb, ab, a, b)
    already_used = (2.0 / 5.0)
    v_a = already_used * diff(eta_V_symb_s, a)
    v_b = already_used * diff(eta_V_symb_s, b)
    v_a_sum = v_a.subs(lam, 1.0) + v_a.subs(lam, -1.0)
    v_b_sum = v_b.subs(lam, 1.0) + v_b.subs(lam, -1.0)
    symb = (aa, bb, ab, a, b)
    v_a_np = lambdify(symb, v_a_sum, modules='numpy')
    v_b_np = lambdify(symb, v_b_sum, modules='numpy')

    return [v_a_np, v_b_np]
"""


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
                 MyBox: Box,
                 Res: Resolution,
                 max_timesteps: int,
                 dt: float,
                 N_list: List[float],
                 m_list: List[float],
                 a_s_array: np.ndarray,
                 a_dd_array: np.ndarray,
                 a_s_factor: float = 4.0 * np.pi,
                 a_dd_factor: float = 3.0,
                 nA_max: int = 100,
                 dt_func: Optional[Callable] = None,
                 w_x: float = 2.0 * np.pi * 33.0,
                 w_y: float = 2.0 * np.pi * 80.0,
                 w_z: float = 2.0 * np.pi * 167.0,
                 imag_time: bool = True,
                 mu_arr: Optional[np.ndarray] = None,
                 E: float = 1.0,
                 V: Optional[Callable] = functions.v_harmonic_3d,
                 V_interaction: Optional[Callable] = None,
                 psi_0_list: List[np.ndarray] = [functions.psi_gauss_3d],
                 psi_0_noise_list: List[Optional[Callable]] = [functions.noise_mesh],
                 psi_sol_list: List[Optional[Callable]] = [functions.thomas_fermi_3d],
                 mu_sol_list: List[Optional[Callable]] = [functions.mu_3d],
                 input_path: Path = Path("~/supersolids/results").expanduser(),
                 ) -> None:

        self.name: str = "SchroedingerMixtureSummary_"
        self.t: float = 0.0
        self.imag_time: bool = imag_time
        self.dt: float = dt
        self.max_timesteps: int = max_timesteps

        self.nA_max = nA_max

        self.Res, self.Box = functions.check_ResBox(Res, MyBox)
        self.dim: int = self.Box.dim

        self.N_list: List[float] = N_list
        self.m_list: List[float] = m_list

        self.a_s_factor: float = a_s_factor
        self.a_dd_factor: float = a_dd_factor

        self.a_s_array: np.ndarray = a_s_array
        self.a_dd_array: np.ndarray = a_dd_array
        self.dt_func: Optional[Callable] = dt_func

        self.w_x: float = w_x
        self.w_y: float = w_y
        self.w_z: float = w_z

        self.E: float = E

        if mu_arr is None:
            self.mu_arr: np.ndarray = np.array([1.0] * len(self.N_list))
        else:
            self.mu_arr: np.ndarray = mu_arr

        self.psi_0_list: List[np.ndarray] = psi_0_list
        self.psi_sol_list: List[Optional[Callable]] = psi_sol_list
        self.mu_sol_list: List[Optional[Callable]] = mu_sol_list

        self.psi_val_list: List[np.ndarray] = []
        self.psi_sol_val_list: List[np.ndarray] = []
        self.mu_sol_val_list: List[float] = []

        self.input_path: Path = input_path

        if V is None:
            self.V = None
        else:
            self.V: Callable = V

        if V_interaction is None:
            self.V_interaction = None
        else:
            self.V_interaction: Callable = V_interaction

        """
        for mu_sol in self.mu_sol_list:
            if mu_sol is None:
                self.mu_sol_val_list.append(None)
            else:
                if callable(mu_sol):
                    self.mu_sol_val_list.append(mu_sol(self.g))
                else:
                    self.mu_sol_val_list.append(mu_sol)
        """

        self.x, self.dx, self.kx, self.dkx = functions.get_grid_helper(self.Res, self.Box, 0)
        if self.dim >= 2:
            self.y, self.dy, self.ky, self.dky = functions.get_grid_helper(self.Res, self.Box, 1)
        if self.dim >= 3:
            self.z, self.dz, self.kz, self.dkz = functions.get_grid_helper(self.Res, self.Box, 2)

        if self.dim == 1:
            # TODO: complete implementation
            for psi_0, psi_0_noise in zip(psi_0_list, psi_0_noise_list):
                if psi_0_noise is None:
                    self.psi_val_list.append(psi_0(self.x))
                else:
                    self.psi_val_list.append(psi_0_noise * psi_0(self.x))

            self.k_squared: np.ndarray = self.kx ** 2.0

            if V is None:
                self.V_val: Union[float, np.ndarray] = 0.0
            else:
                self.V_val = self.V(self.x)

            if V_interaction is None:
                # For no interaction the identity is needed with respect to 2D
                # * 2D (array with 1.0 everywhere)
                self.V_k_val: np.ndarray = np.full(self.psi_val_list[0].shape, 1.0)
            else:
                if callable(V_interaction):
                    self.V_k_val = V_interaction(self.kx)
                else:
                    self.V_k_val = V_interaction

        elif self.dim == 2:
            # TODO: complete implementation
            self.x_mesh, self.y_mesh, self.pos = functions.get_meshgrid(self.x, self.y)

            for psi_0, psi_0_noise in zip(psi_0_list, psi_0_noise_list):
                if psi_0_noise is None:
                    self.psi_val_list.append(psi_0(self.pos))
                else:
                    self.psi_val_list.append(psi_0_noise * psi_0(self.pos))

            kx_mesh, ky_mesh, _ = functions.get_meshgrid(self.kx, self.ky)
            self.k_squared = kx_mesh ** 2.0 + ky_mesh ** 2.0

            if V is None:
                self.V_val = 0.0
            else:
                self.V_val = self.V(self.pos)

        elif self.dim == 3:
            self.x_mesh, self.y_mesh, self.z_mesh = functions.get_grid(self.Res, self.Box)

            for psi_0, psi_0_noise in zip(psi_0_list, psi_0_noise_list):
                if psi_0_noise is None:
                    self.psi_val_list.append(psi_0(self.x_mesh, self.y_mesh, self.z_mesh))
                else:
                    self.psi_val_list.append(psi_0_noise * psi_0(self.x_mesh,
                                                                 self.y_mesh,
                                                                 self.z_mesh)
                                             )

            for psi_sol in self.psi_sol_list:
                if psi_sol is None:
                    self.psi_sol_val_list.append(None)
                else:
                    if callable(psi_sol):
                        psi_sol_val = psi_sol(self.x_mesh, self.y_mesh, self.z_mesh)
                        self.psi_sol_val_list.append(psi_sol_val)
                        if psi_sol_val is not None:
                            print(f"Norm for psi_sol (trapez integral): "
                                  f"{self.trapez_integral(np.abs(psi_sol_val) ** 2.0)}")

            kx_mesh, ky_mesh, kz_mesh, _ = functions.get_meshgrid_3d(self.kx, self.ky, self.kz)
            self.k_squared = kx_mesh ** 2.0 + ky_mesh ** 2.0 + kz_mesh ** 2.0

            if V is None:
                self.V_val = 0.0
            else:
                self.V_val = self.V(self.x_mesh, self.y_mesh, self.z_mesh)

            if self.V_interaction is None:
                # For no interaction the identity is needed with respect to 2D
                # * 2D (array with 1.0 everywhere)
                self.V_k_val = np.full(self.psi_val.shape, 1.0)
            else:
                if callable(self.V_interaction):
                    self.V_k_val = self.V_interaction(kx_mesh, ky_mesh, kz_mesh)

        if imag_time:
            # Convention: $e^{-iH} = e^{UH}$
            self.U: complex = -1.0
        else:
            self.U = -1.0j

        self.H_kin: np.ndarray = np.exp(self.U * (0.5 * self.k_squared) * self.dt)
        self.H_kin_list = []
        for m in self.m_list:
            self.H_kin_list.append(self.H_kin)

        self.A = np.linspace(0.0, 1.0, self.nA_max)

        func_fa = functools.partial(self.func_f, eta_dVdn=self.eta_dVdna)
        func_fb = functools.partial(self.func_f, eta_dVdn=self.eta_dVdnb)
        mu_v_list = self.mu_lhy_list(func_list=[func_fa, func_fb])

        # V_symb: List[Callable] = func_V_symb()
        # func_fa_symb = functools.partial(self.func_f_symb, func=V_symb[0], eta_a=self.A, eta_b=1.0 - self.A)
        # func_fb_symb = functools.partial(self.func_f_symb, func=V_symb[1], eta_a=self.A, eta_b=1.0 - self.A)
        # mu_v_list_symb = self.mu_lhy_list(func_list=[func_fa_symb, func_fb_symb])

        mu_lhy_helper_list = []
        for mu_v in mu_v_list:
            mu_lhy_helper_list.append(interpolate.interp1d(self.A, mu_v))

        self.mu_lhy_helper_list = mu_lhy_helper_list

        energy_v = quad_vec(self.func_energy, 0.0, 1.0)[0]
        self.energy_helper_function = interpolate.interp1d(self.A, energy_v)

    def func_energy(self, u):
        """
        (V_+)**5/2 + (V_-)**5/2

        """
        eta_array: np.ndarray = (self.a_s_factor * self.a_s_array
                                 + 4.0 * np.pi * self.a_dd_factor * self.a_dd_array * (
                                         u ** 2.0 - 1.0 / 3.0)
                                 )
        eta_aa = eta_array[0, 0]
        eta_ab = eta_array[0, 1]
        eta_bb = eta_array[1, 1]
        energy_prefactor = np.sqrt(2.) / 15. / np.pi ** 2

        return energy_prefactor * np.real(
                    np.lib.scimath.sqrt(self.eta_V(1, eta_aa, eta_bb, eta_ab)) ** 5.0
                    + np.lib.scimath.sqrt(self.eta_V(-1, eta_aa, eta_bb, eta_ab)) ** 5.0
                    )

    def func_f_symb(self, u, func: Callable, eta_a, eta_b):
        eta_array: np.ndarray = (self.a_s_factor * self.a_s_array
                                 + 4.0 * np.pi * self.a_dd_factor * self.a_dd_array * (
                                         u ** 2.0 - 1.0 / 3.0)
                                 )
        eta_aa = eta_array[0, 0]
        eta_ab = eta_array[0, 1]
        eta_bb = eta_array[1, 1]

        # TODO: why do we get nan here for eta_aa = 0.9 for some values
        # eta_aa: float64 = 0.9
        # eta_bb: float64 = 0.9
        # eta_ab: float64 = 0.9
        func_u = np.real(func(eta_aa, eta_bb, eta_ab, eta_a, eta_b))

        return func_u

    def mu_lhy_list(self, func_list: List[Callable]):
        mu_lhy_prefactor = 1.0 / (3.0 * np.sqrt(2.0) * np.pi ** 2.0)
        mu_lhy_list: list = []
        for func in func_list:
            mu_lhy_list.append(mu_lhy_prefactor * quad_vec(func, 0.0, 1.0)[0])

        return mu_lhy_list

    def func_f(self, u, eta_dVdn: Callable):
        eta_array: np.ndarray = (self.a_s_factor * self.a_s_array
                                 + 4.0 * np.pi * self.a_dd_factor * self.a_dd_array * (
                                         u ** 2.0 - 1.0 / 3.0)
                                 )
        eta_aa = eta_array[0, 0]
        eta_ab = eta_array[0, 1]
        eta_bb = eta_array[1, 1]

        term1 = (np.lib.scimath.sqrt(self.eta_V(1, eta_aa, eta_bb, eta_ab)) ** 3.0
                 * eta_dVdn(1, eta_aa, eta_bb, eta_ab)
                 )
        term2 = (np.lib.scimath.sqrt(self.eta_V(-1, eta_aa, eta_bb, eta_ab)) ** 3.0
                 * eta_dVdn(-1, eta_aa, eta_bb, eta_ab)
                 )

        result = np.real(term1 + term2)

        return result

    def eta_V(self, lam, eta_aa, eta_bb, eta_ab):
        return (eta_aa * self.A
                + eta_bb * (1.0 - self.A)
                + lam * np.sqrt((eta_aa * self.A - eta_bb * (1.0 - self.A)) ** 2.0
                                + 4.0 * eta_ab ** 2.0 * self.A * (1.0 - self.A)))

    def eta_dVdna(self, lam, eta_aa, eta_bb, eta_ab):
        return (eta_aa
                + lam * (eta_aa * (eta_aa * self.A - eta_bb * (1 - self.A))
                         + 2 * eta_ab ** 2 * (1 - self.A)
                         )
                / np.sqrt((eta_aa * self.A - eta_bb * (1 - self.A)) ** 2
                          + 4 * eta_ab ** 2 * self.A * (1 - self.A)
                          )
                )

    def eta_dVdnb(self, lam, eta_aa, eta_bb, eta_ab):
        return (eta_bb
                + lam * (eta_bb * (eta_bb * (1 - self.A) - eta_aa * self.A)
                         + 2 * eta_ab ** 2 * self.A)
                / np.sqrt((eta_aa * self.A - eta_bb * (1 - self.A)) ** 2
                          + 4 * eta_ab ** 2 * self.A * (1 - self.A)
                          )
                )

    def get_mu_lhy_list(self, density_list):
        density_A, density_total = self.get_A_density_total(density_list)

        mu_lhy_list: List[np.ndarray] = []
        en_lhy_list: List[np.ndarray] = []
        for mu_lhy_helper in self.mu_lhy_helper_list:
            mu_lhy_list.append(mu_lhy_helper(density_A) * density_total ** 1.5)

        return mu_lhy_list

    def get_energy_lhy(self, density_list):
        density_A, density_total = self.get_A_density_total(density_list)

        return self.energy_helper_function(density_A) * density_total ** 2.5

    def get_A_density_total(self, density_list):
        if len(density_list) > 2:
            sys.exit(f"get_mu_lhy_list is not implemented for mixture with more than 2 components.")

        density_total: np.ndarray = np.full_like(density_list[0], 0.0)
        for density in density_list:
            density_total += density

        # A = n1/(n1+n2)
        density_A = density_list[0] / density_total

        return density_A, density_total

    def get_r_0(self, mu_DEFAULT: float = 9.93):
        # r_0 = 387.4
        r_0 = (161.9 * constants.u_in_kg * mu_0 * (mu_DEFAULT * constants.mu_bohr) ** 2.0
               / (4.0 * np.pi * constants.hbar ** 2.0)
               ) / constants.a_0
        print(f"r_0: {r_0}")

        return r_0

    def save_psi_val(self, input_path, filename_steps, steps_format, frame):
        with open(Path(input_path, filename_steps + steps_format % frame + ".npz"), "wb") as g:
            np.savez_compressed(g, psi_val_list=self.psi_val_list)

    def use_summary(self, summary_name: Optional[str] = None):
        Summary: SchroedingerMixtureSummary = SchroedingerMixtureSummary(self)

        return Summary, summary_name

    def load_summary(self, input_path, steps_format, frame,
                     summary_name: Optional[str] = "SchroedingerMixtureSummary_"):
        try:
            # load SchroedingerSummary
            if summary_name:
                system_summary_path = Path(input_path, summary_name + steps_format % frame + ".pkl")
            else:
                system_summary_path = Path(input_path, self.name + steps_format % frame + ".pkl")

            with open(system_summary_path, "rb") as f:
                SystemSummary: SchroedingerMixtureSummary = dill.load(file=f)
        except Exception:
            print(f"{system_summary_path} not found.")
        finally:
            SystemSummary.copy_to(self)

        return self

    def array_to_tensor_grid(self, arr: np.ndarray):
        number_of_mixtures: int = arr.shape[0]

        arr_grid: List[np.ndarray] = []
        for elem in np.nditer(arr):
            arr_grid.append(np.full((self.Res.x, self.Res.y, self.Res.z), elem))
        tensor_grid_1d = np.array(arr_grid)
        tensor_grid_2d = tensor_grid_1d.reshape(
            (number_of_mixtures, number_of_mixtures, self.Res.x, self.Res.y, self.Res.z))

        return tensor_grid_2d

    def tensor_grid_mult(self, tensor, tensor_vec):
        tensor_vec_result = np.einsum("ij..., j...->i...", tensor, tensor_vec)

        return tensor_vec_result

    def arr_tensor_mult(self, arr, tensor_vec):
        tensor = self.array_to_tensor_grid(arr)
        tensor_result = self.tensor_grid_mult(tensor, tensor_vec)

        return tensor_result

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

    def energy_density_interaction(self, density_list, U_dd_list):
        dV = self.volume_element(fourier_space=False)
        U_dd_array = np.stack(U_dd_list, axis=0)
        density_array = np.stack(density_list, axis=0)

        density_U_dd = np.einsum("i...,j...->ij...", U_dd_array, density_array)
        density_density = np.einsum("i...,j...->ij...", density_array, density_array)

        # a = self.array_to_tensor_grid(self.U_dd_factor_array) * density_U_dd
        a = np.einsum("...ij, kl...->kl...", self.a_dd_array, density_U_dd)
        b = np.einsum("...ij, kl...->kl...", self.a_s_array, density_density)
        en_mf = 0.5 * np.sum(a + b).real

        return self.sum_dV(en_mf, dV=dV)

    def energy(self, density_list, U_dd_list, mu_lhy_list):
        """
        Input psi_1, psi_2 need to be normalized.
        density1 and density2 need to be build by the normalized psi_1, psi_2.

        """
        dV = self.volume_element(fourier_space=False)

        mu_lhy_part_list: List[float] = []
        for mu_lhy, density in zip(mu_lhy_list, density_list):
            mu_lhy_part_list.append(self.sum_dV(mu_lhy * density, dV=dV))
        mu_lhy_part = np.sum(np.array(mu_lhy_part_list))

        p_int = self.energy_density_interaction(density_list, U_dd_list)

        E_lhy = self.sum_dV(self.get_energy_lhy(density_list), dV=dV)

        # p_ext_list: List[float] = []
        # for density in density_list:
        #     p_ext_list.append(self.sum_dV(density * self.V_val, dV=dV))
        # p_ext = np.array(p_ext_list)

        # psi_val_array = np.array(self.psi_val_list)
        # H_kin_array = np.array(self.H_kin_list)
        # E_kin_grid = np.einsum("i...,i...->i...", H_kin_array, psi_val_array)
        # E_kin = np.fromiter(map(functools.partial(self.sum_dV, dV=dV), E_kin_grid.real),
        #                     dtype=float)

        N_array = np.array(self.N_list)
        chem_pot_N = np.dot(self.mu_arr, N_array)

        E = (chem_pot_N + p_int + E_lhy - mu_lhy_part) / np.sum(N_array)

        return E

    def save_psi_val(self, input_path, filename_steps, steps_format, frame):
        with open(Path(input_path, "mixture_" + filename_steps + steps_format % frame + ".npz"),
                  "wb") as g:
            np.savez_compressed(g, psi_val_list=self.psi_val_list)

    def get_density_list(self):
        density_list: List[np.ndarray] = []
        for psi_val, N in zip(self.psi_val_list, self.N_list):
            density_list.append(N * self.get_density(func=psi_val, p=2.0))

        return density_list

    def get_U_dd_list(self, density_list):
        U_dd_list: List[np.ndarray] = []
        for density in density_list:
            U_dd_list.append(np.fft.ifftn(self.V_k_val * np.fft.fftn(density)))

        return U_dd_list

    def get_H_pot_exponent_terms(self, dipol_term, contact_interaction, mu_lhy):
        terms = (self.V_val
                 + self.a_dd_factor * dipol_term
                 + self.a_s_factor * contact_interaction
                 + mu_lhy
                 )

        return terms

    def get_H_pot(self, terms, split_step: float = 0.5):
        H_pot = np.exp(self.U * (split_step * self.dt) * terms)

        return H_pot

    def split_operator_pot(self, split_step: float = 0.5) -> (List[np.ndarray],
                                                              List[np.ndarray]):
        density_list = self.get_density_list()
        density_tensor_vec = np.stack(density_list, axis=0)

        U_dd_list = self.get_U_dd_list(density_list)
        U_dd_tensor_vec = np.stack(U_dd_list, axis=0)

        # update H_pot before use
        contact_interaction_vec = np.einsum("...ij, j...->i...", self.a_s_array, density_tensor_vec)
        dipol_term_vec = np.einsum("...ij, j...->i...", self.a_dd_array, U_dd_tensor_vec)
        # contact_interaction_vec = self.arr_tensor_mult(self.a_s_array, density_tensor_vec)
        # dipol_term_vec = self.arr_tensor_mult(self.a_dd_array, U_dd_tensor_vec)

        mu_lhy_list: List[np.ndarray] = self.get_mu_lhy_list(density_list)
        terms_list: List[np.ndarray] = []
        for contact_interaction, dipol_term, mu_lhy in zip(list(contact_interaction_vec),
                                                           list(dipol_term_vec),
                                                           mu_lhy_list):
            terms_list.append(self.get_H_pot_exponent_terms(dipol_term,
                                                            contact_interaction,
                                                            mu_lhy
                                                            )
                              )

        H_pot_list: List[np.ndarray] = []
        for terms in terms_list:
            H_pot_list.append(self.get_H_pot(terms, split_step=split_step))

        for i, H_pot in enumerate(H_pot_list):
            self.psi_val_list[i] = H_pot * self.psi_val_list[i]

        return density_list, U_dd_list

    def split_operator_kin(self):
        # apply H_kin in k-space (transform back and forth)
        for i in range(0, len(self.psi_val_list)):
            self.psi_val_list[i] = np.fft.fftn(self.psi_val_list[i])

        for i, H_kin in enumerate(self.H_kin_list):
            self.psi_val_list[i] = H_kin * self.psi_val_list[i]

        for i in range(0, len(self.psi_val_list)):
            self.psi_val_list[i] = np.fft.ifftn(self.psi_val_list[i])

    def normalize_psi_val(self) -> List[float]:
        psi_norm_list: List[float] = []
        for i, psi_val in enumerate(self.psi_val_list):
            psi_norm_list.append(self.trapez_integral(np.abs(psi_val) ** 2.0))
            self.psi_val_list[i] = psi_val / np.sqrt(psi_norm_list[i])

        return psi_norm_list

    def time_step(self) -> None:
        """
        Evolves System according Schrödinger Equations by using the
        split operator method with the Trotter-Suzuki approximation.

        """
        # adjust dt, to get the time accuracy when needed
        # self.dt = self.dt_func(self.t, self.dt)
        self.split_operator_pot(split_step=0.5)
        self.split_operator_kin()
        self.split_operator_pot(split_step=0.5)

        self.t = self.t + self.dt

        # for self.imag_time=False, renormalization should be preserved,
        # but we play safe here (regardless of speedup)
        # if self.imag_time:
        psi_norm_list: List[float] = self.normalize_psi_val()
        for i, psi_norm in enumerate(psi_norm_list):
            self.mu_arr[i] = -np.log(psi_norm) / (2.0 * self.dt)

        # update for energy calculation
        density_list = self.get_density_list()
        U_dd_list = self.get_U_dd_list(density_list)
        mu_lhy_list: List[np.ndarray] = self.get_mu_lhy_list(density_list)

        # use normalized inputs for energy
        self.E = self.energy(density_list, U_dd_list, mu_lhy_list)
