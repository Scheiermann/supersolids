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
from scipy import ndimage
from scipy.interpolate import interpolate
from scipy.integrate import quad_vec
from scipy.ndimage import distance_transform_edt
from typing import Optional, Callable, Union, List, Tuple

import supersolids.helper.numbas as numbas
import supersolids.helper.numba_compiled as numba_compiled
from supersolids.Schroedinger import Schroedinger
from supersolids.SchroedingerMixtureSummary import SchroedingerMixtureSummary
from supersolids.helper import functions
from supersolids.helper.Box import Box
from supersolids.helper.Resolution import Resolution
from supersolids.helper.run_time import run_time

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


def get_A_density_total(density_list: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    density_total: np.ndarray = np.copy(density_list[0])
    for density in density_list[1:]:
        density_total += density

    # A = n1/(n1+n2)
    density_A = density_list[0] / density_total

    return density_A, density_total


def get_mu_lhy_integrated_list(func_list: List[Callable]) -> List[np.ndarray]:
    mu_lhy_prefactor = 1.0 / (3.0 * np.sqrt(2.0) * np.pi ** 2.0)
    mu_lhy_list: List[np.ndarray] = []
    for func in func_list:
        mu_lhy_list.append(mu_lhy_prefactor * quad_vec(func, 0.0, 1.0)[0])

    return mu_lhy_list


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
                 t: float = 0.0,
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
        self.t: float = t
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

            kx_mesh, ky_mesh, kz_mesh = np.meshgrid(self.kx, self.ky, self.kz, indexing="ij")
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

        mu_lhy_integrand_a = functools.partial(self.mu_lhy_integrand, eta_dVdn=self.eta_dVdna)
        mu_lhy_integrand_b = functools.partial(self.mu_lhy_integrand, eta_dVdn=self.eta_dVdnb)
        mu_lhy_integrated_list = get_mu_lhy_integrated_list(func_list=[mu_lhy_integrand_a,
                                                                       mu_lhy_integrand_b])

        # V_symb: List[Callable] = func_V_symb()
        # func_fa_symb = functools.partial(self.func_f_symb, func=V_symb[0],
        #                                  eta_a=self.A, eta_b=1.0 - self.A)
        # func_fb_symb = functools.partial(self.func_f_symb, func=V_symb[1],
        #                                  eta_a=self.A, eta_b=1.0 - self.A)
        # mu_v_list_symb = construct_mu_lhy_list(func_list=[func_fa_symb, func_fb_symb])

        mu_lhy_interpolation_list = []
        for mu_lhy_integrated in mu_lhy_integrated_list:
            mu_lhy_interpolation_list.append(interpolate.interp1d(self.A, mu_lhy_integrated))

        self.mu_lhy_interpolation_list = mu_lhy_interpolation_list

        energy_v = quad_vec(self.func_energy, 0.0, 1.0)[0]
        self.energy_helper_function = interpolate.interp1d(self.A, energy_v)

    def func_energy(self, u: float) -> np.ndarray:
        """
        (V_+)**5/2 + (V_-)**5/2

        """
        eta_array: np.ndarray = (self.a_s_factor * self.a_s_array
                                 + self.a_dd_factor * self.a_dd_array * functions.dipol_dipol(u)
                                 )
        eta_aa = eta_array[0, 0]
        eta_ab = eta_array[0, 1]
        eta_bb = eta_array[1, 1]
        energy_prefactor = np.sqrt(2.) / 15. / np.pi ** 2

        return energy_prefactor * np.real(
            np.lib.scimath.sqrt(numba_compiled.f_lam(self.A, 1, eta_aa, eta_bb, eta_ab)) ** 5.0
            + np.lib.scimath.sqrt(numba_compiled.f_lam(self.A, -1, eta_aa, eta_bb, eta_ab)) ** 5.0
        )

    def func_f_symb(self, u: float, func: Callable, eta_a, eta_b):
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

    def mu_lhy_integrand(self, u: float, eta_dVdn: Callable) -> np.ndarray:
        eta_array: np.ndarray = (self.a_s_factor * self.a_s_array
                                 + self.a_dd_factor * self.a_dd_array * functions.dipol_dipol(u)
                                 )
        eta_aa = eta_array[0, 0]
        eta_ab = eta_array[0, 1]
        eta_bb = eta_array[1, 1]

        term1 = (np.lib.scimath.sqrt(numba_compiled.f_lam(self.A, 1, eta_aa, eta_bb, eta_ab)) ** 3.0
                 * eta_dVdn(1, eta_aa, eta_bb, eta_ab)
                 )
        term2 = (np.lib.scimath.sqrt(numba_compiled.f_lam(self.A, -1, eta_aa, eta_bb, eta_ab)) ** 3.0
                 * eta_dVdn(-1, eta_aa, eta_bb, eta_ab)
                 )

        result = np.real(term1 + term2)

        return result

    def eta_dVdna(self, lam: float, eta_aa: float, eta_bb: float, eta_ab: float) -> np.ndarray:
        return numba_compiled.eta_dVdna_jit(self.A, lam, eta_aa, eta_bb, eta_ab)

    def eta_dVdnb(self, lam: float, eta_aa: float, eta_bb: float, eta_ab: float) -> np.ndarray:
        return numba_compiled.eta_dVdnb_jit(self.A, lam, eta_aa, eta_bb, eta_ab)

    def get_mu_lhy_list(self, density_list: List[np.ndarray]) -> List[np.ndarray]:
        density_A, density_total = get_A_density_total(density_list)

        mu_lhy_list: List[np.ndarray] = []
        for mu_lhy_interpolation in self.mu_lhy_interpolation_list:
            mu_lhy = mu_lhy_interpolation(density_A) * density_total ** 1.5
            mu_lhy_list.append(mu_lhy)

        return mu_lhy_list

    def get_energy_lhy(self, density_list: List[np.ndarray]) -> np.ndarray:
        density_A, density_total = get_A_density_total(density_list)

        return self.energy_helper_function(density_A) * density_total ** 2.5

    def save_psi_val(self, input_path: Path, filename_steps: str,
                     steps_format: str, frame: int) -> None:
        with open(Path(input_path, "mixture_" + filename_steps + steps_format % frame + ".npz"),
                  "wb") as g:
            np.savez_compressed(g, psi_val_list=self.psi_val_list)

    def use_summary(self, summary_name: Optional[str] = None) -> Tuple[SchroedingerMixtureSummary,
                                                                       Optional[str]]:
        Summary: SchroedingerMixtureSummary = SchroedingerMixtureSummary(self)

        return Summary, summary_name

    def load_summary(self, input_path: Path, steps_format: str, frame: int,
                     summary_name: Optional[str] = "SchroedingerMixtureSummary_"):
        if summary_name:
            system_summary_path = Path(input_path, summary_name + steps_format % frame + ".pkl")
        else:
            system_summary_path = Path(input_path, self.name + steps_format % frame + ".pkl")

        try:
            # load SchroedingerSummary
            with open(system_summary_path, "rb") as f:
                SystemSummary: SchroedingerMixtureSummary = dill.load(file=f)
        except Exception:
            print(f"{system_summary_path} not found.")
        finally:
            SystemSummary.copy_to(self)

        return self

    def load_mu(self,
                filename_mu_a: str = "interpolator_mu_a.pkl",
                filename_mu_b: str = "interpolator_mu_b.pkl") -> Tuple[np.ndarray, np.ndarray]:
        print("Loading mu from interpolating pickles ... ")
        with open(Path(self.input_path, filename_mu_a), "rb") as f:
            mu_a: np.ndarray = pickle.load(f)
        with open(Path(self.input_path, filename_mu_b), "rb") as f:
            mu_b: np.ndarray = pickle.load(f)
        print("mu pickles sucessfully loaded!")

        return mu_a, mu_b

    def load_lhy(self, filename_lhy: str = "interpolator_lhy_energy.pkl") -> np.ndarray:
        print("Loading lhy from interpolating pickles ... ")
        with open(Path(self.input_path, filename_lhy), "rb") as f:
            lhy_energy: np.ndarray = pickle.load(f)
        print("lhy pickle sucessfully loaded!")

        return lhy_energy

    def energy_density_interaction(self,
                                   density_list: List[np.ndarray],
                                   U_dd_list: List[np.ndarray]) -> float:
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

    def get_E(self) -> None:
        # update for energy calculation
        density_list = self.get_density_list()
        U_dd_list = self.get_U_dd_list(density_list)
        mu_lhy_list: List[np.ndarray] = self.get_mu_lhy_list(density_list)

        # use normalized inputs for energy
        self.E = self.energy(density_list, U_dd_list, mu_lhy_list)

    def energy(self, density_list: List[np.ndarray], U_dd_list: List[np.ndarray],
               mu_lhy_list: List[np.ndarray]) -> float:
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

    def get_density_list(self) -> List[np.ndarray]:
        density_list: List[np.ndarray] = []
        for psi_val, N in zip(self.psi_val_list, self.N_list):
            density_list.append(N * numbas.get_density_jit(psi_val))

        return density_list

    def get_center_of_mass(self,
                           Mx0: Optional[int] = None, Mx1: Optional[int] = None,
                           My0: Optional[int] = None, My1: Optional[int] = None,
                           Mz0: Optional[int] = None, Mz1: Optional[int] = None) -> List[float]:
        """
        Calculates the center of mass of the System.

        """

        x0, x1, y0, y1, z0, z1 = self.slice_default(Mx0, Mx1, My0, My1, Mz0, Mz1)
        prob_list = [density[x0:x1, y0:y1, z0:z1] for density in self.get_density_list()]
        r = self.get_mesh_list(x0, x1, y0, y1, z0, z1)
        com_list = []
        for prob in prob_list:
            center_of_mass_along_axis = [prob * r_i for r_i in r]
            com_list.append([self.trapez_integral(com_along_axis) / self.trapez_integral(prob)
                             for com_along_axis in center_of_mass_along_axis])
        return com_list

    def get_parity(self,
                   axis: int = 2,
                   Mx0: Optional[int] = None, Mx1: Optional[int] = None,
                   My0: Optional[int] = None, My1: Optional[int] = None,
                   Mz0: Optional[int] = None, Mz1: Optional[int] = None) -> List[float]:
        x0, x1, y0, y1, z0, z1 = self.slice_default(Mx0, Mx1, My0, My1, Mz0, Mz1)
        parity_list: List[float] = []
        for psi_val in self.psi_val_list:
            psi_under0, psi_over0 = np.split(psi_val, 2, axis=axis)

            if axis in [0, 1, 2]:
                psi_over0_reversed = psi_over0[::-1]
            else:
                sys.exit(f"No such axis ({axis}). Choose 0, 1 or 2 for axis x, y or z.")

            parity = self.trapez_integral(np.abs(
                psi_under0[x0:x1, y0:y1, z0:z1] - psi_over0_reversed[x0:x1, y0:y1, z0:z1]) ** 2.0)

            parity_list.append(parity)

        return parity_list

    def distmat(self, a: np.ndarray, index: List[float]):
        mask = np.ones(a.shape, dtype=bool)
        mask[index[0], index[1], index[2]] = False
        return distance_transform_edt(mask)

    def get_contrast_old(self,
                   axis: int = 2,
                   Mx0: Optional[int] = None, Mx1: Optional[int] = None,
                   My0: Optional[int] = None, My1: Optional[int] = None,
                   Mz0: Optional[int] = None, Mz1: Optional[int] = None) -> List[float]:
        x0, x1, y0, y1, z0, z1 = self.slice_default(Mx0, Mx1, My0, My1, Mz0, Mz1)
        prob_list = self.get_density_list()
        bec_contrast_list = []
        for N, prob in zip(self.N_list, prob_list):
            mask = np.full(np.shape(prob), False)
            mask[x0:x1, y0:y1, z0:z1] = True

            bec_min_edgeless = ndimage.minimum(prob, labels=mask)
            bec_max_edgeless = ndimage.maximum(prob, labels=mask)
            bec_contrast_edgeless = (bec_max_edgeless - bec_min_edgeless) / (
                    bec_max_edgeless + bec_min_edgeless)
            bec_contrast_list.append(bec_contrast_edgeless)

        return bec_contrast_list

    def get_contrast(self, number_of_peaks: int, prob_min_start, prob_step: float = 0.01,
                     prob_min_edge: float = 0.015, region_threshold: int = 100,
                     ) -> List[float]:
        prob_list = self.get_density_list()
        bec_contrast_list = []
        structures = functions.binary_structures()
        for N, prob in zip(self.N_list, prob_list):
            prob_max = np.max(prob)
            prob_min_lin_reverse = np.arange(0.001, prob_min_start, prob_step)[::-1]
            for prob_min_ratio in prob_min_lin_reverse:
                # get biggest region
                prob_region_outer = np.where(prob >= prob_min_ratio * prob_max, prob, 0)
                label_im_outer, nb_labels_outer = ndimage.label(prob_region_outer,
                                                                structure=structures[0])
                sizes_outer = ndimage.sum(prob_region_outer, label_im_outer,
                                          range(nb_labels_outer + 1))
                mask_outer = sizes_outer > region_threshold
                label_region_outer = mask_outer[label_im_outer]
                region_outer = functions.fill_holes(label_region_outer, *structures[1:])

                # remove edge
                prob_region_inner = np.where(prob >= (prob_min_edge + prob_min_ratio) * prob_max,
                                             prob, 0)
                label_im_inner, nb_labels_inner = ndimage.label(prob_region_inner,
                                                                structure=structures[0])
                sizes_inner = ndimage.sum(prob_region_inner, label_im_inner,
                                          range(nb_labels_inner + 1))
                mask_inner = sizes_inner > region_threshold
                label_region_inner = mask_inner[label_im_inner]
                region_inner = functions.fill_holes(label_region_inner, *structures[1:])

                region_edgeless = np.logical_and(region_outer, region_inner)
                region = functions.fill_holes(region_edgeless, *structures[1:])
                label_im_region, nb_labels_region = ndimage.label(region, structure=structures[0])

                if nb_labels_region == 1:
                    region_shrunk = ndimage.binary_erosion(region)
                    region_diff = np.logical_xor(region_shrunk, region)
                    label_im_region_diff, nb_labels_region_diff = ndimage.label(
                        region_diff, structure=structures[0])

                    peak_list = self.get_peak_neighborhood(prob, prob_max * prob_min_ratio,
                                                           number_of_peaks=number_of_peaks)
                    if len(peak_list) == 1:
                        # no peak_neighborhood to calculate dist
                        prob_one = np.where(prob > 0.8 * prob_max, prob, 0)
                        label_im_one, nb_labels_one = ndimage.label(prob_one,
                                                                    structure=structures[0])
                        bec_min = ndimage.minimum(prob_one, labels=label_im_one)
                    else:
                        peaks_max_index = [ndimage.maximum_position(peak,
                                                                    labels=np.array(peak,
                                                                                    dtype=bool))
                                           for peak in peak_list
                                           ]
                        max_dists = [self.distmat(prob, peak_max_index)
                                     for peak_max_index in peaks_max_index]
                        max_dists_sum = sum(max_dists)
                        bec_min_index = ndimage.minimum_position(max_dists_sum, labels=region_diff)
                        bec_min = prob[bec_min_index]

                    bec_max = ndimage.maximum(prob, labels=region)
                    bec_contrast = (bec_max - bec_min) / (bec_max + bec_min)
                    break

            if nb_labels_region != 1:
                print(f"Could not connect maxima regions. prob_step: {prob_step} to high.")
                # sys.exit(f"Could not connect maxima regions. prob_step: {prob_step} to high.")

            bec_contrast_list.append(bec_contrast)

        return bec_contrast_list

    def get_U_dd_list(self, density_list: List[np.ndarray]) -> List[np.ndarray]:
        U_dd_list: List[np.ndarray] = []
        for density in density_list:
            U_dd_list.append(np.fft.ifftn(self.V_k_val * np.fft.fftn(density)))

        return U_dd_list


    def get_H_pot(self, terms: np.ndarray, split_step: float = 0.5) -> np.ndarray:
        H_pot = np.exp(self.U * (split_step * self.dt) * terms)

        return H_pot

    def split_operator_pot(self, split_step: float = 0.5) -> Tuple[List[np.ndarray],
                                                                   List[np.ndarray]]:
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
        for i, (contact_interaction, dipol_term, mu_lhy) in enumerate(zip(
                list(contact_interaction_vec),
                list(dipol_term_vec),
                mu_lhy_list)):
            term = numba_compiled.get_H_pot_exponent_terms_jit(self.V_val,
                                                               self.a_dd_factor,
                                                               self.a_s_factor,
                                                               dipol_term,
                                                               contact_interaction,
                                                               mu_lhy
                                                               )
            H_pot = numba_compiled.get_H_pot_jit(self.U, self.dt, term, split_step)
            self.psi_val_list[i] = H_pot * self.psi_val_list[i]

        return density_list, U_dd_list

    def split_operator_kin(self) -> None:
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
            psi_norm_list.append(self.get_norm(func_val=psi_val))
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
