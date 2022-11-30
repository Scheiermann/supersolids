#!/usr/bin/env python

# author: Daniel Scheiermann
# email: daniel.scheiermann@itp.uni-hannover.de
# license: MIT
# Please feel free to use and modify this, but keep the above information.

"""
Numerical solver for non-linear time-dependent Schrodinger equation (eGPE) for dipolar mixtures.


"""
from copy import deepcopy
import functools
import pickle
import sys

import dill
import numpy as np
from pathlib import Path
from typing import Optional, Callable, Union, List, Tuple

from scipy import ndimage
from scipy.interpolate import interpolate
from scipy.integrate import quad_vec
from scipy.ndimage import distance_transform_edt

from supersolids.helper import constants, functions, get_path, get_version

__GPU_OFF_ENV__, __GPU_INDEX_ENV__ = get_version.get_env_variables()
cp, cupy_used, cuda_used, numba_used = get_version.check_cp_nb(np,
                                                               gpu_off=__GPU_OFF_ENV__,
                                                               gpu_index=__GPU_INDEX_ENV__)
import supersolids.helper.numbas as numbas

from supersolids.Schroedinger import Schroedinger
from supersolids.SchroedingerMixtureSummary import SchroedingerMixtureSummary
from supersolids.helper import constants, functions
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


def smaller_slice(val0, val1):
    # decrease interval from left and right by one

    # make sure updated val1 will be bigger than updated val0
    if val0 < val1 - 1:
        changed = True
        val0 = val0 + 1
        val1 = val1 - 1
        if val0 == val1:
            val1 = val1 + 1
    else:
        changed = False

    return changed, val0, val1

def get_A_density_total(density_list: List[cp.ndarray]) -> Tuple[cp.ndarray, cp.ndarray]:
    if cupy_used:
        try:
            density_list[0].get() 
            density_total: cp.ndarray = cp.copy(density_list[0])
        except:
            try:
                for i, density in enumerate(density_list):
                    density_list[i] = cp.asarray(density)
                density_total: cp.ndarray = cp.copy(density_list[0])
            except:
                density_total: np.ndarray = np.copy(density_list[0])
    else:
        density_total: np.ndarray = np.copy(density_list[0])

    for density in density_list[1:]:
        density_total += density

    # A = n1/(n1+n2)
    density_A = density_list[0] / density_total
    # remove nan, because of possible 0/0 
    try:
        density_A = cp.where(density_A != cp.nan, density_A, 0.0)
    except:
        try:
            density_A = np.where(density_A != cp.nan, density_A, 0.0)
        except:
            density_A = np.where(density_A != np.nan, density_A, 0.0)


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
                 tilt: float = 0.0,
                 stack_shift: float = 0.0,
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
        
        self.tilt = tilt
        self.stack_shift = stack_shift

        self.E: float = E

        if mu_arr is None:
            self.mu_arr: np.ndarray = np.array([1.0] * len(self.N_list))
        else:
            self.mu_arr: np.ndarray = mu_arr

        self.psi_0_list: List[np.ndarray] = psi_0_list
        self.psi_sol_list: List[Optional[Callable]] = psi_sol_list
        self.mu_sol_list: List[Optional[Callable]] = mu_sol_list

        self.psi_val_list: List[cp.ndarray] = []
        self.psi_sol_val_list: List[cp.ndarray] = []
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
            if psi_0_list:
                for psi_0, psi_0_noise in zip(psi_0_list, psi_0_noise_list):
                    if cupy_used:
                        psi_val: cp.ndarray = cp.asarray(psi_0(self.x))
                    else:
                        psi_val: cp.ndarray = psi_0(self.x)

                    if psi_0_noise is None:
                        self.psi_val_list.append(psi_val)
                    else:
                        self.psi_val_list.append(psi_0_noise * psi_val)
            else:
                # no initialization
                psi_val: cp.ndarray = self.x
                self.psi_val_list.append(psi_val)

            self.k_squared: np.ndarray = self.kx ** 2.0

            if V is None:
                self.V_val: Union[float, cp.ndarray] = 0.0
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

            if psi_0_list:
                for psi_0, psi_0_noise in zip(psi_0_list, psi_0_noise_list):
                    if cupy_used:
                        psi_val: cp.ndarray = cp.asarray(psi_0(self.pos))
                    else:
                        psi_val: cp.ndarray = psi_0(self.pos)

                    if psi_0_noise is None:
                        self.psi_val_list.append(psi_val)
                    else:
                        self.psi_val_list.append(psi_0_noise * psi_val)
            else:
                # no initialization
                psi_val: cp.ndarray = self.x_mesh
                self.psi_val_list.append(psi_val)


            kx_mesh, ky_mesh, _ = functions.get_meshgrid(self.kx, self.ky)
            self.k_squared = kx_mesh ** 2.0 + ky_mesh ** 2.0

            if V is None:
                self.V_val = 0.0
            else:
                if cupy_used:
                    self.V_val: cp.ndarray = cp.asarray(self.V(self.pos))
                else:
                    self.V_val: np.ndarray = self.V(self.pos)

        elif self.dim == 3:
            self.x_mesh, self.y_mesh, self.z_mesh = functions.get_grid(self.Res, self.Box)
            # if cupy_used:
            #     x_mesh, y_mesh, z_mesh = cp.asarray(x_mesh), cp.asarray(y_mesh), cp.asarray(z_mesh)
            #     x_mesh, y_mesh, z_mesh = self.x_mesh.get(), self.y_mesh.get(), self.z_mesh.get()
     
            if psi_0_list:
                for psi_0, psi_0_noise in zip(psi_0_list, psi_0_noise_list):
                    if cupy_used:
                        psi_val: cp.ndarray = cp.asarray(psi_0(self.x_mesh, self.y_mesh, self.z_mesh))
                        if psi_0_noise is None:
                            self.psi_val_list.append(psi_val)
                        else:
                            # psi_0_noise: cp.ndarray = cp.asarray(psi_0_noise)
                            # psi_val_noisy = psi_0_noise * psi_val
                            psi_val_noisy = psi_0_noise * psi_val.get()
                            self.psi_val_list.append(cp.asarray(psi_val_noisy))
                    else:
                        psi_val: np.ndarray = psi_0(self.x_mesh, self.y_mesh, self.z_mesh)
                        if psi_0_noise is None:
                            self.psi_val_list.append(psi_val)
                        else:
                            psi_val_noisy = psi_0_noise * psi_val
                            self.psi_val_list.append(psi_val_noisy)
            else:
                # no initialization
                psi_val: cp.ndarray = self.x_mesh
                self.psi_val_list.append(psi_val)

            for psi_sol in self.psi_sol_list:
                if psi_sol is None:
                    self.psi_sol_val_list.append(None)
                else:
                    if callable(psi_sol):
                        if cupy_used:
                            psi_sol_val = cp.asarray(psi_sol(self.x_mesh, self.y_mesh, self.z_mesh))
                        else:
                            psi_sol_val = psi_sol(self.x_mesh, self.y_mesh, self.z_mesh)
                        self.psi_sol_val_list.append(psi_sol_val)
                        if psi_sol_val is not None:
                            print(f"Norm for psi_sol (trapez integral): "
                                  f"{self.trapez_integral(cp.abs(psi_sol_val) ** 2.0)}")

            kx_mesh, ky_mesh, kz_mesh = np.meshgrid(self.kx, self.ky, self.kz, indexing="ij")
            self.k_squared: cp.ndarray = np.power(kx_mesh, 2) + np.power(ky_mesh, 2) + np.power(kz_mesh, 2)
            if cupy_used:
                self.kz_mesh = cp.asarray(kz_mesh)
            else:
                self.kz_mesh = kz_mesh

            if V is None:
                self.V_val = 0.0
            else:
                if cupy_used:
                    self.V_val: cp.ndarray = cp.asarray(self.V(self.x_mesh,
                                                               self.y_mesh,
                                                               self.z_mesh))
                else:
                    self.V_val: cp.ndarray = self.V(self.x_mesh, self.y_mesh, self.z_mesh)

            if self.V_interaction is None:
                # For no interaction the identity is needed with respect to 2D
                # * 2D (array with 1.0 everywhere)
                self.V_k_val: cp.ndarray = np.full(self.psi_val.shape, 1.0)
            else:
                if callable(self.V_interaction):
                    self.V_k_val: np.ndarray = self.V_interaction(kx_mesh, ky_mesh, kz_mesh)
                    if cupy_used:
                        self.V_k_val: cp.ndarray = cp.asarray(self.V_k_val)

        if imag_time:
            # Convention: $e^{-iH} = e^{UH}$
            self.U: complex = -1.0
        else:
            self.U = -1.0j

        self.H_kin: np.ndarray = np.exp(self.U * (0.5 * self.k_squared) * self.dt)

        self.A = np.linspace(0.0, 1.0, self.nA_max)

        eta_dVdna = functools.partial(numbas.eta_dVdna_jit, A=self.A)
        eta_dVdnb = functools.partial(numbas.eta_dVdnb_jit, A=self.A)
 
        mu_lhy_integrand_a = functools.partial(self.mu_lhy_integrand, eta_dVdn=eta_dVdna)
        mu_lhy_integrand_b = functools.partial(self.mu_lhy_integrand, eta_dVdn=eta_dVdnb)
        
        ######## testing
        eta_array: np.ndarray = (self.a_s_factor * self.a_s_array
                                 + self.a_dd_factor * self.a_dd_array * quad_vec(functions.dipol_dipol, 0.0, 1.0)[0]
                                 )
        eta_aa = eta_array[0, 0]
        eta_ab = eta_array[0, 1]
        eta_bb = eta_array[1, 1]
        c_t1 = eta_dVdna(lam=1, eta_aa=eta_aa, eta_bb=eta_bb, eta_ab=eta_ab)
        c_t2 = eta_dVdna(lam=-1, eta_aa=eta_aa, eta_bb=eta_bb, eta_ab=eta_ab)
        d_t1 = eta_dVdnb(lam=1, eta_aa=eta_aa, eta_bb=eta_bb, eta_ab=eta_ab)
        d_t2 = eta_dVdnb(lam=-1, eta_aa=eta_aa, eta_bb=eta_bb, eta_ab=eta_ab)
        b_t = quad_vec(mu_lhy_integrand_b, 0.0, 1.0)[0]
        a_t = quad_vec(mu_lhy_integrand_a, 0.0, 1.0)[0]
        ######## testing

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

        if cupy_used:
            self.k_squared = cp.asarray(self.k_squared)
            self.H_kin = cp.asarray(self.H_kin)

        self.H_kin_list = []
        for m in self.m_list:
            self.H_kin_list.append(self.H_kin)


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
            np.lib.scimath.sqrt(numbas.f_lam(self.A, 1, eta_aa, eta_bb, eta_ab)) ** 5.0
            + np.lib.scimath.sqrt(numbas.f_lam(self.A, -1, eta_aa, eta_bb, eta_ab)) ** 5.0
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
        term1 = (np.lib.scimath.sqrt(numbas.f_lam(self.A, 1, eta_aa, eta_bb, eta_ab)) ** 3.0
                 * eta_dVdn(lam=1, eta_aa=eta_aa, eta_bb=eta_bb, eta_ab=eta_ab)
                 )
        term2 = (np.lib.scimath.sqrt(numbas.f_lam(self.A, -1, eta_aa, eta_bb, eta_ab)) ** 3.0
                 * eta_dVdn(lam=-1, eta_aa=eta_aa, eta_bb=eta_bb, eta_ab=eta_ab)
                 )

        result = np.real(term1 + term2)

        return result

    def eta_dVdna(self, lam: float, eta_aa: float, eta_bb: float, eta_ab: float) -> np.ndarray:
        return numbas.eta_dVdna_jit(self.A, lam, eta_aa, eta_bb, eta_ab)

    def eta_dVdnb(self, lam: float, eta_aa: float, eta_bb: float, eta_ab: float) -> np.ndarray:
        return numbas.eta_dVdnb_jit(self.A, lam, eta_aa, eta_bb, eta_ab)

    def get_mu_lhy_list(self, density_list: List[cp.ndarray]) -> List[cp.ndarray]:
        density_A, density_total = get_A_density_total(density_list)

        mu_lhy_list: List[cp.ndarray] = []
        for mu_lhy_interpolation in self.mu_lhy_interpolation_list:
            if cupy_used:
                mu_lhy = cp.asarray(mu_lhy_interpolation(density_A.get()) * density_total.get() ** 1.5)
            else:
                mu_lhy = mu_lhy_interpolation(density_A) * density_total ** 1.5

            mu_lhy_list.append(mu_lhy)

        return mu_lhy_list

    def get_energy_lhy(self, density_list: List[cp.ndarray]) -> cp.ndarray:
        density_A, density_total = get_A_density_total(density_list)
        if cupy_used:
            energy_lhy = self.energy_helper_function(density_A.get()) * density_total.get() ** 2.5
        else:
            energy_lhy = self.energy_helper_function(density_A) * density_total ** 2.5

        return energy_lhy

    def save_psi_val(self, input_path: Path, filename_steps: str,
                     steps_format: str, frame: int, arr_list = None) -> None:
        if arr_list is None:
            arr_list: List[cp.ndarray] = self.psi_val_list
        path_output = Path(input_path, filename_steps + f"{steps_format % frame}" + ".npz")
        with open(path_output, "wb") as g:
            if cupy_used:
                try:
                    psi_val_list: np.ndarray = [psi_val.get() for psi_val in arr_list]
                except Exception:
                    # cupy is installed, but data was saved as numpy array
                    psi_val_list: np.ndarray = [psi_val for psi_val in arr_list]
            else:
                psi_val_list: np.ndarray = arr_list
            cp.savez_compressed(g, psi_val_list=psi_val_list)

    def use_summary(self, summary_name: Optional[str] = None) -> Tuple[SchroedingerMixtureSummary,
                                                                       Optional[str]]:
        Summary: SchroedingerMixtureSummary = SchroedingerMixtureSummary(self)

        return Summary, summary_name

    def load_summary(self, input_path: Path, steps_format: str, frame: int,
                     summary_name: Optional[str] = "SchroedingerMixtureSummary_",
                     host=None):
        if summary_name:
            system_summary_path = Path(input_path, summary_name + steps_format % frame + ".pkl")
        else:
            system_summary_path = Path(input_path, self.name + steps_format % frame + ".pkl")

        try:
            # load SchroedingerSummary
            if host:
                sftp = host.sftp()
                with sftp.file(str(system_summary_path), "rb") as f:
                    SystemSummary: SchroedingerMixtureSummary = dill.load(file=f)
                    SystemSummary.copy_to(self)
            else:
                with open(system_summary_path, "rb") as f:
                    SystemSummary: SchroedingerMixtureSummary = dill.load(file=f)
                    SystemSummary.copy_to(self)
        except Exception as e:
            print(f"{system_summary_path} not found.\n {e}")

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

    def energy_density_interaction_explicit(self, density_list: List[np.ndarray]) -> float:
        dV = self.volume_element(fourier_space=False)
        density_array = cp.stack(density_list, axis=0)

        density_density = cp.einsum("i...,j...->ij...", density_array, density_array)

        # density_density and density_U_dd are (2, 2, res_x, res_y, res_z)
        # the (2,2) part needs to be summed after one-by-one multiplication
        # [[a11 * b11, a12 * a12], [a21 * b21, a22 * b22]] with (2, 2) matrices
        # formally, as we sum in real space dV is just a constant multiplication, so summation
        # over all 5 dimensions is fine
        energy_scattering = cp.einsum("ij..., ijklm->ij", self.a_s_array, density_density)

        # if stack_shift is used U_dd has higher dimensionality, but resulting density_U_dd the same
        if self.stack_shift != 0.0:
            U_dd = self.get_U_dd(density_list)
            density_U_dd = cp.einsum("ijklm, jklm->ijklm", U_dd, density_array)
        else:
            U_dd_list = self.get_U_dd_list(density_list)
            U_dd_array = cp.stack(U_dd_list, axis=0)
            density_U_dd = cp.einsum("i...,j...->ij...", U_dd_array, density_array)

        energy_dipol_dipol = cp.einsum("ij..., ijklm->ij", self.a_dd_array, density_U_dd)

        # E_scatter = 0.5 * np.sum(energy_scattering).real * dV
        # E_dd = np.sum(energy_dipol_dipol).real * dV
        E_scatter = 0.5 * np.sum(energy_scattering) * dV
        E_dd = 0.5 * np.sum(energy_dipol_dipol).real * dV
        if cupy_used:
            E_scatter = cp.asnumpy(E_scatter)
            E_dd = cp.asnumpy(E_dd)

        return E_dd, E_scatter

    def energy_density_interaction(self, density_list: List[np.ndarray]) -> float:
        dV = self.volume_element(fourier_space=False)
        density_array = cp.stack(density_list, axis=0)

        density_density = cp.einsum("i...,j...->ij...", density_array, density_array)

        # density_density and density_U_dd are (2, 2, res_x, res_y, res_z)
        # the (2,2) part needs to be summed after one-by-one multiplication
        # [[a11 * b11, a12 * a12], [a21 * b21, a22 * b22]] with (2, 2) matrices
        # formally, as we sum in real space dV is just a constant multiplication, so summation
        # over all 5 dimensions is fine
        
        # there are 3 ways when to sum (verison c is the fastest):
        # a. getting a (2, 2, res_x, res_y, res_z) matrix
        # b. getting a (res_x, res_y, res_z) matrix, where (2,2) grid part is already summed
        # c. getting a (2, 2) matrix, where the (res_x, res_y, res_z) is already summed

        # energy_dipol_dipol_a = np.einsum("ij..., ijklm->ijklm", self.a_s_array, density_density)
        # energy_dipol_dipol_b = np.einsum("ij..., ijklm->klm", self.a_s_array, density_density)
        energy_scattering = cp.einsum("ij..., ijklm->ij", self.a_s_array, density_density)

        # if stack_shift is used U_dd has higher dimensionality, but resulting density_U_dd the same
        if self.stack_shift != 0.0:
            U_dd = self.get_U_dd(density_list)
            density_U_dd = cp.einsum("ijklm, jklm->ijklm", U_dd, density_array)
        else:
            U_dd_list = self.get_U_dd_list(density_list)
            U_dd_array = cp.stack(U_dd_list, axis=0)
            density_U_dd = cp.einsum("i...,j...->ij...", U_dd_array, density_array)

        energy_dipol_dipol = cp.einsum("ij..., ijklm->ij", self.a_dd_array, density_U_dd)

        E_dd = 0.5 * np.sum(energy_scattering + energy_dipol_dipol).real * dV

        return E_dd

    def get_E(self) -> None:
        # update for energy calculation
        density_list = self.get_density_list(jit=numba_used)
        mu_lhy_list: List[np.ndarray] = self.get_mu_lhy_list(density_list)

        # use normalized inputs for energy
        self.E = self.energy(density_list, mu_lhy_list)


    def get_E_pot(self, density_list: List[cp.ndarray], dV: float = None) -> None:
        E_pot_ext_list: List[float] = []
        for density in density_list:
            if cupy_used:
                E_pot_ext_list.append(self.sum_dV(cp.asarray(density) * cp.asarray(self.V_val), dV=dV))
            else:
                try:
                    E_pot_ext_list.append(self.sum_dV(density * self.V_val, dV=dV))
                except:
                    E_pot_ext_list.append(self.sum_dV(density.get() * self.V_val, dV=dV))

        try:
            E_pot_arr = cp.array(E_pot_ext_list).get()
        except:
            E_pot_arr = np.array(E_pot_ext_list)
        # add all components
        E_pot = np.sum(E_pot_arr)
        
        return E_pot


    def get_E_kin(self, dV: float = None) -> None:
        psi_val_array = np.array(self.psi_val_list)
        fft_dim = range(1, psi_val_array.ndim)
        psi_val_array_k = np.fft.fftn(psi_val_array, axes=fft_dim)
        H_kin_array = np.array(self.H_kin_list)
        E_kin_grid_helper = np.einsum("i...,i...->i...", H_kin_array, psi_val_array_k)
        E_kin_grid_helper_fft = np.fft.ifftn(E_kin_grid_helper, axes=fft_dim)
        E_kin_grid = np.einsum("i...,i...->i...", np.conjugate(psi_val_array), E_kin_grid_helper_fft)
        E_kin_arr = np.fromiter(map(functools.partial(self.sum_dV, dV=dV), E_kin_grid.real),
                                 dtype=float) * np.array(self.N_list)
        E_kin = np.sum(E_kin_arr)

        return E_kin

    def get_mu_explicit(self, psi_previous_list) -> None:
        mu_list = []
        for (psi_val, psi_previous) in zip(self.psi_val_list, psi_previous_list):
            integrand = np.conjugate(psi_previous) * (psi_val - psi_previous)
            mu_list.append((1 / self.dt) * self.sum_dV(integrand))
        
        return mu_list

    def get_E_explicit(self) -> None:
        dV = self.volume_element(fourier_space=False)

        # update for energy calculation
        density_list = self.get_density_list(jit=numba_used)
        # density_list_per_particle = []
        # for density, N in zip(density_list, self.N_list):
        #     # norm
            # density_list_per_particle.append((1.0 / N) * density)

        E_lhy = self.sum_dV(self.get_energy_lhy(density_list), dV=dV)
        E_dd, E_scatter = self.energy_density_interaction_explicit(density_list)
        E_pot = self.get_E_pot(density_list, dV=dV)
        E_kin = self.get_E_kin(dV=dV)

        # use normalized inputs for energy
        N = np.sum(self.N_list)
        E = (E_kin + E_pot + E_lhy + E_dd + E_scatter) / N
        print(f"Check E: {E}")
        
        return E

    def energy(self, density_list: List[cp.ndarray], mu_lhy_list: List[cp.ndarray]) -> float:
        """
        Input psi_1, psi_2 need to be normalized.
        density1 and density2 need to be build by the normalized psi_1, psi_2.

        """
        dV = self.volume_element(fourier_space=False)

        mu_lhy_part_list: List[float] = []
        for mu_lhy, density in zip(mu_lhy_list, density_list):
            mu_lhy_part_list.append(self.sum_dV(mu_lhy * density, dV=dV))
        mu_lhy_part = cp.sum(cp.array(mu_lhy_part_list))

        p_int = self.energy_density_interaction(density_list)

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

    def get_density_list(self, jit: bool = True, cupy_used: bool = False) -> List[cp.ndarray]:
        density_list: List[np.ndarray] = []
        for psi_val, N in zip(self.psi_val_list, self.N_list):
            if jit:
                density_N: np.ndarray = N * numbas.get_density_jit(psi_val, p=2.0)
            else:
                try:
                    density_N = N * self.get_density(psi_val, jit=jit)
                except Exception:
                    try:
                        density_N: np.ndarray = N * self.get_density(psi_val, jit=jit).get()
                    except Exception:
                        density_N: np.ndarray = N * self.get_density(psi_val, jit=jit)
                if cupy_used:
                    density_N: cp.ndarray = cp.asarray(density_N)

            density_list.append(density_N)

        return density_list

    def get_center_of_mass(self,
                           Mx0: Optional[int] = None, Mx1: Optional[int] = None,
                           My0: Optional[int] = None, My1: Optional[int] = None,
                           Mz0: Optional[int] = None, Mz1: Optional[int] = None,
                           p: float = 1.0) -> List[float]:
        """
        Calculates the center of mass of the System.

        """

        x0, x1, y0, y1, z0, z1 = self.slice_default(Mx0, Mx1, My0, My1, Mz0, Mz1)
        prob_list = [density[x0:x1, y0:y1, z0:z1]
                     for density in self.get_density_list(jit=numba_used)]
        r = self.get_mesh_list(x0, x1, y0, y1, z0, z1)
        com_list = []
        for prob in prob_list:
            center_of_mass_along_axis = [prob * r_i ** p for r_i in r]
            com_list.append([self.trapez_integral(com_along_axis) / self.trapez_integral(prob)
                             for com_along_axis in center_of_mass_along_axis])
        return com_list

    def get_parity(self,
                   axis: Optional[int] = None,
                   Mx0: Optional[int] = None, Mx1: Optional[int] = None,
                   My0: Optional[int] = None, My1: Optional[int] = None,
                   Mz0: Optional[int] = None, Mz1: Optional[int] = None) -> List[float]:
        x0, x1, y0, y1, z0, z1 = self.slice_default(Mx0, Mx1, My0, My1, Mz0, Mz1)
        if axis is None:
            axis_list = range(self.dim)
        else:
            axis_list = [axis]
        parity_list: List[float] = []
        for psi_val in self.psi_val_list:
            parity_axis_list = []
            for axis in axis_list:
                psi_under0, psi_over0 = np.split(psi_val, 2, axis=axis)

                if axis in [0, 1, 2]:
                    psi_over0_reversed = psi_over0[::-1]
                else:
                    sys.exit(f"No such axis ({axis}). Choose 0, 1 or 2 for axis x, y or z.")

                parity = self.trapez_integral(np.abs(
                    psi_under0[x0:x1, y0:y1, z0:z1] - psi_over0_reversed[x0:x1, y0:y1, z0:z1]) ** 2.0)
                parity_axis_list.append(parity)
            parity_list.append(parity_axis_list)

        return parity_list

    def check_N(self) -> List[float]:
        """
        Calculates :math:`N_i = \int \\mathcal{d}\vec{r} |\psi_{i}(r)|^2`
        Returns [N_1, N_2, N]
        """
        N_list_check: List[float] = []
        for i, (psi_val, N) in enumerate(zip(self.psi_val_list, self.N_list)):
            N_list_check.append(N * self.get_norm(func_val=psi_val))
        N_list_check.append(sum(N_list_check))

        return N_list_check


    def distmat(self, a: cp.ndarray, index: List[float]):
        mask = cp.ones(a.shape, dtype=bool)
        mask[index[0], index[1], index[2]] = False

        return distance_transform_edt(mask)

    def get_contrast_old(self,
                   axis: int = 2,
                   Mx0: Optional[int] = None, Mx1: Optional[int] = None,
                   My0: Optional[int] = None, My1: Optional[int] = None,
                   Mz0: Optional[int] = None, Mz1: Optional[int] = None) -> List[float]:
        x0, x1, y0, y1, z0, z1 = self.slice_default(Mx0, Mx1, My0, My1, Mz0, Mz1)
        prob_list = self.get_density_list(jit=numba_used)
        bec_contrast_list = []
        for N, prob in zip(self.N_list, prob_list):
            mask = cp.full(cp.shape(prob), False)
            mask[x0:x1, y0:y1, z0:z1] = True

            bec_min_edgeless = ndimage.minimum(prob, labels=mask)
            bec_max_edgeless = ndimage.maximum(prob, labels=mask)
            bec_contrast_edgeless = (bec_max_edgeless - bec_min_edgeless) / (
                    bec_max_edgeless + bec_min_edgeless)
            bec_contrast_list.append(bec_contrast_edgeless)

        return bec_contrast_list

    def get_contrast_old_smart(self,
                               axis: int = 2,
                               Mx0: Optional[int] = None, Mx1: Optional[int] = None,
                               My0: Optional[int] = None, My1: Optional[int] = None,
                               Mz0: Optional[int] = None, Mz1: Optional[int] = None) -> List[float]:
        x0, x1, y0, y1, z0, z1 = self.slice_default(Mx0, Mx1, My0, My1, Mz0, Mz1)

        prob_list = self.get_density_list(jit=numba_used)

        bec_contrast_list = []
        for N, prob in zip(self.N_list, prob_list):
            mask = cp.full(cp.shape(prob), False)
            mask[x0:x1, y0:y1, z0:z1] = True

            min_on_edge_bool = True
            slice_x0, slice_x1, slice_y0, slice_y1, slice_z0, slice_z1 = x0, x1, y0, y1, z0, z1
            while min_on_edge_bool:
                mask_slice = cp.full(cp.shape(prob), False)
                mask_slice[slice_x0:slice_x1, slice_y0:slice_y1, slice_z0:slice_z1] = True
                bec_min_edgeless_pos = ndimage.minimum_position(prob, labels=mask_slice)

                (min_on_edge_bool,
                slice_x0, slice_x1,
                slice_y0, slice_y1,
                slice_z0, slice_z1) = self.on_edge(bec_min_edgeless_pos,
                                                   slice_x0, slice_x1,
                                                   slice_y0, slice_y1,
                                                   slice_z0, slice_z1)

            # if minimum is on edge of region, it is not a local minima,
            # but depends choice of the region. Thus contrast is meaningless (set to 0).
            if min_on_edge_bool:
                bec_contrast_edgeless = 0.0
            else:
                bec_min_edgeless = ndimage.minimum(prob, labels=mask)
                bec_max_edgeless = ndimage.maximum(prob, labels=mask)
                bec_contrast_edgeless = (bec_max_edgeless - bec_min_edgeless) / (
                        bec_max_edgeless + bec_min_edgeless)

            bec_contrast_list.append(bec_contrast_edgeless)

        return bec_contrast_list

    def on_edge(self, indices,
                 Mx0: Optional[int] = None, Mx1: Optional[int] = None,
                 My0: Optional[int] = None, My1: Optional[int] = None,
                 Mz0: Optional[int] = None, Mz1: Optional[int] = None) -> List[float]:
        # test if indices are on th edge

        x0, x1, y0, y1, z0, z1 = self.slice_default(Mx0, Mx1, My0, My1, Mz0, Mz1)
        x_bounds = [x0, x1]
        y_bounds = [y0, y1]
        z_bounds = [z0, z1]

        # return True whenever it was changed (it was on edge), gives back smaller slice
        if indices[0] in x_bounds:
            changed, x0_new, x1_new = smaller_slice(x0, x1)
            if changed:
                return True, x0_new, x1_new, y0, y1, z0, z1

        if indices[1] in y_bounds:
            changed, y0_new, y1_new = smaller_slice(y0, y1)
            if changed:
                return True, x0, x1, y0_new, y1_new, z0, z1

        # x, y not changed
        if indices[2] in z_bounds:
            changed, z0_new, z1_new = smaller_slice(z0, z1)
            if changed:
                return True, x0, x1, y0, y1, z0_new, z1_new
            else:
                # x, y, z not changed, meaning: no change in slice
                return False, x0, x1, y0, y1, z0, z1


    def get_contrast(self, number_of_peaks: int, prob_min_start, prob_step: float = 0.01,
                     prob_min_edge: float = 0.015, region_threshold: int = 100,
                     ) -> List[float]:
        prob_list = self.get_density_list(jit=numba_used)
        bec_contrast_list = []
        structures = functions.binary_structures()
        for N, prob in zip(self.N_list, prob_list):
            prob_max = cp.max(prob)
            prob_min_lin_reverse = cp.arange(0.001, prob_min_start, prob_step)[::-1]
            for prob_min_ratio in prob_min_lin_reverse:
                # get biggest region
                prob_region_outer = cp.where(prob >= prob_min_ratio * prob_max, prob, 0)
                label_im_outer, nb_labels_outer = ndimage.label(prob_region_outer,
                                                                structure=structures[0])
                sizes_outer = ndimage.sum(prob_region_outer, label_im_outer,
                                          range(nb_labels_outer + 1))
                mask_outer = sizes_outer > region_threshold
                label_region_outer = mask_outer[label_im_outer]
                region_outer = functions.fill_holes(label_region_outer, *structures[1:])

                # remove edge
                prob_region_inner = cp.where(prob >= (prob_min_edge + prob_min_ratio) * prob_max,
                                             prob, 0)
                label_im_inner, nb_labels_inner = ndimage.label(prob_region_inner,
                                                                structure=structures[0])
                sizes_inner = ndimage.sum(prob_region_inner, label_im_inner,
                                          range(nb_labels_inner + 1))
                mask_inner = sizes_inner > region_threshold
                label_region_inner = mask_inner[label_im_inner]
                region_inner = functions.fill_holes(label_region_inner, *structures[1:])

                region_edgeless = cp.logical_and(region_outer, region_inner)
                region = functions.fill_holes(region_edgeless, *structures[1:])
                label_im_region, nb_labels_region = ndimage.label(region, structure=structures[0])

                if nb_labels_region == 1:
                    region_shrunk = ndimage.binary_erosion(region)
                    region_diff = cp.logical_xor(region_shrunk, region)
                    label_im_region_diff, nb_labels_region_diff = ndimage.label(
                        region_diff, structure=structures[0])

                    peak_list = self.get_peak_neighborhood(prob, prob_max * prob_min_ratio,
                                                           number_of_peaks=number_of_peaks)
                    if len(peak_list) == 1:
                        # no peak_neighborhood to calculate dist
                        prob_one = cp.where(prob > 0.8 * prob_max, prob, 0)
                        label_im_one, nb_labels_one = ndimage.label(prob_one,
                                                                    structure=structures[0])
                        bec_min = ndimage.minimum(prob_one, labels=label_im_one)
                    else:
                        peaks_max_index = [ndimage.maximum_position(peak,
                                                                    labels=cp.array(peak,
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

    def get_polarization(self, input_path: Path, filename_steps: str,
                         steps_format: str, numerator_cut_off: float, divisor_cut_off: float,
                         frame: int):
        """
        Calculates the polarization of a two component mixture: P = density2/density1.

        :param divisor_cut_off: Cutoff for density1 to prohibit division by 0.

        :param frame: Frame number of the npz to save to.

        """
        density_list = self.get_density_list()
        val2 = np.where(density_list[1] > numerator_cut_off, density_list[1], 0.0) 
        val1 = np.where(density_list[0] > divisor_cut_off, density_list[0], 0.0) 
        polarization = np.divide(val2, val1, out=np.zeros_like(val1), where=val1!=0)
        print(f"Saving Polarization into System.psi_val_list with shape {np.shape(polarization)} "
              + f"in: {input_path}")
        self.save_psi_val(input_path, filename_steps, steps_format, frame,
                          arr_list=[polarization,
                                    np.zeros(shape=np.shape(polarization))
                                    ]
                          )
        polarization_max_index = ndimage.maximum_position(polarization)
        polarization_max = polarization[polarization_max_index]

        return polarization_max

    def get_U_dd(self, density_list: List[cp.ndarray]) -> List[cp.ndarray]:
        """
        Calculates :math:`U_{dd, ij} = \\mathcal{F}^{-1}(\\mathcal{F}(S_{ij} |\psi_{j}|^{2}) V_{k})`
        with :math: `V_{k} = epsilon_{dd} g (3 (k_z / k)^2 - 1)`,
        :math: `S_{ij} = exp{-ikM}` and
        :math: `M_{ij} = stack_shift * \sigma_{x}` with :math: `\sigma_{x}` is the x-Pauli matrix

        """
        pauli_x = cp.array([[0, 1], [1, 0]])
        stack_mat = self.stack_shift * pauli_x
        stack_exponent_mat = cp.einsum("ij..., klm->ijklm", stack_mat, self.kz_mesh)
        stack_shift_op_mat = cp.exp((-1) * 1.0j * stack_exponent_mat)

        density_fft_V_k = self.V_k_val * cp.array([cp.fft.fftn(density_list[0]),
                                                   cp.fft.fftn(density_list[1])])
        shifted = cp.einsum("ijklm, jklm->ijklm", stack_shift_op_mat, density_fft_V_k)
        U_dd = cp.array([[cp.fft.ifftn(shifted[0, 0]), cp.fft.ifftn(shifted[0, 1])],
                         [cp.fft.ifftn(shifted[1, 0]), cp.fft.ifftn(shifted[1, 1])]])
        
        return U_dd

    def get_U_dd_list(self, density_list: List[cp.ndarray]) -> List[cp.ndarray]:
        U_dd_list: List[cp.ndarray] = []
        for density in density_list:
            try:
                U_dd_list.append(cp.fft.ifftn(cp.asarray(self.V_k_val) * cp.fft.fftn(density)))
            except Exception as e:
                print({e})
                try:
                    U_dd_list.append(np.fft.ifftn(self.V_k_val.get() * np.fft.fftn(density.get())))
                except Exception as e:
                    print({e})
                    try:
                        U_dd_list.append(np.fft.ifftn(self.V_k_val * np.fft.fftn(density.get())))
                    except Exception as e:
                        print({e})


        return U_dd_list

    def get_H_pot(self, terms: cp.ndarray, split_step: float = 0.5) -> cp.ndarray:
        H_pot = cp.exp(self.U * (split_step * self.dt) * terms)

        return H_pot

    def get_H_pot_exponent_terms(self,
                                 dipol_term: cp.ndarray,
                                 contact_interaction: cp.ndarray,
                                 mu_lhy: cp.ndarray) -> cp.ndarray:
        return (self.V_val
                + self.a_dd_factor * dipol_term
                + self.a_s_factor * contact_interaction
                + mu_lhy
                )

    def split_operator_pot(self, split_step: float = 0.5,
            jit: bool = True, cupy_used: bool = False) -> Tuple[List[cp.ndarray], List[cp.ndarray]]:
        if cupy_used:
            jit = False
        density_list: List[cp.ndarray] = self.get_density_list(jit=jit, cupy_used=cupy_used)
        try:
            # density_tensor_vec: cp.ndarray = cp.stack(density_list, axis=0)
            density_tensor_vec: cp.ndarray = cp.array(density_list)
        except Exception:
            density_list_np = []
            for density in density_list:
                density_list_np.append(density.get())
            density_tensor_vec: np.ndarray = np.stack(density_list_np, axis=0)

        # if stack_shift is used U_dd has higher dimensionality (get_U_dd),
        # but resulting dipol_term_vec the same (else use get_U_dd_list)
        if self.stack_shift != 0.0:
            U_dd = self.get_U_dd(density_list)
            dipol_term_vec: cp.ndarray = cp.einsum("ij..., ij...->i...",
                                                   self.a_dd_array,
                                                   U_dd)
        else:
            U_dd_list: List[cp.ndarray] = self.get_U_dd_list(density_list)
            try:
                # U_dd_tensor_vec: cp.ndarray = cp.stack(U_dd_list, axis=0)
                U_dd_tensor_vec: cp.ndarray = cp.array(U_dd_list)
            except Exception:
                U_dd_list_np = []
                for U_dd in U_dd_list:
                    U_dd_list_np.append(U_dd.get())
                U_dd_tensor_vec: np.ndarray = np.stack(U_dd_list_np, axis=0)
            dipol_term_vec: cp.ndarray = cp.einsum("...ij, j...->i...",
                                                   self.a_dd_array,
                                                   U_dd_tensor_vec)


        # update H_pot before use
        contact_interaction_vec: cp.ndarray = cp.einsum("...ij, j...->i...",
                                                        self.a_s_array,
                                                        density_tensor_vec)

        # contact_interaction_vec = self.arr_tensor_mult(self.a_s_array, density_tensor_vec)
        # dipol_term_vec = self.arr_tensor_mult(self.a_dd_array, U_dd_tensor_vec)

        mu_lhy_list: List[cp.ndarray] = self.get_mu_lhy_list(density_list)
        for i, (contact_interaction, dipol_term, mu_lhy) in enumerate(zip(
                list(contact_interaction_vec),
                list(dipol_term_vec),
                mu_lhy_list)):
            if cupy_used:
                tilt_term = (-1) ** i * self.tilt * cp.array(self.x_mesh)
            else:
                tilt_term = (-1) ** i * self.tilt * self.x_mesh
 
            if jit: 
                term: cp.ndarray = numbas.get_H_pot_exponent_terms_jit(self.V_val,
                                                                       self.a_dd_factor,
                                                                       self.a_s_factor,
                                                                       dipol_term,
                                                                       contact_interaction,
                                                                       mu_lhy
                                                                       )
                term_and_tilt = term + tilt_term
                H_pot_propagator: cp.ndarray = numbas.get_H_pot_jit(self.U, self.dt, term_and_tilt, split_step)
            else:
                term: cp.ndarray = self.get_H_pot_exponent_terms(dipol_term,
                                                                 contact_interaction,
                                                                 mu_lhy
                                                                 )
                term_and_tilt = term + tilt_term
                H_pot_propagator: cp.ndarray = self.get_H_pot(term_and_tilt, split_step)

            if cupy_used:
                self.psi_val_list[i] = (cp.asarray(H_pot_propagator) * cp.asarray(self.psi_val_list[i]))
            else:
                self.psi_val_list[i] = H_pot_propagator * self.psi_val_list[i]

    def split_operator_kin(self) -> None:
        # apply H_kin in k-space (transform back and forth)
        for i in range(0, len(self.psi_val_list)):
            psi_val: cp.ndarray = self.psi_val_list[i]
            self.psi_val_list[i] = cp.fft.fftn(psi_val)

        for i, H_kin in enumerate(self.H_kin_list):
            psi_val: cp.ndarray = self.psi_val_list[i]
            self.psi_val_list[i] = H_kin * psi_val

        for i in range(0, len(self.psi_val_list)):
            psi_val: cp.ndarray = self.psi_val_list[i]
            self.psi_val_list[i] = cp.fft.ifftn(psi_val)

    def normalize_psi_val(self) -> List[float]:
        psi_norm_list: List[float] = []
        for i, psi_val in enumerate(self.psi_val_list):
            psi_norm_list.append(self.get_norm(func_val=psi_val))
            self.psi_val_list[i] = psi_val / cp.sqrt(psi_norm_list[i])

        return psi_norm_list

    def time_step(self, numba_used, cupy_used) -> None:
        """
        Evolves System according Schrödinger Equations by using the
        split operator method with the Trotter-Suzuki approximation.

        """
        # psi_previous_list = []
        # for previous in self.psi_val_list:
        #     psi_previous_list.append(deepcopy(previous))
        # self.get_E_explicit()

        # adjust dt, to get the time accuracy when needed
        # self.dt = self.dt_func(self.t, self.dt)
        self.split_operator_pot(split_step=0.5, jit=numba_used, cupy_used=cupy_used)
        self.split_operator_kin()
        self.split_operator_pot(split_step=0.5, jit=numba_used, cupy_used=cupy_used)

        self.t = self.t + self.dt
        # self.get_mu_explicit(psi_previous_list)
        # self.get_E_explicit()

        # for self.imag_time=False, renormalization should be preserved,
        # but we play safe here (regardless of speedup)
        # if self.imag_time:
        psi_norm_list: List[float] = self.normalize_psi_val()
        for i, psi_norm in enumerate(psi_norm_list):
            self.mu_arr[i] = -cp.log(psi_norm) / (2.0 * self.dt)
