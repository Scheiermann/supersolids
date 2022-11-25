#!/usr/bin/env python

# author: Daniel Scheiermann
# email: daniel.scheiermann@itp.uni-hannover.de
# license: MIT
# Please feel free to use and modify this, but keep the above information.

"""
Numerical solver for non-linear time-dependent Schrodinger equation (eGPE) for dipolar mixtures.


"""
import os
import numpy as np
import dask.array as da
from dask import compute, delayed
from pathlib import Path
from typing import Optional, Callable, List, Tuple

from supersolids.helper import functions, get_version

__GPU_OFF_ENV__ = bool(os.environ.get("SUPERSOLIDS_GPU_OFF", False))
cp, cupy_used, cuda_used, numba_used = get_version.check_cp_nb(np, gpu_off=__GPU_OFF_ENV__)

from supersolids.SchroedingerMixture import SchroedingerMixture
from supersolids.helper.Box import Box
from supersolids.helper.Resolution import Resolution


class SchroedingerMixtureDask(SchroedingerMixture):
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
        super().__init__(MyBox, Res, max_timesteps, dt, N_list, m_list, a_s_array, a_dd_array,
                         t, a_s_factor, a_dd_factor, nA_max, dt_func, w_x, w_y, w_z, imag_time,
                         tilt, stack_shift, mu_arr, E, V, V_interaction,
                         psi_0_list, psi_0_noise_list, psi_sol_list, mu_sol_list, input_path,
                         )

        # convert to dask arrays
        self.chunks = (Res.x, Res.y, Res.z)
        # chunks = "auto"

        self.psi_val_list = list_np_to_dask(self.psi_val_list, chunks=self.chunks)
        self.H_kin_list = list_np_to_dask(self.H_kin_list, chunks=self.chunks)
        self.V_k_val = da.from_array(self.V_k_val, chunks=self.chunks)
        self.V_val = da.from_array(self.V_val, chunks=self.chunks)
        self.k_squared = da.from_array(self.k_squared, chunks=self.chunks)

        self.a_s_array = da.from_array(self.a_s_array)
        self.a_dd_array = da.from_array(self.a_dd_array)

    def get_density_arr(self) -> List[cp.ndarray]:
        psi_density_arr = N * cp.abs(self.psi_val_arr) ** p

        return psi_density_arr

    def get_density_list(self) -> List[cp.ndarray]:
        density_list: List[np.ndarray] = []
        for psi_val, N in zip(self.psi_val_list, self.N_list):
            density_N = N * self.get_density(psi_val)
            density_list.append(density_N)

        # run the lazily values, reomve 1dim tuple
        output = compute(density_list)[0]

        return output

    @delayed
    def get_density(self, func_val: Optional[cp.ndarray] = None, p: float = 2.0,
                    jit: bool = True) -> cp.ndarray:
        """
        Calculates :math:`|\psi|^p` for 1D, 2D or 3D (depending on self.dim).

        :param p: Exponent of :math:`|\psi|`. Use p=2.0 for density.

        :param func_val: Array of function values to get p-norm for.

        :return: :math:`|\psi|^p`

        """
        if self.dim <= 3:
            if func_val is None:
                psi_density = cp.abs(self.psi_val) ** p
            else:
                psi_density = cp.abs(func_val) ** p
        else:
            sys.exit("Spatial dimension over 3. This is not implemented.")

        return psi_density


    def get_U_dd_list(self, density_list):
        U_dd_list = []
        for i, density in enumerate(density_list):
            if self.stack_shift == 0.0:
                U_dd_list.append(da.fft.ifftn(self.V_k_val * da.fft.fftn(density)))
            else:
                stack_shift_op = cp.exp((-1) ** i * 1.0j * self.kz_mesh * self.stack_shift)
                U_dd_list.append(da.fft.ifftn(self.V_k_val * da.fft.fftn(density) * stack_shift_op))

        return compute(U_dd_list)[0]

    def split_operator_kin(self) -> None:
        # apply H_kin in k-space (transform back and forth)
        for i in range(0, len(self.psi_val_list)):
            self.psi_val_list[i] = da.fft.fftn(self.psi_val_list[i])

        for i, H_kin in enumerate(self.H_kin_list):
            self.psi_val_list[i] = H_kin * self.psi_val_list[i]

        for i in range(0, len(self.psi_val_list)):
            self.psi_val_list[i] = da.fft.ifftn(self.psi_val_list[i]).compute(scheduler='processes')
            self.psi_val_list[i] = da.from_array(self.psi_val_list[i], chunks=self.chunks)

    def split_operator_pot(self, split_step: float = 0.5, jit: bool = True,
        cupy_used: bool = False):
        density_list = self.get_density_list()
        density_list = list_np_to_dask(density_list, chunks=self.chunks)
        density_tensor_vec = cp.stack(density_list, axis=0)

        U_dd_list = self.get_U_dd_list(density_list)
        U_dd_tensor_vec = cp.stack(U_dd_list, axis=0)

        # update H_pot before use
        contact_interaction_vec = da.einsum("...ij, j...->i...",
                                            self.a_s_array,
                                            density_tensor_vec).compute(scheduler='processes')
        dipol_term_vec = da.einsum("...ij, j...->i...",
                                   self.a_dd_array,
                                   U_dd_tensor_vec).compute(scheduler='processes')

        mu_lhy_list = self.get_mu_lhy_list(density_list)
        for i, (contact_interaction, dipol_term, mu_lhy) in enumerate(zip(
                list(contact_interaction_vec),
                list(dipol_term_vec),
                mu_lhy_list)):

            term: cp.ndarray = self.get_H_pot_exponent_terms(dipol_term,
                                                             contact_interaction,
                                                             mu_lhy
                                                             )
            if self.tilt == 0.0:
                H_pot_propagator: cp.ndarray = self.get_H_pot(term, split_step)
            else:
                tilt_term = (-1) ** i * self.tilt * self.x_mesh
                term_and_tilt = term + tilt_term
                H_pot_propagator: cp.ndarray = self.get_H_pot(term_and_tilt, split_step)

            self.psi_val_list[i] = (H_pot_propagator * self.psi_val_list[i]).compute(scheduler='processes')
            self.psi_val_list[i] = da.from_array(self.psi_val_list[i], chunks=self.chunks)


    def get_A_density_total(self, density_list: List[cp.ndarray]) -> Tuple[cp.ndarray, cp.ndarray]:
        density_total = da.Array.copy(density_list[0])
        for density in density_list[1:]:
            density_total += density

        # A = n1/(n1+n2)
        density_A = density_list[0] / density_total
        # remove nan, because of possible 0/0
        density_A = cp.where(density_A != cp.nan, density_A, 0.0)

        return density_A, density_total

    def normalize_psi_val(self) -> List[float]:
        psi_norm_list: List[float] = []
        for i, psi_val in enumerate(self.psi_val_list):
            psi_norm = self.get_norm(func_val=psi_val).compute()
            self.psi_val_list[i] = psi_val / cp.sqrt(psi_norm)
            psi_norm_list.append(psi_norm)

        return psi_norm_list

    def get_E(self) -> None:
        # update for energy calculation
        density_list = self.get_density_list()
        density_list = list_np_to_dask(density_list, chunks=self.chunks)
        U_dd_list = self.get_U_dd_list(density_list)
        mu_lhy_list: List[np.ndarray] = self.get_mu_lhy_list(density_list)

        # use normalized inputs for energy
        self.E = self.energy(density_list, U_dd_list, mu_lhy_list)

def list_np_to_dask(list_np, chunks):
    for i in range(0, len(list_np)):
        list_np[i] = da.from_array(list_np[i], chunks=chunks)

    return list_np

def list_dask_to_np(list_dask):
    for i in range(0, len(list_dask)):
        list_dask[i] = np.asarray(list_dask[i])

    return list_dask


