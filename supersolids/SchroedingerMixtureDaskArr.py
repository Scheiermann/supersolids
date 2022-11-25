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
from dask.graph_manipulation import bind
from pathlib import Path
from typing import Optional, Callable, List, Tuple

from mpi4py import MPI
from mpi4py_fft import PFFT, newDistArray

from supersolids.helper import functions, get_version
from supersolids.helper.run_time import run_time

__GPU_OFF_ENV__ = bool(os.environ.get("SUPERSOLIDS_GPU_OFF", False))
cp, cupy_used, cuda_used, numba_used = get_version.check_cp_nb(np, gpu_off=__GPU_OFF_ENV__)

from supersolids.SchroedingerMixture import SchroedingerMixture
from supersolids.helper.Box import Box
from supersolids.helper.Resolution import Resolution


class SchroedingerMixtureDaskArr(SchroedingerMixture):
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
        # self.chunks = (int(Res.x / 2), int(Res.y / 2), Res.z)
        self.chunks = (int(Res.x / 2), Res.y, Res.z)
        # self.chunks = (Res.x, Res.y, Res.z)
        self.chunks_fft = (Res.x, Res.y, Res.z)
        # self.chunks = "auto"

        self.psi_val_list = list_np_to_dask(self.psi_val_list, chunks=self.chunks)
        self.H_kin_list = list_np_to_dask(self.H_kin_list, chunks=self.chunks)

        self.H_kin_arr = cp.stack(self.H_kin_list, axis=0)
        self.psi_val_arr = cp.stack(self.psi_val_list, axis=0)

        self.V_k_val = da.from_array(self.V_k_val, chunks=self.chunks)
        self.V_val = da.from_array(self.V_val, chunks=self.chunks)
        self.k_squared = da.from_array(self.k_squared, chunks=self.chunks)

        self.N_arr = da.from_array(np.array(self.N_list))
        self.a_s_array = da.from_array(self.a_s_array)
        self.a_dd_array = da.from_array(self.a_dd_array)

    def get_density_arr(self, p: float = 2.0) -> cp.ndarray:
        psi_density_arr = da.einsum("...i, i...->i...",self.N_arr, cp.abs(self.psi_val_arr) ** p)

        return psi_density_arr

    def get_U_dd_arr_mpi(self, density_arr):
        axes = range(1, self.dim + 1)
        components = len(self.N_list)
        fft_plan = PFFT(MPI.COMM_WORLD, np.array([components, *self.Res.to_array()]),
                        axes=axes, dtype=np.complex, grid=(-1,))
        density_arr = da.rechunk(density_arr, chunks=tuple((components, *self.chunks_fft)))
        if self.stack_shift == 0.0:
            input = newDistArray(fft_plan, False)
            input[:] = density_arr
            fft_result = fft_plan.forward(input, normalize=True)
            ffted = self.V_k_val * fft_result
            input[:] = ffted
            U_dd = fft_plan.backward(input, normalize=True)
        else:
            stack_shift_op = cp.exp((-1) ** i * 1.0j * self.kz_mesh * self.stack_shift)
            input = newDistArray(fft_plan, False)
            input[:] = density_arr
            fft_result = fft_plan.forward(input, normalize=True)
            ffted = self.V_k_val * fft_result * stack_shift_op
            input[:] = ffted
            U_dd = fft_plan.backward(input, normalize=True)

        return U_dd

    def get_U_dd_arr(self, density_arr):
        axes = range(1, self.dim + 1)
        components = len(self.N_list)
        density_arr = da.rechunk(density_arr, chunks=tuple((components, *self.chunks_fft)))
        if self.stack_shift == 0.0:
            ffted = self.V_k_val * da.fft.fftn(density_arr, axes=axes)
            ffted = da.rechunk(ffted, chunks=tuple((components, *self.chunks_fft)))
            U_dd = da.fft.ifftn(ffted , axes=axes)
        else:
            stack_shift_op = cp.exp((-1) ** i * 1.0j * self.kz_mesh * self.stack_shift)
            U_dd = da.fft.ifftn(self.V_k_val * da.fft.fftn(density_arr, axes=axes) * stack_shift_op,
                                axes=axes)

        return U_dd

    def get_mu_lhy_arr(self, density_arr: cp.ndarray) -> cp.ndarray:
        density_A, density_total = self.get_A_density_total_arr(density_arr)

        mu_lhy_list: List[cp.ndarray] = []
        for mu_lhy_interpolation in self.mu_lhy_interpolation_list:
            if cupy_used:
                mu_lhy = cp.asarray(mu_lhy_interpolation(density_A.get()) * density_total.get() ** 1.5)
                # mu_lhy = cp.asarray(delayed(mu_lhy_interpolation)(density_A.get()) * density_total.get() ** 1.5)
            else:
                mu_lhy = mu_lhy_interpolation(density_A) * density_total ** 1.5
                # mu_lhy = delayed(mu_lhy_interpolation)(density_A) * density_total ** 1.5

            mu_lhy_list.append(mu_lhy)
        # mu_lhy_arr = cp.array(compute(mu_lhy_list)[0])
        mu_lhy_arr = cp.array(mu_lhy_list)

        return mu_lhy_arr

    def get_A_density_total_arr(self, density_arr: cp.ndarray) -> Tuple[cp.ndarray, cp.ndarray]:
        density_total = da.sum(density_arr, axis=0)

        # A = n1/(n1+n2)
        density_A = (density_arr[0] / density_total)
        # remove nan, because of possible 0/0
        density_A = cp.where(density_A != cp.nan, density_A, 0.0)

        density_A = density_A.compute()
        density_total = density_total.compute()

        return density_A, density_total

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

    def split_operator_kin(self) -> None:
        axes = range(1, self.dim + 1)
        components = len(self.N_list)

        # apply H_kin in k-space (transform back and forth)

        ffted = (self.H_kin_arr * da.fft.fftn(self.psi_val_arr, axes=axes))
        ffted = da.rechunk(ffted, chunks=tuple((components, *self.chunks_fft)))
        self.psi_val_arr = da.fft.ifftn(ffted, axes=axes).compute()
        self.psi_val_arr = da.from_array(self.psi_val_arr)

    def split_operator_pot(self, split_step: float = 0.5, jit: bool = True,
                           cupy_used: bool = False):
        density_arr = self.get_density_arr()
        U_dd_arr = self.get_U_dd_arr(density_arr)

        # update H_pot before use
        contact_interaction_arr = da.einsum("...ij, j...->i...",
                                            self.a_s_array,
                                            density_arr)
        dipol_term_arr = da.einsum("...ij, j...->i...",
                                   self.a_dd_array,
                                   U_dd_arr)

        # with run_time(name="mu_lhy_arr"):
        mu_lhy_arr = self.get_mu_lhy_arr(density_arr)

        term: cp.ndarray = self.get_H_pot_exponent_terms(dipol_term_arr,
                                                         contact_interaction_arr,
                                                         mu_lhy_arr
                                                         )
        if self.tilt == 0.0:
            H_pot_propagator: cp.ndarray = self.get_H_pot(term, split_step)
        else:
            tilt_term = da.einsum('...i, ...->i...', cp.array([1.0, -1.0]), 10.0 * self.x_mesh)
            term_and_tilt = term + tilt_term
            H_pot_propagator: cp.ndarray = self.get_H_pot(term_and_tilt, split_step)

        self.psi_val_arr = (H_pot_propagator * self.psi_val_arr).compute()
        self.psi_val_arr = da.from_array(self.psi_val_arr)


    def normalize_psi_val_arr(self) -> List[float]:
        # axis=0 is stacking the components
        axis = tuple(range(1, self.dim + 1))
        density_arr = self.get_density_arr()
        psi_norm_arr: float = self.sum_dV(density_arr, axis=axis)
        self.psi_val_arr = da.einsum('...i,...->...', 1.0 / cp.sqrt(psi_norm_arr), self.psi_val_arr)

        return psi_norm_arr

    def get_E(self) -> None:
        # update for energy calculation
        density_arr = self.get_density_arr()
        U_dd_arr = self.get_U_dd_arr(density_arr)
        # with run_time(name="get_E mu_lhy_arr"):
        mu_lhy_arr: cp.ndarray = self.get_mu_lhy_arr(density_arr)

        # use normalized inputs for energy
        self.E = self.energy(density_arr, U_dd_arr, mu_lhy_arr)

    def energy(self, density_arr: cp.ndarray, U_dd_arr: cp.ndarray,
               mu_lhy_arr: cp.ndarray) -> float:
        """
        Input psi_1, psi_2 need to be normalized.
        density1 and density2 need to be build by the normalized psi_1, psi_2.

        """
        dV = self.volume_element(fourier_space=False)
        mu_lhy_part: float = cp.sum(mu_lhy_arr * density_arr) * dV
        p_int = self.energy_density_interaction(density_arr, U_dd_arr)

        E_lhy = self.sum_dV(self.get_energy_lhy(density_arr), dV=dV)

        N_array = cp.array(self.N_list)
        chem_pot_N = cp.dot(self.mu_arr, N_array)

        E = (chem_pot_N + p_int + E_lhy - mu_lhy_part) / cp.sum(N_array)

        return E

    def get_energy_lhy(self, density_arr: cp.ndarray) -> cp.ndarray:
        density_A, density_total = self.get_A_density_total_arr(density_arr)
        if cupy_used:
            energy_lhy = self.energy_helper_function(density_A.get()) * density_total.get() ** 2.5
        else:
            energy_lhy = self.energy_helper_function(density_A) * density_total ** 2.5

        return energy_lhy

    def energy_density_interaction(self, density_arr: cp.ndarray, U_dd_arr: cp.ndarray) -> float:
        dV = self.volume_element(fourier_space=False)
        density_U_dd = cp.einsum("i...,j...->ij...", U_dd_arr, density_arr)
        density_density = cp.einsum("i...,j...->ij...", density_arr, density_arr)

        # a = self.array_to_tensor_grid(self.U_dd_factor_array) * density_U_dd
        a = np.einsum("...ij, kl...->kl...", self.a_dd_array, density_U_dd)
        b = np.einsum("...ij, kl...->kl...", self.a_s_array, density_density)
        en_mf = 0.5 * np.sum(a + b).real

        return self.sum_dV(en_mf, dV=dV)
    def time_step(self, numba_used, cupy_used) -> None:
        """
        Evolves System according Schrödinger Equations by using the
        split operator method with the Trotter-Suzuki approximation.

        """

        # with run_time(name="pot"):
        self.split_operator_pot(split_step=0.5, jit=numba_used, cupy_used=cupy_used)
        # with run_time(name="kin"):
        self.split_operator_kin()
        # with run_time(name="pot"):
        self.split_operator_pot(split_step=0.5, jit=numba_used, cupy_used=cupy_used)

        self.t = self.t + self.dt

        if self.imag_time:
            psi_norm_arr: List[float] = self.normalize_psi_val_arr()
            self.mu_arr = (-cp.log(psi_norm_arr) / (2.0 * self.dt)).compute()

def list_np_to_dask(list_np, chunks):
    for i in range(0, len(list_np)):
        list_np[i] = da.from_array(list_np[i], chunks=chunks)

    return list_np

def check_equal(A, B, dim=2):
    for i in range(dim):
        if dim == 1:
            A_buffer = A
            B_buffer = B
        else:
            A_buffer = A[i]
            B_buffer = B[i]

        try:
            C = A_buffer.compute()
            D = B_buffer.compute()
        except Exception:
            C = A_buffer
            D = B_buffer
        print(f"{C == D}")
        print(f"{np.sum(C == D)}")
        print(f"{np.prod((C == D).shape)}")

