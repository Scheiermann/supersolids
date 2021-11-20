#!/usr/bin/env python

# author: Daniel Scheiermann
# email: daniel.scheiermann@itp.uni-hannover.de
# license: MIT
# Please feel free to use and modify this, but keep the above information.

"""
Numerical solver for non-linear time-dependent Schrodinger equation.

"""

import functools
import sys
from typing import Callable, Union, Optional, List
from pathlib import Path

import dill
import numpy as np
import scipy.signal

from supersolids.SchroedingerSummary import SchroedingerSummary
from supersolids.helper import constants, functions, get_path
from supersolids.helper.Resolution import Resolution
from supersolids.helper.Box import Box
from supersolids.helper.get_path import get_step_index_from_list
from supersolids.helper.load_script import reload_files
from supersolids.helper.save_script import save_script


class Schroedinger:
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
    """

    def __init__(self,
                 N: int,
                 MyBox: Box,
                 Res: Resolution,
                 max_timesteps: int,
                 dt: float,
                 dt_func: Optional[Callable] = None,
                 g: float = 0.0,
                 g_qf: float = 0.0,
                 w_x: float = 2.0 * np.pi * 33.0,
                 w_y: float = 2.0 * np.pi * 80.0,
                 w_z: float = 2.0 * np.pi * 167.0,
                 a_s: float = 85.0 * constants.a_0,
                 e_dd: float = 1.0,
                 imag_time: bool = True,
                 mu_arr: np.ndarray = np.array([1.1]),
                 E: float = 1.0,
                 psi_0: Callable = functions.psi_gauss_3d,
                 psi_0_noise: np.ndarray = functions.noise_mesh,
                 V: Optional[Callable] = functions.v_harmonic_3d,
                 V_interaction: Optional[Callable] = None,
                 psi_sol: Optional[Callable] = functions.thomas_fermi_3d,
                 mu_sol: Optional[Callable] = functions.mu_3d,
                 ) -> None:
        """
        Schrödinger equations for the specified system.

        :param MyBox: Keyword x0 is minimum in x direction and
            x1 is maximum. Same for y and z. For 1D just use x0, x1.
            For 2D x0, x1, y0, y1.
            For 3D x0, x1, y0, y1, z0, z1.
            Dimension of simulation is constructed from this dictionary.

        :param Res: Res
            Number of grid points in x, y, z direction.
            Needs to have half size of box dictionary.
            Keywords x, y, z are used.

        :param max_timesteps: Maximum timesteps  with length dt for the animation.

        """
        self.name: str = "SchroedingerSummary_"
        self.t: float = 0.0

        self.Res, self.Box = functions.check_ResBox(Res, MyBox)
        self.dim: int = self.Box.dim

        self.N: int = N
        self.w_x: float = w_x
        self.w_y: float = w_y
        self.w_z: float = w_z
        self.a_s: float = a_s

        self.max_timesteps: int = max_timesteps

        self.dt: float = dt
        self.dt_func: Optional[Callable] = dt_func
        self.g: float = g
        self.g_qf: float = g_qf
        self.e_dd: float = e_dd
        self.imag_time: bool = imag_time

        self.mu_arr: np.ndarray = mu_arr
        self.E: float = E

        self.psi_0: Callable = psi_0
        self.psi_0_noise: np.ndarray = psi_0_noise

        if V is None:
            self.V = None
        else:
            self.V: Callable = V

        if V_interaction is None:
            self.V_interaction = None
        else:
            self.V_interaction: Callable = V_interaction

        if psi_sol is None:
            self.psi_sol = None
        else:
            if callable(psi_sol):
                self.psi_sol: Callable = functools.partial(psi_sol, g=self.g)
            else:
                self.psi_sol = psi_sol

        if mu_sol is None:
            self.mu_sol = None
        else:
            if callable(mu_sol):
                self.mu_sol: Callable = mu_sol(self.g)
            else:
                self.mu_sol = mu_sol

        self.x, self.dx, self.kx, self.dkx = functions.get_grid_helper(self.Res, self.Box, 0)
        if self.dim >= 2:
            self.y, self.dy, self.ky, self.dky = functions.get_grid_helper(self.Res, self.Box, 1)
        if self.dim >= 3:
            self.z, self.dz, self.kz, self.dkz = functions.get_grid_helper(self.Res, self.Box, 2)

        if imag_time:
            # Convention: $e^{-iH} = e^{UH}$
            self.U: complex = -1.0
        else:
            self.U = -1.0j

        if self.dim == 1:
            if psi_0_noise is None:
                self.psi_val: np.ndarray = self.psi_0(self.x)
            else:
                self.psi_val = psi_0_noise * self.psi_0(self.x)

            if V is None:
                self.V_val: Union[float, np.ndarray] = 0.0
            else:
                self.V_val = self.V(self.x)

            if self.psi_sol is None:
                self.psi_sol_val = None
            else:
                self.psi_sol_val: np.ndarray = self.psi_sol(self.x)

            self.k_squared: np.ndarray = self.kx ** 2.0

            if V_interaction is None:
                # For no interaction the identity is needed with respect to 2D
                # * 2D (array with 1.0 everywhere)
                self.V_k_val: np.ndarray = np.full(self.psi_val.shape, 1.0)
            else:
                if callable(V_interaction):
                    self.V_k_val = V_interaction(self.kx, g=self.g)
                else:
                    self.V_k_val = V_interaction

        elif self.dim == 2:
            self.x_mesh, self.y_mesh, self.pos = functions.get_meshgrid(self.x, self.y)

            if psi_0_noise is None:
                self.psi_val = self.psi_0(self.pos)
            else:
                self.psi_val = psi_0_noise * self.psi_0(self.pos)

            if V is None:
                self.V_val = 0.0
            else:
                self.V_val = self.V(self.pos)

            if self.psi_sol is None:
                self.psi_sol_val = None
            else:
                self.psi_sol_val = self.psi_sol(self.pos)

            kx_mesh, ky_mesh, _ = functions.get_meshgrid(self.kx, self.ky)
            self.k_squared = kx_mesh ** 2.0 + ky_mesh ** 2.0

            if V_interaction is None:
                # For no interaction the identity is needed with respect to 2D
                # * 2D (array with 1.0 everywhere)
                self.V_k_val = np.full(self.psi_val.shape, 1.0)
            else:
                self.V_k_val = V_interaction(kx_mesh, ky_mesh, g=self.g)

        elif self.dim == 3:
            self.x_mesh, self.y_mesh, self.z_mesh = functions.get_grid(self.Res, self.Box)

            if psi_0_noise is None:
                self.psi_val = self.psi_0(self.x_mesh, self.y_mesh, self.z_mesh)
            else:
                self.psi_val = psi_0_noise * self.psi_0(self.x_mesh, self.y_mesh, self.z_mesh)

            if self.psi_sol is None:
                self.psi_sol_val = None
            else:
                if callable(self.psi_sol):
                    self.psi_sol_val = self.psi_sol(self.x_mesh, self.y_mesh, self.z_mesh)
                    if self.psi_sol_val is not None:
                        print(f"Norm for psi_sol (trapez integral): "
                              f"{self.trapez_integral(np.abs(self.psi_sol_val) ** 2.0)}")

            kx_mesh, ky_mesh, kz_mesh = np.meshgrid(self.kx, self.ky, self.kz, indexing="ij")
            self.k_squared = kx_mesh ** 2.0 + ky_mesh ** 2.0 + kz_mesh ** 2.0

            if V is None:
                self.V_val = 0.0
            else:
                self.V_val = self.V(self.x_mesh, self.y_mesh, self.z_mesh)

            if V_interaction is None:
                # For no interaction the identity is needed with respect to 2D
                # * 2D (array with 1.0 everywhere)
                self.V_k_val = np.full(self.psi_val.shape, 1.0)
            else:
                if callable(V_interaction):
                    self.V_k_val = V_interaction(kx_mesh, ky_mesh, kz_mesh, self.z_mesh)

        # here a number (U) is multiplied elementwise with an (1D, 2D or 3D) array (k_squared)
        self.H_kin: np.ndarray = np.exp(self.U * (0.5 * self.k_squared) * self.dt)

    def get_density(self, func=None, p: float = 2.0) -> np.ndarray:
        """
        Calculates :math:`|\psi|^p` for 1D, 2D or 3D (depending on self.dim).

        :param p: Exponent of :math:`|\psi|`. Use p=2.0 for density.

        :return: :math:`|\psi|^p`
        """
        if self.dim <= 3:
            if func is None:
                psi_density: np.ndarray = np.abs(self.psi_val) ** p
            else:
                psi_density: np.ndarray = np.abs(func) ** p

        else:
            sys.exit("Spatial dimension over 3. This is not implemented.")

        return psi_density

    def volume_element(self, fourier_space: bool = False):
        if self.dim == 1:
            if fourier_space:
                dV: float = self.dkx
            else:
                dV: float = self.dx
        elif self.dim == 2:
            if fourier_space:
                dV: float = self.dkx * self.dky
            else:
                dV: float = self.dx * self.dy
        elif self.dim == 3:
            if fourier_space:
                dV: float = self.dkx * self.dky * self.dkz
            else:
                dV: float = self.dx * self.dy * self.dz
        else:
            sys.exit("Spatial dimension over 3. This is not implemented.")

        return dV

    def sum_dV(self, func, fourier_space: bool = False, dV: float = None):
        if dV is None:
            dV = self.volume_element(fourier_space=fourier_space)

        psi_norm: float = np.sum(func) * dV

        return psi_norm

    def get_norm(self, func=None, p: float = 2.0, fourier_space: bool = False) -> float:
        """
        Calculates :math:`\int |\psi|^p \\mathrm{dV}` for 1D, 2D or 3D
        (depending on self.dim). For p=2 it is the 2-norm.

        :param func: If func is not provided self.get_density(p=p) is used.

        :param p: Exponent of |\psi|. Use p=2.0 for density.

        :param fourier_space: Flag to use fourier volume element as dV,
        so dV = d^3 k.

        :return: \int |\psi|^p dV

        """
        if func is None:
            func = self.get_density(p=p)
        else:
            func = np.abs(func) ** p

        if fourier_space:
            func_norm = ((np.sqrt(2.0 * np.pi) ** float(self.dim))
                         * (1 / self.volume_element(fourier_space=fourier_space))
                         * (1 / self.psi_val.size))
            func = func * func_norm

        psi_norm: float = self.sum_dV(func, fourier_space=fourier_space)

        return psi_norm

    def trapez_integral(self, func_val: Callable) -> float:
        """
        Calculates the integral over func_val. If :math:`func_val = |\psi|^p`, then
        :math:`\int |\psi|^p \\mathrm{dV}` for 1D, 2D or 3D
        (depending on self.dim) by using the trapez rule.

        For 1D: :math:`h (f(a) + f(a+h)) / 2`

        For 2D: :math:`h (f(a, b) + f(a+h, b) + f(a, b+h) + f(a+h, b+h)) / 2`

        For 3D there are 8 entries in the same manner
        :math:`(a, b, c) ... (a+h, b+h, c+h)`

        :param func_val: Grid sampled values of the function to integrate.

        :return: :math:`\int |\psi|^p \\mathrm{dV}` according to trapez rule
        """

        # TODO: Implement fourier_space (remember fftfreq ordering of func_val)
        dV = self.volume_element()
        if self.dim == 1:
            return dV * np.sum(func_val[0:-1] + func_val[1:]) / 2.0

        elif self.dim == 2:
            return dV * np.sum(func_val[0:-1, 0:-1]
                               + func_val[0:-1, 1:]
                               + func_val[1:, 0:-1]
                               + func_val[1:, 1:]
                               ) / 4.0

        elif self.dim == 3:
            return dV * np.sum(func_val[0:-1, 0:-1, 0:-1]
                               + func_val[0:-1, 0:-1, 1:]
                               + func_val[0:-1, 1:, 0:-1]
                               + func_val[0:-1, 1:, 1:]
                               + func_val[1:, 0:-1, 0:-1]
                               + func_val[1:, 0:-1, 1:]
                               + func_val[1:, 1:, 0:-1]
                               + func_val[1:, 1:, 1:]
                               ) / 8.0

        else:
            sys.exit(f"Trapez integral not implemented for dimension {self.dim}, "
                     "choose dimension smaller than 4.")

    def get_r2(self):
        if self.dim == 1:
            r2 = self.x_mesh ** 2.0
        elif self.dim == 2:
            r2 = self.x_mesh ** 2.0 + self.y_mesh ** 2.0
        elif self.dim == 3:
            r2 = self.x_mesh ** 2.0 + self.y_mesh ** 2.0 + self.z_mesh ** 2.0
        else:
            sys.exit(f"Spatial dimension {self.dim} is over 3. This is not implemented.")

        return r2

    def get_mesh_list(self, x0=None, x1=None, y0=None, y1=None, z0=None, z1=None):
        if self.dim == 1:
            r = self.x_mesh[x0:x1]
        elif self.dim == 2:
            r = [self.x_mesh[x0:x1, y0:y1], self.y_mesh[x0:x1, y0:y1]]
        elif self.dim == 3:
            r = [self.x_mesh[x0:x1, y0:y1, z0:z1],
                 self.y_mesh[x0:x1, y0:y1, z0:z1],
                 self.z_mesh[x0:x1, y0:y1, z0:z1]]
        else:
            sys.exit("Spatial dimension over 3. This is not implemented.")

        return r

    def get_peaks_along(self, axis=0, height=0.05):
        prob = np.abs(self.psi_val) ** 2.0
        res_x_middle = int(self.Res.x / 2)
        res_y_middle = int(self.Res.y / 2)
        res_z_middle = int(self.Res.z / 2)
        if axis == 0:
            peaks_indices, properties = scipy.signal.find_peaks(prob[:, res_y_middle, res_z_middle],
                                                                height=height)
        elif axis == 1:
            peaks_indices, properties = scipy.signal.find_peaks(prob[res_x_middle, :, res_z_middle],
                                                                height=height)
        elif axis == 2:
            peaks_indices, properties = scipy.signal.find_peaks(prob[res_x_middle, res_y_middle, :],
                                                                height=height)
        else:
            sys.exit(f"No such axis (){axis}. Choose 0, 1 or 2 for axis x, y or z.")

        # get the highest peaks in a sorted fashion (the n biggest, where n is number_of_peaks)
        peaks_height = properties['peak_heights']

        return peaks_indices, peaks_height

    def get_peak_positions_along(self, axis=0, height=0.05, number_of_peaks=4):
        peaks_indices, _ = self.get_peaks_along(height=height, axis=axis)
        if axis == 0:
            positions = self.Box.lengths()[axis] * (peaks_indices / self.Res.x) + self.Box.x0
        elif axis == 1:
            positions = self.Box.lengths()[axis] * (peaks_indices / self.Res.y) + self.Box.y0
        elif axis == 2:
            positions = self.Box.lengths()[axis] * (peaks_indices / self.Res.z) + self.Box.z0
        else:
            sys.exit(f"No such axis. Choose 0, 1 or 2 for axis x, y or z.")

        return positions

    def get_peak_distances_along(self, axis=0, height=0.05):
        """
        Calculates the distances between the peaks in terms of box units.

        """
        peaks_indices, _ = self.get_peaks_along(axis=axis, height=height)
        distances_indices = np.diff(peaks_indices)
        if axis == 0:
            distances = self.Box.lengths()[axis] * (distances_indices / self.Res.x)
        elif axis == 1:
            distances = self.Box.lengths()[axis] * (distances_indices / self.Res.y)
        elif axis == 2:
            distances = self.Box.lengths()[axis] * (distances_indices / self.Res.z)
        else:
            sys.exit(f"No such axis ({axis}). Choose 0, 1 or 2 for axis x, y or z.")

        return distances

    def get_peak_neighborhood_along(self, axis=0, height=0.05, number_of_peaks=4, fraction=0.1,
                                    peak_distances_cutoff=0.5):
        """
        Calculates the neighborhood of the peaks,
        which has at least the given fraction of the maximum probability :math:`|\psi|^2`.

        """
        peaks_indices, peaks_height = self.get_peaks_along(axis=axis, height=height)
        peaks_sorted_indices, peaks_sorted_height = functions.peaks_sort_along(peaks_indices,
                                                                               peaks_height,
                                                                               number_of_peaks,
                                                                               axis,
                                                                               )

        distances_indices = np.diff(np.sort(peaks_sorted_indices.T))
        # extend one element at beginning and end, according to first/last element
        distances_indices = [np.pad(distances_indices[i], (1, 1), 'edge')
                             for i in range(0, len(distances_indices))]

        prob_min = fraction * np.max(peaks_sorted_height)
        prob = np.abs(self.psi_val) ** 2.0
        bool_grid = (prob_min <= prob)
        bool_grid_list = []
        for i, peak_index in enumerate(peaks_sorted_indices):
            # peak_radius = peak_distances_cutoff * np.abs(distances_indices)[i]
            peak_radius = peak_distances_cutoff * np.abs(np.array(distances_indices).T[i, :])
            if axis == 0:
                bound_left = int(max(peak_index - peak_radius, 0))
                bound_right = int(min(peak_index + peak_radius, self.Res.x))
                bool_grid_sliced = bool_grid[bound_left:bound_right, 0:self.Res.y, 0:self.Res.z]
                pad_right = self.Res.x - bound_right
                bool_grid_padded = np.pad(bool_grid_sliced, ((bound_left, pad_right),
                                                             (0, 0),
                                                             (0, 0)), 'constant')
            elif axis == 1:
                bound_left = int(max(peak_index - peak_radius, 0))
                bound_right = int(min(peak_index + peak_radius, self.Res.y))
                bool_grid_sliced = bool_grid[0:self.Res.x, bound_left:bound_right, 0:self.Res.z]
                pad_right = self.Res.y - bound_right
                bool_grid_padded = np.pad(bool_grid_sliced, ((0, 0),
                                                             (bound_left, pad_right),
                                                             (0, 0)), 'constant')
            elif axis == 2:
                bound_left = int(max(peak_index - peak_radius, 0))
                bound_right = int(min(peak_index + peak_radius, self.Res.z))
                bool_grid_sliced = bool_grid[0:self.Res.x, 0:self.Res.y, bound_left:bound_right]
                pad_right = self.Res.z - bound_right
                bool_grid_padded = np.pad(bool_grid_sliced, ((0, 0),
                                                             (0, 0),
                                                             (bound_left, pad_right)), 'constant')
            else:
                sys.exit("Choose axis from [0, 1, 2] or use get_peak_neighborhood.")

            bool_grid_list.append(bool_grid_padded)

        return bool_grid_list

    def get_peak_neighborhood(self, prob, prob_min, number_of_peaks):
        """
        Calculates the neighborhood of the peaks,
        which has at least the given fraction of the maximum probability :math:`|\psi|^2`.

        """
        peaks_indices, peaks_height = functions.get_peaks(prob)
        peaks_sorted_indices, peaks_sorted_height = functions.peaks_sort(peaks_indices,
                                                                         peaks_height,
                                                                         number_of_peaks)

        bool_grid_list = []
        for i, peak_index in enumerate(peaks_sorted_indices):
            prob_droplets = np.where(prob > prob_min, prob, 0)
            single_droplet, edges = functions.extract_droplet(prob_droplets, peaks_sorted_indices[i])

            pad_width = []
            for j, res_axis in enumerate(np.array([self.Res.x, self.Res.y, self.Res.z])):
                edge_left = np.asarray(edges)[j, 0]
                edge_right = np.asarray(edges)[j, 1]
                pad_right = res_axis - edge_right
                pad_width.append((edge_left, pad_right))
            bool_grid_padded = np.pad(single_droplet, pad_width, 'constant')

            bool_grid_list.append(bool_grid_padded)

        return bool_grid_list

    def get_N_in_droplets(self, prob_min, number_of_peaks):
        """

        Parameters
        ----------
        prob_min :
        number_of_peaks :

        Returns
        -------
        The first number_of_peaks entries are the number of particles in droplets
        (defined by :math:`|\psi|^2 > \\mathrm{prob_min}`) on the x-axis from left to right.
        The last entry is the sum of particles of those droplets.

        """
        bool_grid_list = self.get_peak_neighborhood(
            prob=np.abs(self.psi_val) ** 2.0,
            prob_min=prob_min,
            number_of_peaks=number_of_peaks,
        )

        N_in_droplets = []
        for k in range(0, number_of_peaks):
            psi_val_droplets = np.where(bool_grid_list[k],
                                        self.psi_val,
                                        np.zeros(shape=np.shape(self.psi_val)))

            droplets_density = self.trapez_integral(np.abs(psi_val_droplets) ** 2.0)
            N_in_droplets.append(self.N * droplets_density)

        N_in_droplets.append(np.sum(N_in_droplets))

        return N_in_droplets

    def slice_default(self, x0=None, x1=None, y0=None, y1=None, z0=None, z1=None):
        if (x0 is None) and (x1 is None):
            x0 = 0
            x1 = self.Res.x - 1
        else:
            if (x0 < 0) or ((x0 or x1) > self.Res.x):
                sys.exit(f"ERROR: Slice indices ({x0}, {x1}) for x out of bound. "
                         f"Bounds are (0, {self.Res.x})\n")

        if (y0 is None) and (y1 is None):
            y0 = 0
            y1 = self.Res.y - 1
        else:
            if (y0 < 0) or ((y0 or y1) > self.Res.y):
                sys.exit(f"ERROR: Slice indices ({y0}, {y1}) for y out of bound. "
                         f"Bounds are (0, {self.Res.y})\n")

        if (z0 is None) and (z1 is None):
            z0 = 0
            z1 = self.Res.z - 1
        else:
            if (z0 < 0) or ((z0 or z1) > self.Res.z):
                sys.exit(f"ERROR: Slice indices ({z0}, {z1}) for z out of bound. "
                         f"Bounds are (0, {self.Res.z})\n")

        return x0, x1, y0, y1, z0, z1

    def get_center_of_mass(self, x0=None, x1=None, y0=None, y1=None, z0=None, z1=None):
        """
        Calculates the center of mass of the System.

        """

        x0, x1, y0, y1, z0, z1 = self.slice_default(x0, x1, y0, y1, z0, z1)
        prob = self.get_density(p=2.0)[x0:x1, y0:y1, z0:z1]
        r = self.get_mesh_list(x0, x1, y0, y1, z0, z1)
        center_of_mass_along_axis = [prob * r_i for r_i in r]
        com = [self.trapez_integral(com_along_axis) / self.trapez_integral(prob) for com_along_axis in
               center_of_mass_along_axis]
        return com

    def get_parity(self, axis=2, x0=None, x1=None, y0=None, y1=None, z0=None, z1=None):
        x0, x1, y0, y1, z0, z1 = self.slice_default(x0, x1, y0, y1, z0, z1)
        psi_under0, psi_over0 = np.split(self.psi_val, 2, axis=axis)

        if axis in [0, 1, 2]:
            psi_over0_reversed = psi_over0[::-1]
        else:
            sys.exit(f"No such axis ({axis}). Choose 0, 1 or 2 for axis x, y or z.")

        parity = self.trapez_integral(np.abs(
            psi_under0[x0:x1, y0:y1, z0:z1] - psi_over0_reversed[x0:x1, y0:y1, z0:z1]) ** 2.0)

        return parity

    def get_phase_var_neighborhood(self, prob_min, number_of_peaks):
        """
        Calculates the variance of the phase of the System.

        """
        prob = np.abs(self.psi_val) ** 2.0,
        bool_grid_list = self.get_peak_neighborhood(prob, prob_min, number_of_peaks)
        bool_grid = np.logical_or(bool_grid_list[:-1], bool_grid_list[-1])

        norm = self.get_norm()
        prob = bool_grid * self.get_density(p=2.0) / norm
        psi_val_bool_grid = bool_grid * self.psi_val
        angle = np.angle(psi_val_bool_grid)
        angle_cos = np.cos(angle + np.pi)

        phase = self.trapez_integral(prob * angle_cos)
        phase2 = self.trapez_integral(prob * angle_cos ** 2.0)
        phase_var = np.sqrt(np.abs(phase2 - phase ** 2.0))

        return phase_var

    def get_phase_var(self, x0, x1, y0, y1, z0, z1):
        """
        Calculates the variance of the phase of the System by cos(phi).

        """
        norm = self.get_norm(func=self.psi_val[x0:x1, y0:y1, z0:z1])

        prob_cropped = self.get_density(p=2.0)[x0:x1, y0:y1, z0:z1] / norm
        psi_val_cropped = self.psi_val[x0:x1, y0:y1, z0:z1]
        angle = np.angle(psi_val_cropped)
        angle_cos = np.cos(angle + np.pi)

        phase = self.trapez_integral(prob_cropped * angle_cos)
        phase2 = self.trapez_integral(prob_cropped * angle_cos ** 2.0)

        phase_var = np.sqrt(np.abs(phase2 - phase ** 2.0))

        return phase_var

    def time_step(self) -> None:
        """
        Evolves System according Schrödinger Equations by using the
        split operator method with the Trotter-Suzuki approximation.

        """
        # adjust dt, to get the time accuracy when needed
        # self.dt = self.dt_func(self.t, self.dt)

        # Calculate the interaction by applying it to the psi_2 in k-space
        # (transform back and forth)
        psi_2: np.ndarray = self.get_density(p=2.0)
        psi_3: np.ndarray = self.get_density(p=3.0)
        U_dd: np.ndarray = np.fft.ifftn(self.V_k_val * np.fft.fftn(psi_2))

        # update H_pot before use
        H_pot: np.ndarray = np.exp(self.U
                                   * (0.5 * self.dt)
                                   * (self.V_val
                                      + self.g * psi_2
                                      + self.g_qf * psi_3
                                      + self.g * self.e_dd * U_dd))
        # multiply element-wise the (1D, 2D or 3D) arrays with each other
        self.psi_val = H_pot * self.psi_val

        self.psi_val = np.fft.fftn(self.psi_val)
        # H_kin is just dependent on U and the grid-points, which are constants,
        # so it does not need to be recalculated
        # multiply element-wise the (1D, 2D or 3D) array (H_kin) with psi_val
        # (1D, 2D or 3D)
        self.psi_val = self.H_kin * self.psi_val
        self.psi_val = np.fft.ifftn(self.psi_val)

        # update H_pot, psi_2, U_dd before use
        psi_2 = self.get_density(p=2.0)
        psi_3 = self.get_density(p=3.0)
        U_dd = np.fft.ifftn(self.V_k_val * np.fft.fftn(psi_2))
        H_pot = np.exp(self.U
                       * (0.5 * self.dt)
                       * (self.V_val
                          + self.g * psi_2
                          + self.g_qf * psi_3
                          + self.g * self.e_dd * U_dd))

        # multiply element-wise the (1D, 2D or 3D) arrays with each other
        self.psi_val = H_pot * self.psi_val

        self.t = self.t + self.dt

        # for self.imag_time=False, renormalization should be preserved,
        # but we play safe here (regardless of speedup)
        # if self.imag_time:
        psi_norm_after_evolution: float = self.trapez_integral(np.abs(self.psi_val) ** 2.0)
        self.psi_val = self.psi_val / np.sqrt(psi_norm_after_evolution)

        psi_quadratic_int = self.get_norm(p=4.0)
        psi_quintic_int = self.get_norm(p=5.0)

        self.mu_arr = np.array([-np.log(psi_norm_after_evolution)
                                / (2.0 * self.dt)])

        psi_val_k = np.fft.fftn(self.psi_val)
        psi_norm_k: float = self.get_norm(psi_val_k, fourier_space=True)
        psi_val_k = psi_val_k / np.sqrt(psi_norm_k)
        E_kin = self.get_norm(0.5 * self.k_squared * psi_val_k, fourier_space=True)

        if self.V_interaction:
            E_U_dd = (1 / np.sqrt(2.0 * np.pi) ** 3.0) * self.sum_dV(
                self.V_k_val * np.abs(np.fft.fftn(psi_2)) ** 2.0, fourier_space=True)
        else:
            E_U_dd = 0.0

        self.E = (self.mu_arr - 0.5 * self.g * psi_quadratic_int
                  - 0.5 * E_U_dd
                  - (3.0 / 5.0) * self.g_qf * psi_quintic_int)

    def use_summary(self, summary_name: Optional[str] = None):
        Summary: SchroedingerSummary = SchroedingerSummary(self)

        return Summary, summary_name

    def load_summary(self, input_path, steps_format, frame,
                     summary_name: Optional[str] = "SchroedingerSummary_"):
        if summary_name:
            system_summary_path = Path(input_path, summary_name + steps_format % frame + ".pkl")
        else:
            try:
                # needed because old versions had no self.name
                system_summary_path = Path(input_path, self.name + steps_format % frame + ".pkl")
            except:
                system_summary_path = None
        try:
            # load SchroedingerSummary
            with open(system_summary_path, "rb") as f:
                SystemSummary: SchroedingerSummary = dill.load(file=f)
                SystemSummary.copy_to(self)
        except Exception:
            print(f"{system_summary_path} not found.")

        return self

    def save_psi_val(self, input_path, filename_steps, steps_format, frame):
        with open(Path(input_path, filename_steps + steps_format % frame + ".npz"), "wb") as g:
            np.savez_compressed(g, psi_val=self.psi_val)

    def simulate_raw(self,
                     accuracy: float = 10 ** -6,
                     dir_path: Path = Path.home().joinpath("supersolids", "results"),
                     dir_name_load: str = "",
                     dir_name_result: str = "",
                     filename_schroedinger: str = "schroedinger.pkl",
                     filename_steps: str = "step_",
                     steps_format: str = "%07d",
                     steps_per_npz: int = 10,
                     frame_start: int = 0,
                     script_name: str = "script",
                     script_args: str = "",
                     script_number_regex: str = '*',
                     script_extensions: Optional[List[str]] = None,
                     script_extensions_index: int = 0,
                     ):
        if script_extensions is None:
            script_extensions = [".pkl", ".txt"]
            script_extensions_index = 0

        print(f"Accuracy goal: {accuracy}")

        # Create a results dir, if there is none
        if not dir_path.is_dir():
            dir_path.mkdir(parents=True)

        # Initialize mu_rel
        mu_rel = self.mu_arr

        if dir_name_result == "":
            _, last_index, dir_name, counting_format = get_path.get_path(dir_path)
            input_path = Path(dir_path, dir_name + counting_format % (last_index + 1))
        else:
            input_path = Path(dir_path, dir_name_result)

        # Create a movie dir, if there is none
        if not input_path.is_dir():
            input_path.mkdir(parents=True)

        # per default index 0 takes list with pkl
        script_list = reload_files(
            dir_path, dir_name_load, input_path, script_name,
            script_number_regex=script_number_regex,
            script_extensions=script_extensions
            )[script_extensions_index]

        script_count_old = get_step_index_from_list(
            script_list,
            filename_prefix=script_name+"_",
            file_pattern=script_extensions[script_extensions_index]
            )

        # per default index 0 takes list with pkl
        filename_schroedinger_prefix = filename_schroedinger.split(".")[0]
        schroedinger_list = reload_files(
            dir_path, dir_name_load, input_path, filename_schroedinger_prefix,
            script_number_regex="*",
            script_extensions=script_extensions
            )[script_extensions_index]

        save_script(script_count_old, input_path, script_name, script_args, txt_version=True)
        save_script(script_count_old, input_path, filename_schroedinger_prefix, self)

        # save used Schroedinger
        with open(Path(input_path, filename_schroedinger), "wb") as f:
            dill.dump(obj=self, file=f)

        frame_end = frame_start + self.max_timesteps
        for frame in range(frame_start, frame_end):
            mu_old = np.copy(self.mu_arr)
            self.time_step()

            SystemSummary, summary_name = self.use_summary()

            # save SchroedingerSummary not Schroedinger to save disk space
            if ((frame % steps_per_npz) == 0) or (frame == frame_end - 1):
                with open(Path(input_path, self.name + steps_format % frame + ".pkl"),
                          "wb") as f:
                    dill.dump(obj=SystemSummary, file=f)

            # save psi_val after steps_per_npz steps of dt (to save disk space)
            if ((frame % steps_per_npz) == 0) or (frame == frame_end - 1):
                self.save_psi_val(input_path, filename_steps, steps_format, frame)

            print(f"t={self.t:07.05f}, mu_rel={mu_rel}, "
                  f"processed={(frame - frame_start) / self.max_timesteps:05.03f}%")

            mu_rel = np.abs((self.mu_arr - mu_old) / self.mu_arr)

            # Stop animation when accuracy is reached
            if np.all(np.where(mu_rel < accuracy, True, False)):
                print(f"Accuracy reached: {mu_rel}")
                break

            elif np.any(np.isnan(mu_rel) & np.isnan(self.mu_arr)):
                print(f"Accuracy NOT reached! System diverged.")
                assert np.any(np.isnan(self.E)), ("E should be nan, when mu is nan."
                                                  "Then the system is divergent.")
                break

            if frame == (frame_end - 1):
                # Animation stops at the next step, to actually show the last step
                print(f"Maximum timesteps are reached. Animation is stopped.")
