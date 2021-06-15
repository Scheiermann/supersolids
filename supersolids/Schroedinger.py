#!/usr/bin/env python

# author: Daniel Scheiermann
# email: daniel.scheiermann@stud.uni-hannover.de
# license: MIT
# Please feel free to use and modify this, but keep the above information.

"""
Numerical solver for non-linear time-dependent Schrodinger equation.

"""

import functools
import sys
from typing import Callable, Union, Optional
from pathlib import Path

import dill
import numpy as np
import scipy.signal

from supersolids.helper import constants, functions, get_path
from supersolids.helper.Resolution import Resolution
from supersolids.helper.Box import Box

class Schroedinger:
    """
    Implements a numerical solution of the dimensionless time-dependent
    non-linear Schrodinger equation for an arbitrary potential:

    .. math::
    
       i \\partial_t \psi = [&-\\frac{1}{2} \\nabla ^2
                              + \\frac{1}{2} (x^2 + (y \\alpha_y)^2 + (z \\alpha_z)^2) \\\\
                             &+ g |\psi|^2  + g_{qf} |\psi|^3 + U_{dd}] \psi \\\\

    With :math:`U_{dd} = \\mathcal{F}^{-1}(\\mathcal{F}(H_{pot} \psi) \epsilon_{dd} g ((3 k_z / k^2) - 1))`

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
                 mu: float = 1.1,
                 E: float = 1.0,
                 psi_0: Callable = functions.psi_gauss_3d,
                 V: Optional[Callable] = functions.v_harmonic_3d,
                 V_interaction: Optional[Callable] = None,
                 psi_sol: Optional[Callable] = functions.thomas_fermi_3d,
                 mu_sol: Optional[Callable] = functions.mu_3d,
                 psi_0_noise: np.ndarray = functions.noise_mesh,
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
        assert isinstance(Res, Resolution), (f"box: {type(Res)} is not type {type(Resolution)}")

        self.N: int = N
        self.w_x: float = w_x
        self.w_y: float = w_y
        self.w_z: float = w_z
        self.a_s: float = a_s
        self.Res: Resolution = Res
        self.max_timesteps: int = max_timesteps

        assert isinstance(MyBox, Box), (
            f"box: {type(MyBox)} is not type {type(Box)}")

        self.Box: Box = MyBox
        self.dt: float = dt
        self.dt_func: Optional[Callable] = dt_func
        self.g: float = g
        self.g_qf: float = g_qf
        self.e_dd: float = e_dd
        self.imag_time: float = imag_time

        assert self.Box.dim == self.Res.dim, (
            f"Dimension of Box ({self.Box.dim}) and "
            f"Res ({self.Res.dim}) needs to be equal.")
        self.dim: int = self.Box.dim

        # mu = - ln(N) / (2 * dtau), where N is the norm of the :math:`\psi`
        self.mu: float = mu

        # E = mu - 0.5 * g * integral psi_val ** 2
        self.E: float = E

        self.psi: Callable = psi_0

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
            self.psi_sol: Callable = functools.partial(psi_sol, g=self.g)

        if mu_sol is None:
            self.mu_sol = None
        else:
            self.mu_sol: Callable = mu_sol(self.g)

        try:
            box_x_len = (self.Box.x1 - self.Box.x0)
            self.x: np.ndarray = np.linspace(self.Box.x0,
                                             self.Box.x1,
                                             self.Res.x)
            self.dx: float = (box_x_len / self.Res.x)
            self.dkx: float = np.pi / (box_x_len / 2.0)
            self.kx: np.ndarray = np.fft.fftfreq(self.Res.x,
                d=1.0 / (self.dkx * self.Res.x))

        except KeyError:
            sys.exit(
                f"Keys x0, x1 of box needed, "
                f"but it has the keys: {self.Box.keys()}, "
                f"Key x of res needed, "
                f"but it has the keys: {self.Res.keys()}")

        if imag_time:
            # Convention: $e^{-iH} = e^{UH}$
            self.U: complex = -1.0
        else:
            self.U = -1.0j

        # Add attributes as soon as they are needed (e.g. for dimension 3, all
        # besides the error are needed)
        if self.dim >= 2:
            try:
                box_y_len = self.Box.y1 - self.Box.y0
                self.y: np.ndarray = np.linspace(self.Box.y0,
                                                 self.Box.y1,
                                                 self.Res.y)
                self.dy: float = box_y_len / self.Res.y
                self.dky: float = np.pi / (box_y_len / 2.0)
                self.ky: np.ndarray = np.fft.fftfreq(self.Res.y,
                    d=1.0 / (self.dky * self.Res.y))

            except KeyError:
                sys.exit(
                    f"Keys y0, y1 of box needed, "
                    f"but it has the keys: {self.Box.keys()}, "
                    f"Key y of res needed, "
                    f"but it has the keys: {self.Res.keys()}")

        if self.dim >= 3:
            try:
                box_z_len = MyBox.z1 - MyBox.z0
                self.z: np.ndarray = np.linspace(self.Box.z0,
                                                 self.Box.z1,
                                                 self.Res.z)
                self.dz: float = box_z_len / self.Res.z
                self.dkz: float = np.pi / (box_z_len / 2.0)
                self.kz: np.ndarray = np.fft.fftfreq(self.Res.z,
                    d=1.0 / (self.dkz * self.Res.z))

            except KeyError:
                sys.exit(
                    f"Keys z0, z1 of box needed, "
                    f"but it has the keys: {self.Box.keys()}, "
                    f"Key z of res needed, "
                    f"but it has the keys: {self.Res.keys()}")

        if self.dim > 3:
            sys.exit("Spatial dimension over 3. This is not implemented.")

        if self.dim == 1:
            if psi_0_noise is None:
                self.psi_val: np.ndarray = self.psi(self.x)
            else:
                self.psi_val = psi_0_noise * self.psi(self.x)

            if V is None:
                self.V_val: Union[float, np.ndarray] = 0.0
            else:
                self.V_val = self.V(self.x)

            if self.psi_sol is None:
                self.psi_sol_val = None
            else:
                self.psi_sol_val: np.ndarray = self.psi_sol(self.x)

            self.k_squared: np.ndarray = self.kx ** 2.0
            self.H_kin: np.ndarray = np.exp(
                self.U * (0.5 * self.k_squared) * self.dt)

            if V_interaction is None:
                # For no interaction the identity is needed with respect to 2D
                # * 2D (array with 1.0 everywhere)
                self.V_k_val: np.ndarray = np.full(self.psi_val.shape, 1.0)

        elif self.dim == 2:
            self.x_mesh, self.y_mesh, self.pos = functions.get_meshgrid(self.x,
                                                                        self.y)

            if psi_0_noise is None:
                self.psi_val = self.psi(self.pos)
            else:
                self.psi_val = psi_0_noise * self.psi(self.pos)

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
            # here a number (U) is multiplied elementwise with an (1D, 2D or
            # 3D) array (k_squared)
            self.H_kin = np.exp(self.U * (0.5 * self.k_squared) * self.dt)

            if V_interaction is None:
                # For no interaction the identity is needed with respect to 2D
                # * 2D (array with 1.0 everywhere)
                self.V_k_val = np.full(self.psi_val.shape, 1.0)
            else:
                self.V_k_val = V_interaction(kx_mesh, ky_mesh, g=self.g)

        elif self.dim == 3:
            try:
                self.x_mesh, self.y_mesh, self.z_mesh = np.mgrid[
                                                        self.Box.x0: self.Box.x1:
                                                        complex(0, self.Res.x),
                                                        self.Box.y0: self.Box.y1:
                                                        complex(0, self.Res.y),
                                                        self.Box.z0: self.Box.z1:
                                                        complex(0, self.Res.z)
                                                        ]
            except KeyError:
                sys.exit(
                    f"Keys x0, x1, y0, y1, z0, z1 of box needed, "
                    f"but it has the keys: {self.Box.keys()}, "
                    f"Keys x, y, z of res needed, "
                    f"but it has the keys: {self.Res.keys()}")

            if psi_0_noise is None:
                self.psi_val = self.psi(self.x_mesh, self.y_mesh, self.z_mesh)
            else:
                self.psi_val = psi_0_noise * self.psi(self.x_mesh,
                                                      self.y_mesh,
                                                      self.z_mesh)

            if V is None:
                self.V_val = 0.0
            else:
                self.V_val = self.V(self.x_mesh, self.y_mesh, self.z_mesh)

            if self.psi_sol is None:
                self.psi_sol_val = None
            else:
                self.psi_sol_val = self.psi_sol(self.x_mesh,
                                                self.y_mesh,
                                                self.z_mesh)
                print(f"Norm for psi_sol (trapez integral): "
                      f"{self.trapez_integral(np.abs(self.psi_sol_val) ** 2.0)}")

            kx_mesh, ky_mesh, kz_mesh, _ = functions.get_meshgrid_3d(self.kx,
                                                                     self.ky,
                                                                     self.kz)
            self.k_squared = kx_mesh ** 2.0 + ky_mesh ** 2.0 + kz_mesh ** 2.0

            # here a number (U) is multiplied elementwise with an (1D, 2D or
            # 3D) array (k_squared)
            self.H_kin = np.exp(self.U * (0.5 * self.k_squared) * self.dt)

            if V_interaction is None:
                # For no interaction the identity is needed with respect to 2D
                # * 2D (array with 1.0 everywhere)
                self.V_k_val = np.full(self.psi_val.shape, 1.0)
            else:
                self.V_k_val = V_interaction(kx_mesh, ky_mesh, kz_mesh)

        # attributes for animation
        self.t: float = 0.0

    def get_density(self, p: float = 2.0) -> np.ndarray:
        """
        Calculates :math:`|\psi|^p` for 1D, 2D or 3D (depending on self.dim).

        :param p: Exponent of :math:`|\psi|`. Use p=2.0 for density.

        :return: :math:`|\psi|^p`
        """
        if self.dim <= 3:
            psi_density: np.ndarray = np.abs(self.psi_val) ** p
        else:
            sys.exit("Spatial dimension over 3. This is not implemented.")

        return psi_density

    def get_norm(self, p: float = 2.0) -> float:
        """
        Calculates :math:`\int |\psi|^p \\mathrm{dV}` for 1D, 2D or 3D
        (depending on self.dim). For p=2 it is the 2-norm.

        :param p: Exponent of :math:`|\psi|`. Use p=2.0 for density.

        :return: :math:`\int |\psi|^p \\mathrm{dV}`
        """

        if self.dim == 1:
            dV: float = self.dx
        elif self.dim == 2:
            dV = self.dx * self.dy
        elif self.dim == 3:
            dV = self.dx * self.dy * self.dz
        else:
            sys.exit("Spatial dimension over 3. This is not implemented.")

        psi_norm: float = np.sum(self.get_density(p=p)) * dV

        return psi_norm

    def trapez_integral(self, func_val: Callable) -> float:
        """
        Calculates :math:`\int |\psi|^p \\mathrm{dV}` for 1D, 2D or 3D
        (depending on self.dim) by using the trapez rule.

        For 1D: :math:`h (f(a) + f(a+h)) / 2`

        For 2D: :math:`h (f(a, b) + f(a+h, b) + f(a, b+h) + f(a+h, b+h)) / 2`

        For 3D there are 8 entries in the same manner
        :math:`(a, b, c) ... (a+h, b+h, c+h)`

        :param func_val: Grid sampled values of the function to integrate.

        :return: :math:`\int |\psi|^p \\mathrm{dV}` according to trapez rule
        """

        if self.dim == 1:
            dV: float = self.dx
            return dV * np.sum(func_val[0:-1] + func_val[1:]) / 2.0

        elif self.dim == 2:
            dV = self.dx * self.dy
            return dV * np.sum(func_val[0:-1, 0:-1]
                               + func_val[0:-1, 1:]
                               + func_val[1:, 0:-1]
                               + func_val[1:, 1:]
                               ) / 4.0

        elif self.dim == 3:
            dV = self.dx * self.dy * self.dz
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
            print(f"Not implemented yet.")
            pass

    def get_r2(self):
        if self.dim == 1:
            r2 = self.x_mesh ** 2.0
        elif self.dim == 2:
            r2 = self.x_mesh ** 2.0 + self.y_mesh ** 2.0
        elif self.dim == 3:
            r2 = self.x_mesh ** 2.0 + self.y_mesh ** 2.0 + self.z_mesh ** 2.0
        else:
            sys.exit("Spatial dimension over 3. This is not implemented.")

        return r2

    def get_r_vector(self):
        if self.dim == 1:
            r = self.x_mesh.ravel()
        elif self.dim == 2:
            r = (self.x_mesh.ravel(), self.y_mesh.ravel())
        elif self.dim == 3:
            r = np.array(
                [self.x_mesh.ravel(), self.y_mesh.ravel(), self.z_mesh.ravel()])
        else:
            sys.exit("Spatial dimension over 3. This is not implemented.")

        return r

    def get_peaks_along(self, axis=0, height=0.05, amount=4):
        prob = np.abs(self.psi_val) ** 2.0
        res_x_middle = int(self.Res.x / 2)
        res_y_middle = int(self.Res.y / 2)
        res_z_middle = int(self.Res.z / 2)
        if axis == 0:
            peaks, properties = scipy.signal.find_peaks(prob[:, res_y_middle, res_z_middle],
                                                        height=height)
        elif axis == 1:
            peaks, properties = scipy.signal.find_peaks(prob[res_x_middle, :, res_z_middle],
                                                        height=height)
        elif axis == 2:
            peaks, properties = scipy.signal.find_peaks(prob[res_x_middle, res_y_middle, :],
                                                        height=height)
        else:
            peaks, properties = scipy.signal.find_peaks(prob, height=height)

        # get the highest peaks in a sorted fashion (the n biggest, where n is amount)
        sorting_indices = np.argsort(properties['peak_heights'])[-amount:]
        peaks_sorted_indices = peaks[sorting_indices]

        # sort properties in the same fashion as peaks
        properties_sorted = properties.copy()
        for key, value in properties.items():
            properties_sorted[key] = value[sorting_indices]

        return peaks_sorted_indices, properties_sorted

    def get_peak_positions(self, axis=0, height=0.05, amount=4):
        peaks_indices, _ = self.get_peaks_along(height=height, amount=amount, axis=axis)
        if axis == 0:
            positions = self.Box.lengths()[axis] * (peaks_indices / self.Res.x) + self.Box.x0
        elif axis == 1:
            positions = self.Box.lengths()[axis] * (peaks_indices / self.Res.y) + self.Box.y0
        elif axis == 2:
            positions = self.Box.lengths()[axis] * (peaks_indices / self.Res.z) + self.Box.z0
        else:
            sys.exit(f"No such axis. Choose 0, 1 or 2 for axis x, y or z.")

        return positions

    def get_peak_distances(self, axis=0, height=0.05, amount=3):
        """
        Calculates the distances between the peaks in terms of box units.

        """
        peaks_indices, _ = self.get_peaks_along(axis=axis, height=height, amount=amount)
        distances_indices = np.diff(peaks_indices)
        if axis == 0:
            distances = self.Box.lengths()[axis] * (distances_indices / self.Res.x)
        elif axis == 1:
            distances = self.Box.lengths()[axis] * (distances_indices / self.Res.y)
        elif axis == 2:
            distances = self.Box.lengths()[axis] * (distances_indices / self.Res.z)
        else:
            sys.exit(f"No such axis. Choose 0, 1 or 2 for axis x, y or z.")

        return distances

    def get_peak_neighborhood(self, axis=0, height=0.05, amount=4, fraction=0.1,
                              peak_distances_cutoff=0.5):
        """
        Calculates the neighborhood of the peaks,
        which has at least the given fraction of the maximum probability :math:`|\psi|^2`.

        """
        peaks_sorted_indices, properties_sorted = self.get_peaks_along(axis=axis,
                                                                       height=height,
                                                                       amount=amount)
        distances_indices = np.diff(np.sort(peaks_sorted_indices))
        # extend one element at beginning and end, according to first/last element
        distances_indices = np.pad(distances_indices, (1, 1), 'edge')

        cut_off = fraction * np.max(properties_sorted['peak_heights'])
        prob = np.abs(self.psi_val) ** 2.0
        bool_grid = (cut_off <= prob)
        bool_grid_list = []
        for i, peak_index in enumerate(peaks_sorted_indices):
            peak_radius = peak_distances_cutoff * np.abs(distances_indices)[i]
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
                sys.exit("Not implemented yet. Choose axis 0, 1, 2.")

            bool_grid_list.append(bool_grid_padded)

        return bool_grid_list

    def get_center_of_mass(self):
        """
        Calculates the center of mass of the System.

        """
        return np.average(self.get_density(p=2.0).ravel() * self.get_r_vector(), axis=1)

    def get_parity(self, axis=2):
        psi_under0, psi_over0 = np.split(self.psi_val, 2, axis=axis)
        if axis == 0:
            psi_over0_reversed = psi_over0[::-1]
        elif axis == 1:
            psi_over0_reversed = psi_over0[::-1]
        elif axis == 2:
            psi_over0_reversed = psi_over0[::-1]
        else:
            sys.exit(f"No such axis. Choose 0, 1 or 2 for axis x, y or z.")

        parity = self.trapez_integral(np.abs(psi_under0 - psi_over0_reversed) ** 2.0)

        return parity

    def get_phase_var_neighborhood(self, axis=0, height=0.05, amount=4, fraction=0.1,
                                   peak_distances_cutoff=0.5):
        """
        Calculates the variance of the phase of the System.

        """
        bool_grid_list = self.get_peak_neighborhood(
            axis=axis,
            height=height,
            amount=amount,
            fraction=fraction,
            peak_distances_cutoff=peak_distances_cutoff,
        )
        bool_grid = np.logical_or(bool_grid_list[:-1], bool_grid_list[-1])

        prob_cropped = bool_grid * self.get_density(p=2.0)
        psi_val_cropped = bool_grid * self.psi_val
        angle = np.angle(psi_val_cropped)

        phase = self.trapez_integral(prob_cropped * (angle + np.pi))
        phase2 = self.trapez_integral(prob_cropped * (angle + np.pi) ** 2.0)
        phase_var = np.sqrt(np.abs(phase2 - phase ** 2.0))

        return phase_var

    def get_phase_var(self, x0, x1, y0, y1, z0, z1):
        """
        Calculates the variance of the phase of the System.

        """
        prob_cropped = self.get_density(p=2.0)[x0:x1, y0:y1, z0:z1]
        psi_val_cropped = self.psi_val[x0:x1, y0:y1, z0:z1]
        angle = np.angle(psi_val_cropped)

        phase = self.trapez_integral(prob_cropped * (angle + np.pi))
        phase2 = self.trapez_integral(prob_cropped * (angle + np.pi) ** 2.0)
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

        self.mu = - np.log(psi_norm_after_evolution) / (2.0 * self.dt)
        U_dd_int = self.trapez_integral(U_dd * psi_2)
        self.E = (self.mu - 0.5 * self.g * psi_quadratic_int
                          - 0.5 * U_dd_int
                          - (3.0 / 5.0) * self.g_qf * psi_quintic_int)

    def simulate_raw(self,
                     accuracy: float = 10 ** -6,
                     dir_path: Path = Path.home().joinpath("supersolids", "results"),
                     dir_name_result: str = "",
                     filename_schroedinger=f"schroedinger.pkl",
                     filename_steps=f"step_",
                     steps_format: str = "%06d",
                     steps_per_npz: int = 10,
                     frame_start: int = 0,
                     ):

        print(f"Accuracy goal: {accuracy}")

        # Create a results dir, if there is none
        if not dir_path.is_dir():
            dir_path.mkdir(parents=True)

        # Initialize mu_rel
        mu_rel = self.mu

        if dir_name_result == "":
            _, last_index, dir_name, counting_format = get_path.get_path(dir_path)
            input_path = Path(dir_path, dir_name + counting_format % (last_index + 1))
        else:
            input_path = Path(dir_path, dir_name_result)

        # Create a movie dir, if there is none
        if not input_path.is_dir():
            input_path.mkdir(parents=True)

        # save used Schroedinger
        with open(Path(input_path, filename_schroedinger), "wb") as f:
            dill.dump(obj=self, file=f)

        frame_end = frame_start + self.max_timesteps
        for frame in range(frame_start, frame_end):
            mu_old = self.mu
            self.time_step()

            # save psi_val after steps_per_pickle steps of dt (to save disk space)
            if ((frame % steps_per_npz) == 0) or (frame == frame_end - 1):
                with open(Path(input_path, filename_steps + steps_format % frame + ".npz"),
                          "wb") as g:
                    np.savez_compressed(g, psi_val=self.psi_val)

            print(f"t={self.t:07.05f}, mu_rel={mu_rel:+05.05e}, "
                  f"processed={(frame - frame_start) / self.max_timesteps:05.03f}%")

            mu_rel = np.abs((self.mu - mu_old) / self.mu)

            # Stop animation when accuracy is reached
            if mu_rel < accuracy:
                print(f"Accuracy reached: {mu_rel}")
                break

            elif np.isnan(mu_rel) and np.isnan(self.mu):
                assert np.isnan(self.E), ("E should be nan, when mu is nan."
                                          "Then the system is divergent.")
                print(f"Accuracy NOT reached! System diverged.")
                break

            if frame == (self.max_timesteps - 1):
                # Animation stops at the next step, to actually show the last step
                print(f"Maximum timesteps are reached. Animation is stopped.")
