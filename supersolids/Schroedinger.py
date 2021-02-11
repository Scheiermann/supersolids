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

import numpy as np

from supersolids.helper import constants, functions


class Schroedinger:
    """
    Implements a numerical solution of the dimensionless time-dependent
    non-linear Schrodinger equation for an arbitrary potential:

    .. math::
    
       i \\partial_t \psi = [&-\\frac{1}{2} \\nabla ^2
                              + \\frac{1}{2} (x^2 + (y \\alpha_y)^2 + (z \\alpha_z)^2) \\\\
                             &+ g |\psi|^2  + g_{qf} |\psi|^3 + U_{dd}] \psi \\\\

    With :math:`U_{dd} =  \\mathcal{F}^{-1}(\\mathcal{F}(H_{pot} \psi) \epsilon_{dd} g ((3 k_z / k^2) - 1))`

    The split operator method with the Trotter-Suzuki approximation
    for the commutator relation (:math:`H = H_{pot} + H_{kin}`) is used.
    Hence the accuracy is proportional to :math:`dt^4`
    The approximation is needed because of the Baker-Campell-Hausdorff formula.
    """

    def __init__(self,
                 N: int,
                 Box: functions.Box,
                 Res: functions.Resolution,
                 max_timesteps: int,
                 dt: float,
                 dt_func: Optional[Callable] = None,
                 g: float = 0.0,
                 g_qf: float = 0.0,
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

        :param Box: Keyword x0 is minimum in x direction and
            x1 is maximum. Same for y and z. For 1D just use x0, x1.
            For 2D x0, x1, y0, y1.
            For 3D x0, x1, y0, y1, z0, z1.
            Dimension of simulation is constructed from this dictionary.

        :param Res: functoins.Res
            Number of grid points in x, y, z direction.
            Needs to have half size of box dictionary.
            Keywords x, y, z are used.

        :param max_timesteps: Maximum timesteps  with length dt for the animation.

        """
        assert isinstance(Res, functions.Resolution), (
            f"box: {type(Res)} is not type {type(functions.Resolution)}")

        self.N: int = N
        self.w_y: float = w_y
        self.w_z: float = w_z
        self.a_s: float = a_s
        self.Res: functions.Resolution = Res
        self.max_timesteps: int = max_timesteps

        assert isinstance(Box, functions.Box), (
            f"box: {type(Box)} is not type {type(functions.Box)}")

        self.Box: functions.Box = Box
        self.dt: float = dt
        self.dt_func: Optional[Callable] = dt_func
        self.g: float = g
        self.g_qf: float = g_qf
        self.e_dd: float = e_dd
        self.imag_time: float = imag_time

        assert self.Box.dim == self.Res.dim, (f"Dimension of Box ({self.Box.dim}) and "
                                    f"Res ({self.Res.dim}) needs to be equal.")
        self.dim: int = self.Box.dim

        # mu = - ln(N) / (2 * dtau), where N is the norm of the :math:`\psi`
        self.mu: float = mu

        # E = mu - 0.5 * g * integral psi_val ** 2
        self.E: float = E

        self.psi: Callable = psi_0

        if V is not None:
            self.V: Callable = V
        else:
            self.V = None

        if psi_sol is not None:
            self.psi_sol: Callable = functools.partial(psi_sol, g=self.g)
        else:
            self.psi_sol = None

        if mu_sol is not None:
            self.mu_sol: Callable = mu_sol(self.g)
        else:
            self.mu_sol = None

        try:
            box_x_len = (self.Box.x1 - self.Box.x0)
            self.x: np.ndarray = np.linspace(self.Box.x0, self.Box.x1,
                                             self.Res.x)
            self.dx: float = (box_x_len / self.Res.x)
            self.dkx: float = np.pi / (box_x_len / 2.0)
            self.kx: np.ndarray = np.fft.fftfreq(self.Res.x,
                                                 d=1.0 / (
                                                         self.dkx * self.Res.x))

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
                                                     d=1.0 / (
                                                             self.dky * self.Res.y))

            except KeyError:
                sys.exit(
                    f"Keys y0, y1 of box needed, "
                    f"but it has the keys: {self.Box.keys()}, "
                    f"Key y of res needed, "
                    f"but it has the keys: {self.Res.keys()}")

        if self.dim >= 3:
            try:
                box_z_len = Box.z1 - Box.z0
                self.z: np.ndarray = np.linspace(self.Box.z0,
                                                 self.Box.z1,
                                                 self.Res.z)
                self.dz: float = box_z_len / self.Res.z
                self.dkz: float = np.pi / (box_z_len / 2.0)
                self.kz: np.ndarray = np.fft.fftfreq(self.Res.z,
                                                     d=1.0 / (
                                                             self.dkz * self.Res.z))

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
                print(f"Trapez Norm for psi_sol: "
                      f"{self.get_norm_trapez(self.psi_sol_val)}")

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

    def get_norm_trapez(self, func_val: Callable) -> float:
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
        psi_norm_after_evolution: float = self.get_norm_trapez(np.abs(self.psi_val) ** 2.0)
        # psi_norm_after_evolution: float = self.get_norm(p=2.0)
        self.psi_val = self.psi_val / np.sqrt(psi_norm_after_evolution)

        psi_quadratic_int = self.get_norm(p=4.0)

        # TODO: adjust for DDI
        self.mu = - np.log(psi_norm_after_evolution) / (2.0 * self.dt)
        self.E = self.mu - 0.5 * self.g * psi_quadratic_int

        # print(f"Sol norm: {self.get_norm_trapez(self.psi_sol_val)}")

        # TODO: These formulas for mu.sol and E are not for all cases correct
        # print(f"mu: {self.mu}")
        # if self.g != 0:
        #     print(f"E: {self.E}, "
        #           f"E_sol: {self.mu_sol - 0.5 * self.g * psi_quadratic_int}")
        # else:
        #     print(f"E: {self.E}")
