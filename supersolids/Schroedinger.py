#!/usr/bin/env python

# author: Daniel Scheiermann
# email: daniel.scheiermann@stud.uni-hannover.de
# license: MIT
# Please feel free to use and modify this, but keep the above information. Thanks!

"""
Numerical solver for non-linear time-dependent Schrodinger equation.

"""

import functools
import sys
from typing import Callable, Dict, Union, Optional, NamedTuple

import numpy as np

from supersolids import functions


class Schroedinger(object):
    """
    Implements a numerical solution of the dimensionless time-dependent
    non-linear Schrodinger equation for an arbitrary potential:

    With :math:`U_{dd} =  iFFT(FFT(H_{pot} \psi) \epsilon_{dd} g ((3 k_z / k^2) - 1))`

    :math:`i \\partial_t \psi = [-0.5 \\nabla ^2 + 0.5 (x^2 + (y \\alpha_y)^2 + (z \\alpha_z)^2) + g |\psi|^2 + g_{qf} |\psi|^3 + U_{dd}] \psi`

    The split operator method with the Trotter-Suzuki approximation
    for the commutator relation (:math:`H = H_{pot} + H_{kin}`) is used.
    Hence the accuracy is proportional to :math:`dt^4`
    The approximation is needed because of the Baker-Campell-Hausdorff formula.
    """

    def __init__(self,
                 box: NamedTuple,
                 res: NamedTuple,
                 max_timesteps: int,
                 dt: float,
                 g: float = 0.0,
                 g_qf: float = 0.0,
                 e_dd: float = 1.0,
                 imag_time: bool = True,
                 mu: float = 1.1,
                 E: float = 1.0,
                 psi_0: Callable = functions.psi_gauss_3d,
                 V: Optional[Callable] = functions.v_harmonic_3d,
                 V_interaction: Optional[Callable] = None,
                 psi_sol: Optional[Callable] = functions.thomas_fermi_3d,
                 mu_sol: Optional[Callable] = functions.mu_3d,
                 psi_0_noise: Optional[Callable] = functions.noise_mesh,
                 alpha_psi: float = 0.8,
                 alpha_psi_sol: float = 0.53,
                 alpha_V: float = 0.3,
                 ) -> None:
        """
        Schrödinger equations for the specified system.

        Parameters

        box : NamedTuple
            Keyword x0 is minimum in x direction and
            x1 is maximum. Same for y and z. For 1D just use x0, x1.
            For 2D x0, x1, y0, y1.
            For 3D x0, x1, y0, y1, z0, z1.
            Dimension of simulation is constructed from this dictionary.

        res : NamedTuple
            NamedTuple for the number of grid points in x, y, z direction.
            Needs to have half size of box dictionary.
            Keywords x, y z are used.

        max_timesteps : int
            Maximum timesteps  with length dt for the animation.

        alpha_psi : float
            Alpha value for plot transparency of :math:`\psi`

        alpha_psi_sol : float
            Alpha value for plot transparency of :math:`\psi_{sol}`

        alpha_V : float
            Alpha value for plot transparency of V

        """
        assert isinstance(res, functions.Resolution), (
            f"box: {type(res)} is not type {type(functions.Resolution)}")

        self.res: NamedTuple = res
        self.max_timesteps: int = max_timesteps

        assert isinstance(box, functions.Box), (
            f"box: {type(box)} is not type {type(functions.Box)}")

        self.box: NamedTuple = box
        self.dt: float = dt
        self.g: float = g
        self.g_qf: float = g_qf
        self.e_dd: float = e_dd
        self.imag_time: float = imag_time

        box_dim = int(len(self.box) / 2.0)
        res_dim = len(self.res)
        assert box_dim == res_dim, (f"Dimension of box ({box_dim}) and "
                                    f"res ({res_dim}) needs to be equal.")
        self.dim: int = box_dim

        # mu = - ln(N) / (2 * dtau), where N is the norm of the :math:`\psi`
        self.mu: float = mu

        # E = mu - 0.5 * g * integral psi_val ** 2
        self.E: float = E

        self.psi: Callable = psi_0

        if V is not None:
            self.V: Callable = V

        if psi_sol is not None:
            self.psi_sol: Callable = functools.partial(psi_sol, g=self.g)

        if mu_sol is not None:
            self.mu_sol: Callable = mu_sol(self.g)

        try:
            box_x_len = (box.x1 - box.x0)
            self.x: np.ndarray = np.linspace(self.box.x0, self.box.x1,
                                             self.res.x)
            self.dx: float = (box_x_len / self.res.x)
            self.dkx: float = np.pi / (box_x_len / 2.0)
            self.kx: np.ndarray = np.fft.fftfreq(self.res.x,
                                                 d=1.0 / (
                                                         self.dkx * self.res.x))

        except KeyError:
            sys.exit(
                f"Keys x0, x1 of box needed, "
                f"but it has the keys: {self.box.keys()}, "
                f"Key x of res needed, "
                f"but it has the keys: {self.res.keys()}")

        if imag_time:
            # Convention: $e^{-iH} = e^{UH}$
            self.U: complex = -1.0
        else:
            self.U = -1.0j

        # Add attributes as soon as they are needed (e.g. for dimension 3, all
        # besides the error are needed)
        if self.dim >= 2:
            try:
                box_y_len = box.y1 - box.y0
                self.y: np.ndarray = np.linspace(self.box.y0,
                                                 self.box.y1,
                                                 self.res.y)
                self.dy: float = box_y_len / self.res.y
                self.dky: float = np.pi / (box_y_len / 2.0)
                self.ky: np.ndarray = np.fft.fftfreq(self.res.y,
                                                     d=1.0 / (
                                                             self.dky * self.res.y))

            except KeyError:
                sys.exit(
                    f"Keys y0, y1 of box needed, "
                    f"but it has the keys: {self.box.keys()}, "
                    f"Key y of res needed, "
                    f"but it has the keys: {self.res.keys()}")

        if self.dim >= 3:
            try:
                box_z_len = box.z1 - box.z0
                self.z: np.ndarray = np.linspace(self.box.z0,
                                                 self.box.z1,
                                                 self.res.z)
                self.dz: float = box_z_len / self.res.z
                self.dkz: float = np.pi / (box_z_len / 2.0)
                self.kz: np.ndarray = np.fft.fftfreq(self.res.z,
                                                     d=1.0 / (
                                                             self.dkz * self.res.z))

            except KeyError:
                sys.exit(
                    f"Keys z0, z1 of box needed, "
                    f"but it has the keys: {self.box.keys()}, "
                    f"Key z of res needed, "
                    f"but it has the keys: {self.res.keys()}")

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

            if self.psi_sol is not None:
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

            if self.psi_sol is not None:
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
                                                        self.box.x0: self.box.x1:
                                                        complex(0, self.res.x),
                                                        self.box.y0: self.box.y1:
                                                        complex(0, self.res.y),
                                                        self.box.z0: self.box.z1:
                                                        complex(0, self.res.z)
                                                        ]
            except KeyError:
                sys.exit(
                    f"Keys x0, x1, y0, y1, z0, z1 of box needed, "
                    f"but it has the keys: {self.box.keys()}, "
                    f"Keys x, y, z of res needed, "
                    f"but it has the keys: {self.res.keys()}")

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

            if self.psi_sol is not None:
                self.psi_sol_val = self.psi_sol(self.x_mesh,
                                                self.y_mesh,
                                                self.z_mesh)

            kx_mesh, ky_mesh, kz_mesh = np.mgrid[
                                        self.kx[0]:self.kx[-1]:
                                        complex(0, self.res.x),
                                        self.ky[0]:self.ky[-1]:
                                        complex(0, self.res.y),
                                        self.kz[0]:self.kz[-1]:
                                        complex(0, self.res.z)
                                        ]
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

        self.alpha_psi: float = alpha_psi
        self.alpha_psi_sol: float = alpha_psi_sol
        self.alpha_V: float = alpha_V

    def get_density(self, p: float = 2.0) -> np.ndarray:
        """
        Calculates :math:`|\psi|^p` for 1D, 2D or 3D.

        Parameters

        p : float
            Exponent of :math:`|\psi|`. Use p=2.0 for density.

        Returns

        psi_density : np.ndarray
            :math:`|\psi|^p`
        """
        if self.dim <= 3:
            psi_density: np.ndarray = np.abs(self.psi_val) ** p
        else:
            sys.exit("Spatial dimension over 3. This is not implemented.")

        return psi_density

    def get_norm(self, p: float = 2.0) -> float:
        """
        Calculates :math:`\int |\psi|^p \\mathrm{dV}` for 1D, 2D or 3D.
        For p=2 it is the 2-norm.

        Parameters

        p : float
            Exponent of :math:`|\psi|`. Use p=2.0 for density.

        Returns

        psi_norm : float
            p-norm of :math:`\int |\psi|^p \\mathrm{dV}`
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

    def time_step(self) -> None:
        # Here we use half steps in real space, but will use it before and
        # after H_kin with normal steps

        # Calculate the interaction by applying it to the psi_2 in k-space
        # (transform back and forth)
        psi_2: np.ndarray = self.get_density(p=2.0)
        psi_3: np.ndarray = self.get_density(p=3.0)
        U_dd: np.ndarray = np.fft.ifftn(
            self.V_k_val * np.fft.fftn(psi_2))
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
        psi_norm_after_evolution: float = self.get_norm(p=2.0)
        self.psi_val = self.psi_val / np.sqrt(psi_norm_after_evolution)

        psi_quadratic_int = self.get_norm(p=4.0)

        # TODO: adjust for DDI
        self.mu = - np.log(psi_norm_after_evolution) / (2.0 * self.dt)
        self.E = self.mu - 0.5 * self.g * psi_quadratic_int

        # TODO: These formulas for mu.sol and E are not for all cases correct
        # print(f"mu: {self.mu}")
        # if self.g != 0:
        #     print(f"E: {self.E}, "
        #           f"E_sol: {self.mu_sol - 0.5 * self.g * psi_quadratic_int}")
        # else:
        #     print(f"E: {self.E}")
