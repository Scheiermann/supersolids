#!/usr/bin/env python

"""
Numerical solver for non-linear time-dependent Schrodinger's equation.

author: Daniel Scheiermann
email: daniel.scheiermann@stud.uni-hannover.de
license: MIT
Please feel free to use and modify this, but keep the above information. Thanks!
"""
import sys

import functools
import numpy as np
from typing import Callable

from supersolids import functions


class Schroedinger(object):
    """
    Implements a numerical solution of the dimensionless time-dependent
    non-linear Schrodinger equation for an arbitrary potential, where D[., t] is a partial derivative to t:
    i D(psi, t) = [-0.5 * (D[., x] ** 2 + D[., y] ** 2 + D[., z] ** 2)
                   + 0.5 * (x ** 2 + (alpha_y * y) ** 2 + (alpha_z * z) ** 2)
                   + g |psi| ** 2
                   + g_qf * |psi| ** 3
                   + U_dd] psi

    With U_dd = iFFT( FFT(|psi| ** 2) * e_dd * g * ((3 * k_z / k ** 2) - 1.0) )

    We will first implement the split operator without commutator relation ($H = H_{pot} + H_{kin}$)
    WARNING: We don't use Baker-Campell-Hausdorff formula, hence the accuracy is small. This is just a draft.
    """

    def __init__(self, resolution: int, max_timesteps: int, L: float, dt: float, g: float = 0.0, g_qf: float = 0.0,
                 imag_time: bool = True, s: float = 1.1, E: float = 1.0,
                 dim: int = 3,
                 psi_0: Callable = functions.psi_gauss_3d,
                 V: Callable = functions.v_harmonic_3d,
                 V_interaction: Callable = None,
                 psi_sol: Callable = functions.thomas_fermi_3d,
                 mu_sol: Callable = functions.mu_3d,
                 alpha_psi: float = 0.8,
                 alpha_psi_sol: float = 0.5,
                 alpha_V: float = 0.3,
                 ):
        """
        Schrödinger equations for the specified system.

        Parameters
        ----------
        resolution : int
            number of grid points in one direction

        max_timesteps : int
            Maximum timesteps  with length dt for the animation.

        alpha_psi : float
            Alpha value for plot transparency of psi

        alpha_psi_sol : float
            Alpha value for plot transparency of psi_sol

        alpha_V : float
            Alpha value for plot transparency of V

        """
        self.resolution = int(resolution)
        self.max_timesteps = int(max_timesteps)

        self.L = float(L)
        self.dt = float(dt)
        self.g = float(g)
        self.g_qf = float(g_qf)
        self.imag_time = imag_time
        self.dim = dim

        # mu = - ln(N) / (2 * dtau), where N is the norm of the psi
        self.mu = s

        # E = mu - 0.5 * g * int psi_val ** 2
        self.E = E

        self.psi = psi_0
        self.V = V
        if psi_sol is not None:
            self.psi_sol = functools.partial(psi_sol, g=self.g)
            self.mu_sol = mu_sol(self.g)
        else:
            self.psi_sol = None
            self.mu_sol = None

        self.x = np.linspace(-self.L, self.L, self.resolution)
        self.dx = float(2.0 * L / self.resolution)
        self.dkx = float(np.pi / self.L)

        self.kx = np.fft.fftfreq(resolution, d=1.0/(self.dkx * self.resolution))

        if imag_time:
            # Convention: $e^{-iH} = e^{UH}$
            self.U = -1.0
        else:
            self.U = -1.0j

        # Add attributes as soon as they are needed (e.g. for dimension 3, all besides the error are needed)
        if dim >= 2:
            self.y = np.linspace(-self.L, self.L, self.resolution)
            self.dy = float(2.0 * L / self.resolution)
            self.dky = float(np.pi / self.L)
            self.ky = np.fft.fftfreq(resolution, d=1.0 / (self.dky * self.resolution))
        if dim >= 3:
            self.z = np.linspace(-self.L, self.L, self.resolution)
            self.dz = float(2.0 * L / self.resolution)
            self.dkz = float(np.pi / self.L)
            self.kz = np.fft.fftfreq(resolution, d=1.0 / (self.dkz * self.resolution))
        if dim > 3:
            print("Spatial dimension over 3. This is not implemented.", file=sys.stderr)
            sys.exit(1)

        if dim == 1:
            self.psi_val = self.psi(self.x)
            self.V_val = self.V(self.x)
            if self.psi_sol is not None:
                self.psi_sol_val = self.psi_sol(self.x)
            else:
                self.psi_sol_val = None

            self.k_squared = self.kx ** 2.0
            self.H_kin = np.exp(self.U * (0.5 * self.k_squared) * self.dt)

            if V_interaction is None:
                # For no interaction the identity is needed with respect to 2D * 2D (array with 1.0 everywhere)
                self.V_k_val = np.full(self.psi_val.shape, 1.0)

        elif dim == 2:
            self.x_mesh, self.y_mesh, self.pos = functions.get_meshgrid(self.x, self.y)
            self.psi_val = self.psi(self.pos)
            self.V_val = self.V(self.pos)
            if self.psi_sol is not None:
                self.psi_sol_val = self.psi_sol(self.pos)
            else:
                self.psi_sol_val = None

            kx_mesh, ky_mesh, _ = functions.get_meshgrid(self.kx, self.ky)
            self.k_squared = kx_mesh ** 2.0 + ky_mesh ** 2.0
            # here a number (U) is multiplied elementwise with an (1D, 2D or 3D) array (k_squared)
            self.H_kin = np.exp(self.U * (0.5 * self.k_squared) * self.dt)

            if V_interaction is None:
                # For no interaction the identity is needed with respect to 2D * 2D (array with 1.0 everywhere)
                self.V_k_val = np.full(self.psi_val.shape, 1.0)
            else:
                self.V_k_val = V_interaction(kx_mesh, ky_mesh, g=self.g)

        elif dim == 3:
            self.x_mesh, self.y_mesh, self.z_mesh = np.mgrid[self.x[0]:self.x[-1]:complex(0, self.resolution),
                                                             self.y[0]:self.y[-1]:complex(0, self.resolution),
                                                             self.z[0]:self.z[-1]:complex(0, self.resolution)
                                                             ]
            self.psi_val = self.psi(self.x_mesh, self.y_mesh, self.z_mesh)
            self.V_val = self.V(self.x_mesh, self.y_mesh, self.z_mesh)

            if self.psi_sol is not None:
                self.psi_sol_val = self.psi_sol(self.x_mesh, self.y_mesh, self.z_mesh)
            else:
                self.psi_sol_val = None

            kx_mesh, ky_mesh, kz_mesh, _ = functions.get_meshgrid_3d(self.kx, self.ky, self.kz)
            self.k_squared = kx_mesh ** 2.0 + ky_mesh ** 2.0 + kz_mesh ** 2.0
            # here a number (U) is multiplied elementwise with an (1D, 2D or 3D) array (k_squared)
            self.H_kin = np.exp(self.U * (0.5 * self.k_squared) * self.dt)

            if V_interaction is None:
                # For no interaction the identity is needed with respect to 2D * 2D (array with 1.0 everywhere)
                self.V_k_val = np.full(self.psi_val.shape, 1.0)
            else:
                self.V_k_val = V_interaction(kx_mesh, ky_mesh, kz_mesh, g=self.g)

       # attributes for animation
        self.t = 0.0

        self.alpha_psi = alpha_psi
        self.alpha_psi_sol = alpha_psi_sol
        self.alpha_V = alpha_V

    def get_density(self, p: float = 2.0) -> np.ndarray:
        if self.dim <= 3:
            psi_density = np.abs(self.psi_val) ** p
        else:
            print("Spatial dimension over 3. This is not implemented.", file=sys.stderr)
            sys.exit(1)

        return psi_density

    def get_norm(self, p: float = 2.0) -> float:
        if self.dim == 1:
            dV = self.dx
        elif self.dim == 2:
            dV = self.dx * self.dy
        elif self.dim == 3:
            dV = self.dx * self.dy * self.dz
        else:
            print("Spatial dimension over 3. This is not implemented.", file=sys.stderr)
            sys.exit(1)

        psi_norm = np.sum(self.get_density(p=p)) * dV

        return psi_norm

    def time_step(self):
        # Here we use half steps in real space, but will use it before and after H_kin with normal steps

        # Calculate the interaction by appling it to the psi_2 in k-space (transform back and forth)
        psi_2 = self.get_density(p=2.0)
        psi_3 = self.get_density(p=3.0)
        U_interaction = np.fft.ifftn(self.V_k_val * np.fft.fftn(psi_2))
        # update H_pot before use
        H_pot = np.exp(self.U * (self.V_val + self.g * psi_2 + U_interaction + self.g_qf * psi_3)
                       * (0.5 * self.dt))
        # multiply element-wise the (1D, 2D or 3D) arrays with each other
        self.psi_val = H_pot * self.psi_val

        self.psi_val = np.fft.fftn(self.psi_val)
        # H_kin is just dependent on U and the grid-points, which are constants, so it does not need to be recalculated
        # multiply element-wise the (1D, 2D or 3D) array (H_kin) with psi_val (1D, 2D or 3D)
        self.psi_val = self.H_kin * self.psi_val
        self.psi_val = np.fft.ifftn(self.psi_val)

        # update H_pot, psi_2, U_interaction before use
        psi_2 = self.get_density(p=2.0)
        psi_3 = self.get_density(p=3.0)
        U_interaction = np.fft.ifftn(self.V_k_val * np.fft.fftn(psi_2))
        H_pot = np.exp(self.U * (self.V_val + self.g * psi_2 + U_interaction + self.g_qf * psi_3)
                       * (0.5 * self.dt))
        # multiply element-wise the (1D, 2D or 3D) arrays with each other
        self.psi_val = H_pot * self.psi_val

        self.t += self.dt

        # for self.imag_time=False, renormalization should be preserved, but we play safe here (regardless of speedup)
        # if self.imag_time:
        psi_norm_after_evolution = self.get_norm(p=2.0)
        self.psi_val /= np.sqrt(psi_norm_after_evolution)

        psi_quadratic_integral = self.get_norm(p=4.0)

        # TODO: adjust for DDI
        self.mu = - np.log(psi_norm_after_evolution) / (2.0 * self.dt)
        self.E = self.mu - 0.5 * self.g * psi_quadratic_integral

        # print(f"mu: {self.mu}")
        # if self.g != 0:
        #     print(f"E: {self.E}, E_sol: {self.mu_sol - 0.5 * self.g * psi_quadratic_integral}")
        # else:
        #     print(f"E: {self.E}")
