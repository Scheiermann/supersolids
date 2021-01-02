#!/usr/bin/env python

"""
Numerical solver for non-linear time-dependent Schrodinger's equation.

author: Daniel Scheiermann
email: daniel.scheiermann@stud.uni-hannover.de
license: MIT
Please feel free to use and modify this, but keep the above information. Thanks!
"""

import numpy as np
import sys

from supersolids import functions


class Schroedinger(object):
    """
    Implements a numerical solution of the time-dependent
    non-linear Schrodinger equation for an arbitrary potential:
    i \hbar \frac{\partial}{\partial t} \psi(r,t) = \left(\frac{-\hbar^{2}}{2 m} \nabla^{2}
                                                    + V(r) + g |\psi(x,t)|^{2} \right) \psi(x,t)

    For the moment we aim to solve:
    \mu \phi_{0}(x) = \left(\frac{-1}{2} \frac{\partial^{2}}{\partial x^{2}}
                      + \frac{1}{2} x^{2} + \tilde{g} |\phi_{0}(x)|^{2} \right) \phi_{0}(x)

    We will first implement the split operator without commutator relation ($H = H_{pot} + H_{kin}$)
    WARNING: We don't use Baker-Campell-Hausdorff formula, hence the accuracy is small. This is just a draft.
    """

    def __init__(self, resolution, timesteps, L, dt, g=0.0, imag_time=False, dim=1, s=1.0,
                 psi_0=functions.psi_gauss_1d,
                 V=functions.v_harmonic_1d,
                 psi_sol=functions.thomas_fermi
                 ):
        """
        Parameters
        ----------
        x: array_like, float
            description
        """
        self.resolution = int(resolution)
        self.timesteps = int(timesteps)

        self.L = float(L)
        self.dt = float(dt)
        self.g = float(g)
        self.imag_time = imag_time
        self.dim = dim
        # s = - ln(N) / (2 * dtau), where N is the norm of the psi
        self.s = s

        self.psi = psi_0
        self.V = V
        self.psi_sol = psi_sol

        self.x = np.linspace(-self.L, self.L, self.resolution)
        self.dx = float(2.0 * L / self.resolution)
        self.dkx = float(np.pi / self.L)

        # TODO: This can probably be done with sp.ttf.fftshift
        # self.kx = sp.fft.fftshift(self.psi_val)
        k_over_0 = np.arange(0.0, resolution / 2.0, 1.0)
        k_under_0 = np.arange(-resolution / 2.0, 0.0, 1.0)

        self.kx = np.concatenate((k_over_0, k_under_0), axis=0) * self.dkx
        self.k_squared = self.kx ** 2.0

        if imag_time:
            # Convention: $e^{-iH} = e^{UH}$
            self.U = -1.0
        else:
            self.U = -1.0j

        # Add attributes as soon as they are needed (e.g. for dimension 3, all besides the error are needed)
        if dim >= 2:
            self.y = np.linspace(-self.L, self.L, self.resolution)
            self.dy = float(2 * L / self.resolution)
            self.dky = float(np.pi / self.L)
            self.ky = np.concatenate((k_over_0, k_under_0), axis=0) * self.dky
            self.k_squared += self.ky ** 2.0
        if dim >= 3:
            self.z = np.linspace(-self.L, self.L, self.resolution)
            self.dz = float(2 * L / self.resolution)
            self.dkz = float(np.pi / self.L)
            self.kz = np.concatenate((k_over_0, k_under_0), axis=0) * self.dkz
            self.k_squared += self.kz ** 2.0
        if dim > 3:
            print("Spatial dimension over 3. This is not implemented.", file=sys.stderr)
            sys.exit(1)

        if dim == 1:
            self.psi_val = self.psi(self.x)
            self.V_val = self.V(self.x)
            self.psi_sol_val = self.psi_sol(self.x)
            self.H_kin = np.exp(self.U * (0.5 * self.k_squared) * self.dt)
            # Here we use half steps in real space, but will use it before and after H_kin with normal steps
            self.H_pot = np.exp(self.U * (self.V_val + self.g * np.abs(self.psi_val) ** 2.0) * (0.5 * self.dt))

        elif dim == 2:
            self.x_mesh, self.y_mesh, self.pos = functions.get_meshgrid(self.x, self.y)
            self.psi_val = self.psi(self.pos)
            self.V_val = self.V(self.pos)
            self.psi_sol_val = self.psi_sol(self.pos)

            # here np.multiply is needed to multiply a number with an 1D array
            self.H_kin = np.exp(np.multiply(self.U, (0.5 * self.k_squared)) * self.dt)

            # here np.multiply is needed two times to multiply a number with an 2D array
            # Here we use half steps in real space, but will use it before and after H_kin with normal steps
            self.H_pot = np.exp(np.multiply(self.U, (self.V_val + np.multiply(self.g, np.abs(self.psi_val)) ** 2.0))
                                * (0.5 * self.dt))

        elif dim == 3:
            self.psi_val = self.psi(self.x, self.y, self.z)
            self.V_val = self.V(self.x, self.y, self.z)


        # attributes for animation
        self.t = 0.0

        self.psi_line = None
        self.alpha_psi = 0.5

        self.V_line = None
        self.alpha_V = 0.5

        self.psi_sol_line = None

        self.psi_x_line = None
        self.psi_y_line = None
        self.psi_z_line = None

        self.V_z_line = None

    def get_norm(self):
        if self.dim == 1:
            psi_norm = np.sum(np.abs(self.psi_val) ** 2.0) * self.dx
        elif self.dim == 2:
            psi_norm = np.sum(np.abs(self.psi_val) ** 2.0) * self.dx * self.dy
        elif self.dim == 3:
            psi_norm = np.sum(np.abs(self.psi_val) ** 2.0) * self.dx * self.dy * self.dz
        else:
            print("Spatial dimension over 3. This is not implemented.", file=sys.stderr)
            sys.exit(1)

        return psi_norm

    def time_step(self):
        # H_kin is just dependend on U and the gridpoints, which are constants, so it does not need to be recaculated
        # update H_pot before use
        self.H_pot = np.exp(self.U * (self.V_val + self.g * np.abs(self.psi_val) ** 2.0) * (0.5 * self.dt))

        # multiply element-wise the 2D with each other (not np.multiply)
        self.psi_val = self.H_pot * self.psi_val

        self.psi_val = np.fft.fft(self.psi_val)

        # multiply element-wise the 2D with each other (not np.multiply)
        self.psi_val = self.H_kin * self.psi_val

        self.psi_val = np.fft.ifft(self.psi_val)

        # update H_pot before use
        self.H_pot = np.exp(self.U * (self.V_val + self.g * np.abs(self.psi_val) ** 2.0) * (0.5 * self.dt))

        # multiply element-wise the 2D with each other (not np.multiply)
        self.psi_val = self.H_pot * self.psi_val

        self.t += self.dt

        # for self.imag_time=False, renormalization should be preserved, but we play safe here (regardless of speedup)
        # if self.imag_time:
        psi_norm_after_evolution = self.get_norm()

        self.psi_val /= np.sqrt(psi_norm_after_evolution)

        # self.s = - np.log(self.get_norm()) / (2.0 * self.dt)
