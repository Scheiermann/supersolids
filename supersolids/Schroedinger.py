#!/usr/bin/env python

"""
Numerical solver for non-linear time-dependent Schrodinger's equation.

author: Daniel Scheiermann
email: daniel.scheiermann@stud.uni-hannover.de
license: MIT
Please feel free to use and modify this, but keep the above information. Thanks!
"""

import numpy as np
import scipy as sp
from sympy import Symbol, lambdify


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
    WARNING: Normalization of $\psi$ at every step needs to be checked, but is NOT implemented.
    """

    def __init__(self, resolution, L, timesteps, dx, dk, dt, psi_0=None, V=None, g=1, imag_time=False):
        """
        Parameters
        ----------
        x : array_like, float
            description
        """
        self.resolution = resolution
        self.L = L
        self.timesteps = timesteps
        self.dx = dx
        self.dk = dk
        self.dt = dt
        self.g = g
        self.imag_time = imag_time

        self.x = np.linspace(-self.L, self.L, self.resolution)
        k_over_0 = np.arange(0, resolution / 2, 1)
        k_under_0 = np.arange(-resolution / 2, 0, 1)

        self.k = np.concatenate((k_over_0, k_under_0), axis=0) * (np.pi / L)

        if imag_time:
            # Convention: $e^{-iH} = e^{UH}$
            self.U = -1
        else:
            self.U = -1.0j

        x_real = Symbol('x', real=True)

        # Makes V callable, if it is not
        if callable(V):
            self.V = V
        else:
            self.V = lambdify(x_real, V, "numpy")

        # Gets values of psi even if its callable
        if callable(psi_0):
            self.psi = psi_0(self.x)
        else:
            self.psi = psi_0

        self.H_kin = np.exp(self.U * (-0.5 * self.k ** 2) * self.dt)

        # Here we use half steps in real space, but will use it before and after H_kin with normal steps
        self.H_pot = np.exp(self.U * (self.V(self.x) + self.g * np.abs(self.psi) ** 2) * (0.5 * self.dt))

        self.t = 0.0
        self.psi_x_line = None
        self.psi_k_line = None
        self.V_x_line = None

    def time_step(self):
        self.psi = self.H_pot * self.psi
        self.psi = sp.fft.fft(self.psi)
        self.psi = self.H_kin * self.psi
        self.psi = sp.fft.ifft(self.psi)
        self.psi = self.H_pot * self.psi

        self.t += self.dt

        # for self.imag_time=False, renormalization should be preserved, but we play safe here (regardless of speedup)
        # if self.imag_time:
        psi_norm = np.sum(np.abs(self.psi) ** 2) * self.dx
        self.psi /= np.sqrt(psi_norm)

