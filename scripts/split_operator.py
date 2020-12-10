#!/usr/bin/env python

"""
Numerical Solver for non-linear Time-Dependent Schrodinger's equation.

author: Scheiermann
email: daniel.scheiermann@stud.uni-hannover.de
license: MIT
Please feel free to use and modify this, but keep the above information. Thanks!
"""

import numpy as np
import scipy as sp
from scipy.stats import norm
from sympy import Symbol, symbols, Function, lambdify
from matplotlib import pyplot as plt
from matplotlib import animation


# Animate plot
def init():
    psi_x_line.set_data([], [])
    V_x_line.set_data([], [])

    title.set_text("")
    print(f"Init:\n {Harmonic.psi}")
    return (psi_x_line, V_x_line, title)


def animate(i):
    Harmonic.time_step()
    if i % 10 == 0:
        print(f"Round {i}")
    # print(f"Round {i}:\n {Harmonic.psi}")
    psi_x_line.set_data(Harmonic.x, np.abs(Harmonic.psi))
    V_x_line.set_data(Harmonic.x, Harmonic.V(Harmonic.x))

    title.set_text("t = %.2f" % Harmonic.t)
    return (psi_x_line, V_x_line, title)


class Schroedinger(object):
    """
    Implements a numerical solution of the time-dependent
    non-linear Schrodinger equation for an arbitrary potential:
    i \hbar \frac{\partial}{\partial t} \psi(r,t) = \left(\frac{-\hbar^{2}}{2 m} \nabla^{2}
                                                    + V(r) + g |\psi(x,t)|^{2} \right) \psi(x,t)

    For the momemnt we aim to solve:
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

        self.x = np.linspace(0, self.L, self.resolution)
        self.k = np.linspace(-self.L / 2, self.L / 2, self.resolution)

        if imag_time:
            self.U = -1
        else:
            self.U = 1.0j

        if psi_0 or V:
            x_real = Symbol('x', real=True)

        if V:
            self.V = lambdify(x_real, V, "numpy")
        else:
            V = 1 / 2 * (x - L / 2) ** 2
            self.V = lambdify(x_real, V, "numpy")

        self.psi = norm.pdf(self.x, loc=0.5 * L, scale=0.1)

        self.H_kin = np.exp(self.U * 0.5 * self.k ** 2 * self.dt)
        self.H_pot = np.exp(self.U * (self.V(self.x) + self.g * np.abs(self.psi)) * self.dt)

        self.t = 0.0
        self.psi_x_line = None
        self.psi_k_line = None
        self.V_x_line = None

    def time_step(self):
        # for i in np.arange(0, self.timesteps):
        self.psi = self.H_pot * self.psi
        self.psi = sp.fft.fft(self.psi)
        self.psi = self.H_kin * self.psi
        self.psi = sp.fft.ifft(self.psi)
        self.psi = self.H_pot * self.psi

        self.t += self.dt
        # print(f"Round {i}:\n {self.psi}")

        if self.imag_time:
            norm = np.sum(np.abs(self.psi)) * self.x
            self.psi /= norm
            print(f"norm: {norm}")
            print(f"Normed psi:\n {self.psi}")


if __name__ == '__main__':
    # Script runs, if script is run as main script (called by python *.py)
    x, y, z, t = symbols('x y z t')
    k, m = symbols('k m', integer=True)
    f, g, h = symbols('f g h', cls=Function)

    L = 10
    datapoints_exponent: int = 5
    resolution = 2 ** datapoints_exponent

    V = 1/2 * (x - L/2) ** 2
    # psi_0 = syp.exp(- 0.5 * (x - L/2) ** 2)
    Harmonic = Schroedinger(resolution, L, timesteps=1000, dx=(2*L/resolution), dk=(np.pi/L), dt=0.01,
                            V=V, imag_time=False)

    ######################################################################
    fig = plt.figure()

    xlim = (0, L)
    ymin = abs(Harmonic.psi).min()
    ymax = abs(Harmonic.psi).max()

    ax1 = fig.add_subplot(111, xlim=xlim,
                          ylim=(ymin - 0.2 * (ymax - ymin),
                                ymax + 0.2 * (ymax - ymin)))
    psi_x_line, = ax1.plot([], [], "x--", c="r", label=r'$|\psi(x)|$')
    V_x_line, = ax1.plot([], [], ".-", c="k", label=r'$V(x)$')

    title = ax1.set_title("")
    ax1.legend(prop=dict(size=12))
    ax1.set_xlabel('$x$')
    ax1.set_ylabel(r'$|\psi(x)|$')
    ax1.grid()

    V_x_line.set_data(Harmonic.x, Harmonic.V(Harmonic.x))

    # call the animator. blit=True means only re-draw the parts that have changed.
    anim = animation.FuncAnimation(fig, animate, init_func=init, frames=Harmonic.timesteps,
                                   interval=30, blit=True)

    # requires either mencoder or ffmpeg to be installed on your system
    anim.save('split.mp4', fps=15, extra_args=['-vcodec', 'libx264'])

