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
    psi_line.set_data([], [])
    V_line.set_data([], [])

    title.set_text("")
    print(f"Init:\n {Harmonic.psi}")
    return (psi_line, V_line, title)


def animate(i):
    Harmonic.time_step()
    if i % 10 == 0:
        print(f"Round {i}")
    # print(f"Round {i}:\n {Harmonic.psi}")
    psi_line.set_data(Harmonic.x, np.abs(Harmonic.psi) ** 2)
    x_V = np.linspace(-0.2 * L, 0.2 * L, resolution)
    V_line.set_data(x_V, Harmonic.V(x_V))

    title.set_text("t = %.2f" % Harmonic.t)
    return (psi_line, V_line, title)


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

        self.x = np.linspace(-self.L, self.L, self.resolution)
        k_over_0 = np.arange(0, resolution/2, 1)
        k_under_0 = np.arange(-resolution/2, 0, 1)
        self.k = np.concatenate((k_over_0, k_under_0), axis=0) * (np.pi / L)

        if imag_time:
            # Convention: $e^{-iH} = e^{UH}$
            self.U = -1
        else:
            self.U = -1.0j

        if psi_0 or V:
            x_real = Symbol('x', real=True)

        if V:
            self.V = lambdify(x_real, V, "numpy")
        else:
            V = 1 / 2 * x ** 2
            self.V = lambdify(x_real, V, "numpy")

        self.psi = norm.pdf(self.x, loc=-0.1 * L, scale=1.0)

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

        # if self.imag_time:
        norm = np.sum(np.abs(self.psi) ** 2) * self.dx
        self.psi /= np.sqrt(norm)
        # print(f"norm: {norm}")
        # print(f"Normed psi:\n {self.psi}")


if __name__ == '__main__':
    # Script runs, if script is run as main script (called by python *.py)
    x, y, z, t = symbols('x y z t')
    k, m = symbols('k m', integer=True)
    f, g, h = symbols('f g h', cls=Function)

    L = 10
    datapoints_exponent: int = 5
    resolution = 2 ** datapoints_exponent

    V = 0.5 * x ** 2
    # psi_0 = syp.exp(- 0.5 * x ** 2)
    Harmonic = Schroedinger(resolution, L, timesteps=300, dx=(2*L/resolution), dk=(np.pi/L), dt=0.05,
                            V=V, g=0, imag_time=False)

    ######################################################################
    fig = plt.figure()
    xlim = (-L, L)
    psi_abs = np.abs(Harmonic.psi)
    psi_prob = psi_abs ** 2
    psi_abs_max = psi_abs.max()
    psi_prob_max = psi_prob.max()
    if psi_prob_max < psi_abs_max:
        ymin = psi_abs.min()
        ymax = psi_abs_max
    else:
        ymin = psi_prob.min()
        ymax = psi_prob_max

    ax1 = fig.add_subplot(111, xlim=xlim,
                          ylim=(ymin - 0.2 * (ymax - ymin),
                                ymax + 0.2 * (ymax - ymin)))
    psi_line, = ax1.plot([], [], "x--", c="r", label=r'$|\psi(x)|^2$')
    V_line, = ax1.plot([], [], ".-", c="k", label=r'$V(x)$')

    title = ax1.set_title("")
    ax1.legend(prop=dict(size=12))
    ax1.set_xlabel('$x$')
    ax1.set_ylabel(r'$|\psi(x)|^2$')
    ax1.grid()

    x_V = np.linspace(-0.2 * L, 0.2 * L, resolution)
    V_line.set_data(x_V, Harmonic.V(x_V))

    # call the animator. blit=True means only re-draw the parts that have changed.
    anim = animation.FuncAnimation(fig, animate, init_func=init, frames=Harmonic.timesteps,
                                   interval=30, blit=True)

    # requires either mencoder or ffmpeg to be installed on your system
    anim.save('split.mp4', fps=15, extra_args=['-vcodec', 'libx264'])
