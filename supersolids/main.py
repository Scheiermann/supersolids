#!/usr/bin/env python

"""
Animation for the numerical solver for non-linear time-dependent Schrodinger's equation.

author: Daniel Scheiermann
email: daniel.scheiermann@stud.uni-hannover.de
license: MIT
Please feel free to use and modify this, but keep the above information. Thanks!
"""

import functools
import numpy as np
from sympy import symbols

import Animation
import functions
import parallel
import Schroedinger

# Script runs, if script is run as main script (called by python *.py)
if __name__ == '__main__':
    # constants needed for the Schroedinger equation
    # box length [-L,L]
    L = 10
    # due to fft of the points the resolution needs to be 2 ** datapoints_exponent
    datapoints_exponent: int = 6
    resolution: int = 2 ** datapoints_exponent

    # functions needed for the Schroedinger equation (e.g. potential: V, initial wave function: psi_0)
    x = symbols('x')
    V = 0.5 * x ** 2
    # functools.partial sets all arguments except x, as multiple arguments for Schroedinger aren't implement yet
    # psi_0 = functools.partial(functions.psi_0_rect, x_min=-1.00, x_max=-0.50, a=2)
    # psi_0 = functools.partial(functions.psi_0_pdf, loc=0.0, scale=1.0)
    psi_0 = functools.partial(functions.psi_0_gauss, a=4, x_0=0, k_0=0)

    with parallel.run_time():
        Harmonic = Schroedinger.Schroedinger(resolution, L, timesteps=500, dx=(2*L/resolution), dk=(np.pi/L),
            dt=0.00005, psi_0=psi_0, V=V, g=100.0, imag_time=True)

    ani = Animation.Animation()
    ani.set_limits(0, 0, -L, L, 0, 0.2)
    # ani.set_limits_smart(0, Harmonic)

    with parallel.run_time():
        ani.start(Harmonic)