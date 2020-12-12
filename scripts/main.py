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

import functions
import Schroedinger
import Animation

# Script runs, if script is run as main script (called by python *.py)
if __name__ == '__main__':
    x = symbols('x')

    L = 10
    datapoints_exponent: int = 6
    resolution = 2 ** datapoints_exponent

    V = 0.5 * x ** 2
    psi_0 = functools.partial(functions.psi_0_rect, x_min=-0.00, x_max=0.60, a=2)
    # psi_0 = functools.partial(functions.psi_0_pdf, L=10.0)
    Harmonic = Schroedinger.Schroedinger(resolution, L, timesteps=500, dx=(2*L/resolution), dk=(np.pi/L), dt=0.005,
        psi_0=psi_0, V=V, g=1, imag_time=True)

    ani = Animation.Animation()
    ani.set_limits(0, -L, L, 0, 2.5)
    # ani.set_limits_smart(0, Harmonic)
    ani.start(Harmonic)
