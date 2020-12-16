#!/usr/bin/env python

"""
Animation for the numerical solver for non-linear time-dependent Schrodinger's equation.

author: Daniel Scheiermann
email: daniel.scheiermann@stud.uni-hannover.de
license: MIT
Please feel free to use and modify this, but keep the above information. Thanks!
"""

import itertools
import functools
import numpy as np
from sympy import symbols

from supersolids import Animation
from supersolids import functions
from supersolids import parallel
from supersolids import Schroedinger

# Script runs, if script is run as main script (called by python *.py)
if __name__ == '__main__':
    # due to fft of the points the resolution needs to be 2 ** datapoints_exponent
    datapoints_exponent: int = 6
    resolution: int = 2 ** datapoints_exponent

    # constants needed for the Schroedinger equation
    # box length [-L,L]
    L = 10
    timesteps = 10
    dx = (2 * L / resolution)
    dk = (np.pi / L)
    dt = 0.00005
    g = 100.0

    # functions needed for the Schroedinger equation (e.g. potential: V, initial wave function: psi_0)
    # x = symbols('x')
    # V = functions.v_harmonic(x)

    # functools.partial sets all arguments except x, as multiple arguments for Schroedinger aren't implement yet
    # psi_0 = functools.partial(functions.psi_0_rect, x_min=-1.00, x_max=-0.50, a=2)
    # psi_0 = functools.partial(functions.psi_0_pdf, loc=0.0, scale=1.0)
    # psi_0 = functools.partial(functions.psi_0_gauss, a=4, x_0=0, k_0=0)

    # L = (5, 8, 10)
    G = (i for i in range(1, 100, 10))
    DT = (0.00005 * 10 ** i for i in range(1, 3, 1))
    # cases = itertools.product(L, G, DT)
    #
    # for i, case in enumerate(cases):
    #     print(f"i: {i}, case = {case}")

    Animation.simulate_case(resolution, timesteps, dx, dk, L, dt, g, imag_time=False,
                            psi_0=functions.psi_0_pdf, V=functions.v_harmonic)
