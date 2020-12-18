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
import psutil
from sympy import symbols
from concurrent import futures

from supersolids import Animation
from supersolids import functions

# Script runs, if script is run as main script (called by python *.py)
if __name__ == '__main__':
    # due to fft of the points the resolution needs to be 2 ** datapoints_exponent
    datapoints_exponent: int = 6
    resolution: int = 2 ** datapoints_exponent

    # constants needed for the Schroedinger equation
    timesteps = 200
    dt = 0.05

    # box length [-L,L]
    # generators for L, g, dt to compute for different parameters
    L_generator = (10,)
    G = (i for i in range(100, 110, 10))
    DT = (dt * 1.1 ** i for i in range(0, -8, -1))
    cases = itertools.product(L_generator, G, DT)

    # functions needed for the Schroedinger equation (e.g. potential: V, initial wave function: psi_0)
    V = functions.v_harmonic

    # functools.partial sets all arguments except x, as multiple arguments for Schroedinger aren't implement yet
    # psi_0 = functools.partial(functions.psi_0_rect, x_min=-1.00, x_max=-0.50, a=2)
    psi_0 = functools.partial(functions.psi_0_gauss, a=1, x_0=1, k_0=0)

    i: int = 0
    with futures.ProcessPoolExecutor(max_workers=psutil.cpu_count(logical=False)) as e:
        for L, g, dt in cases:
            i = i + 1
            print(f"i={i}, L={L}, g={g}, dt={dt}")
            file_name = "split_{:03}.mp4".format(i)
            e.submit(Animation.simulate_case, resolution, timesteps, L=L, g=g, dt=dt, imag_time=True,
                     psi_0=psi_0, V=V, file_name=file_name)
