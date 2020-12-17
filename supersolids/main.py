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

from concurrent import futures

from supersolids import Animation
from supersolids import functions

# Script runs, if script is run as main script (called by python *.py)
if __name__ == '__main__':
    # for parallelization (use all cores)
    max_workers = psutil.cpu_count(logical=False)

    # due to fft of the points the resolution needs to be 2 ** datapoints_exponent
    datapoints_exponent: int = 6
    resolution: int = 2 ** datapoints_exponent

    # constants needed for the Schroedinger equation
    timesteps = 8000
    dt = 0.00005

    # box length [-L,L]
    # generators for L, g, dt to compute for different parameters
    L_generator = (10,)
    G = (i for i in range(100, 110, 10))
    # DT = (dt * 1.1 ** i for i in range(0, -maxworkers, -1))
    factors = np.linspace(0.2, 0.3, max_workers)
    DT = (i * dt for i in factors)
    cases = itertools.product(L_generator, G, DT)

    # functions needed for the Schroedinger equation (e.g. potential: V, initial wave function: psi_0)
    V = functions.v_harmonic

    # functools.partial sets all arguments except x, as multiple arguments for Schroedinger aren't implement yet
    # psi_0 = functools.partial(functions.psi_0_rect, x_min=-1.00, x_max=-0.50, a=2)
    psi_0 = functools.partial(functions.psi_gauss, a=4, x_0=0, k_0=0)

    # psi_0_3d = functools.partial(functions.psi_gauss_3d, a=4, x_0=0, y_0=0, z_0=0, k_0=0)
    L = 10
    psi_0_2d = functions.psi_gauss_2d(resolution, x_min=-L, x_max=L, y_min=-L, y_max=L,
                                      mu_x=0.0, mu_y=0.0, var_x=1.0, var_y=1.0)
    Animation.plot_2d(*psi_0_2d, L)

    i: int = 0
    with futures.ProcessPoolExecutor(max_workers=max_workers) as e:
        for L, g, dt in cases:
            i = i + 1
            print(f"i={i}, L={L}, g={g}, dt={dt}")
            file_name = f"split_{i:03}.mp4"
            e.submit(Animation.simulate_case, resolution, timesteps, L=L, g=g, dt=dt, imag_time=True,
                     psi_0=psi_0, V=V, file_name=file_name)
