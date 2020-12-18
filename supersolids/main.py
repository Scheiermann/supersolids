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
    timesteps = 200
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
    V_1d = functions.v_harmonic_1d
    V_2d = functions.v_harmonic_2d
    V_3d = functions.v_harmonic_3d

    # functools.partial sets all arguments except x, as multiple arguments for Schroedinger aren't implement yet
    # psi_0 = functools.partial(functions.psi_0_rect, x_min=-1.00, x_max=-0.50, a=2)
    psi_0_1d = functools.partial(functions.psi_gauss, a=4, x_0=0, k_0=0)
    L = 12
    psi_0_2d = functools.partial(functions.psi_gauss_2d, mu=np.array([0.0, 0.0]), var=np.array([1.0, 1.0]))
    psi_0_3d = functools.partial(functions.psi_gauss_3d, a=4, x_0=0, y_0=0, z_0=0, k_0=0)

    x = np.linspace(-L, L, resolution)
    y = np.linspace(-L, L, resolution)
    X, Y = np.meshgrid(x, y)
    # Animation.plot_2d(X, Y, functions.psi_gauss_2d(x, y), L)

    i: int = 0
    with futures.ProcessPoolExecutor(max_workers=1) as e:
        for L, g, dt in cases:
            i = i + 1
            print(f"i={i}, L={L}, g={g}, dt={dt}")
            file_name = f"split_{i:03}.mp4"
            e.submit(Animation.simulate_case, resolution, timesteps, L=L, g=g, dt=dt, imag_time=True,
                     psi_0=psi_0_2d, V=V_2d, dim=2, file_name=file_name)
