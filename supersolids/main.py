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
from supersolids import parallel
from supersolids import Schroedinger


def simulate_case(resolution, timesteps, L, dt, g, imag_time=False, dim=1, s=1,
                  psi_0=functions.psi_gauss_1d,
                  V=functions.v_harmonic_1d,
                  psi_sol=functions.thomas_fermi,
                  file_name="split.mp4",
                  x_lim=(-1, 1),
                  y_lim=(-1, 1),
                  z_lim=(0, 1.0)
                  ):

    with parallel.run_time():
        Harmonic = Schroedinger.Schroedinger(resolution, timesteps, L, dt, g=g, imag_time=imag_time, dim=dim, s=s,
                                             psi_0=psi_0, V=V,
                                             psi_sol=psi_sol,
                                             )

    ani = Animation.Animation(dim=dim)

    if ani.dim == 1:
        ani.set_limits(0, 0, *x_lim, *y_lim)
    else:
        ani.ax.set_xlim(*x_lim)
        ani.ax.set_ylim(*y_lim)
        ani.ax.set_zlim(*z_lim)
        ani.ax.view_init(20.0, 45.0)
        ani.ax.dist = 10.0

    # ani.set_limits_smart(0, Harmonic)

    with parallel.run_time():
        ani.start(Harmonic, file_name)


# Script runs, if script is run as main script (called by python *.py)
if __name__ == '__main__':
    # for parallelization (use all cores)
    max_workers = psutil.cpu_count(logical=False)

    # due to fft of the points the resolution needs to be 2 ** datapoints_exponent
    datapoints_exponent: int = 6
    resolution: int = 2 ** datapoints_exponent

    # constants needed for the Schroedinger equation
    g = 1.0
    g_step = 10
    dt = 0.5

    # box length [-L,L]
    # generators for L, g, dt to compute for different parameters
    L_generator = (10,)
    G = (i for i in np.arange(g, g + g_step, g_step))
    factors = np.linspace(0.2, 0.3, max_workers)
    DT = (i * dt for i in factors)
    cases = itertools.product(L_generator, G, DT)

    # functions needed for the Schroedinger equation (e.g. potential: V, initial wave function: psi_0)
    V_1d = functions.v_harmonic_1d
    V_2d = functions.v_harmonic_2d
    V_3d = functions.v_harmonic_3d

    # functools.partial sets all arguments except x, as multiple arguments for Schroedinger aren't implement yet
    # psi_0 = functools.partial(functions.psi_0_rect, x_min=-1.00, x_max=-0.50, a=2)
    psi_0_1d = functools.partial(functions.psi_gauss_1d, a=1, x_0=0, k_0=0)
    psi_0_2d = functools.partial(functions.psi_gauss_2d, mu=np.array([1.0, 0.0]), var=np.array([1.0, 1.0]))
    psi_0_3d = functools.partial(functions.psi_gauss_3d, a=1, x_0=0, y_0=0, z_0=0, k_0=0)

    i: int = 0
    with futures.ProcessPoolExecutor(max_workers=max_workers) as e:
        for L, g, dt in cases:
            i = i + 1
            print(f"i={i}, L={L}, g={g}, dt={dt}")
            file_name = f"split_{i:03}.mp4"
            psi_sol = functools.partial(functions.thomas_fermi, g=g)
            e.submit(simulate_case, resolution, timesteps=100, L=L, g=g, dt=dt, imag_time=True, dim=2, s=1,
                     psi_0=psi_0_2d, V=V_2d, psi_sol=psi_sol, file_name=file_name,
                     x_lim=(-2, 2),
                     y_lim=(-2, 2),
                     z_lim=(0, 0.025)
                     )
