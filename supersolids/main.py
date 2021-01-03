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
from concurrent import futures
import psutil

import numpy as np
from mayavi import mlab

from supersolids import Animation
from supersolids import functions
from supersolids import MayaviAnimation
from supersolids import run_time
from supersolids import Schroedinger


def simulate_case(resolution, timesteps, L, g, dt, imag_time=False, dim=1, s=1.1, E=1.0, accuracy=10**-6,
                  psi_0=functions.psi_gauss_1d,
                  V=functions.v_harmonic_1d,
                  psi_sol=functions.thomas_fermi,
                  mu_sol=functions.mu_3d,
                  file_name="split.mp4",
                  x_lim=(-1, 1),
                  y_lim=(-1, 1),
                  z_lim=(0, 1.0),
                  slice_x_index=0, slice_y_index=0,
                  view_height=20.0,
                  view_angle=45.0,
                  view_distance=10.0
                  ):
    with run_time.run_time():
        Harmonic = Schroedinger.Schroedinger(resolution, timesteps, L, dt, g=g, imag_time=imag_time, dim=dim,
                                             s=s, E=E,
                                             psi_0=psi_0, V=V,
                                             psi_sol=psi_sol,
                                             mu_sol=mu_sol,
                                             )

    if dim < 3:
        # matplotlib for 1D and 2D
        ani = Animation.Animation(dim=dim)

        if ani.dim == 1:
            ani.set_limits(0, 0, *x_lim, *y_lim)
        elif ani.dim == 2:
            ani.ax.set_xlim(*x_lim)
            ani.ax.set_ylim(*y_lim)
            ani.ax.set_zlim(*z_lim)
            ani.ax.view_init(view_height, view_angle)
            ani.ax.dist = view_distance

        # ani.set_limits_smart(0, Harmonic)

        with run_time.run_time():
            ani.start(Harmonic, file_name)
    else:
        # mayavi for 3D
        may = MayaviAnimation.MayaviAnimation(dim=dim)
        with run_time.run_time():
            # may.animate(Harmonic)
            may.animate(Harmonic, accuracy=accuracy, x_lim=x_lim, y_lim=y_lim, z_lim=z_lim,
                        slice_x_index=slice_x_index, slice_y_index=slice_y_index)
        mlab.show()
        # TODO: close window after last frame
        # print(f"{Harmonic.t}, {Harmonic.dt * Harmonic.timesteps}")
        # if Harmonic.t >= Harmonic.dt * Harmonic.timesteps:
        #     mlab.close()
        may.create_movie(input_data_file_pattern="*.png", filename=file_name)


# Script runs, if script is run as main script (called by python *.py)
if __name__ == "__main__":
    # for parallelization (use all cores)
    max_workers = psutil.cpu_count(logical=False)

    # due to fft of the points the resolution needs to be 2 ** datapoints_exponent
    datapoints_exponent: int = 6
    resolution: int = 2 ** datapoints_exponent

    # constants needed for the Schroedinger equation
    g = 10.0
    g_step = 10
    dt = 0.1

    # box length [-L,L]
    # generators for L, g, dt to compute for different parameters
    L_generator = (4,)
    g_generator = (i for i in np.arange(g, g + g_step, g_step))
    mu_sol_list = [functions.mu_3d for g in g_generator]
    factors = np.linspace(0.2, 0.3, max_workers)
    dt_generator = (i * dt for i in factors)
    cases = itertools.product(L_generator, g_generator, dt_generator, mu_sol_list)

    # functions needed for the Schroedinger equation (e.g. potential: V, initial wave function: psi_0)
    V_1d = functions.v_harmonic_1d
    V_2d = functools.partial(functions.v_harmonic_2d, alpha_y=1.0)
    V_3d = functools.partial(functions.v_harmonic_3d, alpha_y=1.0, alpha_z=1.0)

    # functools.partial sets all arguments except x, as multiple arguments for Schroedinger aren't implement yet
    # psi_0 = functools.partial(functions.psi_0_rect, x_min=-1.00, x_max=-0.50, a=2)
    psi_0_1d = functools.partial(functions.psi_gauss_1d, a=1, x_0=0, k_0=0)
    psi_0_2d = functools.partial(functions.psi_gauss_2d_pdf, mu=[0.0, 0.0], var=np.array([[1.0, 0.0], [0.0, 1.0]]))
    psi_0_3d = functools.partial(functions.psi_gauss_3d, a=1, x_0=0, y_0=0, z_0=0, k_0=0)

    # TODO: get mayavi lim to work
    # 3D works in single core mode
    simulate_case(resolution, timesteps=500, L=L_generator[0], g=g, dt=dt, imag_time=True, dim=3,
                  s=1.1, E=1.0,
                  psi_0=psi_0_3d, V=V_3d,
                  psi_sol=functions.thomas_fermi, mu_sol=mu_sol_list[0],
                  accuracy=10 ** -6,
                  file_name="anim.mp4",
                  x_lim=(-2, 2), y_lim=(-2, 2), z_lim=(0, 0.4),
                  slice_x_index=resolution//3, slice_y_index=resolution//3,
                  view_height=15.0, view_angle=75.0, view_distance=10.0
                  )
    print("Single core done")

    # TODO: get mayavi concurrent to work
    i: int = 0
    with futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        for L, g, dt, mu_sol in cases:
            i = i + 1
            print(f"i={i}, L={L}, g={g}, dt={dt}, mu={mu_sol}")
            file_name = f"split_{i:03}.mp4"
            psi_sol = functools.partial(functions.thomas_fermi, g=g)
            executor.submit(simulate_case, resolution, timesteps=30, L=L, g=g, dt=dt, imag_time=True,
                            dim=3, s=1.1, accuracy=10**-6,
                            psi_0=psi_0_3d, V=V_3d,
                            psi_sol=psi_sol, mu_sol=mu_sol,
                            file_name=file_name,
                            x_lim=(-8, 8),
                            y_lim=(-5, 5),
                            z_lim=(0, 0.4),
                            slice_x_index=0, slice_y_index=0,
                            view_height=15.0,
                            view_angle=75.0,
                            view_distance=10.0
                            )
