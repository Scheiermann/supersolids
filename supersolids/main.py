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


def simulate_case(resolution, timesteps, L, g, dt, imag_time=False, s=1.1, E=1.0, accuracy=10**-6,
                  dim=1,
                  psi_0=functions.psi_gauss_3d,
                  V=functions.v_harmonic_3d,
                  V_interaction=None,
                  psi_sol=functions.thomas_fermi_3d,
                  mu_sol=functions.mu_3d,
                  alpha_psi=0.8,
                  alpha_psi_sol=0.5,
                  alpha_V=0.3,
                  file_name="split.mp4",
                  x_lim=(-1.0, 1.0),
                  y_lim=(-1.0, 1.0),
                  z_lim=(-1.0, 1.0),
                  slice_x_index=0,
                  slice_y_index=0,
                  r_func=None,
                  phi_func=functools.partial(functions.camera_func_phi, phi_per_frame=20.0),
                  z_func=None,
                  camera_r=10.0,
                  camera_phi=75.0,
                  camera_z=20.0,
                  delete_input=True):
    with run_time.run_time():
        Harmonic = Schroedinger.Schroedinger(resolution, timesteps, L, dt, g=g, imag_time=imag_time,
                                             s=s, E=E,
                                             dim=dim,
                                             psi_0=psi_0,
                                             V=V,
                                             V_interaction=V_interaction,
                                             psi_sol=psi_sol,
                                             mu_sol=mu_sol,
                                             alpha_psi=alpha_psi,
                                             alpha_V=alpha_V
                                             )
    if dim < 3:
        # matplotlib for 1D and 2D
        ani = Animation.Animation(dim=dim,
                                  r_func=r_func,
                                  phi_func=phi_func,
                                  z_func=z_func,
                                  camera_r=camera_r,
                                  camera_phi=camera_phi,
                                  camera_z=camera_z)

        if ani.dim == 1:
            ani.set_limits(0, 0, *x_lim, *y_lim)
        elif ani.dim == 2:
            ani.ax.set_xlim(*x_lim)
            ani.ax.set_ylim(*y_lim)
            ani.ax.set_zlim(*z_lim)

        # ani.set_limits_smart(0, Harmonic)

        with run_time.run_time():
            ani.start(Harmonic, file_name)
    else:
        # mayavi for 3D
        may = MayaviAnimation.MayaviAnimation(dim=3)
        with run_time.run_time():
            # may.animate(Harmonic)
            may.animate(Harmonic, accuracy=accuracy, x_lim=x_lim, y_lim=y_lim, z_lim=z_lim,
                        slice_x_index=slice_x_index, slice_y_index=slice_y_index)
        mlab.show()
        # TODO: close window after last frame
        # print(f"{Harmonic.t}, {Harmonic.dt * Harmonic.timesteps}")
        # if Harmonic.t >= Harmonic.dt * Harmonic.timesteps:
        #     mlab.close()
        may.create_movie(input_data_file_pattern="*.png", filename=file_name, delete_input=delete_input)


# Script runs, if script is run as main script (called by python *.py)
if __name__ == "__main__":
    # for parallelization (use all cores)
    max_workers = psutil.cpu_count(logical=False)

    # due to fft of the points the resolution needs to be 2 ** datapoints_exponent
    datapoints_exponent: int = 6
    resolution: int = 2 ** datapoints_exponent

    # constants needed for the Schroedinger equation
    g = 100.0
    g_step = 10
    dt = 0.001

    # box length in 1D: [-L,L], in 2D: [-L,L, -L,L], , in 3D: [-L,L, -L,L, -L,L]
    # generators for L, g, dt to compute for different parameters
    L_generator = (4,)
    g_generator = (i for i in np.arange(g, g + g_step, g_step))
    factors = np.linspace(0.2, 0.3, max_workers)
    dt_generator = (i * dt for i in factors)
    cases = itertools.product(L_generator, g_generator, dt_generator)

    # functions needed for the Schroedinger equation (e.g. potential: V, initial wave function: psi_0)
    V_1d = functions.v_harmonic_1d
    V_2d = functools.partial(functions.v_harmonic_2d, alpha_y=1.0)
    V_3d = functools.partial(functions.v_harmonic_3d, alpha_y=1.0, alpha_z=1.0)

    V_3d_ddi = functools.partial(functions.dipol_dipol_interaction, d=1.0, epsilon_dd=1.0)

    # functools.partial sets all arguments except x, y, z, as multiple arguments for Schroedinger aren't implement yet
    # psi_0 = functools.partial(functions.psi_0_rect, x_min=-0.25, x_max=-0.25, a=2.0)
    psi_0_1d = functools.partial(functions.psi_gauss_1d, a=3.0, x_0=2.0, k_0=0.0)
    psi_0_2d = functools.partial(functions.psi_gauss_2d_pdf, mu=[1.0, 0.0], var=np.array([[1.0, 0.0], [0.0, 1.0]]))
    psi_0_3d = functools.partial(functions.psi_gauss_3d, a=1.0, x_0=0.0, y_0=0.0, z_0=0.0, k_0=0.0)

    # Used to remember that 2D need the special pos function (g is set inside of Schoerdinger for convenicence)
    psi_sol_1d = functions.thomas_fermi_1d
    psi_sol_2d = functions.thomas_fermi_2d_pos
    psi_sol_3d = functions.thomas_fermi_3d

    # TODO: get mayavi lim to work
    # 3D works in single core mode
    simulate_case(resolution, timesteps=800, L=L_generator[0], g=g, dt=dt, imag_time=True,
                  s=1.1, E=1.0,
                  dim=3,
                  psi_0=psi_0_3d,
                  V=V_3d,
                  V_interaction=V_3d_ddi,
                  psi_sol=psi_sol_3d,
                  mu_sol=functions.mu_3d,
                  accuracy=10**-8,
                  alpha_psi=0.8,
                  alpha_psi_sol=0.5,
                  alpha_V=0.3,
                  file_name="anim.mp4",
                  x_lim=(-2.0, 2.0), y_lim=(-2.0, 2.0), z_lim=(0, 0.5),
                  slice_x_index=resolution//3, slice_y_index=resolution//3,  # just for mayavi (3D)
                  r_func=functools.partial(functions.camera_func_r, r_0=10.0, r_per_frame=0.0), # from here just 2D
                  phi_func=functools.partial(functions.camera_func_phi, phi_0=45.0, phi_per_frame=10.0),
                  z_func=functools.partial(functions.camera_func_r, r_0=20.0, r_per_frame=0.0),
                  camera_r=10.0,
                  camera_phi=45.0,
                  camera_z=20.0,
                  delete_input=True
                 )
    print("Single core done")

    # TODO: get mayavi concurrent to work (problem with mlab.figure())
    # i: int = 0
    # with futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
    #     for L, g, dt in cases:
    #         i = i + 1
    #         print(f"i={i}, L={L}, g={g}, dt={dt}")
    #         file_name = f"split_{i:03}.mp4"
    #         executor.submit(simulate_case, resolution, timesteps=30, L=L, g=g, dt=dt, imag_time=True,
    #                         s=1.1, E=1.0,
    #                         dim=2,
    #                         psi_0=psi_0_2d,
    #                         V=V_2d,
    #                         psi_sol=psi_sol_2d,
    #                         mu_sol=functions.mu_2d,
    #                         accuracy=10 ** -6,
    #                         alpha_psi=0.8,
    #                         alpha_V=0.3,
    #                         file_name=file_name,
    #                         x_lim=(-2.0, 2.0), y_lim=(-2.0, 2.0), z_lim=(0, 0.5),
    #                         slice_x_index=resolution // 2, slice_y_index=resolution // 2,  # just for mayavi (3D)
    #                         r_func=functools.partial(functions.camera_func_r, r_0=10.0, r_per_frame=0.0),
    #                         phi_func=functools.partial(functions.camera_func_phi, phi_0=45.0, phi_per_frame=10.0),
    #                         z_func=functools.partial(functions.camera_func_r, r_0=20.0, r_per_frame=0.0),
    #                         camera_r=10.0,
    #                         camera_phi=45.0,
    #                         camera_z=20.0,
    #                         delete_input=True
    #                         )
