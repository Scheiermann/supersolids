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
from typing import Callable, Tuple

from supersolids import Animation
from supersolids import constants
from supersolids import functions
from supersolids import MayaviAnimation
from supersolids import run_time
from supersolids import Schroedinger


def simulate_case(resolution: int, max_timesteps: int, L: float, dt: float, g: float = 0.0, g_qf: float = 0.0,
                  epsilon_dd: float = 1.0, imag_time: bool = False, s: float = 1.1, E: float = 1.0,
                  dim: int = 1,
                  psi_0: Callable = functions.psi_gauss_3d,
                  V: Callable = functions.v_harmonic_3d,
                  V_interaction: Callable = None,
                  psi_sol: Callable = functions.thomas_fermi_3d,
                  mu_sol: Callable = functions.mu_3d,
                  plot_psi_sol: bool = False,
                  plot_V: bool = True,
                  psi_0_noise: Callable = functions.noise_mesh,
                  alpha_psi: float = 0.8,
                  alpha_psi_sol: float = 0.5,
                  alpha_V: float = 0.3,
                  accuracy: float = 10 ** -6,
                  filename: str = "split.mp4",
                  x_lim: Tuple[float, float] = (-1.0, 1.0),
                  y_lim: Tuple[float, float] = (-1.0, 1.0),
                  z_lim: Tuple[float, float] = (-1.0, 1.0),
                  slice_x_index: int = 0,
                  slice_y_index: int = 0,
                  slice_z_index: int = 0,
                  camera_r_func: Callable = None,
                  camera_phi_func: Callable = functools.partial(functions.camera_func_phi, phi_per_frame=20.0),
                  camera_z_func: Callable = None,
                  camera_r_0: float = 10.0,
                  camera_phi_0: float = 75.0,
                  camera_z_0: float = 20.0,
                  delete_input: bool = True) -> None:
    """
    Wrapper for Animation and Schroedinger to get a working Animation of a System
    through the equations given by Schroedinger.

    Parameters
    ----------
    resolution : int
                 number of grid points in one direction

    max_timesteps : int
        Maximum timesteps  with length dt for the animation.

    accuracy : float
        Convergence is reached when relative error of s ios smaller than accuracy,
        where s is System.s = - np.log(psi_norm_after_evolution) / (2.0 * self.dt)

    plot_psi_sol :
        Condition if psi_sol should be plotted.

    plot_V : bool
        Condition if V should be plotted.

    x_lim : Tuple[float, float]
        Limits of plot in x direction

    y_lim : Tuple[float, float]
        Limits of plot in y direction

    z_lim : Tuple[float, float]
        Limits of plot in z direction

    alpha_psi : float
        Alpha value for plot transparency of psi

    alpha_psi_sol : float
        Alpha value for plot transparency of psi_sol

    alpha_V : float
        Alpha value for plot transparency of V

    filename : str
        Filename with filetype to save the movie to

    slice_x_index : int
        Index of grid point in x direction to produce a slice/plane in mayavi,
        where psi_prob = |psi| ** 2 is used for the slice

    slice_y_index : int
        Index of grid point in y direction to produce a slice/plane in mayavi,
        where psi_prob = |psi| ** 2 is used for the slice

    camera_r_func : Callable or None
        r component of the movement of the camera.

    camera_phi_func : Callable or None
        phi component of the movement of the camera.

    camera_z_func : Callable or None
        z component of the movement of the camera.

    camera_r_0 : float
        r component of the starting point of the camera movement.

    camera_phi_0 :
        phi component of the starting point of the camera movement.

    camera_z_0 :
        z component of the starting point of the camera movement.

    delete_input : bool
        Condition if the input pictures should be deleted, after creation the creation of the animation as e.g. mp4

    Returns
    -------
    """
    with run_time.run_time():
        Harmonic = Schroedinger.Schroedinger(resolution, max_timesteps, L, dt, g=g, g_qf=g_qf, epsilon_dd=epsilon_dd,
                                             imag_time=imag_time,
                                             mu=s, E=E,
                                             dim=dim,
                                             psi_0=psi_0,
                                             V=V,
                                             V_interaction=V_interaction,
                                             psi_sol=psi_sol,
                                             mu_sol=mu_sol,
                                             psi_0_noise=psi_0_noise,
                                             alpha_psi=alpha_psi,
                                             alpha_psi_sol=alpha_psi_sol,
                                             alpha_V=alpha_V
                                             )
    if dim < 3:
        # matplotlib for 1D and 2D
        ani = Animation.Animation(dim=dim,
                                  camera_r_func=camera_r_func,
                                  camera_phi_func=camera_phi_func,
                                  camera_z_func=camera_z_func,
                                  camera_r_0=camera_r_0,
                                  camera_phi_0=camera_phi_0,
                                  camera_z_0=camera_z_0
                                  )

        if ani.dim == 1:
            ani.set_limits(0, 0, *x_lim, *y_lim)
        elif ani.dim == 2:
            ani.ax.set_xlim(*x_lim)
            ani.ax.set_ylim(*y_lim)
            ani.ax.set_zlim(*z_lim)

        # ani.set_limits_smart(0, Harmonic)

        with run_time.run_time():
            ani.start(Harmonic, filename, accuracy=accuracy, plot_psi_sol=plot_psi_sol, plot_V=plot_V)
    else:
        # mayavi for 3D
        may = MayaviAnimation.MayaviAnimation(dim=3)
        with run_time.run_time():
            may.animate(Harmonic, accuracy=accuracy, plot_V=plot_V, plot_psi_sol=plot_V,
                        x_lim=x_lim, y_lim=y_lim, z_lim=z_lim,
                        slice_x_index=slice_x_index, slice_y_index=slice_y_index, slice_z_index=slice_z_index
                        )
        mlab.show()
        # TODO: close window after last frame
        # print(f"{Harmonic.t}, {Harmonic.dt * Harmonic.max_timesteps}")
        # if Harmonic.t >= Harmonic.dt * Harmonic.max_timesteps:
        #     mlab.close()
        may.create_movie(input_data_file_pattern="*.png", filename=filename, delete_input=delete_input)


# Script runs, if script is run as main script (called by python *.py)
if __name__ == "__main__":
    # for parallelization (use all cores)
    max_workers = psutil.cpu_count(logical=False)

    # due to fft of the points the resolution needs to be 2 ** datapoints_exponent
    datapoints_exponent: int = 6
    resolution: int = 2 ** datapoints_exponent

    # constants needed for the Schroedinger equation
    # box length in 1D: [-L,L], in 2D: [-L,L, -L,L], , in 3D: [-L,L, -L,L, -L,L]
    L = 8
    dt: float = 0.001
    N: int = 4 * 10 ** 4
    m: float = 164.0 * constants.u_in_kg
    a_s: float = 85.0 * constants.a_0
    a_dd: float = 130.0 * constants.a_0

    w_x: float = 2.0 * np.pi * 30.0
    w_y: float = 2.0 * np.pi * 60.0
    w_z: float = 2.0 * np.pi * 140.0

    alpha_y, alpha_z = functions.get_alphas(w_x=w_x, w_y=w_y, w_z=w_z)
    g, g_qf, epsilon_dd = functions.get_parameters(N=N, m=m, a_s=a_s, a_dd=a_dd, w_x=w_x)
    print(f"g, g_qf, epsilon_dd, alpha_y, alpha_z: {g, g_qf, epsilon_dd, alpha_y, alpha_z}")

    # functions needed for the Schroedinger equation (e.g. potential: V, initial wave function: psi_0)
    V_1d = functions.v_harmonic_1d
    V_2d = functools.partial(functions.v_harmonic_2d, alpha_y=alpha_y)
    V_3d = functools.partial(functions.v_harmonic_3d, alpha_y=alpha_y, alpha_z=alpha_z)

    V_3d_ddi = functools.partial(functions.dipol_dipol_interaction)

    # functools.partial sets all arguments except x, y, z, as multiple arguments for Schroedinger aren't implement yet
    # psi_0 = functools.partial(functions.psi_0_rect, x_min=-0.25, x_max=-0.25, a=2.0)
    psi_0_1d = functools.partial(functions.psi_gauss_1d, a=3.0, x_0=2.0, k_0=0.0)
    psi_0_2d = functools.partial(functions.psi_gauss_2d_pdf, mu=[0.0, 0.0], var=np.array([[1.0, 0.0], [0.0, 1.0]]))
    psi_0_3d = functools.partial(functions.psi_gauss_3d, a=3.0, x_0=0.0, y_0=0.0, z_0=0.0, k_0=0.0)

    # Used to remember that 2D need the special pos function (g is set inside of Schoerdinger for convenicence)
    psi_sol_1d = functions.thomas_fermi_1d
    psi_sol_2d = functions.thomas_fermi_2d_pos
    psi_sol_3d = functions.thomas_fermi_3d

    # TODO: get mayavi lim to work
    # 3D works in single core mode
    simulate_case(resolution, max_timesteps=800, L=L, dt=dt, g=g, g_qf=g_qf, epsilon_dd=epsilon_dd, imag_time=True,
                  s=1.1, E=1.0,
                  dim=3,
                  psi_0=psi_0_3d,
                  V=V_3d,
                  V_interaction=V_3d_ddi,
                  psi_sol=psi_sol_3d,
                  mu_sol=functions.mu_3d,
                  plot_psi_sol=False,
                  plot_V=False,
                  psi_0_noise=functions.noise_mesh(min=0.8, max=1.4, shape=(resolution, resolution, resolution)),
                  alpha_psi=0.8,
                  alpha_psi_sol=0.5,
                  alpha_V=0.3,
                  accuracy=10 ** -7,
                  filename="anim.mp4",
                  x_lim=(-2.0, 2.0), y_lim=(-2.0, 2.0), z_lim=(0, 0.5),
                  slice_x_index=resolution // 3,  # just for mayavi (3D)
                  slice_y_index=resolution // 3,
                  slice_z_index=resolution // 3,
                  camera_r_func=functools.partial(functions.camera_func_r, r_0=10.0, r_per_frame=0.0),  # camera just 2D
                  camera_phi_func=functools.partial(functions.camera_func_phi, phi_0=45.0, phi_per_frame=10.0),
                  camera_z_func=functools.partial(functions.camera_func_r, r_0=20.0, r_per_frame=0.0),
                  camera_r_0=10.0,
                  camera_phi_0=45.0,
                  camera_z_0=20.0,
                  delete_input=True
                  )
    print("Single core done")

    # # TODO: As g is proportional to N * a_s/w_x, changing g, means V, g_qf are different (maybe other variables too)
    # # Creating multiple cases (Schroedinger with different parameters) for multi-core
    # # box length in 1D: [-L,L], in 2D: [-L,L, -L,L], , in 3D: [-L,L, -L,L, -L,L]
    # # generators for L, g, dt to compute for different parameters
    # g_step = 10
    # L_generator = (4,)
    # g_generator = (i for i in np.arange(g, g + g_step, g_step))
    # factors = np.linspace(0.2, 0.3, max_workers)
    # dt_generator = (i * dt for i in factors)
    # cases = itertools.product(L_generator, g_generator, dt_generator)

    # TODO: get mayavi concurrent to work (problem with mlab.figure())
    # i: int = 0
    # with futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
    #     for L, g, dt in cases:
    #         i = i + 1
    #         print(f"i={i}, L={L}, g={g}, dt={dt}")
    #         file_name = f"split_{i:03}.mp4"
    #         executor.submit(simulate_case, resolution, max_timesteps=30, L=L, g=g, dt=dt, imag_time=True,
    #                         s=1.1, E=1.0,
    #                         dim=2,
    #                         psi_0=psi_0_2d,
    #                         V=V_2d,
    #                         psi_sol=psi_sol_2d,
    #                         mu_sol=functions.mu_2d,
    #                         alpha_psi=0.8,
    #                         alpha_V=0.3,
    #                         file_name=file_name,
    #                         accuracy=10 ** -6,
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
