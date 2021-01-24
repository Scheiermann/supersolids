#!/usr/bin/env python

# author: Daniel Scheiermann
# email: daniel.scheiermann@stud.uni-hannover.de
# license: MIT
# Please feel free to use and modify this, but keep the above information. Thanks!

"""
Animation for the numerical solver for the non-linear
time-dependent Schrodinger equation.

"""

import itertools
import functools
from concurrent import futures
import psutil

import numpy as np
from mayavi import mlab
from typing import Callable, Tuple, NamedTuple
from matplotlib import pyplot as plt

from supersolids import Animation
from supersolids import constants
from supersolids import functions
from supersolids import MayaviAnimation
from supersolids import run_time
from supersolids import Schroedinger


def simulate_case(box: NamedTuple,
                  res: NamedTuple,
                  max_timesteps: int,
                  dt: float,
                  g: float = 0.0,
                  g_qf: float = 0.0,
                  e_dd: float = 1.0,
                  imag_time: bool = False,
                  mu: float = 1.1,
                  E: float = 1.0,
                  psi_0: Callable = functions.psi_gauss_3d,
                  V: Callable = functions.v_harmonic_3d,
                  V_interaction: Callable = None,
                  psi_sol: Callable = functions.thomas_fermi_3d,
                  mu_sol: Callable = functions.mu_3d,
                  plot_psi_sol: bool = False,
                  psi_sol_3d_cut_x: Callable = None,
                  psi_sol_3d_cut_z: Callable = None,
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
                  interactive: bool = True,
                  camera_r_func: Callable = None,
                  camera_phi_func: Callable = functools.partial(
                      functions.camera_func_phi, phi_per_frame=20.0),
                  camera_z_func: Callable = None,
                  delete_input: bool = True) -> None:
    """
    Wrapper for Animation and Schroedinger to get a working Animation
    of a System through the equations given by Schroedinger.

    Parameters

    box : NamedTuple
        Endpoints of box where to simulate the Schroedinger equation.
        Keyword x0 is minimum in x direction and x1 is maximum.
        Same for y and z. For 1D just use x0, x1.
        For 2D x0, x1, y0, y1.
        For 3D x0, x1, y0, y1, z0, z1.
        Dimension of simulation is constructed from this dictionary.

    res : NamedTuple
        NamedTuple for the number of grid points in x, y, z direction.
        Needs to have half size of box dictionary.
        Keywords x, y z are used.

    max_timesteps : int
        Maximum timesteps  with length dt for the animation.

    accuracy : float
        Convergence is reached when relative error of mu is smaller
        than accuracy, where :math:`\mu = - \\log(\psi_{normed}) / (2 dt)`

    plot_psi_sol :
        Condition if :math:`\psi_{sol}` should be plotted.

    plot_V : bool
        Condition if V should be plotted.

    x_lim : Tuple[float, float]
        Limits of plot in x direction

    y_lim : Tuple[float, float]
        Limits of plot in y direction

    z_lim : Tuple[float, float]
        Limits of plot in z direction

    alpha_psi : float
        Alpha value for plot transparency of :math:`\psi`

    alpha_psi_sol : float
        Alpha value for plot transparency of :math:`\psi_sol`

    alpha_V : float
        Alpha value for plot transparency of V

    filename : str
        Filename with filetype to save the movie to

    slice_x_index : int
        Index of grid point in x direction to produce a slice/plane in mayavi,
        where :math:`\psi_{prob} = |\psi|^2` is used for the slice

    slice_y_index : int
        Index of grid point in y direction to produce a slice/plane in mayavi,
        where :math:`\psi_{prob} = |\psi|^2` is used for the slice

    slice_z_index : int
        Index of grid point in z direction to produce a slice/plane in mayavi,
        where :math:`\psi_{prob} = |\psi|^2` is used for the slice

    interactive : bool
        Condition for interactive mode. When camera functions are used,
        then interaction is not possible. So interactive=True turn the usage
        of camera functions off.

    camera_r_func : Callable or None
        r component of the movement of the camera.

    camera_phi_func : Callable or None
        phi component of the movement of the camera.

    camera_z_func : Callable or None
        z component of the movement of the camera.

    delete_input : bool
        Condition if the input pictures should be deleted,
        after creation the creation of the animation as e.g. mp4

    Returns

    """
    with run_time.run_time():
        Harmonic = Schroedinger.Schroedinger(box,
                                             res,
                                             max_timesteps,
                                             dt,
                                             g=g,
                                             g_qf=g_qf,
                                             e_dd=e_dd,
                                             imag_time=imag_time,
                                             mu=mu, E=E,
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
    if Harmonic.dim < 3:
        # matplotlib for 1D and 2D
        ani = Animation.Animation(dim=Harmonic.dim,
                                  camera_r_func=camera_r_func,
                                  camera_phi_func=camera_phi_func,
                                  camera_z_func=camera_z_func,
                                  )

        if ani.dim == 1:
            ani.set_limits(0, 0, *x_lim, *y_lim)
        elif ani.dim == 2:
            ani.ax.set_xlim(*x_lim)
            ani.ax.set_ylim(*y_lim)
            ani.ax.set_zlim(*z_lim)

        # ani.set_limits_smart(0, Harmonic)

        with run_time.run_time():
            ani.start(
                Harmonic,
                filename,
                accuracy=accuracy,
                plot_psi_sol=plot_psi_sol,
                plot_V=plot_V)
    else:
        # mayavi for 3D
        may = MayaviAnimation.MayaviAnimation(dim=Harmonic.dim)
        with run_time.run_time():
            may.animate(Harmonic,
                        accuracy=accuracy,
                        plot_V=plot_V,
                        plot_psi_sol=plot_psi_sol,
                        x_lim=x_lim,
                        y_lim=y_lim,
                        z_lim=z_lim,
                        slice_x_index=slice_x_index,
                        slice_y_index=slice_y_index,
                        slice_z_index=slice_z_index,
                        interactive=interactive,
                        camera_r_func=camera_r_func,
                        camera_phi_func=camera_phi_func,
                        camera_z_func=camera_z_func,
                        )
        mlab.show()
        # TODO: close window after last frame
        # print(f"{Harmonic.t}, {Harmonic.dt * Harmonic.max_timesteps}")
        # if Harmonic.t >= Harmonic.dt * Harmonic.max_timesteps:
        #     mlab.close()
        cut_x = np.linspace(Harmonic.box.x0, Harmonic.box.x1, Harmonic.res.x)
        cut_z = np.linspace(Harmonic.box.z0, Harmonic.box.z1, Harmonic.res.z)
        prob_mitte_x = np.abs(Harmonic.psi_val[:]
                              [Harmonic.res.y // 2]
                              [Harmonic.res.z // 2]) ** 2.0
        prob_mitte_z = np.abs(Harmonic.psi_val[Harmonic.res.x // 2]
                                              [Harmonic.res.y // 2][:]) ** 2.0

        plt.plot(cut_x, prob_mitte_x, "x-", color="tab:blue", label="x cut")
        plt.plot(cut_z, prob_mitte_z, "--", color="tab:orange", label="z cut")
        plt.plot(cut_x, psi_sol_3d_cut_x(cut_x), ".-", color="tab:cyan",
                 label="x cut sol")
        plt.plot(cut_z, psi_sol_3d_cut_z(z=cut_z), "o-", color="tab:olive",
                 label="z cut sol")
        plt.ylim([0.0, 0.01])
        plt.legend()
        plt.grid()
        plt.show()

        may.create_movie(
            input_data_file_pattern="*.png",
            filename=filename,
            delete_input=delete_input)


# Script runs, if script is run as main script (called by python *.py)
if __name__ == "__main__":
    # for parallelization (use all cores)
    max_workers = psutil.cpu_count(logical=False)

    # constants needed for the Schroedinger equation

    # due to fft of the points the res
    # needs to be 2 ** resolution_exponent
    res = functions.Resolution(x=2 ** 6, y=2 ** 6, z=2 ** 6)

    box = functions.Box(x0=-12.0, x1=12.0,
                        y0=-12.0, y1=12.0,
                        z0=-3.0, z1=3.0)

    dt: float = 0.02
    N: int = 10 ** 5
    m: float = 164.0 * constants.u_in_kg
    a_dd: float = 130.0 * constants.a_0
    a_s: float = (130.0 / 0.8) * constants.a_0

    w_x: float = 2.0 * np.pi * 30.0
    w_y: float = w_x
    w_z: float = 2.0 * np.pi * 140.0

    alpha_y, alpha_z = functions.get_alphas(w_x=w_x, w_y=w_y, w_z=w_z)
    g, g_qf, e_dd, a_s_l_ho_ratio = functions.get_parameters(
        N=N, m=m, a_s=a_s, a_dd=a_dd, w_x=w_x)
    print(f"g, g_qf, epsilon_dd, alpha_y, alpha_z: "
          f"{g, g_qf, e_dd, alpha_y, alpha_z}")

    # functions needed for the Schroedinger equation (e.g. potential: V,
    # initial wave function: psi_0)
    V_1d = functions.v_harmonic_1d
    V_2d = functools.partial(functions.v_harmonic_2d, alpha_y=alpha_y)
    V_3d = functools.partial(
        functions.v_harmonic_3d,
        alpha_y=alpha_y,
        alpha_z=alpha_z)

    box_length = [(box.x1 - box.x0), (box.y1 - box.y0), (box.z1 - box.z0)]
    V_3d_ddi = functools.partial(functions.dipol_dipol_interaction,
                                 R=1000 * min(box_length) / 2.0)

    # functools.partial sets all arguments except x, y, z,
    # as multiple arguments for Schroedinger aren't implement yet
    # psi_0_1d = functools.partial(
    #     functions.psi_0_rect, x_min=-0.25, x_max=-0.25, a=2.0)
    psi_0_1d = functools.partial(
        functions.psi_gauss_1d, a=3.0, x_0=0.0, k_0=0.0)
    psi_0_2d = functools.partial(functions.psi_gauss_2d_pdf,
                                 mu=[0.0, 0.0],
                                 var=np.array([[1.0, 0.0], [0.0, 1.0]])
                                 )

    psi_0_3d = functools.partial(
        functions.psi_gauss_3d,
        a_x=4.0,
        a_y=4.0,
        a_z=4.0,
        x_0=0.0,
        y_0=0.0,
        z_0=0.0,
        k_0=0.0)

    psi_0_noise_3d = functions.noise_mesh(
        min=0.8, max=1.4, shape=(res.x, res.y, res.z))

    # Used to remember that 2D need the special pos function (g is set inside
    # of Schroedinger for convenience)
    psi_sol_1d = functions.thomas_fermi_1d
    psi_sol_2d = functions.thomas_fermi_2d_pos

    # psi_sol_3d = functions.thomas_fermi_3d
    kappa = functions.get_kappa(alpha_z=alpha_z, e_dd=e_dd,
                                x_min=3.0, x_max=5.0, res=1000)
    R_r, R_z = functions.get_R_rz(kappa=kappa, e_dd=e_dd, N=N,
                                  a_s_l_ho_ratio=a_s_l_ho_ratio)
    psi_sol_3d = functools.partial(functions.density_in_trap,
                                   R_r=R_r, R_z=R_z)
    print(f"kappa: {kappa}, R_r: {R_r}, R_z: {R_z}")

    psi_sol_3d_cut_x = functools.partial(functions.density_in_trap,
                                         y=0, z=0, R_r=R_r, R_z=R_z)

    psi_sol_3d_cut_z = functools.partial(functions.density_in_trap,
                                         x=0, y=0, R_r=R_r, R_z=R_z)

    # TODO: get mayavi lim to work
    # 3D works in single core mode
    simulate_case(box,
                  res,
                  max_timesteps=101,
                  dt=dt,
                  g=g,
                  g_qf=g_qf,
                  e_dd=e_dd,
                  imag_time=True,
                  mu=1.1,
                  E=1.0,
                  psi_0=psi_0_3d,
                  V=V_3d,
                  V_interaction=V_3d_ddi,
                  psi_sol=psi_sol_3d,
                  mu_sol=functions.mu_3d,
                  plot_psi_sol=True,
                  psi_sol_3d_cut_x=psi_sol_3d_cut_x,
                  psi_sol_3d_cut_z=psi_sol_3d_cut_z,
                  plot_V=False,
                  psi_0_noise=psi_0_noise_3d,
                  alpha_psi=0.8,
                  alpha_psi_sol=0.5,
                  alpha_V=0.3,
                  accuracy=10 ** -7,
                  filename="anim.mp4",
                  x_lim=(-2.0, 2.0),
                  y_lim=(-2.0, 2.0),
                  z_lim=(0, 0.5),
                  slice_x_index=int(res.x / 8),  # just for mayavi (3D)
                  slice_y_index=int(res.y / 8),
                  slice_z_index=int(res.z / 2),
                  interactive=True,
                  camera_r_func=functools.partial(
                      functions.camera_func_r,
                      r_0=40.0, phi_0=45.0, z_0=50.0, r_per_frame=0.0),
                  camera_phi_func=functools.partial(
                      functions.camera_func_phi,
                      r_0=40.0, phi_0=45.0, z_0=50.0, phi_per_frame=5.0),
                  camera_z_func=functools.partial(
                      functions.camera_func_z,
                      r_0=40.0, phi_0=45.0, z_0=50.0, z_per_frame=0.0),
                  delete_input=False
                  )
    print("Single core done")

    # # TODO: As g is proportional to N * a_s/w_x,
    # # changing g, V, g_qf are different (maybe other variables too)
    # # Multi-core: multiple cases (Schroedinger with different parameters)
    # # box length in 1D: [-L,L], in 2D: [-L,L, -L,L], in 3D: [-L,L, -L,L, -L,L]
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
    #         executor.submit(simulate_case,
    #                         box,
    #                         res,
    #                         max_timesteps=800,
    #                         dt=dt,
    #                         g=g,
    #                         g_qf=g_qf,
    #                         e_dd=e_dd,
    #                         imag_time=True,
    #                         mu=1.1,
    #                         E=1.0,
    #                         psi_0=psi_0_3d,
    #                         V=V_3d,
    #                         V_interaction=V_3d_ddi,
    #                         psi_sol=psi_sol_3d,
    #                         mu_sol=functions.mu_3d,
    #                         plot_psi_sol=False,
    #                         plot_V=False,
    #                         psi_0_noise=psi_0_noise_3d,
    #                         alpha_psi=0.8,
    #                         alpha_psi_sol=0.5,
    #                         alpha_V=0.3,
    #                         accuracy=10 ** -8,
    #                         filename="anim.mp4",
    #                         x_lim=(-2.0, 2.0),
    #                         y_lim=(-2.0, 2.0),
    #                         z_lim=(0, 0.5),
    #                         slice_x_index=res["x"] // 10,
    #                         slice_y_index=res["y"] // 10,
    #                         slice_z_index=res["z"] // 2,
    #                         camera_r_func = functools.partial(
    #                             functions.camera_func_r,
    #                             r_0=40.0, phi_0=45.0, z_0=50.0,
    #                             r_per_frame=0.0),
    #                         camera_phi_func = functools.partial(
    #                             functions.camera_func_phi,
    #                             r_0=40.0, phi_0=45.0, z_0=50.0,
    #                             phi_per_frame=5.0),
    #                         camera_z_func = functools.partial(
    #                             functions.camera_func_z,
    #                             r_0=40.0, phi_0=45.0, z_0=50.0,
    #                             z_per_frame=0.0),
    #                        delete_input=True
    #                         )
