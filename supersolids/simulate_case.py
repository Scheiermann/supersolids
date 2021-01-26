#!/usr/bin/env python

# author: Daniel Scheiermann
# email: daniel.scheiermann@stud.uni-hannover.de
# license: MIT
# Please feel free to use and modify this, but keep the above information.

"""
Animation for the numerical solver for the non-linear
time-dependent Schrodinger equation.

"""

import functools

import numpy as np
from mayavi import mlab
from typing import Callable, Tuple, NamedTuple
from matplotlib import pyplot as plt

from supersolids import Animation
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

        with run_time.run_time(name="ani.start"):
            ani.start(
                Harmonic,
                filename,
                accuracy=accuracy,
                plot_psi_sol=plot_psi_sol,
                plot_V=plot_V)
    else:
        # mayavi for 3D
        may = MayaviAnimation.MayaviAnimation(dim=Harmonic.dim)
        with run_time.run_time(name="may.animate"):
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
        with run_time.run_time(name="mlab.show"):
            mlab.show()
        # TODO: close window after last frame
        # print(f"{Harmonic.t}, {Harmonic.dt * Harmonic.max_timesteps}")
        # if Harmonic.t >= Harmonic.dt * Harmonic.max_timesteps:
        #     mlab.close()

        cut_x = np.linspace(Harmonic.box.x0, Harmonic.box.x1, Harmonic.res.x)
        cut_y = np.linspace(Harmonic.box.y0, Harmonic.box.y1, Harmonic.res.y)
        cut_z = np.linspace(Harmonic.box.z0, Harmonic.box.z1, Harmonic.res.z)

        prob_mitte_x = np.abs(Harmonic.psi_val[:, Harmonic.res.y // 2, Harmonic.res.z // 2]) ** 2.0
        prob_mitte_y = np.abs(Harmonic.psi_val[Harmonic.res.x // 2, :, Harmonic.res.z // 2]) ** 2.0
        prob_mitte_z = np.abs(Harmonic.psi_val[Harmonic.res.x // 2, Harmonic.res.y // 2, :]) ** 2.0

        plt.plot(cut_x, prob_mitte_x, "x-", color="tab:blue", label="x cut")
        plt.plot(cut_y, prob_mitte_y, "x-", color="tab:grey", label="y cut")
        plt.plot(cut_z, prob_mitte_z, "x-", color="tab:orange", label="z cut")
        plt.plot(cut_x, psi_sol_3d_cut_x(cut_x), "x-", color="tab:cyan",
                 label="x cut sol")
        plt.plot(cut_z, psi_sol_3d_cut_z(z=cut_z), "x-", color="tab:olive",
                 label="z cut sol")
        plt.ylim([0.0, 0.005])
        plt.legend()
        plt.grid()
        plt.show()

        may.create_movie(
            input_data_file_pattern="*.png",
            filename=filename,
            delete_input=delete_input)
