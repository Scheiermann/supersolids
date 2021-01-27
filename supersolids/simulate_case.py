#!/usr/bin/env python

# author: Daniel Scheiermann
# email: daniel.scheiermann@stud.uni-hannover.de
# license: MIT
# Please feel free to use and modify this, but keep the above information.

"""
Animation for the numerical solver for the non-linear
time-dependent Schrodinger equation for 1D, 2D and 3D in single-core.

"""

import functools

from mayavi import mlab
from typing import Callable, Tuple

from supersolids.Animation import Animation, MayaviAnimation, MatplotlibAnimation
from supersolids.Schroedinger import Schroedinger
from supersolids.psi_cut_1d import psi_cut_1d
from supersolids import run_time


def simulate_case(System: Schroedinger,
                  Anim: Animation.Animation,
                  accuracy: float = 10 ** -6,
                  plot_psi_sol: bool = False,
                  psi_sol_3d_cut_x: Callable = None,
                  psi_sol_3d_cut_y: Callable = None,
                  psi_sol_3d_cut_z: Callable = None,
                  plot_V: bool = True,
                  filename: str = "split.mp4",
                  x_lim: Tuple[float, float] = (-1.0, 1.0),
                  y_lim: Tuple[float, float] = (-1.0, 1.0),
                  z_lim: Tuple[float, float] = (-1.0, 1.0),
                  slice_x_index: int = 0,
                  slice_y_index: int = 0,
                  slice_z_index: int = 0,
                  interactive: bool = True,
                  delete_input: bool = True) -> None:
    """
    Wrapper for Animation and Schroedinger to get a working Animation
    of a System through the equations given by Schroedinger.

    Parameters

    System : Schroedinger.Schroedinger
        Schrödinger equations for the specified system

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
    if System.dim < 3:
        # matplotlib for 1D and 2D
        MatplotlibAnim = MatplotlibAnimation.MatplotlibAnimation(Anim)
        if MatplotlibAnim.dim == 1:
            MatplotlibAnim.set_limits(0, 0, *x_lim, *y_lim)
        elif MatplotlibAnim.dim == 2:
            MatplotlibAnim.ax.set_xlim(*x_lim)
            MatplotlibAnim.ax.set_ylim(*y_lim)
            MatplotlibAnim.ax.set_zlim(*z_lim)

        # Animation.set_limits_smart(0, System)

        with run_time.run_time(name="Animation.start"):
            MatplotlibAnim.start(System,
                                 filename,
                                 accuracy=accuracy,
                                 plot_psi_sol=plot_psi_sol,
                                 plot_V=plot_V)
    else:
        # mayavi for 3D
        MayAnim = MayaviAnimation.MayaviAnimation(Anim)
        with run_time.run_time(name="MayaviAnimation.animate"):
            MayAnim.animate(System,
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
                            )

        psi_cut_1d(System,
                   psi_sol_3d_cut_x,
                   psi_sol_3d_cut_y,
                   psi_sol_3d_cut_z,
                   y_lim=(0.0, 0.05))

        with run_time.run_time(name="mlab.show"):
            mlab.show()
        # TODO: close window after last frame
        # print(f"{System.t}, {System.dt * System.max_timesteps}")
        # if System.t >= System.dt * System.max_timesteps:
        #     mlab.close()

        MayAnim.create_movie(input_data_file_pattern="*.png",
                             filename=filename,
                             delete_input=delete_input)
