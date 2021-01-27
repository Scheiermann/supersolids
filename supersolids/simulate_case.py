#!/usr/bin/env python

# author: Daniel Scheiermann
# email: daniel.scheiermann@stud.uni-hannover.de
# license: MIT
# Please feel free to use and modify this, but keep the above information.

"""
Animation for the numerical solver for the non-linear
time-dependent Schrodinger equation for 1D, 2D and 3D in single-core.

"""

from pathlib import Path

from mayavi import mlab
from typing import Tuple, List

from supersolids.Animation import Animation, MayaviAnimation, MatplotlibAnimation
from supersolids.Schroedinger import Schroedinger
from supersolids import run_time


def simulate_case(System: Schroedinger,
                  Anim: Animation.Animation,
                  accuracy: float = 10 ** -6,
                  slice_indices: List[int] = [0, 0, 0],
                  interactive: bool = True,
                  delete_input: bool = True,
                  x_lim: Tuple[float, float] = (-1.0, 1.0),
                  y_lim: Tuple[float, float] = (-1.0, 1.0),
                  z_lim: Tuple[float, float] = (-1.0, 1.0),
                  ) -> Schroedinger:
    """
    Wrapper for Animation and Schroedinger to get a working Animation
    of a System through the equations given by Schroedinger.

    Parameters

    System : Schroedinger.Schroedinger
        Schrödinger equations for the specified system

    accuracy : float
        Convergence is reached when relative error of mu is smaller
        than accuracy, where :math:`\mu = - \\log(\psi_{normed}) / (2 dt)`

    x_lim : Tuple[float, float]
        Limits of plot in x direction

    y_lim : Tuple[float, float]
        Limits of plot in y direction

    z_lim : Tuple[float, float]
        Limits of plot in z direction

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
        then interaction is not possible. So interactive=True turns the usage
        of camera functions off.

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
                                 accuracy=accuracy,
                                 )
    else:
        # mayavi for 3D
        MayAnim = MayaviAnimation.MayaviAnimation(Anim,
                                                  slice_indices=slice_indices,
                                                  dir_path=Path(__file__).parent.joinpath("results")
                                                  )
        with run_time.run_time(name="MayaviAnimation.animate"):
            MayAnim.animate(System,
                            accuracy=accuracy,
                            interactive=interactive,
                            )

        with run_time.run_time(name="mlab.show"):
            mlab.show()
        # TODO: close window after last frame
        # print(f"{System.t}, {System.dt * System.max_timesteps}")
        # if System.t >= System.dt * System.max_timesteps:
        #     mlab.close()

        MayAnim.create_movie(input_data_file_pattern="*.png",
                             delete_input=delete_input)

        return System
