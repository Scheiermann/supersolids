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

import numpy as np
from mayavi import mlab
from typing import Tuple

from supersolids.Animation import Animation, MayaviAnimation, MatplotlibAnimation
from supersolids.Schroedinger import Schroedinger
from supersolids.tools import run_time
from supersolids.tools.cut_1d import cut_1d


def simulate_case(System: Schroedinger,
                  Anim: Animation.Animation,
                  accuracy: float = 10 ** -6,
                  delete_input: bool = True,
                  dir_path: Path = Path.home().joinpath("supersolids", "results"),
                  dir_name_result: str = "",
                  slice_indices: np.ndarray = [0, 0, 0],
                  offscreen: bool = False,
                  x_lim: Tuple[float, float] = (-1.0, 1.0),
                  y_lim: Tuple[float, float] = (-1.0, 1.0),
                  z_lim: Tuple[float, float] = (-1.0, 1.0),
                  steps_per_npz: int = 10,
                  frame_start: int = 0,
                  ) -> Schroedinger:
    """
    Wrapper for Animation and Schroedinger to get a working Animation
    of a System through the equations given by Schroedinger.

    :param System: Schrödinger equations for the specified system

    :param Anim: :class: Animation with configured properties

    :param accuracy: Convergence is reached when relative error of mu is smaller
        than accuracy, where :math:`\mu = - \\log(\psi_{normed}) / (2 dt)`

    :param offscreen: Condition for interactive mode. When camera functions are used,
        then interaction is not possible. So interactive=True turn the usage
        of camera functions off.

    :param delete_input: Condition if the input pictures should be deleted,
        after creation the creation of the animation as e.g. mp4

    :param dir_path: Path where to look for old directories (movie data)

    :param dir_name_result: Name of directory where to save the results at. For example the
        standard naming convention is movie002")

    :param slice_indices: Numpy array with indices of grid points
        in the directions x, y, z (in terms of System.x, System.y, System.z)
        to produce a slice/plane in mayavi,
        where :math:`\psi_{prob}` = :math:`|\psi|^2` is used for the slice
        Max values is for e.g. System.Res.x - 1.

    :param x_lim: Limits of plot in x direction

    :param y_lim: Limits of plot in y direction

    :param z_lim: Limits of plot in z direction

    :param steps_per_npz: Number of dt steps skipped between saved npz.

    :param frame_start: Number of named file, where psi_val is loaded from. For example
        the standard naming convention is step_000001.npz

    :return: Reference to Schroedinger System

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
            MatplotlibAnim.start(
                System,
                accuracy=accuracy,
            )
    else:
        if not offscreen:
            # mayavi for 3D
            MayAnim = MayaviAnimation.MayaviAnimation(
                Anim,
                slice_indices=slice_indices,
                dir_path=dir_path,
                offscreen=offscreen,
            )

            with run_time.run_time(name="MayaviAnimation.animate"):
                MayAnimator = MayAnim.animate(System, accuracy=accuracy,
                                              interactive=(not offscreen),
                                              )

            with run_time.run_time(name="mlab.show"):
                mlab.show()

            result_path = MayAnim.create_movie(dir_path=dir_path,
                                               input_data_file_pattern="*.png",
                                               delete_input=delete_input)

            cut_1d(System, slice_indices=slice_indices,
                   dir_path=result_path, y_lim=(0.0, 0.05))
        else:
            System.simulate_raw(accuracy=accuracy,
                                dir_path=dir_path,
                                dir_name_result=dir_name_result,
                                steps_per_npz=steps_per_npz,
                                frame_start=frame_start,
                                )

        return System
