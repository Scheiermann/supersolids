#!/usr/bin/env python

"""
Functions for Potential and initial wave function psi_0

author: Daniel Scheiermann
email: daniel.scheiermann@stud.uni-hannover.de
license: MIT
Please feel free to use and modify this, but keep the above information. Thanks!
"""

from pathlib import Path

import numpy as np
import ffmpeg
from mayavi import mlab
from typing import Tuple

from supersolids import Animation, functions, Schroedinger


def get_image_path(dir_path: Path, dir_name: str = "movie", counting_format: str = "%03d"):
    """
    Looks up all directories with matching dir_name and counting format in dir_path.
    Gets the highest number and returns a path with dir_name counted one up (prevents colliding with old data).

    Parameters
    ----------
    dir_path : Path
               Path where to look for old directories (movie data)
    dir_name : str
               General name of the directories without the counter
    counting_format : str
                      Format of counter of the directories

    Returns
    -------
    Path for the new directory (not colliding with old data)
    """
    # "movie" and "%03d" strings are hardcoded in mayavi movie_maker _update_subdir
    existing = sorted([x for x in dir_path.glob(dir_name + "*") if x.is_dir()])
    try:
        last_index = int(existing[-1].name.split(dir_name)[1])
    except Exception as e:
        assert last_index is not None, ("Extracting last index from dir_path failed")
    input_path = Path(dir_path, dir_name + counting_format % last_index)

    return input_path


@mlab.animate(delay=10, ui=True)
def animate(System: Schroedinger.Schroedinger, accuracy: float = 10 ** -6,
            plot_psi_sol: bool = False,
            plot_V: bool = True,
            x_lim: Tuple[float, float] = (-1, 1),
            y_lim: Tuple[float, float] = (-1, 1),
            z_lim: Tuple[float, float] = (-1, 1),
            slice_x_index: int = 0,
            slice_y_index: int = 0
            ):
    """
    Animates solving of the Schroedinger equations of System with mayavi in 3D.
    Animation is limited to System.max_timesteps or the convergence according to accuracy.

    Parameters
    ----------
    System : Schroedinger.Schoredinger
             Schrödinger equations for the specified system

    accuracy : float
               Convergence is reached when relative error of s ios smaller than accuracy,
               where s is System.s = - np.log(psi_norm_after_evolution) / (2.0 * self.dt)

    x_lim : Tuple[float, float]
    y_lim : Tuple[float, float]
    z_lim : Tuple[float, float]
    slice_x_index : int
                    Index of projection in terms of indexes of System.x
    slice_y_index : int
                    Index of projection in terms of indexes of System.y
    plot_V : bool
             Condition if V should be plotted.
    plot_psi_sol :
             Condition if psi_sol should be plotted.

    Returns
    -------

    """
    prob_3d = np.abs(System.psi_val) ** 2
    prob_plot = mlab.contour3d(System.x_mesh, System.y_mesh, System.z_mesh, prob_3d,
                               colormap="spectral", opacity=System.alpha_psi, transparent=True)

    slice_x_plot = mlab.volume_slice(System.x_mesh, System.y_mesh, System.z_mesh, prob_3d, colormap="spectral",
                                plane_orientation="x_axes",
                                slice_index=slice_x_index,
                                extent=[*x_lim, *y_lim, *z_lim])
    slice_y_plot = mlab.volume_slice(System.x_mesh, System.y_mesh, System.z_mesh, prob_3d, colormap="spectral",
                                plane_orientation="y_axes",
                                slice_index=slice_y_index,
                                extent=[*x_lim, *y_lim, *z_lim])

    if plot_V:
        V_plot = mlab.contour3d(System.x_mesh, System.y_mesh, System.z_mesh, System.V_val,
                                colormap="hot", opacity=System.alpha_V, transparent=True)

    if System.psi_sol_val is not None:
        if plot_psi_sol:
            psi_sol_plot = mlab.contour3d(System.x_mesh, System.y_mesh, System.z_mesh, System.psi_sol_val,
                                          colormap="cool", opacity=System.alpha_psi_sol, transparent=True)

    for i in range(0, System.max_timesteps):
        mu_old = System.mu
        System.time_step()
        mu_rel = np.abs((System.mu - mu_old) / System.mu)
        print(f"mu_rel: {mu_rel}")
        if mu_rel < accuracy:
            print(f"accuracy reached: {mu_rel}")
            break
        prob_3d = np.abs(System.psi_val) ** 2
        slice_x_plot.mlab_source.trait_set(scalars=prob_3d)
        slice_y_plot.mlab_source.trait_set(scalars=prob_3d)
        prob_plot.mlab_source.trait_set(scalars=prob_3d)
        yield


class MayaviAnimation:
    mayavi_counter: int = 0
    animate = staticmethod(animate)

    def __init__(self, dim: float = 3, dir_path: Path = Path(__file__).parent.joinpath("results")):
        """
        Creates an Animation with mayavi for a Schroedinger equation
        Methods need the object Schroedinger with the parameters of the equation
        """
        if not dir_path.is_dir():
            dir_path.mkdir(parents=True)

        MayaviAnimation.mayavi_counter += 1
        self.dim = dim

        self.fig = mlab.figure(f"{MayaviAnimation.mayavi_counter:02d}")
        mlab.title(f"{MayaviAnimation.mayavi_counter:02d}")
        self.ax = mlab.axes(line_width=2, nb_labels=5)
        self.ax.axes.visibility = True

        # dir_path need to be saved to access it after the figure closed
        self.dir_path = dir_path

        self.fig.scene.disable_render = False
        # anti_aliasing default is 8, and removes resolution issues when downscaling, but takes longer
        self.fig.scene.anti_aliasing_frames = 8
        self.fig.scene.movie_maker.record = True
        # set dir_path to save images to
        self.fig.scene.movie_maker.directory = dir_path
        self.fig.scene.show_axes = True

    def create_movie(self,
                     dir_path: Path = None,
                     input_data_file_pattern: str = "*.png",
                     filename: str = "anim.mp4",
                     delete_input: bool = True):
        """
        Creates movie filename with all matching pictures from input_data_file_pattern.
        By default deletes all input pictures after creation of movie to save disk space.

        Parameters
        ----------
        delete_input : bool
                       Flag to delete input_data after creation of movie
        dir_path : Path
                   Path where to look for old directories (movie data)

        input_data_file_pattern : str
                                  regex pattern to find all input data

        filename : str
                  filename with filetype to save the movie to

        Returns
        -------

        """
        if dir_path is None:
            input_path = get_image_path(self.dir_path)
        else:
            input_path = get_image_path(dir_path)

        input_data = Path(input_path, input_data_file_pattern)
        output_path = Path(input_path, filename)
        print(f"input_data: {input_data}")

        # requires either mencoder or ffmpeg to be installed on your system
        # from command line:
        # ffmpeg -f image2 -r 10 -i anim%05d.png -qscale 0 anim.mp4 -pass 2
        ffmpeg.input(input_data, pattern_type="glob", framerate=25).output(str(output_path)).run()

        if delete_input:
            # remove all input files (pictures), after animation is created and saved
            input_data_used = [x for x in input_path.glob(input_data_file_pattern) if x.is_file()]
            for trash_file in input_data_used:
                trash_file.unlink()


# Script runs, if script is run as main script (called by python *.py)
if __name__ == "__main__":
    Harmonic = Schroedinger.Schroedinger(resolution=2 ** 6, max_timesteps=100, L=3, dt=1.0, g=1.0, imag_time=True,
                                         s=1.1, E=1.0,
                                         dim=3,
                                         psi_0=functions.psi_gauss_3d,
                                         V=functions.v_harmonic_3d,
                                         psi_sol=functions.thomas_fermi_3d
                                         )
    may = MayaviAnimation(dim=Harmonic.dim, dir_path=Path(__file__).parent.joinpath("results"))
    may.animate(Harmonic, x_lim=(-10, 5), y_lim=(-1, 1), z_lim=(-1, 1))
    mlab.show()
    may.create_movie(dir_path=None, input_data_file_pattern="*.png", filename="anim.mp4")
