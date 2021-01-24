#!/usr/bin/env python

# author: Daniel Scheiermann
# email: daniel.scheiermann@stud.uni-hannover.de
# license: MIT
# Please feel free to use and modify this, but keep the above information. Thanks!

"""
Functions for Potential and initial wave function :math:`\psi_0`

"""

from pathlib import Path

import numpy as np
from ffmpeg import input
from mayavi import mlab
from typing import Tuple, Dict

from supersolids import Animation, functions, Schroedinger


def get_image_path(dir_path: Path,
                   dir_name: str = "movie",
                   counting_format: str = "%03d") -> Path:
    """
    Looks up all directories with matching dir_name
    and counting format in dir_path.
    Gets the highest number and returns a path with dir_name counted one up
    (prevents colliding with old data).

    Parameters

    dir_path : Path
        Path where to look for old directories (movie data)
    dir_name : str
        General name of the directories without the counter
    counting_format : str
        Format of counter of the directories

    Returns

    input_path : Path
    Path for the new directory (not colliding with old data)
    """
    # "movie" and "%03d" strings are hardcoded
    # in mayavi movie_maker _update_subdir
    existing = sorted([x for x in dir_path.glob(dir_name + "*") if x.is_dir()])
    try:
        last_index = int(existing[-1].name.split(dir_name)[1])
    except IndexError as e:
        assert last_index is not None, (
            "Extracting last index from dir_path failed")

    input_path = Path(dir_path, dir_name + counting_format % last_index)

    return input_path


def axes_style():
    ax = mlab.axes(line_width=2, nb_labels=5)
    ax.axes.visibility = True
    ax.label_text_property.font_size = 8
    ax.label_text_property.color = (0.0, 0.0, 0.0)
    ax.title_text_property.color = (0.0, 0.0, 0.0)
    ax.property.color = (0.0, 0.0, 0.0)
    ax.property.line_width = 2.5



@mlab.animate(delay=10, ui=True)
def animate(System: Schroedinger.Schroedinger,
            accuracy: float = 10 ** -6,
            plot_psi_sol: bool = False,
            plot_V: bool = True,
            x_lim: Tuple[float, float] = (-1, 1),
            y_lim: Tuple[float, float] = (-1, 1),
            z_lim: Tuple[float, float] = (-1, 1),
            slice_x_index: int = 0,
            slice_y_index: int = 0,
            slice_z_index: int = 0,
            interactive: bool = True,
            camera_r_func=None,
            camera_phi_func=None,
            camera_z_func=None,
           ):
    """
    Animates solving of the Schroedinger equations of System with mayavi in 3D.
    Animation is limited to System.max_timesteps or
    the convergence according to accuracy.

    Parameters

    System : Schroedinger.Schroedinger
        Schrödinger equations for the specified system

    accuracy : float
        Convergence is reached when relative error of mu is smaller
        than accuracy, where :math:`\mu = - \\log(\psi_{normed}) / (2 dt)`

    plot_psi_sol :
        Condition if :math:`\psi_sol` should be plotted.

    plot_V : bool
        Condition if V should be plotted.

    x_lim : Tuple[float, float]
        Limits of plot in x direction

    y_lim : Tuple[float, float]
        Limits of plot in y direction

    z_lim : Tuple[float, float]
        Limits of plot in z direction

    slice_x_index : int
        Index of grid point in x direction (in terms of System.x)
        to produce a slice/plane in mayavi,
        where :math:`\psi_{prob}` = :math:`|\psi|^2` is used for the slice

    slice_y_index : int
        Index of grid point in y  (in terms of System.y) direction
        to produce a slice/plane in mayavi,
        where :math:`\psi_{prob} = |\psi|^2` is used for the slice

    slice_z_index : int
        Index of grid point in z  (in terms of System.z) direction
        to produce a slice/plane in mayavi,
        where :math:`\psi_{prob} = |\psi|^2` is used for the slice

    interactive : bool
        Condition for interactive mode. When camera functions are used,
        then interaction is not possible. So interactive=True turn the usage
        of camera functions off.

    camera_r_func : Callable, function
        r component of the movement of the camera.

    camera_phi_func : Callable, function
        phi component of the movement of the camera.

    camera_z_func : Callable, function
        z component of the movement of the camera.

    Returns


    """
    prob_3d = np.abs(System.psi_val) ** 2
    prob_plot = mlab.contour3d(System.x_mesh,
                               System.y_mesh,
                               System.z_mesh,
                               prob_3d,
                               colormap="spectral",
                               opacity=System.alpha_psi,
                               transparent=True)

    slice_x_plot = mlab.volume_slice(System.x_mesh,
                                     System.y_mesh,
                                     System.z_mesh,
                                     prob_3d,
                                     colormap="spectral",
                                     plane_orientation="x_axes",
                                     slice_index=slice_x_index,
                                     # extent=[*x_lim, *y_lim, *z_lim]
                                     )

    slice_y_plot = mlab.volume_slice(System.x_mesh,
                                     System.y_mesh,
                                     System.z_mesh,
                                     prob_3d,
                                     colormap="spectral",
                                     plane_orientation="y_axes",
                                     slice_index=slice_y_index,
                                     # extent=[*x_lim, *y_lim, *z_lim]
                                     )

    slice_z_plot = mlab.volume_slice(System.x_mesh,
                                     System.y_mesh,
                                     System.z_mesh,
                                     prob_3d,
                                     colormap="spectral",
                                     plane_orientation="z_axes",
                                     slice_index=slice_z_index,
                                     # extent=[*x_lim, *y_lim, *z_lim]
                                     )

    if plot_V:
        V_plot = mlab.contour3d(System.x_mesh,
                                System.y_mesh,
                                System.z_mesh,
                                System.V_val,
                                colormap="hot",
                                opacity=System.alpha_V,
                                transparent=True)

    if System.psi_sol_val is not None:
        if plot_psi_sol:
            psi_sol_plot = mlab.contour3d(System.x_mesh,
                                          System.y_mesh,
                                          System.z_mesh,
                                          System.psi_sol_val,
                                          colormap="cool",
                                          opacity=System.alpha_psi_sol,
                                          transparent=True)

    axes_style()
    for i in range(0, System.max_timesteps):
        if not interactive:
            # rotate camera
            camera_r, camera_phi, camera_z = functions.camera_3d_trajectory(
                i,
                r_func=camera_r_func,
                phi_func=camera_phi_func,
                z_func=camera_z_func
                )

            mlab.view(distance=camera_r,
                      azimuth=camera_phi,
                      elevation=camera_z)

        # Initialize mu_rel
        mu_rel = System.mu

        # The initial plot needs to be shown first,
        # also a timestep is needed for mu_rel
        if i > 0:
            mu_old = System.mu
            System.time_step()

            mu_rel = np.abs((System.mu - mu_old) / System.mu)
            print(f"mu_rel: {mu_rel}")

            # Stop animation when accuracy is reached
            if mu_rel < accuracy:
                print(f"accuracy reached: {mu_rel}")
                yield None
                break

        if i == (System.max_timesteps - 1):
            print(f"Maximum timesteps are reached. Animation is stopped.")

        # Update plot functions
        prob_3d = np.abs(System.psi_val) ** 2
        slice_x_plot.mlab_source.trait_set(scalars=prob_3d)
        slice_y_plot.mlab_source.trait_set(scalars=prob_3d)
        slice_z_plot.mlab_source.trait_set(scalars=prob_3d)
        prob_plot.mlab_source.trait_set(scalars=prob_3d)

        # Update legend (especially time)
        mlab.title(f"g = {System.g:.2}, dt = {System.dt:.6}, "
                   f"max_timesteps = {System.max_timesteps:d}, "
                   f"imag_time = {System.imag_time}, "
                   f"mu_rel = {mu_rel:02.05e}, "
                   f"t = {System.t:02.05f}",
                   height=0.95,
                   line_width=1.0,
                   size=0.3,
                   color=(0, 0, 0))

        yield


class MayaviAnimation:
    mayavi_counter: int = 0
    animate = staticmethod(animate)

    def __init__(self,
                 dim: float = 3,
                 dir_path: Path = Path(__file__).parent.joinpath("results")):
        """
        Creates an Animation with mayavi for a Schroedinger equation
        Methods need the object Schroedinger with the parameters of the equation
        """
        if not dir_path.is_dir():
            dir_path.mkdir(parents=True)

        MayaviAnimation.mayavi_counter += 1
        self.dim = dim

        self.fig = mlab.figure(f"{MayaviAnimation.mayavi_counter:02d}")

        # dir_path need to be saved to access it after the figure closed
        self.dir_path = dir_path

        self.fig.scene.disable_render = False
        # anti_aliasing default is 8,
        # and removes res issues when downscaling, but takes longer
        self.fig.scene.anti_aliasing_frames = 8
        self.fig.scene.movie_maker.record = True
        # set dir_path to save images to
        self.fig.scene.movie_maker.directory = dir_path
        self.fig.scene.show_axes = True

    def create_movie(self,
                     dir_path: Path = None,
                     input_data_file_pattern: str = "*.png",
                     filename: str = "anim.mp4",
                     delete_input: bool = True) -> None:
        """
        Creates movie filename with all matching pictures from
        input_data_file_pattern.
        By default deletes all input pictures after creation of movie
        to save disk space.

        Parameters

        dir_path : Path
            Path where to look for old directories (movie data)

        input_data_file_pattern : str
            Regex pattern to find all input data

        filename : str
            Filename with filetype to save the movie to

        delete_input : bool
            Condition if the input pictures should be deleted,
            after creation the creation of the animation as e.g. mp4

        Returns


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
        input(input_data,
              pattern_type="glob",
              framerate=25).output(str(output_path)).run()

        if delete_input:
            # remove all input files (pictures),
            # after animation is created and saved
            input_data_used = [x
                               for x in input_path.glob(input_data_file_pattern)
                               if x.is_file()]
            for trash_file in input_data_used:
                trash_file.unlink()


# Script runs, if script is run as main script (called by python *.py)
if __name__ == "__main__":
    box: Dict[str, float] = {"x0": -5, "x1": 5,
                             "y0": -5, "y1": 5,
                             "z0": -5, "z1": 5}
    resolution: Dict[str, int] = {"x": 2 ** 6,
                                  "y": 2 ** 6,
                                  "z": 2 ** 6}
    psi_0_noise_3d = functions.noise_mesh(min=0.8,
                                          max=1.4,
                                          shape=(resolution["x"],
                                                 resolution["y"],
                                                 resolution["z"]))

    Harmonic = Schroedinger.Schroedinger(box=box,
                                         res=resolution,
                                         max_timesteps=101,
                                         dt=1.0,
                                         g=1.0,
                                         g_qf=0.0,
                                         e_dd=1.0,
                                         imag_time=True,
                                         mu=1.1, E=1.0,
                                         dim=3,
                                         psi_0=functions.psi_gauss_3d,
                                         V=functions.v_harmonic_3d,
                                         V_interaction=None,
                                         psi_sol=functions.thomas_fermi_3d,
                                         mu_sol=functions.mu_3d,
                                         psi_0_noise=psi_0_noise_3d,
                                         alpha_psi=0.8,
                                         alpha_psi_sol=0.53,
                                         alpha_V=0.3
                                         )

    may = MayaviAnimation(dim=Harmonic.dim,
                          dir_path=Path(__file__).parent.joinpath("results"))
    may.animate(Harmonic, accuracy=10**-6, plot_psi_sol=False, plot_V=True,
                x_lim=(-10, 5), y_lim=(-1, 1), z_lim=(-1, 1),
                slice_x_index=0, slice_y_index=0, slice_z_index=0
                )
    mlab.show()
    may.create_movie(dir_path=None,
                     input_data_file_pattern="*.png",
                     filename="anim.mp4")
