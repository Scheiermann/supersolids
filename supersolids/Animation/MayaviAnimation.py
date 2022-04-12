#!/usr/bin/env python

# author: Daniel Scheiermann
# email: daniel.scheiermann@itp.uni-hannover.de
# license: MIT
# Please feel free to use and modify this, but keep the above information.

"""
Functions for Potential and initial wave function :math:`\psi_0`

"""
import sys
from pathlib import Path
import zipfile
from typing import Optional, List

import dill
import numpy as np
from ffmpeg import input
from mayavi import mlab
from functools import partial

from supersolids.Animation import Animation
from supersolids.Schroedinger import Schroedinger
from supersolids.SchroedingerMixture import SchroedingerMixture
from supersolids.helper import functions, constants, get_path
from supersolids.helper.get_version import check_cp_nb, get_version
cp, cupy_used, cuda_used, numba_used = check_cp_nb(np)
if numba_used:
    import supersolids.helper.numbas as numbas


def get_legend(System, frame, frame_start, supersolids_version, mu_rel=None):
    if isinstance(System, SchroedingerMixture):
        # Update legend (especially time)
        format_1d = partial(np.format_float_scientific, pad_left=5, precision=6, sign=True)
        text = (f"version={supersolids_version}, "
                f"N={System.N_list}, "
                f"Box={System.Box}, "
                f"Res={System.Res}, "
                f"max_timesteps={System.max_timesteps:d}, "
                f"dt={System.dt:.6f}, "
                f"a_s={System.a_s_array}, "
                f"a_dd={System.a_dd_array}, "
                f"w_x/2pi={System.w_x / (2.0 * np.pi):05.02f}, "
                f"w_y/2pi={System.w_y / (2.0 * np.pi):05.02f}, "
                f"w_z/2pi={System.w_z / (2.0 * np.pi):05.02f}, "
                f"imag_time={System.imag_time}, "
                f"t={System.t:07.05f}, "
                f"processed={(frame - frame_start) / System.max_timesteps:05.03f}%, "
                f"E={np.format_float_scientific(System.E, pad_left=5, precision=6, sign=True)}, "
                f"mu={list(map(format_1d, System.mu_arr))}"
                )
    else:
        # Update legend (especially time)
        text = (f"version={supersolids_version}, "
                f"N={System.N}, "
                f"Box={System.Box}, "
                f"Res={System.Res}, "
                f"max_timesteps={System.max_timesteps:d}, "
                f"dt={System.dt:.6f}, "
                f"g={System.g:.2}, "
                f"g_qf={System.g_qf:.2}, "
                f"e_dd={System.e_dd:05.03f},\n"
                f"a_s/a_0={System.a_s / constants.a_0:05.02f}, "
                f"w_x/2pi={System.w_x / (2 * np.pi):05.02f}, "
                f"w_y/2pi={System.w_y / (2 * np.pi):05.02f}, "
                f"w_z/2pi={System.w_z / (2 * np.pi):05.02f}, "
                f"imag_time={System.imag_time}, "
                f"t={System.t:07.05f}, "
                f"processed={(frame - frame_start) / System.max_timesteps:05.03f}%, "
                f"E={np.format_float_scientific(System.E, pad_left=5, precision=6, sign=True)}, "
                f"mu={np.format_float_scientific(System.mu_arr, pad_left=5, precision=6, sign=True)}, "
                )

    if mu_rel is not None:
        text = text + f"mu_rel={mu_rel:+05.05e}"

    return text


def axes_style():
    ax = mlab.axes(line_width=2, nb_labels=5)
    ax.axes.visibility = True
    ax.label_text_property.font_size = 8
    ax.label_text_property.color = (0.0, 0.0, 0.0)
    ax.title_text_property.color = (0.0, 0.0, 0.0)
    # throws error, maybe deprecated
    # ax.property_tuple.color = (0.0, 0.0, 0.0)
    # ax.property_tuple.line_width = 2.5


class MayaviAnimation(Animation.Animation):
    mayavi_counter: int = 0

    def __init__(self,
                 Anim: Animation.Animation,
                 slice_indices: np.ndarray = [0, 0, 0],
                 dir_path: Path = Path.home().joinpath("supersolids", "results"),
                 offscreen: bool = False,
                 ):
        """
        Creates an Animation with mayavi for a Schroedinger equation
        Methods need the object Schroedinger with the parameters of the equation

        :param Anim: Base class Animation with configured properties for the animation.

        :param slice_indices: Numpy array with indices of grid points
            in the directions x, y, z (in terms of System.x, System.y, System.z)
            to produce a slice/plane in mayavi,
            where :math:`\psi_{prob}` = :math:`|\psi|^2` is used for the slice
            Max values is for e.g. System.Res.x - 1.

        :param dir_path: Path where to look for old directories (movie data)

        """
        super().__init__(Res=Anim.Res,
                         plot_V=Anim.plot_V,
                         alpha_psi_list=Anim.alpha_psi_list,
                         alpha_psi_sol_list=Anim.alpha_psi_sol_list,
                         alpha_V=Anim.alpha_V,
                         camera_r_func=Anim.camera_r_func,
                         camera_phi_func=Anim.camera_phi_func,
                         camera_z_func=Anim.camera_z_func,
                         filename=Anim.filename,
                         )

        if not dir_path.is_dir():
            dir_path.mkdir(parents=True)

        MayaviAnimation.mayavi_counter += 1
        self.slice_indices = slice_indices
        self.offscreen = offscreen
        # dir_path need to be saved to access it after the figure closed
        self.dir_path = dir_path

        if not self.offscreen:
            mlab.options.offscreen = self.offscreen
            self.fig = mlab.figure(f"{MayaviAnimation.mayavi_counter:02d}")

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
                     delete_input: bool = True) -> Path:
        """
        Creates movie filename with all matching pictures from
        input_data_file_pattern.
        By default deletes all input pictures after creation of movie
        to save disk space.

        :param dir_path: Path where to look for old directories (movie data)

        :param input_data_file_pattern: Regex pattern to find all input data

        :param delete_input: Condition if the input pictures should be deleted,
            after creation the creation of the animation as e.g. mp4

        """
        if dir_path is None:
            input_path, _, _, _ = get_path.get_path(self.dir_path)
        else:
            input_path, _, _, _ = get_path.get_path(dir_path)

        input_data = Path(input_path, input_data_file_pattern)
        output_path = Path(input_path, self.filename)
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

        return input_path

    def prepare(self, System: Schroedinger, mixture_slice_index: int = 0):
        if cupy_used:
            x_mesh: np.ndarray = System.x_mesh.get()
            y_mesh: np.ndarray = System.y_mesh.get()
            z_mesh: np.ndarray = System.z_mesh.get()
            V_val: np.ndarray = System.V_val.get()

        else:
            x_mesh: np.ndarray = System.x_mesh
            y_mesh: np.ndarray = System.y_mesh
            z_mesh: np.ndarray = System.z_mesh
            V_val: np.ndarray = System.V_val

        if isinstance(System, SchroedingerMixture):
            prob_plots: List[cp.ndarray] = []
            if len(System.psi_val_list) != len(self.alpha_psi_list):
                sys.exit(f"System.psi_val_list ({len(System.psi_val_list)}) and Anim.alpha_psi_list "
                         f"({len(self.alpha_psi_list)}) need to have same length.")
            for i, (psi_val, alpha_psi) in enumerate(zip(System.psi_val_list, self.alpha_psi_list)):
                colormap: str = "spectral"
                if i == mixture_slice_index:
                    colormap = "cool"
                    if cupy_used:
                        prob1_3d: np.ndarray = (cp.abs(psi_val) ** 2.0).get()
                    else:
                        prob1_3d: np.ndarray = np.abs(psi_val) ** 2.0

                if cupy_used:
                    prob: np.ndarray = (cp.abs(psi_val) ** 2.0).get()
                else:
                    prob: np.ndarray = np.abs(psi_val) ** 2.0


                prob_plots.append(mlab.contour3d(x_mesh,
                                                 y_mesh,
                                                 z_mesh,
                                                 prob,
                                                 colormap=colormap,
                                                 opacity=alpha_psi,
                                                 transparent=True)
                                  )
        else:
            if cupy_used:
                prob1_3d = (cp.abs(System.psi_val) ** 2.0).get()
            else:
                prob1_3d = np.abs(System.psi_val) ** 2.0

            prob1_plot = mlab.contour3d(x_mesh,
                                        y_mesh,
                                        z_mesh,
                                        prob1_3d,
                                        colormap="spectral",
                                        opacity=self.alpha_psi_list[0],
                                        transparent=True)

            prob_plots = [prob1_plot]

        slice_x_plot = mlab.volume_slice(x_mesh,
                                         y_mesh,
                                         z_mesh,
                                         prob1_3d,
                                         colormap="spectral",
                                         plane_orientation="x_axes",
                                         slice_index=self.slice_indices[0],
                                         )

        slice_y_plot = mlab.volume_slice(x_mesh,
                                         y_mesh,
                                         z_mesh,
                                         prob1_3d,
                                         colormap="spectral",
                                         plane_orientation="y_axes",
                                         slice_index=self.slice_indices[1],
                                         )

        slice_z_plot = mlab.volume_slice(x_mesh,
                                         y_mesh,
                                         z_mesh,
                                         prob1_3d,
                                         colormap="spectral",
                                         plane_orientation="z_axes",
                                         slice_index=self.slice_indices[2],
                                         )

        if self.plot_V:
            V_plot = mlab.contour3d(x_mesh,
                                    y_mesh,
                                    z_mesh,
                                    V_val,
                                    colormap="hot",
                                    opacity=self.alpha_V,
                                    transparent=True)
        else:
            V_plot = None

        if isinstance(System, SchroedingerMixture):
            psi_sol_plot = None
            pass
        else:
            if System.psi_sol_val is not None:
                if cupy_used:
                    psi_sol_val: np.ndarray = System.psi_sol_val.get()
                else:
                    psi_sol_val: np.ndarray = System.psi_sol_val

                psi_sol_plot = mlab.contour3d(x_mesh,
                                              y_mesh,
                                              z_mesh,
                                              psi_sol_val,
                                              colormap="cool",
                                              opacity=self.alpha_psi_sol_list[0],
                                              transparent=True)
            else:
                psi_sol_plot = None

        axes_style()

        return prob_plots, slice_x_plot, slice_y_plot, slice_z_plot, V_plot, psi_sol_plot

    # @mlab.animate(delay=10, ui=True)
    def animate_npz(self,
                    dir_path: Path = None,
                    dir_name: str = None,
                    filename_schroedinger: str = f"schroedinger.pkl",
                    filename_steps: str = f"step_",
                    steps_format: str = "%06d",
                    steps_per_npz: int = 10,
                    frame_start: int = 0,
                    arg_slices: bool = False,
                    azimuth: float = 0.0,
                    elevation: float = 0.0,
                    distance: float = 60.0,
                    sum_along: Optional[float] = None,
                    summary_name: Optional[str] = None,
                    mixture_slice_index: int = 0,
                    no_legend: bool = False,
                    ):
        """
        Animates solving of the Schroedinger equations of System with mayavi in 3D.
        Loaded from npz-files.

        :param no_legend: Option to add legend as text to every frame.

        :param mixture_slice_index: Index of component of which the slices are taken.

        """

        supersolids_version = get_version()

        if (dir_path is None) or (dir_path == Path("~/supersolids/results").expanduser()):
            if dir_name is not None:
                input_path = Path(self.dir_path, dir_name)
            else:
                input_path, _, _, _ = get_path.get_path(self.dir_path)
        else:
            if dir_name is not None:
                input_path = Path(self.dir_path, dir_name)
            else:
                input_path, _, _, _ = get_path.get_path(dir_path)

        self.dir_path = input_path
        self.fig.scene.movie_maker.directory = self.dir_path
        _, last_index, _, _ = get_path.get_path(self.dir_path,
                                                search_prefix=filename_steps,
                                                file_pattern=".npz"
                                                )

        print("Load schroedinger")
        with open(Path(input_path, filename_schroedinger), "rb") as f:
            # WARNING: this is just the input Schroedinger at t=0
            System: Schroedinger = dill.load(file=f)

        (prob_plots, slice_x_plot, slice_y_plot, slice_z_plot,
         V_plot, psi_sol_plot) = self.prepare(System, mixture_slice_index=mixture_slice_index)

        yield

        # read new frames until Exception (last frame read)
        frame = frame_start
        while True:
            print(f"frame={frame}")
            try:
                # get the psi_val of Schroedinger at other timesteps (t!=0)
                psi_val_path = Path(input_path, filename_steps + steps_format % frame + ".npz")
                if isinstance(System, SchroedingerMixture):
                    with open(psi_val_path, "rb") as f:
                        System.psi_val_list = np.load(file=f)["psi_val_list"]
                else:
                    with open(psi_val_path, "rb") as f:
                        System.psi_val = np.load(file=f)["psi_val"]

                if not (summary_name is None):
                    try:
                        System = System.load_summary(input_path, steps_format, frame,
                                                     summary_name=summary_name)
                    except Exception:
                        print(f"Could not load {summary_name}!")

                if not no_legend:
                    text = get_legend(System, frame, frame_start, supersolids_version)

                if frame == frame_start:
                    if not no_legend:
                        # create title for first frame
                        title = mlab.title(text=text,
                                           height=0.95,
                                           line_width=1.0,
                                           size=1.0,
                                           color=(0, 0, 0),
                                           )
                    mlab.view(azimuth=azimuth, elevation=elevation, distance=distance)

                    if arg_slices:
                        cbar = mlab.colorbar(object=slice_x_plot, orientation='vertical', nb_labels=10)
                        cbar.use_default_range = False
                        cbar.data_range = np.array([0.0, 2.0 * np.pi])

                if not no_legend:
                    title.set(text=text)
                if isinstance(System, SchroedingerMixture):
                    densities: List[np.ndarray] = []
                    for i, psi_val in enumerate(System.psi_val_list):
                        densities.append(System.get_density(func_val=psi_val, p=2.0,
                                                            jit=numba_used))
                        if i == mixture_slice_index:
                            if sum_along is None:
                                psi_val1 = psi_val
                                density1 = densities[i]
                            else:
                                psi_val1 = psi_val
                                density1 = System.sum_along(func_val=psi_val, axis=sum_along,
                                                            l_0=None)
                    if arg_slices:
                        psi_arg = np.angle(psi_val1) + np.pi
                        slice_x_plot.mlab_source.trait_set(scalars=psi_arg)
                        slice_y_plot.mlab_source.trait_set(scalars=psi_arg)
                        slice_z_plot.mlab_source.trait_set(scalars=psi_arg)
                    else:
                        slice_x_plot.mlab_source.trait_set(scalars=density1)
                        slice_y_plot.mlab_source.trait_set(scalars=density1)
                        slice_z_plot.mlab_source.trait_set(scalars=density1)
                else:
                    # Update plot functions
                    if sum_along is None:
                        density1: np.ndarray = System.get_density(func_val=System.psi_val, p=2.0,
                                                                  jit=numba_used)
                    else:
                        density1 = System.sum_along(func_val=System.psi_val, axis=sum_along,
                                                    l_0=None)

                    densities = [density1]
                    if arg_slices:
                        psi_arg = np.angle(System.psi_val) + np.pi
                        slice_x_plot.mlab_source.trait_set(scalars=psi_arg)
                        slice_y_plot.mlab_source.trait_set(scalars=psi_arg)
                        slice_z_plot.mlab_source.trait_set(scalars=psi_arg)
                    else:
                        slice_x_plot.mlab_source.trait_set(scalars=density1)
                        slice_y_plot.mlab_source.trait_set(scalars=density1)
                        slice_z_plot.mlab_source.trait_set(scalars=density1)

                for prob_plot, density in zip(prob_plots, densities):
                    prob_plot.mlab_source.trait_set(scalars=density)

                yield

            except zipfile.BadZipFile:
                print(f"Zipfile with frame {frame} can't be read. Maybe the simulation "
                      "was stopped before file was successfully created."
                      "Animation is built until, but without that frame.")
                yield None
                break

            except FileNotFoundError:
                yield None
                break

            frame = frame + steps_per_npz
            if frame == last_index + steps_per_npz:
                yield None
                break
            elif frame > last_index:
                frame = last_index

        # Finally close
        mlab.close(all=True)

    @mlab.animate(delay=10, ui=True)
    def animate(self, System: Schroedinger, accuracy: float = 10 ** -6,
                interactive: bool = True,
                mixture_slice_index: int = 0,
                no_legend: bool = False):
        """
        Animates solving of the Schroedinger equations of System with mayavi in 3D.
        Animation is limited to System.max_timesteps or
        the convergence according to accuracy.

        :param System: Schrödinger equations for the specified system

        :param accuracy: Convergence is reached when relative error of mu is smaller
            than accuracy, where :math:`\mu = - \\log(\psi_{normed}) / (2 dt)`

        :param slice_indices: Numpy array with indices of grid points
            in the directions x, y, z (in terms of System.x, System.y, System.z)
            to produce a slice/plane in mayavi,
            where :math:`\psi_{prob}` = :math:`|\psi|^2` is used for the slice
            Max values is for e.g. System.Res.x - 1.

        :param interactive: Condition for interactive mode. When camera functions are used,
            then interaction is not possible. So interactive=True turn the usage
            of camera functions off.

        :param no_legend: Option to add legend as text to every frame.

        :param mixture_slice_index: Index of component of which the slices are taken.

        """
        (prob_plots, slice_x_plot, slice_y_plot, slice_z_plot,
         V_plot, psi_sol_plot) = self.prepare(System, mixture_slice_index=0)

        supersolids_version = get_version()

        for frame in range(0, System.max_timesteps):
            if not interactive:
                # rotate camera
                camera_r, camera_phi, camera_z = functions.camera_3d_trajectory(
                    frame,
                    r_func=self.camera_r_func,
                    phi_func=self.camera_phi_func,
                    z_func=self.camera_z_func
                )

                mlab.view(distance=camera_r,
                          azimuth=camera_phi,
                          elevation=camera_z)

            # Initialize mu_rel
            mu_rel = 1

            # The initial plot needs to be shown first,
            # also a timestep is needed for mu_rel
            if frame > 0:
                mu_old = System.mu_arr
                System.time_step()

                mu_rel = np.abs((System.mu_arr - mu_old) / System.mu_arr)

                # Stop animation when accuracy is reached
                if mu_rel < accuracy:
                    print(f"Accuracy reached: {mu_rel}")
                    yield None
                    break

                elif np.isnan(mu_rel) and np.isnan(System.mu_arr):
                    assert np.isnan(System.E), ("E should be nan, when mu is nan."
                                                "Then the system is divergent.")
                    print(f"Accuracy NOT reached! System diverged.")
                    yield None
                    break

            if frame == (System.max_timesteps - 1):
                # Animation stops at the next step, to actually show the last step
                print(f"Maximum timesteps are reached. Animation is stopped.")

            if not no_legend:
                # Update legend (especially time)
                text = get_legend(System, frame, 0, supersolids_version, mu_rel)

            if frame == 0:
                if not no_legend:
                    # create title for first frame
                    title = mlab.title(text=text,
                                       height=0.95,
                                       line_width=1.0,
                                       size=1.0,
                                       color=(0, 0, 0),
                                       )

            if not no_legend:
                title.set(text=text)

            # Update plot functions
            density1: np.ndarray = System.get_density(func_val=System.psi_val, p=2.0,
                                                      jit=numba_used)
            densities = [density1]
            if isinstance(System, SchroedingerMixture):
                for i, psi_val in enumerate(System.psi_val_list):
                    densities.append(System.get_density(func_val=psi_val, p=2.0, jit=numba_used))
                    if i == mixture_slice_index:
                        psi_val1 = psi_val
                        density1 = densities[i]

            for prob_plot, density in zip(prob_plots, densities):
                prob_plot.mlab_source.trait_set(scalars=density)

            slice_x_plot.mlab_source.trait_set(scalars=density1)
            slice_y_plot.mlab_source.trait_set(scalars=density1)
            slice_z_plot.mlab_source.trait_set(scalars=density1)

            yield

        # Finally close
        mlab.close(all=True)
