#!/usr/bin/env python

import traceback
import dill
import numpy as np
import shutil
import sys

from matplotlib import pyplot as plt
from pathlib import Path
from PIL import Image
from typing import List, Optional, Tuple

from supersolids.helper import cut_1d
from supersolids.helper.cut_1d import plot_cuts_tuples, zip_meshes
from supersolids.helper.functions import fft_plot
from supersolids.helper.property_load_npz import property_load_npz
from supersolids.helper.get_path import get_last_png_in_last_anim, get_path, get_step_index
from supersolids.helper.merge_meshes import check_if_further, merge_meshes
from supersolids.helper.periodic_system import paste_together, paste_together_videos
from supersolids.tools.track_property import get_dim


def path_mesh_remap(path_mesh: np.ndarray):
    path_mesh_new: List[Path] = np.roll(path_mesh, 1, axis=1)

    return path_mesh_new


def last_frame(frame: Optional[int],
               var1_list,
               var2_list,
               experiment_suffix: str,
               movie_take_last_list: List[int],
               path_anchor_output_list: List[Path],
               suffix_list: List[str],
               merge_suffix: str,
               filename_out_list: List[str],
               path_anchor_input_list: List[Path],
               movie_list_list: List[List[int]],
               check_further_list: int,
               movie_skip: int,
               dir_name: str = "movie",
               counting_format: str = "%03d", 
               nrow_components: int = 1,
               ncol_components: int = 1,
               dpi_ratio: float = 1.0,
               dpi_ratio_all: float = 1.0,
               use_edited: bool = False,
               dir_name_png: str = "movie",
               counting_format_png: str = "%03d",
               filename_pattern: str = "anim",
               filename_format: str = "%05d",
               filename_extension: str = ".png",
               frame_format: str = "%07d",
               video: bool = False,
               margin: int = 10,
               width: int = 1920,
               height: int = 1200,
               fps: float = 0.1,
               mesh_remap_index_list: List[int] = [],
               y_lim: Tuple[int] = (0, 1),
               cut_names: List[str] = ["cut_x"],
               mixture_slice_index_list: List[int] = [0],
               filename_steps_list: List[str] = ["mixture_step_"],
               normed_plots: bool = True,
               property_filename_list_list: List[str] = [],
               property_filename_list: List[str] = [],
               list_of_arrays_list: List[bool] = [],
               dir_all: str = "all",
               dir_cuts: str = "cuts",
               dir_frames: str = "frames",
               dir_scripts: str = "scripts",
               dir_videos: str = "videos",
               fft_start: int = 1,
               fft_end: Optional[int] = None,
               path_overlay: Optional[Path] = None,
               ):
    path_graphs = Path(path_anchor_input_list[0].parent, "graphs")
    number_of_movies_list = [len(movie_list_i) for movie_list_i in movie_list_list]

    labels_list = [[filename + f"{slice_index}_" + name for name in cut_names]
                    for slice_index, filename in zip(mixture_slice_index_list, filename_steps_list)]

    # construct list of all paths (path_mesh_list)
    path_mesh_list = []
    path_list: List[Path] = []
    for i, (path_anchor_input, movie_list,
            number_of_movies, check_further) in enumerate(zip(path_anchor_input_list,
                                                              movie_list_list,
                                                              number_of_movies_list,
                                                              check_further_list)):
    # for i, (path_anchor_input, movie_start, movie_end,
    #         number_of_movies, check_further) in enumerate(zip(path_anchor_input_list,
    #                                                           movie_start_list,
    #                                                           movie_end_list,
    #                                                           number_of_movies_list,
    #                                                           check_further_list)):
        path_inner_list: List[Path] = []
        for movie_number in movie_list:
        # for movie_number in range(movie_start, movie_end + 1):
            if movie_number == movie_skip:
                continue
            path_movie = Path(path_anchor_input, f"{dir_name}{counting_format % movie_number}"),
            if check_further != 0:
                path_movie = check_if_further(path_anchor_input, dir_name, counting_format,
                                              movie_number, experiment_step=number_of_movies,
                                              check_further=check_further)
            path_inner_list.append(path_movie)

        path_list.append(path_inner_list)
        path_mesh_list.append(np.array(path_inner_list).reshape(len(var2_list[i]),
                                                                len(var1_list[i])))
    print(f"path_mesh_list created!")

    var_mesh_list = [np.meshgrid(var1, var2, indexing="ij") for var1, var2 in zip(var1_list,
                                                                                  var2_list)]
    var_mesh_x, var_mesh_y, path_mesh = merge_meshes(var_mesh_list,
                                                     path_mesh_list,
                                                     len(path_anchor_input_list))
    dir_name_list = path_mesh.ravel()

    # construct path_mesh_new with path to the last png in each movie
    path_out_periodic_list: List[Path] = []
    path_dirname_list = Path(path_graphs, f"dir_name_list_{experiment_suffix}")
    with open(path_dirname_list.with_suffix(".pkl"), "wb") as f:
        dill.dump(obj=dir_name_list, file=f)
    with open(path_dirname_list.with_suffix(".txt"), "w") as f:
        f.write(f"{dir_name_list}\n")

    frame_last_list = []
    for movie_take_last, path_anchor_output, suffix, filename_out in zip(movie_take_last_list,
                                                                         path_anchor_output_list,
                                                                         suffix_list,
                                                                         filename_out_list):
        dir_pics = f"{counting_format_png % movie_take_last}"
        path_anchor_all: Path = Path(path_anchor_output, dir_all)
        path_anchor_cuts: Path = Path(path_anchor_output, dir_cuts)
        path_anchor_frames: Path = Path(path_anchor_output, dir_frames)
        path_anchor_pics: Path = Path(path_anchor_output, dir_pics)
        path_anchor_scripts: Path = Path(path_anchor_output, dir_scripts)
        path_anchor_videos: Path = Path(path_anchor_output, dir_videos)
        path_anchor_dir_list = [path_anchor_all, path_anchor_cuts, path_anchor_frames,
                                path_anchor_pics, path_anchor_scripts, path_anchor_videos]

        if frame:
            path_output_frame: Path = Path(path_anchor_frames,
                                           f"frame_{frame_format % frame}")
        else:
            path_output_frame: Path = Path(path_anchor_frames, f"frame_last")
        

        if not path_output_frame.is_dir():
            path_output_frame.mkdir(parents=True)
        
        for path_anchor_dir in path_anchor_dir_list:
            if not path_anchor_dir.is_dir():
                path_anchor_dir.mkdir(parents=True)

        # use path_mesh to get last png of every animation
        path_mesh_new = path_mesh.copy()
        path_mesh_cuts = [path_mesh.copy() for _ in range(0, len(filename_steps_list))]

        for ix, iy in np.ndindex(path_mesh.shape):
            if frame is None:
                path_mesh_new[ix, iy] = get_last_png_in_last_anim(path_mesh[ix, iy],
                                                                  dir_name_png, counting_format_png,
                                                                  movie_take_last,
                                                                  filename_pattern,
                                                                  filename_format,
                                                                  filename_extension,
                                                                  frame_format)
                for i, (filename_step, mixture_slice_index) in enumerate(zip(filename_steps_list,
                                                                             mixture_slice_index_list)):
                    path_mesh_cuts[i][ix, iy] = []
                    for cut_name in cut_names:
                        path_mesh_cuts[i][ix, iy].append(get_last_png_in_last_anim(path_mesh[ix, iy],
                                                                                dir_name_png,
                                                                                counting_format_png,
                                                                                movie_take_last,
                                                                                filename_step
                                                                                + str(mixture_slice_index)
                                                                                + "_" + cut_name + "_",
                                                                                frame_format,
                                                                                ".npz")
                                                      )

            else:
                path_last_movie_png, _, _, _ = get_path(
                    path_mesh[ix, iy],
                    search_prefix=f"{dir_name_png}",
                    counting_format=counting_format_png,
                    file_pattern="",
                    take_last=movie_take_last,
                    )

                path_mesh_new[ix, iy] = Path(path_last_movie_png,
                                             filename_pattern + f"{frame_format % frame}"
                                             + filename_extension)

                for i, (filename_step, mixture_slice_index) in enumerate(zip(filename_steps_list,
                                                                             mixture_slice_index_list)):
                    path_mesh_cuts[i][ix, iy] = []
                    for cut_name in cut_names:
                        path_mesh_cuts[i][ix, iy].append(
                            Path(path_last_movie_png, filename_step + str(mixture_slice_index)
                                 + "_" + cut_name + "_" + frame_format % frame + ".npz"
                                )
                            )


            path_currently_old: Path = path_mesh[ix, iy]
            if frame: 
                path_out: Path = Path(path_output_frame,
                                      f"{filename_out}"
                                      + f"_{path_currently_old.stem}{filename_extension}")
            else:
                frame_last = get_step_index(path_mesh_new[ix, iy], filename_prefix=filename_pattern,
                                            file_pattern=filename_extension)
                frame_last_list.append(frame_last)

                path_out: Path = Path(path_output_frame,
                                      f"{filename_out}"
                                      + f"_{path_currently_old.stem}"
                                      + f"_frame_{frame_format % frame_last}{filename_extension}"
                                      )

            # else:
            #     path_out: Path = Path(path_anchor_output,
            #                           f"{path_currently_old.parent.stem}_{path_currently_old.stem}"
            #                           + f"_{filename_out}{filename_extension}")

            if use_edited:
                # to use png from folders with png copied together (which you could have edited before)
                path_mesh_new[ix, iy] = path_out
            else:
                path_currently_new: Path = path_mesh_new[ix, iy]
                if path_currently_new is not None:
                    try:
                        shutil.copy(path_mesh_new[ix, iy], path_out)
                    except Exception as e: 
                        print(f"Copying failed: {e}")

            script_filename = "script"
            script_extension = ".txt"
            path_out_script: Path = Path(path_anchor_scripts,
                                         f"{script_filename}"
                                         + f"_{path_currently_old.parent.stem}"
                                         + f"_{path_currently_old.stem}{script_extension}")
            path_in_script: Path = Path(path_currently_old, f"{script_filename}{script_extension}")
            shutil.copy(path_in_script, path_out_script)

        print(f"frame: {frame}, movie_take_last: {movie_take_last}")
        if frame:
            path_out_periodic: Path = Path(path_anchor_output, dir_pics,
                                           f"periodic_system_merge{suffix}_{experiment_suffix}"
                                           + f"_{frame_format % frame}" + ".png")
        else:
            path_out_periodic: Path = Path(path_anchor_output, dir_pics,
                                           f"periodic_system_merge{suffix}_{experiment_suffix}.png")

        if mesh_remap_index_list:
            for mesh_remap_index in mesh_remap_index_list:
                path_mesh_cuts[mesh_remap_index] = path_mesh_remap(path_mesh_cuts[mesh_remap_index])
        path_out_mesh_cuts = path_mesh_cuts[0].copy()
        probs_cuts_max = path_mesh_cuts[0].copy()
        probs_cuts_middle = path_mesh_cuts[0].copy()
        for j, (ix, iy) in enumerate(np.ndindex(path_mesh_cuts[0].shape)):
            dir_paths = [path_mesh_cuts[i][ix, iy] for i in range(len(path_mesh_cuts))]
            if frame:
                frame_formatted = f"{frame_format % frame}"
            else:
                frame_formatted = f"{frame_format % frame_last_list[j]}"

            (path_out_mesh_cuts[ix, iy], probs_cuts_middle[ix, iy],
             probs_cuts_max[ix, iy]) = plot_cuts_tuples(dir_paths,
                                                        path_output_frame,
                                                        frame_formatted,
                                                        y_lim=y_lim,
                                                        labels_list=labels_list,
                                                        normed=normed_plots,
                                                        )

        path_out_periodic_list.append(path_out_periodic)
        nrow, ncol = path_mesh_new.shape
        # flip needed as appending pictures start from left top corner,
        # but enumeration of movies from left bottom corner
        np.set_printoptions(linewidth=500)
        path_mesh_new_mirrored: List[Path] = np.flip(path_mesh_new, axis=0)
        probs_cuts_mirrored_list: List = [np.flip(probs_cuts_middle, axis=0),
                                          np.flip(probs_cuts_max, axis=0)]
        probs_cuts_filenames = ["probs_cuts_middle", "probs_cuts_max"]
        if frame:
            probs_cuts_filenames = [name + f"_{frame_format % frame}"
                                    for name in probs_cuts_filenames]

        for probs_cuts_filename, probs_cuts_mirrored in zip(probs_cuts_filenames,
                                                            probs_cuts_mirrored_list):
            with open(Path(path_anchor_cuts, probs_cuts_filename + ".npz"), "wb") as g:
                np.savez_compressed(g, probs_cuts_mesh=probs_cuts_mirrored)

        if video:
            if frame:
                path_out_video: Path = Path(path_anchor_videos,
                                            f"periodic_video{suffix}_{experiment_suffix}"
                                            + f"_{frame_format % frame}" + ".mp4")
            else:
                path_out_video: Path = Path(path_anchor_videos,
                                            f"periodic_video{suffix}_{experiment_suffix}"
                                            + ".mp4")

            paste_together_videos(path_mesh_new_mirrored, path_out_video, margin, width, height, fps)

        else:
            print("Pasting all images together, may take some time.")
            paste_together(path_mesh_new_mirrored.ravel(), path_out_periodic, nrow, ncol, ratio=dpi_ratio)

    if not video:
        if frame:
            path_out_periodic_all: Path = Path(path_anchor_all,
                                               f"periodic_system_merge_all_{experiment_suffix}"
                                               + f"{merge_suffix}"
                                               + f"_{frame_format % frame}" + ".png")
            path_out_cuts_all: Path = Path(path_anchor_cuts,
                                           f"cuts_all_{experiment_suffix}"
                                           + f"{merge_suffix}"
                                           + f"_{frame_format % frame}" + ".png")

        else:
            path_out_periodic_all: Path = Path(path_anchor_all,
                                               f"periodic_system_merge_all_{experiment_suffix}"
                                               + f"{merge_suffix}.png")
            path_out_cuts_all: Path = Path(path_anchor_cuts,
                                           f"cuts_all_{experiment_suffix}"
                                           + f"{merge_suffix}.png")

        # mayavi view in periodic table
        # turn off decompression bomb checker
        Image.MAX_IMAGE_PIXELS = number_of_movies * Image.MAX_IMAGE_PIXELS
        paste_together(path_in_list=path_out_periodic_list, path_out=path_out_periodic_all,
                       nrow=nrow_components, ncol=ncol_components, ratio=dpi_ratio_all)
        
        # cuts in periodic table
        path_out_mesh_cuts_mirrored: List[Path] = np.flip(path_out_mesh_cuts, axis=0)
        # turn off decompression bomb checker
        Image.MAX_IMAGE_PIXELS = number_of_movies * Image.MAX_IMAGE_PIXELS
        paste_together(path_in_list=path_out_mesh_cuts_mirrored.ravel(), path_out=path_out_cuts_all,
                       nrow=nrow, ncol=ncol, ratio=dpi_ratio_all) 
        


        path_mesh_property_list = []
        for j in range(len(property_filename_list_list[0])):
            path_mesh_property_list.append(path_mesh.copy())
            for path_anchor_input, property_filename_list in zip(path_anchor_input_list,
                                                                      property_filename_list_list):

                for ix, iy in np.ndindex(path_mesh.shape):
                    if path_mesh[ix, iy].parent == path_anchor_input:
                        path_mesh_property_list[j][ix, iy] = Path(path_mesh[ix, iy], f"{property_filename_list[j]}")
            
        if property_filename_list:
            # plot property of last frame along var1 for every var2
            property_length_list = []
            property_dim_list = []
            for k, _ in enumerate(path_anchor_input_list):
                # properties in periodic tables
                # for property_filename, list_of_arrays in zip(property_filename_list, list_of_arrays_list):
                for j, list_of_arrays in enumerate(list_of_arrays_list):
                    # path_mesh_mirrored: List[Path] = np.flip(path_mesh, axis=0)
                    # path_mesh_property = path_mesh_mirrored.copy()
                    # path_mesh_property = path_mesh.copy()
                    # for ix, iy in np.ndindex(path_mesh_mirrored.shape):
                    #     path_mesh_property[ix, iy]: Path = Path(path_mesh_mirrored[ix, iy], f"{property_filename}")
                    property_length_of_movie_list = []
                    property_dim_of_movie_list = []
                    for ix, iy in np.ndindex(path_mesh.shape):
                        # path_mesh_property[ix, iy]: Path = Path(path_mesh[ix, iy], f"{property_filename}")
                        path_property_npz = path_mesh_property_list[j][ix, iy].with_suffix('.npz')
                        property_filename = Path(path_property_npz.stem).name
                        # create 3d array in shape of path_mesh (2D) + property 
                        try:
                            property_00_1 = property_load_npz(path_property_npz)[1]
                            if list_of_arrays:
                                number_of_components = property_00_1.shape[0]
                                property_length = property_00_1.shape[-1]
                                property_length_of_movie_list.append(property_length)
                            else:
                                property_length = property_00_1.shape[0]
                                property_length_of_movie_list.append(property_length)
                            property_dim = get_dim(list_of_arrays, property_00_1)
                            property_dim_of_movie_list.append(property_dim)

                        except Exception as e:
                            print(f"Error: property_length not found. "
                                  f"Make sure that property_length is the same for all movies. {e}")
            
                    property_length_list.append(max(property_length_of_movie_list))
                    property_dim_list.append(max(property_dim_of_movie_list))

            for k, _ in enumerate(path_anchor_input_list):
                title_rows = list(map(str, var1_list[k][::-1]))

            # plot property of last frame along var2 for every var1
            # for k, _ in enumerate(path_anchor_input_list):
            # properties in periodic tables
            for j, (list_of_arrays, property_length,
                    property_dim) in enumerate(zip(list_of_arrays_list,
                                                   property_length_list,
                                                   property_dim_list)):
                for ix, iy in np.ndindex(path_mesh.shape):
                    # path_mesh_property[ix, iy]: Path = Path(path_mesh[ix, iy], f"{property_filename}")
                    path_property_npz = path_mesh_property_list[j][ix, iy].with_suffix('.npz')
                    property_filename = Path(path_property_npz.stem).name
                    # create arrays of right dimensions on first element
                    if ix == iy == 0:
                        # mesh_t = np.zeros((path_mesh_property.shape[0],
                        #                    path_mesh_property.shape[1],
                        mesh_t = np.zeros((path_mesh_property_list[j].shape[0],
                                           path_mesh_property_list[j].shape[1],
                                           property_length))
                        mesh_unpadded = np.zeros((path_mesh_property_list[j].shape[0],
                                                  path_mesh_property_list[j].shape[1],
                                                  ))
                        if property_dim == 1:
                            mesh_property_all = np.zeros((path_mesh_property_list[j].shape[0],
                                                          path_mesh_property_list[j].shape[1],
                                                          property_length))
                        else:
                            if list_of_arrays:
                                mesh_property_all = np.zeros((path_mesh_property_list[j].shape[0],
                                                              path_mesh_property_list[j].shape[1],
                                                              number_of_components,
                                                              property_dim,
                                                              property_length,
                                                              ))
                            else:
                                mesh_property_all = np.zeros((path_mesh_property_list[j].shape[0],
                                                              path_mesh_property_list[j].shape[1],
                                                              property_length,
                                                              property_dim))

                    try:
                        a, b = property_load_npz(path_property_npz) 
                        mesh_unpadded[ix, iy] = b.shape[-1]
                        # mesh_t[ix, iy], mesh_property_all[ix, iy] = property_load_npz(path_property_npz) 
                        mesh_t[ix, iy], mesh_property_all[ix, iy] = a, b
                    except Exception as e:
                        print(f"Problem with {path_property_npz}, "
                              f"check out if list_of_arrays_list needs to be True.\n{e}")
                        traceback.print_tb(e.__traceback__)
                        try:
                            mesh_t[ix, iy] = np.pad(a, (0, len(mesh_t[ix, iy]) - len(a)), 'constant')
                            pad_dim_0 = (len(mesh_property_all[ix, iy].shape) - 1) * [(0, 0)]
                            mesh_property_all[ix, iy] = np.pad(b, (*pad_dim_0, (0, mesh_property_all[ix, iy].shape[-1] - b.shape[-1])), 'constant')
                            print("Padding with 0 worked!")
                        except Exception as e:
                            print("Padding did not work!")
                            sys.exit(1)

                path_out_property_all: Path = Path(path_anchor_output,
                                                   f"property_all_{experiment_suffix}"
                                                   + f"{merge_suffix}"
                                                   + f"_{property_filename}")
                
                path_mesh_all = path_out_property_all.with_suffix('.npz')
                with open(path_mesh_all, "wb") as g:
                    # np.savez_compressed(g, x=mesh_t, y=mesh_property_all, z=path_mesh_mirrored)
                    np.savez_compressed(g, x=mesh_t, y=mesh_property_all, z=path_mesh)

                # turn off decompression bomb checker
                Image.MAX_IMAGE_PIXELS = number_of_movies * Image.MAX_IMAGE_PIXELS
                path_mesh_property_mirrored: List[Path] = np.flip(path_mesh_property_list[j], axis=0)
                paste_together(path_in_list=path_mesh_property_mirrored.ravel(),
                               path_out=path_out_property_all,
                               nrow=nrow, ncol=ncol, ratio=dpi_ratio_all) 

                # rows = path_mesh_mirrored.shape[1]
                rows = path_mesh.shape[1]
                if list_of_arrays:
                    cols = property_dim
                else:
                    cols = 1
                fig, axes = plt.subplots(nrows=rows, ncols=cols, squeeze=False,
                                         sharex='col', figsize=(16,9))
                fig.suptitle(f"{Path(property_filename).stem}")

                # give titles to rows and columns
                for i, ax in enumerate(axes[0, :]):
                    ax.set_title(f"axis {i}")

                # title_rows = list(map(str, var1_list[k][::-1]))
                for ax, title_row in zip(axes[:,0], title_rows):
                    pad = 5
                    ax.annotate(title_row, xy=(0, 0.5), xytext=(-ax.yaxis.labelpad - pad, 0),
                                xycoords=ax.yaxis.label, textcoords='offset points',
                                size='large', ha='right', va='center')
                    # ax.set_ylabel(title_row, rotation=0, size='large')

                # for iy, ax in enumerate(plt.gcf().get_axes()):
                # frequency_plot = False 
                frequency_plot = True 
                max_freq_along_vars = True
                # max_freq_along_vars = False
                if frequency_plot:
                    annotation = False
                    # annotation = True
                    # plot_log = True
                    # plot_log = False
                    plot_log_list = [True, False]
                    if annotation:
                       annotation_decimals = 0
                    # spin = True
                    spin_list = [True, False]
                    w_index = 0 # w with largest amplitude
                    # fft_start = 1
                    # fft_end = int(property_length / 2 / 80)
                    # fft_end = int(property_length / 2)
                    # fft_end = 30
                    fft_length = fft_end - fft_start
                    ampl_arr = np.zeros(shape=(path_mesh.shape[0], path_mesh.shape[1],
                                               number_of_components, property_dim, fft_length))
                    ampl_arr_list = [ampl_arr, np.zeros_like(ampl_arr)]
                
                w0_mesh = np.zeros(shape=(path_mesh.shape[0], path_mesh.shape[1], fft_length))
                if max_freq_along_vars:
                    for iy in range(rows):
                        # fixed a11
                        lambda_frame = -1
                        labels = []

                        for i in range(property_dim):
                            if list_of_arrays:
                                for j in range(number_of_components):
                                    # try a12 vs frequency monopolar
                                    if frequency_plot:
                                        # x_range = var2_list[k]
                                        x_range = var_mesh_y.flatten()
                                        for s, spin in enumerate(spin_list):
                                            if spin:
                                                y_range = mesh_property_all[:, iy, 0, i, :] + (-1) ** j * mesh_property_all[:, iy, 1, i, :]
                                            else:
                                                y_range = mesh_property_all[:, iy, j, i, :]
                                            w_list = []
                                            # collect all ix
                                            for ix in range(path_mesh.shape[0]):
                                                w0_mesh[ix, iy], amplitude0 = fft_plot(mesh_t[ix, iy, :int(mesh_unpadded[ix, iy])],
                                                                          y_range[ix, :int(mesh_unpadded[ix, iy])],
                                                                          start=fft_start, end=fft_end)
                                                ampl_arr_list[s][ix, iy, j, i, :] = np.array(amplitude0)
                                                zipped_sorted_by_max = zip(*sorted(zip(w0_mesh[ix, iy], amplitude0),
                                                                                   key=lambda t: t[1]))
                                                w_max, ampl = map(np.array, zipped_sorted_by_max)
                                                w_list.append(w_max[w_index])
                                        y_range = np.array(w_list)
                                        # x_range = var2_list[k]
                                        x_range = var_mesh_y.flatten()
                                        axes[iy, i].plot(x_range, y_range, "x-")
                                    else:
                                        x_range = var_mesh_y.flatten()
                                        y_range = mesh_property_all[:, iy, j, i, lambda_frame]
                                        axes[iy, i].plot(x_range, y_range, "x-")
                            else:
                                x_range = var_mesh_y.flatten()
                                y_range = mesh_property_all[:, iy, lambda_frame]
                                axes[iy, 0].plot(x_range, y_range, "x-")

                for ax in axes.flatten():
                    ax.grid()
                if list_of_arrays:
                    labels = [f"component {j}" for j in range(number_of_components)]
                    fig.legend(labels, loc='center right')

                path_out_property_ix: Path = Path(path_anchor_output,
                                                      f"property_all_{experiment_suffix}"
                                                      + f"{merge_suffix}"
                                                      + f"_ix_{property_filename}")
                print(f"Save to: {path_out_property_ix}")
                fig.savefig(path_out_property_ix)

            if frequency_plot:
                if path_overlay:
                    # load data for overlay ontop of pcolormesh
                    with open(path_overlay, "rb") as g:
                        excitations = np.load(file=g)
                        a_s_excitations = excitations["a_s"]
                        array_ds_list = excitations["array_ds_list"]
                        parity_total_list = excitations["parity_total_list"]
                        evals_ordered_list = excitations["evals_ordered_list"]

                        array_d = array_ds_list * (1 - parity_total_list)
                        # array_s = (1 - array_ds_list)
                        array_s = (1 - array_ds_list) * (1 - parity_total_list)

                        eigen_den_list = []
                        eigen_spin_list = []
                        for id_ex in range(evals_ordered_list.shape[0]):
                            overlay_data = np.real(evals_ordered_list[id_ex, :])[[np.ravel(np.argwhere(d==1)) for d in array_d][id_ex]]
                            eigen_spin = np.real(evals_ordered_list[id_ex, :])[[np.ravel(np.argwhere(d==1)) for d in array_s][id_ex]]
                            eigen_den_list.append(overlay_data)
                            eigen_spin_list.append(eigen_spin)

                for s, spin in enumerate(spin_list):
                    for plot_log in plot_log_list:
                        for j in range(number_of_components):
                            fig_w, axes_w = plt.subplots(nrows=rows, ncols=cols, squeeze=False,
                                                         sharex='col', figsize=(16,9))
                            if spin:
                                if ((-1) ** j == 1):
                                    if path_overlay:
                                        overlay_list = eigen_den_list
                                    fig_w.suptitle(r"$x^2 \cdot \left(|\psi_{0}|^2 + |\psi_{1}|^2\right)$")
                                else:
                                    if path_overlay:
                                        overlay_list = eigen_spin_list
                                    fig_w.suptitle(r"$x^2 \cdot \left(|\psi_{0}|^2 - |\psi_{1}|^2\right)$")
                            else:
                                overlay_list = []
                                fig_w.suptitle(f"$x^2 \cdot |\psi_{j}|^2$")

                            for iy in range(path_mesh.shape[1]):
                                for i in range(property_dim):
                                    if plot_log:
                                       Z = np.log10(ampl_arr_list[s][:, iy, j, i, :]) 
                                    else:
                                       Z = ampl_arr_list[s][:, iy, j, i, :]
                                    X, _ = np.meshgrid(x_range, w0_mesh[0, 0, :], indexing="ij")
                                    Y = w0_mesh[:, iy, :]
                                    im = axes_w[iy, i].pcolormesh(X, Y, Z, shading="auto")
                                    if iy == 0:
                                        # colorbar for every axis and comp
                                        fig_w.colorbar(im)
                                    if spin:
                                        mesh_plot_name = ("spin_"
                                            + f"property_all_{experiment_suffix}"
                                            + f"{merge_suffix}"
                                            + f"_mesh_ix_comp_{j}"
                                            + f"_{property_filename}"
                                            )
                                    else:
                                        mesh_plot_name = (f"property_all_{experiment_suffix}"
                                            + f"{merge_suffix}"
                                            + f"_mesh_ix_comp_{j}"
                                            + f"_{property_filename}"
                                            )

                                    if path_overlay:
                                        Y_max = np.max(Y)
                                        for overlay_data, a_s_excitation in zip(overlay_list,
                                                                                a_s_excitations):
                                            # crop values to plotted pcolormesh
                                            overlay_data = overlay_data[overlay_data <= Y_max]
                                            overlay_data = overlay_data[0 < overlay_data]
                                            axes_w[iy, i].plot(a_s_excitation * np.ones(overlay_data.shape),
                                                               overlay_data,
                                                               "x", color="blue")
                                        
                                    if plot_log:
                                        mesh_plot_name = "log_" + mesh_plot_name
                                    else:
                                        mesh_plot_name = "lin_" + mesh_plot_name
                                        
                                    path_out_property_ix_meshplot: Path = Path(
                                        path_anchor_output,
                                        mesh_plot_name)
                                        
                                    if annotation:
                                        # text of value for every mesh-point
                                        Z_shape = ampl_arr_list[s][:, iy, j, i, :].shape
                                        for m in range(0, Z_shape[0]):
                                            for n in range(0, Z_shape[1]):
                                                if annotation_decimals == 0:
                                                    vals = np.round(
                                                        ampl_arr_list[s][m, iy, j, i, n],
                                                        annotation_decimals).astype(int)
                                                text = axes_w[iy, i].text(X[m, n], Y[m, n],
                                                        vals,
                                                    ha="center", va="center",
                                                    color="black", size=5,
                                                    rotation="horizontal")

                            # give titles to rows and columns
                            for r, ax in enumerate(axes_w[0, :]):
                                ax.set_title(f"axis {r}")
                            for ax, title_row in zip(axes_w[:,0], title_rows):
                                pad = 5
                                ax.annotate(title_row, xy=(0, 0.5),
                                            xytext=(-ax.yaxis.labelpad - pad, 0),
                                            xycoords=ax.yaxis.label, textcoords='offset points',
                                            size='large', ha='right', va='center')

                            # fig.supxlabel(r"$a_{12}$", fontsize=18)
                            # fig.supxlabel(r"$\frac{\omega}{\omega_{x}}$", fontsize=18)
                            print(f"Save to: {path_out_property_ix_meshplot}")
                            fig_w.savefig(path_out_property_ix_meshplot)
