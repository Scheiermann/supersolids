#!/usr/bin/env python

import dill
import numpy as np
import shutil

from pathlib import Path
from typing import List, Optional, Tuple

from PIL import Image
from supersolids.helper import cut_1d
from supersolids.helper.cut_1d import plot_cuts_tuples, zip_meshes

from supersolids.helper.get_path import get_last_png_in_last_anim, get_path, get_step_index
from supersolids.helper.merge_meshes import check_if_further, merge_meshes
from supersolids.helper.periodic_system import paste_together, paste_together_videos


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
               movie_start_list: List[int],
               movie_end_list: List[int],
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
               ):
    path_graphs = Path(path_anchor_input_list[0].parent, "graphs")
    number_of_movies_list = ((np.array(movie_end_list) + 1) - np.array(movie_start_list)).tolist()

    labels_list = [[filename + f"{slice_index}_" + name for name in cut_names]
                    for slice_index, filename in zip(mixture_slice_index_list, filename_steps_list)]

    # construct list of all paths (path_mesh_list)
    path_mesh_list = []
    path_list: List[Path] = []
    for i, (path_anchor_input, movie_start, movie_end,
            number_of_movies, check_further) in enumerate(zip(path_anchor_input_list,
                                                              movie_start_list,
                                                              movie_end_list,
                                                              number_of_movies_list,
                                                              check_further_list)):
        path_inner_list: List[Path] = []
        for movie_number in range(movie_start, movie_end + 1):
            if movie_number == movie_skip:
                continue
            path_movie = Path(path_anchor_input, f"{dir_name}{counting_format % movie_number}"),
            path_movie = check_if_further(path_anchor_input, dir_name, counting_format,
                                          movie_number, experiment_step=number_of_movies,
                                          check_further=check_further)
            path_inner_list.append(path_movie)

        path_list.append(path_inner_list)
        path_mesh_list.append(np.array(path_inner_list).reshape(len(var2_list[i]),
                                                                len(var1_list[i])))

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
        if frame:
            path_output_frame: Path = Path(path_anchor_output, f"frame_{filename_format % frame}")
        else:
            path_output_frame: Path = Path(path_anchor_output, f"frame_last")

        if not path_output_frame.is_dir():
            path_output_frame.mkdir(parents=True)

        # use path_mesh to get last png of every animation
        path_mesh_new = path_mesh.copy()
        path_mesh_cuts = [path_mesh.copy() for _ in range(0, len(filename_steps_list))]

        for ix, iy in np.ndindex(path_mesh.shape):
            if frame is None:
                path_mesh_new[ix, iy] = get_last_png_in_last_anim(path_mesh[ix, iy],
                                                                  dir_name_png, counting_format_png,
                                                                  movie_take_last,
                                                                  filename_pattern, filename_format,
                                                                  filename_extension)
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
                                             filename_pattern + f"{filename_format % frame}"
                                             + filename_extension)

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
                                      + f"_frame_{frame_last}{filename_extension}"
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
                    shutil.copy(path_mesh_new[ix, iy], path_out)

        print(f"frame: {frame}, movie_take_last: {movie_take_last}")
        if frame:
            path_out_periodic: Path = Path(path_anchor_output,
                                           f"periodic_system_merge{suffix}_{experiment_suffix}"
                                           + f"_{filename_format % frame}" + ".png")
        else:
            path_out_periodic: Path = Path(path_anchor_output,
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
            # cut_1d.cut_1d(System_list,
                          # slice_indices=[self.slice_indices[0],
                                         # self.slice_indices[1],
                                         # self.slice_indices[2]],
                          # psi_sol_3d_cut_x=None,
                          # psi_sol_3d_cut_y=None,
                          # psi_sol_3d_cut_z=None,
                          # # dir_path=self.dir_path,
                          # dir_path=path_output_frame,
                          # y_lim=cut1d_y_lim,
                          # plot_val_list=cut1d_plot_val_list,
                          # frame=frame,
                          # steps_format=frame_format,
                          # mixture_slice_index_list=mixture_slice_index_list,
                          # filename_steps_list=filename_steps_list,
                          # )

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
        path_mesh_mirrored: List[Path] = np.flip(path_mesh_new, axis=0)
        probs_cuts_mirrored_list: List = [np.flip(probs_cuts_middle, axis=0),
                                          np.flip(probs_cuts_max, axis=0)]
        probs_cuts_filenames = ["probs_cuts_middle", "probs_cuts_max"]
        if frame:
            probs_cuts_filenames = [name + f"_{filename_format % frame}"
                                    for name in probs_cuts_filenames]

        for probs_cuts_filename, probs_cuts_mirrored in zip(probs_cuts_filenames,
                                                            probs_cuts_mirrored_list):
            with open(Path(path_anchor_output, probs_cuts_filename + ".npz"), "wb") as g:
                np.savez_compressed(g, probs_cuts_mesh=probs_cuts_mirrored)

        if video:
            if frame:
                path_out_video: Path = Path(path_anchor_output,
                                            f"periodic_video_{suffix}_{experiment_suffix}"
                                            + f"_{filename_format % frame}" + ".mp4")
            else:
                path_out_video: Path = Path(path_anchor_output,
                                            f"periodic_video_{suffix}_{experiment_suffix}"
                                            + ".mp4")

            paste_together_videos(path_mesh_mirrored, path_out_video, margin, width, height, fps)

        else:
            paste_together(path_mesh_mirrored.ravel(), path_out_periodic, nrow, ncol, ratio=dpi_ratio)

    if not video:
        if frame:
            path_out_periodic_all: Path = Path(path_anchor_output,
                                               f"periodic_system_merge_all_{experiment_suffix}"
                                               + f"{merge_suffix}"
                                               + f"_{filename_format % frame}" + ".png")
            path_out_cuts_all: Path = Path(path_anchor_output,
                                           f"cuts_all_{experiment_suffix}"
                                           + f"{merge_suffix}"
                                           + f"_{filename_format % frame}" + ".png")

        else:
            path_out_periodic_all: Path = Path(path_anchor_output,
                                               f"periodic_system_merge_all_{experiment_suffix}"
                                               + f"{merge_suffix}.png")
            path_out_cuts_all: Path = Path(path_anchor_output,
                                           f"cuts_all_{experiment_suffix}"
                                           + f"{merge_suffix}.png")

        # merge
        # turn off decompression bomb checker
        Image.MAX_IMAGE_PIXELS = number_of_movies * Image.MAX_IMAGE_PIXELS
        paste_together(path_in_list=path_out_periodic_list, path_out=path_out_periodic_all,
                       nrow=nrow_components, ncol=ncol_components, ratio=dpi_ratio_all)
        
        path_out_mesh_cuts_mirrored: List[Path] = np.flip(path_out_mesh_cuts, axis=0)
        # cuts
        # turn off decompression bomb checker
        Image.MAX_IMAGE_PIXELS = number_of_movies * Image.MAX_IMAGE_PIXELS
        paste_together(path_in_list=path_out_mesh_cuts_mirrored.ravel(), path_out=path_out_cuts_all,
                       nrow=nrow, ncol=ncol, ratio=dpi_ratio_all) 