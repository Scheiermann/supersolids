#!/usr/bin/env python

import dill
import numpy as np
import shutil

from pathlib import Path
from typing import List

from PIL import Image

from supersolids.helper.get_path import get_last_png_in_last_anim
from supersolids.helper.merge_meshes import check_if_further, merge_meshes
from supersolids.helper.periodic_system import paste_together


# Script runs, if script is run as main script (called by python *.py)
if __name__ == "__main__":
    path_anchor_input_list: List[Path] = []
    var1_list = []
    var2_list = []
    
    # path_anchor_input_list.append(Path("/run/media/dsche/scr2/begin_mixture_19/"))
    # var1_list.append(np.arange(0.005, 0.05, 0.005))
    # var2_list.append(np.arange(0.6, 0.66, 0.05))

    # path_anchor_input_list.append(Path("/run/media/dsche/ITP Transfer/begin_mixture_13/"))
    path_anchor_input_list.append(Path("/run/media/dsche/scr2/begin_mixture_13/"))
    var1_list.append(np.arange(0.6, 0.91, 0.05))
    var2_list.append(np.arange(0.05, 0.51, 0.05))

    # # path_anchor_input_list.append(Path("/run/media/dsche/ITP Transfer/begin_mixture_15_6125/"))
    # path_anchor_input_list.append(Path("/run/media/dsche/scr2/begin_mixture_15_6125/"))
    # var1_list.append(np.arange(0.6125, 0.91, 0.05))
    # var2_list.append(np.arange(0.05, 0.51, 0.05))

    # path_anchor_input_list.append(Path("/run/media/dsche/ITP Transfer/begin_mixture_15/"))
    path_anchor_input_list.append(Path("/run/media/dsche/scr2/begin_mixture_15/"))
    var1_list.append(np.arange(0.625, 0.91, 0.05))
    var2_list.append(np.arange(0.05, 0.51, 0.05))

    # # path_anchor_input_list.append(Path("/run/media/dsche/ITP Transfer/begin_mixture_15_6375/"))
    # path_anchor_input_list.append(Path("/run/media/dsche/scr2/begin_mixture_15_6375/"))
    # var1_list.append(np.arange(0.6375, 0.91, 0.05))
    # var2_list.append(np.arange(0.05, 0.51, 0.05))

    nrow_components = 2
    ncol_components = 1

    movie_take_last_list: int = [2, 1]
    suffix_list = ["_0", "_1"]
    dir_suffix_list = ["last_frame_middle" + suf for suf in suffix_list]
    filename_out_list = ["last_frame_middle" + suf for suf in suffix_list]
    file_suffix = ""
    # file_suffix = "-" + "-".join(map(str, property_args)) + ".png"

    # movie_start_list = [1]
    # movie_end_list = [18]
    movie_start_list = [1, 1]
    movie_end_list = [70, 60]
    # movie_start_list = [1, 1, 1, 1]
    # movie_end_list = [70, 60, 60, 60]
    number_of_movies_list = ((np.array(movie_end_list) + 1) - np.array(movie_start_list)).tolist()
    # if simulation for movie_number was continued in dir with name movie_number + number_of_movies
    # check_further_list = [1, 1, 1, 1]
    check_further_list = [2, 2]
    # check_further_list = [2]

    dir_name = "movie"
    counting_format = "%03d"

    # input for get_last_png_in_last_anim
    dir_name_png = "movie"
    counting_format_png = "%03d"
    filename_pattern: str = "anim"
    filename_format = "%05d"
    filename_extension = ".png"

    path_graphs = Path(path_anchor_input_list[0].parent, "graphs")

    # adjust to shrink images, so RAM gets not overloaded
    dpi_ratio = 0.25
    dpi_ratio_all = 1.0

    ######## END OF USER INPUT #####################################################################

    path_anchor_output_list = [Path(path_graphs, dir_suffix) for dir_suffix in dir_suffix_list]
    # property_filename_suffix_list = [dir_suffix + file_suffix for dir_suffix in dir_suffix_list]

    # Create a results dir, if there is none
    for path_anchor_output in path_anchor_output_list:
        if not path_anchor_output.is_dir():
            path_anchor_output.mkdir(parents=True)

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
    path_dirname_list = Path(path_graphs, "dir_name_list")
    with open(path_dirname_list.with_suffix(".pkl"), "wb") as f:
        dill.dump(obj=dir_name_list, file=f)
    with open(path_dirname_list.with_suffix(".txt"), "w") as f:
        f.write(f"{dir_name_list}\n")


    for movie_take_last, path_anchor_output, suffix, filename_out in zip(movie_take_last_list,
                                                                         path_anchor_output_list,
                                                                         suffix_list,
                                                                         filename_out_list):
        # use path_mesh to get lanst png of every animation
        path_mesh_new = path_mesh.copy()
        for ix, iy in np.ndindex(path_mesh.shape):
            path_mesh_new[ix, iy] = get_last_png_in_last_anim(path_mesh[ix, iy],
                                                              dir_name_png, counting_format_png,
                                                              movie_take_last,
                                                              filename_pattern, filename_format,
                                                              filename_extension)

            path_currently_old: Path = path_mesh[ix, iy]
            path_out: Path = Path(path_anchor_output,
                                  f"{path_currently_old.parent.stem}_{path_currently_old.stem}"
                                  + f"_{filename_out}{filename_extension}")

            path_currently_new: Path = path_mesh_new[ix, iy]
            if path_currently_new is not None:
                shutil.copy(path_mesh_new[ix, iy], path_out)

        print(f"movie_take_last: {movie_take_last}")

        path_out_periodic: Path = Path(path_graphs, f"periodic_system_merge{suffix}.png")
        path_out_periodic_list.append(path_out_periodic)
        nrow, ncol = path_mesh_new.shape
        # flip needed as appending pictures start from left top corner,
        # but enumeration of movies from left bottom corner
        path_mesh_mirrored: List[Path] = np.flip(path_mesh_new, axis=0)
        paste_together(path_mesh_mirrored.ravel(), path_out_periodic, nrow, ncol, ratio=dpi_ratio)

    path_out_periodic_all: Path = Path(path_graphs, f"periodic_system_merge_all.png")
    # turn off decompression bomb checker
    Image.MAX_IMAGE_PIXELS = number_of_movies * Image.MAX_IMAGE_PIXELS
    paste_together(path_in_list=path_out_periodic_list, path_out=path_out_periodic_all,
                   nrow=nrow_components, ncol=ncol_components, ratio=dpi_ratio_all)
