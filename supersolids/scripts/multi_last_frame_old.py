#!/usr/bin/env python
import shutil
from pathlib import Path
from typing import List

from PIL import Image

from supersolids.helper.get_path import get_last_png_in_last_anim
from supersolids.helper.merge_meshes import check_if_further
from supersolids.helper.periodic_system import paste_together, periodic_system


# Script runs, if script is run as main script (called by python *.py)
if __name__ == "__main__":
    # path_anchor_input = Path("/home/dsche/supersolids/results/")
    # path_anchor_input = Path("/run/media/dsche/scr2/begin_mixture_13/")
    # path_anchor_input = Path("/run/media/dsche/ITP Transfer/begin_mixture_13/")
    # path_anchor_input = Path("/run/media/dsche/ITP Transfer/begin_mixture_15_6125/")
    path_anchor_input = Path("/run/media/dsche/ITP Transfer/begin_mixture_15/")
    # path_anchor_input = Path("/run/media/dsche/ITP Transfer/begin_mixture_15_6375/")

    filename_pattern: str = "anim"
    filename_format = "%05d"

    movie_start = 1
    movie_end = 60
    number_of_movies = (movie_end + 1) - movie_start
    # if simulation for movie_number was continued in dir with name movie_number + number_of_movies
    check_further = 1

    nrow = 10
    ncol = 6

    nrow_components = 1
    ncol_components = 2

    # adjust to shrink images, so RAM gets not overloaded
    dpi_ratio = 1.0

    dir_name = "movie"
    counting_format = "%03d"

    dir_name_png = "movie"
    counting_format_png = "%03d"

    # movie_take_last_list: int = [1]
    # suffix_list = [""]
    movie_take_last_list: int = [1, 2]
    # movie_take_last_list: int = [2, 1]
    suffix_list = ["_0", "_1"]
    dir_suffix_list = ["last_frame" + suf for suf in suffix_list]
    filename_out_list = ["last_frame" + suf for suf in suffix_list]
    filename_extension = ".png"

    file_suffix = ""
    # file_suffix = "-" + "-".join(map(str, property_args)) + ".png"
    property_filename_suffix_list = [dir_suffix + file_suffix for dir_suffix in dir_suffix_list]
    path_graphs = Path(path_anchor_input, "graphs")

    path_anchor_output_list = [Path(path_graphs, dir_suffix) for dir_suffix in dir_suffix_list]

    # Create a results dir, if there is none
    for path_anchor_output in path_anchor_output_list:
        if not path_anchor_output.is_dir():
            path_anchor_output.mkdir(parents=True)

    path_out_periodic_list: List[Path] = []
    for movie_take_last, path_anchor_output, suffix, filename_out in zip(movie_take_last_list,
                                                                         path_anchor_output_list,
                                                                         suffix_list,
                                                                         filename_out_list):
        path_out_list: List[Path] = []
        for movie_number in range(movie_start, movie_end + 1):
            path_movie = Path(path_anchor_input, f"{dir_name}{counting_format % movie_number}"),
            path_movie = check_if_further(path_anchor_input, dir_name, counting_format,
                                          movie_number, experiment_step=number_of_movies,
                                          check_further=check_further)

            path_last_png = get_last_png_in_last_anim(
                path_movie, dir_name_png, counting_format_png, movie_take_last,
                filename_pattern, filename_format, filename_extension)
            print(f"{path_last_png}")

            path_out: Path = Path(path_anchor_output,
                                  f"{filename_out}_{movie_number}{filename_extension}")
            path_out_list.append(path_out)
            shutil.copy(path_last_png, path_out)

        path_out_periodic: Path = Path(path_anchor_output, f"periodic_system{suffix}.png")
        path_out_periodic_list.append(path_out_periodic)
        periodic_system(path_in_list=path_out_list, path_out=path_out_periodic,
                        nrow=nrow, ncol=ncol)

    path_out_periodic_all: Path = Path(path_graphs, f"periodic_system_all.png")

    # turn off decompression bomb checker
    Image.MAX_IMAGE_PIXELS = number_of_movies * Image.MAX_IMAGE_PIXELS
    paste_together(path_in_list=path_out_periodic_list, path_out=path_out_periodic_all,
                   nrow=nrow_components, ncol=ncol_components, ratio=dpi_ratio)
