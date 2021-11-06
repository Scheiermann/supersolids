#!/usr/bin/env python
import shutil
from pathlib import Path
from typing import List

from PIL import Image

from supersolids.helper.get_path import get_path


def periodic_system(path_in_list: List[Path], path_out: Path, nrow: int, ncol: int):
    fs = []
    for i, path_in in enumerate(path_in_list):
        im = Image.open(path_in, 'r')
        fs.append(im)

    x, y = fs[0].size
    cvs = Image.new('RGB', (x * ncol, y * nrow))
    for i in range(len(fs)):
        px, py = x * (i % ncol), y * int(i / ncol)
        cvs.paste(fs[i], (px, py))

    cvs.save(path_out, format='png')


# Script runs, if script is run as main script (called by python *.py)
if __name__ == "__main__":
    # path_anchor_input = Path("/home/dsche/supersolids/results/begin_mixture_13/")
    path_anchor_input = Path("/run/media/dsche/scr2/begin_mixture_13/")
    # path_anchor_input = Path("/home/dsche/supersolids/results/")

    filename_pattern: str = "anim"
    filename_format = "%05d"

    frame_start = 0
    steps_per_npz = 1000

    movie_start = 1
    movie_end = 70

    ncomponents = 2
    nrow = 10
    ncol = 7


    dir_name = "movie"
    counting_format = "%03d"

    dir_name_png = "movie"
    counting_format_png = "%03d"

    movie_take_last: int = 2
    dir_suffix = "last_frame_1"
    filename_out = "last_frame_1"
    filename_extension = ".png"

    file_suffix = ""
    # file_suffix = "-" + "-".join(map(str, property_args)) + ".png"
    property_filename_suffix = dir_suffix + file_suffix

    path_anchor_output = Path(path_anchor_input, "graphs", dir_suffix)
    # Create a results dir, if there is none
    if not path_anchor_output.is_dir():
        path_anchor_output.mkdir(parents=True)

    path_out_list: List[Path] = []
    for movie_number in range(movie_start, movie_end + 1):
        # gets last movie with animations of each movie
        path_last_movie_png, _, _, _ = get_path(
            Path(path_anchor_input, f"{dir_name}{counting_format % movie_number}"),
            search_prefix=f"{dir_name_png}",
            counting_format=counting_format_png,
            file_pattern="",
            take_last=movie_take_last,
            )
        # gets last anim.png of each last movie with animations
        path_last_png, _, _, _ = get_path(
            path_last_movie_png,
            search_prefix=filename_pattern,
            counting_format=filename_format,
            file_pattern=filename_extension,
            )
        print(f"{path_last_png}")
        path_out: Path = Path(path_anchor_output, f"{filename_out}_{movie_number}{filename_extension}")
        path_out_list.append(path_out)
        shutil.copy(path_last_png, path_out)

    path_out_periodic: Path = Path(path_anchor_output, f"periodic_system.png")
    periodic_system(path_in_list=path_out_list, path_out=path_out_periodic, nrow=nrow, ncol=ncol)
