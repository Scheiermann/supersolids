#!/usr/bin/env python
import shutil
from pathlib import Path
from supersolids.helper.get_path import get_path


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

    dir_name = "movie"
    counting_format = "%03d"

    dir_name_png = "movie"
    counting_format_png = "%03d"

    dir_suffix = "last_frame"
    filename_out = "last_frame"
    filename_extension = ".png"

    file_suffix = ""
    # file_suffix = "-" + "-".join(map(str, property_args)) + ".png"
    property_filename_suffix = dir_suffix + file_suffix

    path_anchor_output = Path(path_anchor_input, "graphs", dir_suffix)
    # Create a results dir, if there is none
    if not path_anchor_output.is_dir():
        path_anchor_output.mkdir(parents=True)

    for movie_number in range(movie_start, movie_end + 1):
        # gets last movie with animations of each movie
        path_last_movie_png, _, _, _ = get_path(
            Path(path_anchor_input, f"{dir_name}{counting_format % movie_number}"),
            dir_name=f"{dir_name_png}",
            counting_format=counting_format_png,
            file_pattern=""
            )
        # gets last anim.png of each last movie with animations
        path_last_png, _, _, _ = get_path(
            path_last_movie_png,
            dir_name=filename_pattern,
            counting_format=filename_format,
            file_pattern=filename_extension
            )
        print(f"{path_last_png}")
        shutil.copy(path_last_png,
                    Path(path_anchor_output, f"{filename_out}_{movie_number}{filename_extension}")
                    )