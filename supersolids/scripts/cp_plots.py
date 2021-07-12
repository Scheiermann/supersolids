#!/usr/bin/env python

from pathlib import Path
import shutil


def cp_plots(start, number, path_anchor_input, dir_name, filename_in,
             path_anchor_output, filename_out, counting_format="%03d", filename_extension=".png"):
    for i in range(start, start + number):
        movie_number = f"{counting_format % i}"
        # Create a results dir, if there is none
        if not path_anchor_output.is_dir():
            path_anchor_output.mkdir(parents=True)

        print(f"{movie_number}")
        shutil.copy(
            Path(path_anchor_input, dir_name + movie_number, filename_in + filename_extension),
            Path(path_anchor_output, filename_out + "_" + movie_number + filename_extension))


if __name__ == "__main__":
    path_anchor_input = Path("/run/media/dsche/ITP Transfer/joseph_injunction2/y_kick/kick_0.01/")
    path_anchor_output = Path("/run/media/dsche/ITP Transfer/joseph_injunction2/y_kick/kick_0.01/"
                              "graphs/center_of_mass_left/")

    filename_in = "get_center_of_mass_left"
    filename_out = filename_in
    filename_extension = ".png"

    start = 745
    number = 20
    dir_name = "movie"

    cp_plots(start, number, path_anchor_input, dir_name, filename_in,
             path_anchor_output, filename_out, filename_extension=".png")
