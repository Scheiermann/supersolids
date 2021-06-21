#!/usr/bin/env python

from pathlib import Path

from supersolids.tools.load_npz import load_npz


def string_float(s):
    return s, float(s)


# Script runs, if script is run as main script (called by python *.py)
if __name__ == "__main__":
    dir_path = Path("/run/media/dsche/ITP Transfer/joseph_injunction2/real2/")

    frame_start = 1210000
    # frame_start = 1510000
    steps_per_npz = 1000

    movie_string = "movie"
    counting_format = "%03d"
    movie_start = 1
    movie_end = 20

    steps_format = "%07d"

    azimuth = 0.0
    elevation = 0.0

    arg_slices = True
    plot_V = True
    ui = False

    for i in range(movie_start, movie_end + 1):
        command = ["python", "-m", "supersolids.tools.load_npz"]
        flags_non_split = f"""-dir_path={dir_path.as_posix()}"""
        flags = f"""-dir_name={movie_string}{counting_format % i} -frame_start={frame_start} -steps_per_npz={steps_per_npz} -steps_format={steps_format} -slice_indices={{"x":127,"y":63,"z":15}} -azimuth={azimuth} -elevation={elevation}"""

        if arg_slices:
            flags += " --arg_slices"
        if plot_V:
            flags += " --plot_V"
        if ui:
            flags += " --ui"

        flags_splitted = flags.split(" ") + [flags_non_split]

        print(flags_splitted)
        load_npz(flags_splitted)
