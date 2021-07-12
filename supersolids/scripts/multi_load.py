#!/usr/bin/env python

from pathlib import Path

from supersolids.tools.load_npz import load_npz, flags


def string_float(s):
    return s, float(s)


# Script runs, if script is run as main script (called by python *.py)
if __name__ == "__main__":
    path_anchor_input = Path("/run/media/dsche/ITP Transfer/joseph_injunction2/y_kick/kick_fix_0.01/")

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
        flags_given = [f"-dir_path={path_anchor_input}",
                 f"-dir_name={movie_string}{counting_format % i}",
                 f"-frame_start={frame_start}",
                 f"-steps_per_npz={steps_per_npz}",
                 f"-steps_format={steps_format}",
                 f'-slice_indices={{"x":127,"y":63,"z":15}}',
                 f"-azimuth={azimuth}",
                 f"-elevation={elevation}"]

        if arg_slices:
            flags_given.append("--arg_slices")
        if plot_V:
            flags_given.append("--plot_V")
        if ui:
            flags_given.append("--ui")

        flags_parsed = " ".join(flags_given)

        print(flags_given)
        args = flags(flags_given)
        load_npz(args)
