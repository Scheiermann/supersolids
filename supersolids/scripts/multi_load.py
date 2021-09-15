#!/usr/bin/env python

# author: Daniel Scheiermann
# email: daniel.scheiermann@itp.uni-hannover.de
# license: MIT
# Please feel free to use and modify this, but keep the above information.

from pathlib import Path
import numpy as np

from supersolids.helper.get_path import get_step_index
from supersolids.tools.load_npz import load_npz
from supersolids.tools.load_npz import flags


def string_float(s):
    return s, float(s)


# Script runs, if script is run as main script (called by python *.py)
if __name__ == "__main__":
    # path_anchor_input = Path("/run/media/dsche/ITP Transfer/joseph_injunction2/y_kick/kick_fix_0.01/")
    # path_anchor_input = Path("/run/media/dsche/ITP Transfer/scissor/")
    # path_anchor_input = Path("/run/media/dsche/ITP Transfer/test/")
    path_anchor_input = Path("/run/media/dsche/ITP Transfer/begin_alpha/")

    take_last = 30
    # take_last = np.inf

    steps_per_npz = 1000

    movie_string = "movie"
    counting_format = "%03d"
    movie_start = 1
    movie_end = 20

    filename_steps = "step_"
    steps_format = "%07d"
    filename_pattern = ".npz"

    azimuth = 0.0
    elevation = 0.0

    arg_slices = False
    plot_V = True
    ui = False

    for i in range(movie_start, movie_end + 1):
        path_in = Path(path_anchor_input, movie_string + f"{counting_format % i}")
        files = sorted([x for x in path_in.glob("*" + filename_pattern) if x.is_file()])
        if len(files) > take_last:
            files_last = files[-take_last]
        else:
            try:
                files_last = files[0]
            except IndexError:
                # no files in dir
                print(f'{str(Path(path_in)) + "/*" + filename_pattern} not found. Skipping.')
                continue

        frame_start = get_step_index(files_last,
                                     filename_prefix=filename_steps,
                                     file_pattern=filename_pattern)

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
