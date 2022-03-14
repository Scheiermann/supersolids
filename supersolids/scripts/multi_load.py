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
from supersolids.helper.dict2str import dic2str


def string_float(s):
    return s, float(s)


# Script runs, if script is run as main script (called by python *.py)
if __name__ == "__main__":
    # path_anchor_input = Path("/run/media/dsche/ITP Transfer/test/")
    # path_anchor_input = Path("/run/media/dsche/scr2/begin_mixture/")
    # path_anchor_input = Path("/home/dsche/supersolids/results/begin_mixture/")
    # path_anchor_input = Path("/run/media/dsche/scr2/begin_mixture_13/")
    # path_anchor_input = Path("/run/media/dsche/ITP Transfer/frenkel/grid/from_step_1150000/steps120k/")
    # path_anchor_input = Path("/run/media/dsche/ITP Transfer/frenkel/grid/from_step_1270000/steps40k/")
    # path_anchor_input = Path("/run/media/dsche/ITP Transfer/frenkel/cos/from_step_637000/")
    # path_anchor_input = Path("/run/media/dsche/ITP Transfer/frenkel/sin/from_step_999000/")
    # path_anchor_input = Path("/run/media/dsche/ITP Transfer/begin_mixture_15/")
    # path_anchor_input = Path("/run/media/dsche/ITP Transfer/joseph_injunction2/")
    # path_anchor_input = Path("/run/media/dsche/ITP Transfer/joseph_injunction2/no_kick/")
    # path_anchor_input = Path("/run/media/dsche/ITP Transfer/frenkel/sin_hold/")
    # path_anchor_input = Path("/run/media/dsche/ITP Transfer/frenkel/sin_shift/")
    # path_anchor_input = Path("/run/media/dsche/ITP Transfer/frenkel/sin_shift01/")
    # path_anchor_input = Path("/home/dsche/supersolids/results/")
    # path_anchor_input = Path("/home/dsche/supersolids/results/begin_schroedinger/")
    # path_anchor_input = Path("/run/media/dsche/ITP Transfer/begin_schroedinger/")
    # path_anchor_input = Path("/run/media/dsche/ITP Transfer/begin_mixture_13/")
    # path_anchor_input = Path("/run/media/dsche/ITP Transfer/begin_mixture_15_6125/")
    # path_anchor_input = Path("/run/media/dsche/ITP Transfer/begin_mixture_15/")
    # path_anchor_input = Path("/run/media/dsche/ITP Transfer/begin_mixture_15_6375/")
    path_anchor_input = Path("/run/media/dsche/scr2/begin_mixture_15_6375/")
    # path_anchor_input = Path("/run/media/dsche/scr2/begin_mixture_a12/")

    # mixture = False
    mixture = True
    no_legend = True
    # no_legend = False

    take_last = 3
    # take_last = np.inf

    steps_per_npz = 1000

    movie_string = "movie"
    counting_format = "%03d"
    movie_start = 1
    movie_end = 114

    slice_indices = {"x": 127, "y": 63, "z": 15}

    if mixture:
        mixture_slice_index = 0
        filename_steps = "mixture_step_"
        alpha_psi_list = [0.0, 0.0]
        alpha_psi_sol_list = [0.0, 0.0]
    else:
        filename_steps = "step_"
        alpha_psi_list = [0.0]
        alpha_psi_sol_list = [0.0]

    steps_format = "%07d"
    filename_pattern = ".npz"

    azimuth = 0.0
    elevation = 0.0
    distance = 29.0

    alpha_V = 0.0

    sum_along = 2
    arg_slices = False
    plot_V = False
    ui = False

    for i in range(movie_start, movie_end + 1):
        path_in = Path(path_anchor_input, movie_string + f"{counting_format % i}")
        files = sorted([x for x in path_in.glob(filename_steps + "*" + filename_pattern) if x.is_file()])
        if len(files) > take_last:
            files_last = files[-take_last]
        else:
            try:
                files_last = files[0]
            except IndexError:
                # no files in dir
                print(f'{str(Path(path_in, filename_steps + "*" + filename_pattern))} '
                      f'not found. Skipping.')
                continue

        frame_start = get_step_index(files_last,
                                     filename_prefix=filename_steps,
                                     file_pattern=filename_pattern)

        command = ["python", "-m", "supersolids.tools.load_npz"]
        flags_given = [f"-dir_path={path_anchor_input}",
                       f"-dir_name={movie_string}{counting_format % i}",
                       f"-frame_start={frame_start}",
                       f"-filename_steps={filename_steps}",
                       f"-steps_per_npz={steps_per_npz}",
                       f"-steps_format={steps_format}",
                       f"-slice_indices={dic2str(slice_indices, single_quote_wrapped=False)}",
                       f"-azimuth={azimuth}",
                       f"-elevation={elevation}",
                       f"-distance={distance}",
                       f"-sum_along={sum_along}",
                       f"--alpha_V={alpha_V}",
                       ]

        alpha_args_parsed = list(map(str, alpha_psi_list))
        flags_given.append(f"--alpha_psi_list")
        flags_given += alpha_args_parsed

        alpha_args_parsed = list(map(str, alpha_psi_sol_list))
        flags_given.append(f"--alpha_psi_sol_list")
        flags_given += alpha_args_parsed

        if no_legend:
            flags_given.append("--no_legend")
        if arg_slices:
            flags_given.append("--arg_slices")
        if plot_V:
            flags_given.append("--plot_V")
        if ui:
            flags_given.append("--ui")
        if mixture:
            flags_given.append(f"-mixture_slice_index={mixture_slice_index}")

        flags_parsed = " ".join(flags_given)

        print(flags_given)
        args = flags(flags_given)
        load_npz(args)
