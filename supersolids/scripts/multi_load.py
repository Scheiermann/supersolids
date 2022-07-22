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
    path_anchor_input_list = []
    path_anchor_input_list.append(Path("/bigwork/dscheier/results/begin_ramp/"))
    
    # mixture = False
    mixture = True
    # no_legend = True
    no_legend = False

    # take_last = 3
    take_last = np.inf
    # frame_end = 100000
    frame_end = None

    steps_per_npz = 10000
    # steps_per_npz = 1

    movie_string = "movie"
    counting_format = "%03d"
    movie_start = 1
    movie_end = 50

    slice_indices = {"x": 63, "y": 31, "z": 15}
    # slice_indices = {"x": 127, "y": 63, "z": 15}
    # slice_indices = {"x": 127, "y": 31, "z": 15}
    # slice_indices = {"x": 127, "y": 31, "z": 31}

    if mixture:
        # mixture_slice_index_list = [1]
        # mixture_slice_index_list = [0]
        # filename_steps_list = ["step_"]
        # filename_steps_list = ["mixture_step_"]
        # filename_steps_list = ["mixture_mixture_step_pol_"]
        # mixture_slice_index_list = [1, 0]
        mixture_slice_index_list = [0, 1]
        filename_steps_list = ["step_", "step_"]
        # filename_steps_list = ["mixture_step_", "mixture_step_", "mixture_mixture_step_pol_"]
        # filename_steps_list = ["step_", "step_", "pol_"]
        alpha_psi_list = [0.0, 0.0]
        alpha_psi_sol_list = [0.0, 0.0]
    else:
        filename_steps = "step_"
        alpha_psi_list = [0.0]
        alpha_psi_sol_list = [0.0]
        
    cut1d = True
    if cut1d:
        cut1d_y_lim = [0.0, 1.0]
        # cut1d_plot_val_list = [False]
        # cut1d_plot_val_list = [False, True]
        cut1d_plot_val_list = [False, False, True]

    steps_format = "%07d"
    filename_pattern = ".npz"

    azimuth = 0.0
    elevation = 0.0
    # distance = 25.0
    distance = 30.0
    # distance = 60.0

    alpha_V = 0.0

    sum_along = None
    # sum_along = 2
    arg_slices = False
    plot_V = False
    ui = False
    # ui = True

    for path_anchor_input in path_anchor_input_list:
        for i in range(movie_start, movie_end + 1):
            path_in = Path(path_anchor_input, movie_string + f"{counting_format % i}")
            files = sorted([x for x in path_in.glob(filename_steps_list[0]
                            + "*" + filename_pattern) if x.is_file()])
            if len(files) > take_last:
                files_last = files[-take_last]
            else:
                try:
                    files_last = files[0]
                except IndexError:
                    # no files in dir
                    print(f'{str(Path(path_in, filename_steps_list[0] + "*" + filename_pattern))} '
                          f'not found. Skipping.')
                    continue

            frame_start = get_step_index(files_last,
                                         filename_prefix=filename_steps_list[0],
                                         file_pattern=filename_pattern)

            command = ["python", "-m", "supersolids.tools.load_npz"]
            flags_given = [f"-dir_path={path_anchor_input}",
                           f"-dir_name={movie_string}{counting_format % i}",
                           f"-frame_start={frame_start}",
                           f"-steps_per_npz={steps_per_npz}",
                           f"-steps_format={steps_format}",
                           f"-slice_indices={dic2str(slice_indices, single_quote_wrapped=False)}",
                           f"-azimuth={azimuth}",
                           f"-elevation={elevation}",
                           f"-distance={distance}",
                           f"--alpha_V={alpha_V}",
                           ]

            alpha_args_parsed = list(map(str, alpha_psi_list))
            flags_given.append(f"--alpha_psi_list")
            flags_given += alpha_args_parsed

            alpha_sol_args_parsed = list(map(str, alpha_psi_sol_list))
            flags_given.append(f"--alpha_psi_sol_list")
            flags_given += alpha_sol_args_parsed

            if filename_steps_list:
                filename_steps_parsed = list(map(str, filename_steps_list))
                flags_given.append(f"--filename_steps_list")
                flags_given += filename_steps_parsed

            if mixture:
                slice_index_parsed = list(map(str, mixture_slice_index_list))
                flags_given.append(f"--mixture_slice_index_list")
                flags_given += slice_index_parsed

            if cut1d_plot_val_list:
                cut1d_plot_val_parsed = list(map(str, cut1d_plot_val_list))
                flags_given.append("--cut1d_plot_val_list")
                flags_given += cut1d_plot_val_parsed

            if cut1d:
                # flags_given.append(f"-cut1d_y_lim={cut1d_y_lim}")
                cut1_y_lim_args_parsed = list(map(str, cut1d_y_lim))
                flags_given.append(f"--cut1d_y_lim")
                flags_given += cut1_y_lim_args_parsed

            if no_legend:
                flags_given.append("--no_legend")
            if arg_slices:
                flags_given.append("--arg_slices")
            if plot_V:
                flags_given.append("--plot_V")
            if ui:
                flags_given.append("--ui")
            if sum_along:
                flags_given.append(f"-sum_along={sum_along}")
            if frame_end:
                flags_given.append(f"-frame_end={frame_end}")

            flags_parsed = " ".join(flags_given)

            print(flags_given)
            args = flags(flags_given)
            load_npz(args)