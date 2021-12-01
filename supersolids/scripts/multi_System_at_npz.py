#!/usr/bin/env python

import dill
import numpy as np
import subprocess

from pathlib import Path

from supersolids.scripts.cp_plots import cp_plots


def string_float(s):
    return s, float(s)


# Script runs, if script is run as main script (called by python *.py)
if __name__ == "__main__":
    path_anchor_input = Path("/run/media/dsche/scr2/")

    filename_schroedinger: str = "schroedinger.pkl"
    filename_steps: str = "mixture_step_"

    frame = None

    # path_dir_name_list = None
    path_dir_name_list = Path("/run/media/dsche/scr2/graphs/dir_name_list.pkl")
    if not path_dir_name_list:
        movie_start = 1
        movie_end = 70

    var1_arange = (0.05, 0.51, 0.05)
    # var2_arange = (0.60, 0.90, 0.05)
    var2_arange = (0.60, 0.90, 0.0125)
    var1_arange = np.arange(*var1_arange)
    var2_arange = np.arange(*var2_arange)

    dir_name = "movie"
    dir_name_format = "%03d"

    steps_format = "%07d"

    # property_func = False
    # property_name = "E"
    # property_name = "mu_arr"

    property_func = True
    # property_name = "get_center_of_mass"
    # property_name = "get_parity"
    # property_name = "get_peak_distances_along"
    # property_name = "get_peak_positions"
    # property_args = [0]
    # property_args = []

    property_name = "get_contrast_old"
    box = [117, 137, 62, 64, 14, 16]
    property_args = [2, *box]

    # property_name = "get_contrast"
    # prob_min_start = 0.3
    # prob_step = 0.01
    # number_of_peaks = [1, 2, 2, 2, 1, 1, 1,
    #                    2, 2, 3, 3, 1, 1, 1,
    #                    3, 4, 4, 4, 1, 1, 1,
    #                    4, 5, 5, 4, 1, 1, 1,
    #                    5, 5, 5, 5, 1, 1, 1,
    #                    5, 6, 5, 5, 1, 1, 1,
    #                    6, 6, 6, 5, 1, 1, 1,
    #                    5, 5, 5, 5, 1, 1, 1,
    #                    6, 6, 5, 4, 1, 1, 1,
    #                    6, 5, 5, 4, 1, 1, 1,
    #                    ]

    # property_name = "get_phase_var"
    # property_args = [0, 256, 0, 128, 0, 32]
    # property_args = [0, 128, 0, 128, 0, 32]
    # property_args = [128, 256, 0, 128, 0, 32]

    # property_name = "get_phase_var_neighborhood"
    # [prob_min, amount]
    # property_args = [0.02, 4]
    #
    # property_name = "get_N_in_droplets"
    # number_of_peaks = [5, 4, 1, 1, 1, 1, 1, 1,
    #                    5, 4, 3, 1, 1, 1, 1, 1,
    #                    5, 5, 3, 1, 1, 1, 1, 1,
    #                    5, 5, 4, 1, 1, 1, 1, 1,
    #                    6, 5, 6, 3, 1, 1, 1, 1,
    #                    ]

    dir_name_list = []
    property_args_list = []
    if path_dir_name_list:
        with open(path_dir_name_list, "rb") as f:
            # WARNING: this is just the input Schroedinger at t=0
            dir_name_list = dill.load(file=f)
        for dir in dir_name_list:
            property_args_list.append(property_args)

    else:
        for peak_index, movie_number in enumerate(range(movie_start, movie_end + 1)):
            # property_args = [number_of_peaks[peak_index], prob_min_start, prob_step]
            property_args_list.append(property_args)
            dir_name_list.append(f"{dir_name}{dir_name_format % movie_number}")

    command = ["python", "-m", "supersolids.tools.get_System_at_npz"]
    flags = [f"-dir_path={path_anchor_input}",
             f"-filename_schroedinger={filename_schroedinger}",
             f"-filename_steps={filename_steps}",
             f"-steps_format={steps_format}",
             f"-property_name={property_name}",
             ]

    var1_arange_args_parsed = list(map(str, var1_arange))
    flags.append(f"-var1_arange")
    flags += var1_arange_args_parsed

    var2_arange_args_parsed = list(map(str, var2_arange))
    flags.append(f"-var2_arange")
    flags += var2_arange_args_parsed

    if dir_name_list.any():
        property_args_parsed = list(map(str, dir_name_list))
        flags.append(f"-dir_name_list")
        flags += property_args_parsed
    if property_func:
        flags.append("--property_func")
    if property_args_list:
        for property_args in property_args_list:
            property_args_parsed = list(map(str, property_args))
            flags.append(f"--property_args_list")
            flags += property_args_parsed
    if frame:
        flags.append(f"-frame={frame}")

    # command needs flags to be provided as list
    command.extend(flags)

    print(command)
    p = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, shell=False)

    # communicate needs flags to be provided as string seperated by " ".
    flags_parsed = " ".join(flags)
    out, err = p.communicate(flags_parsed.encode())
    print(f"out:\n{out}\n")
    print(f"err:\n{err}\n")
