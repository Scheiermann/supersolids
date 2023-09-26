#!/usr/bin/env python
import subprocess
from pathlib import Path

from supersolids.helper import get_path
from supersolids.scripts.cp_plots import cp_plots


def string_float(s):
    return s, float(s)


# Script runs, if script is run as main script (called by python *.py)
if __name__ == "__main__":
    path_anchor_input_list = []

    # experiment_suffix = "ramp_13_09_10**eps"
    # experiment_suffix = "ramp_21_09_a12=70"
    # experiment_suffix = "ramp_test00"
    # experiment_suffix = "ramp_05_10"
    # experiment_suffix = "ramp_05_10_a12=70"
    # experiment_suffix = "ramp_10_10"
    # path_anchor_input_list.append(Path(f"/bigwork/dscheier/results/begin_{experiment_suffix}/"))
    # experiment_suffix = "ramp_10_10_small"
    # path_anchor_input_list.append(Path(f"/bigwork/dscheier/results/begin_{experiment_suffix}/"))
    # experiment_suffix = "ramp_11_10_65"
    # path_anchor_input_list.append(Path(f"/bigwork/dscheier/results/begin_{experiment_suffix}/"))
    # experiment_suffix = "ramp_11_10_775"
    # path_anchor_input_list.append(Path(f"/bigwork/dscheier/results/begin_{experiment_suffix}/"))
    # experiment_suffix = "ramp_11_10_85"
    # path_anchor_input_list.append(Path(f"/bigwork/dscheier/results/begin_{experiment_suffix}/"))
    # experiment_suffix = "ramp_21_10"
    # path_anchor_input_list.append(Path(f"/bigwork/dscheier/results/begin_{experiment_suffix}/"))
    # experiment_suffix = "ramp_21_10_675"
    # path_anchor_input_list.append(Path(f"/bigwork/dscheier/results/begin_{experiment_suffix}/"))
    # experiment_suffix = "ramp_21_10_725"
    # path_anchor_input_list.append(Path(f"/bigwork/dscheier/results/begin_{experiment_suffix}/"))
    # experiment_suffix = "ramp_21_10_825"
    # path_anchor_input_list.append(Path(f"/bigwork/dscheier/results/begin_{experiment_suffix}/"))
    # experiment_suffix = "ramp_11_10_85_long"
    # path_anchor_input_list.append(Path(f"/bigwork/dscheier/results/begin_{experiment_suffix}/"))
    # experiment_suffix = "stacked_05_10_a11"

    # experiment_suffix = "ramp_24_10_fix_movie008"
    # path_anchor_input_list.append(Path(f"/bigwork/dscheier/results/begin_{experiment_suffix}/"))
    # experiment_suffix = "ramp_24_10_fix_movie015"
    # path_anchor_input_list.append(Path(f"/bigwork/dscheier/results/begin_{experiment_suffix}/"))
    # experiment_suffix = "ramp_21_10_90"
    # path_anchor_input_list.append(Path(f"/bigwork/dscheier/results/begin_{experiment_suffix}/"))
    # experiment_suffix = "ramp_21_10_925"
    # path_anchor_input_list.append(Path(f"/bigwork/dscheier/results/begin_{experiment_suffix}/"))
    # experiment_suffix = "ramp_24_10_long_85"
    # path_anchor_input_list.append(Path(f"/bigwork/dscheier/results/begin_{experiment_suffix}/"))
    # experiment_suffix = "ramp_24_10_long"
    # path_anchor_input_list.append(Path(f"/bigwork/dscheier/results/begin_{experiment_suffix}/"))
    # experiment_suffix = "ramp_28_10_65_long"
    # path_anchor_input_list.append(Path(f"/bigwork/dscheier/results/begin_{experiment_suffix}/"))
    # experiment_suffix = "ramp_28_10_85_long"
    # experiment_suffix = "ramp_11_01_65_long_wide"
    # path_anchor_input_list.append(Path(f"/bigwork/dscheier/results/begin_{experiment_suffix}/"))
    # experiment_suffix = "ramp_11_01_85_long_wide"
    # path_anchor_input_list.append(Path(f"/bigwork/dscheier/results/begin_{experiment_suffix}/"))

    # experiment_suffix = "ramp_11_04_675_long_wide"
    # path_anchor_input_list.append(Path(f"/bigwork/dscheier/results/begin_{experiment_suffix}/"))
    # experiment_suffix = "ramp_11_04_70_long_wide"
    # path_anchor_input_list.append(Path(f"/bigwork/dscheier/results/begin_{experiment_suffix}/"))
    # experiment_suffix = "ramp_11_04_725_long_wide"
    # path_anchor_input_list.append(Path(f"/bigwork/dscheier/results/begin_{experiment_suffix}/"))
    # experiment_suffix = "ramp_11_04_75_775_80_long_wide"
    # path_anchor_input_list.append(Path(f"/bigwork/dscheier/results/begin_{experiment_suffix}/"))

    # experiment_suffix = "gpu_11_18_real_w-1"
    # experiment_suffix = "gpu_12_05"
    # experiment_suffix = "gpu_12_06"
    # experiment_suffix = "gpu_12_07"
    # experiment_suffix = "gpu_12_23"
    # experiment_suffix = "gpu_12_28"
    # experiment_suffix = "gpu_12_28_to_102"
    # experiment_suffix = "gpu_12_28_to_102_dip9"
    # experiment_suffix = "gpu_01_13_dip9"
    # path_anchor_input_list.append(Path(f"/bigwork/dscheier/results/begin_{experiment_suffix}/"))

    # experiment_suffix = "gpu_03_15_no_dipol_no_lhy_1comp_w_paper"
    # path_anchor_input_list.append(Path(f"/home/dscheiermann/results/begin_{experiment_suffix}"))

    # experiment_suffix = "gpu_2023_08_23_part2"
    # experiment_suffix = "gpu_2023_08_23"
    # experiment_suffix = "gpu_2023_09_04"
    experiment_suffix = "gpu_2023_09_04_dip9"
    path_anchor_input_list.append(Path(f"/bigwork/dscheier/results/begin_{experiment_suffix}"))

    mixture = True
    filename_schroedinger: str = "schroedinger.pkl"

    if mixture:
        filename_steps = "step_"
        # filename_steps = "mixture_step_"
    else:
        filename_steps = "step_"

    # frame_start = None
    # frame_start = 0
    # frame_start = 100000
    # frame_start = 110000
    # frame_start = 70000
    # frame_start = 150000
    frame_start = 350000
    if frame_start is None:
        take_last = 4
    else:
        take_last = None
        # frame_start = 0
        # frame_end = 1000
        # frame_end = None
        frame_end = 700000
        # frame_end = 600000
        # frame_end = 500000

    # movie_start_list = [1]
    # movie_end_list = [1]
    # movie_start_list = [306]
    # movie_end_list = [350]
    # movie_start_list = [406]
    # movie_end_list = [450]
    # movie_start_list = [506]
    # movie_end_list = [550]
    movie_start_list = [601]
    movie_end_list = [650]

    dt = 0.0002
    # dt = 0.002
    # dt = 0.0015
    # dt = 0.02

    dir_name = "movie"
    counting_format = "%03d"

    steps_format = "%07d"

    if take_last is None:
        dir_suffix = f"_paper_framestart_{frame_start}"
        # dir_suffix = "_right"
    else:
        dir_suffix = f"_paper_takelast_{take_last}"

    # subplots = False
    # property_func = False
    # property_names = ["E", "mu_arr"]
    # list_of_arrays = False

    # subplots_list = [True, False, False, True, False]
    # property_func_list = [True, False, False, True, True]
    # list_of_arrays_list = [True, False, False, True, False]
    # property_names_list = ["get_center_of_mass", "E", "mu_arr", "get_parity", "check_N"]
    # property_args_list = [[], [], [], [], []]
    # property_args_frame_list = [False, False, False, False, False]

    # subplots_list = [False, False, False, True, True]
    # property_func_list = [False, True, False, True, True]
    # list_of_arrays_list = [False, False, False, True, True]
    # property_names_list = ["E", "get_E_explicit", "mu_arr", "get_center_of_mass", "get_parity"]
    # property_args_list = [[], [], [], [], []]
    # property_args_frame_list = [False, False, False, False, False]

    # func = "lambda x, y: (x, y)"
    # func_mu = "lambda x, y: (x[1:], np.abs(np.diff(y)) / np.abs(y[1:]))"
    # file_suffix_list = ["", "", "", "mu_rel", "", "", "", "_fft"]
    # inbuild_func_list = ["", "", "", "", "", "", "", "fft_plot"]
    # inbuild_func_args_list = [[], [], [], [], [], [], [], []]
    # func_list = ["", "", "", func_mu, "", "", "", ""]
    # # steps_per_npz_list = [100, 1000, 100, 100, 1000, 1000, 100, 100]
    # # steps_per_npz_list = [1, 1, 1, 1, 1, 1, 1, 1]
    # steps_per_npz_list = [1, 1, 1, 1, 1, 1, 1, 1]
    # subplots_list = [False, False, False, False, True, True, True, True]
    # property_func_list = [False, True, False, False, True, True, False, False]
    # list_of_arrays_list = [False, False, False, False, True, True, True, True]
    # property_names_list = ["E", "get_E_explicit", "mu_arr", "mu_arr", "get_center_of_mass", "get_parity", "monopolar", "monopolar"]
    # property_args_list = [[], [], [], [], [], [], [], []]
    # property_args_frame_list = [False, False, False, False, False, False, False, False]

    func = "lambda x, y: (x, y)"
    file_suffix_list = ["", "_fft"]
    inbuild_func_list = ["", "fft_plot"]
    # inbuild_func_args_list = [[], [1, 1000]]
    # inbuild_func_args_list = [[], [1, 200]]
    # inbuild_func_args_list = [[], [1, 60]]
    inbuild_func_args_list = [[], [1, 80]]
    func_list = ["", ""]
    steps_per_npz_list = [100, 100]
    subplots_list = [True, True]
    property_func_list = [False, False]
    list_of_arrays_list = [True, True]
    property_names_list = ["monopolar", "monopolar"]
    property_args_list = [[], []]
    property_args_frame_list = [False, False]

    # func = "lambda x, y: (x, y)"
    # file_suffix_list = ["_fft"]
    # inbuild_func_list = ["fft_plot"]
    # inbuild_func_args_list = [[1, 80]]
    # func_list = [""]
    # steps_per_npz_list = [100]
    # subplots_list = [True]
    # property_func_list = [False]
    # list_of_arrays_list = [True]
    # property_names_list = ["monopolar"]
    # property_args_list = [[]]
    # property_args_frame_list = [False]


    # box = [80, 176, 28, 36, 28, 36]
    # # box = [88, 168, 30, 34, 30, 34]
    # # box = [0, 256, 0, 64, 0, 64]
    # func = "lambda x, y: (x, y)"
    # file_suffix_list = [""]
    # inbuild_func_list = [""]
    # inbuild_func_args_list = [[]]
    # func_list = [""]
    # steps_per_npz_list = [10000]
    # subplots_list = [False]
    # property_func_list = [True]
    # list_of_arrays_list = [False]
    # property_names_list = ["get_contrast_old"]
    # property_args_list = [[2, *box]]
    # property_args_frame_list = [False]



    # func_mu = "lambda x, y: (x[1:], np.abs(np.diff(y)) / np.abs(y[1:]))"
    # file_suffix_list = ["", "mu_rel", "", ""]
    # inbuild_func_list = ["", "", "", ""]
    # func_list = ["", func_mu, "", ""]
    # steps_per_npz_list = [100, 100, 1000, 1000]
    # subplots_list = [False, False, True, True]
    # property_func_list = [False, False, True, True]
    # list_of_arrays_list = [False, False, True, True]
    # property_names_list = ["mu_arr", "mu_arr", "get_center_of_mass", "get_parity"]
    # property_args_list = [[], [], [], []]
    # property_args_frame_list = [False, False, False, False]


    # file_suffix_list = ["", "_fft"]
    # inbuild_func_list = ["", "fft_plot"]
    # func_list = ["", ""]
    # steps_per_npz_list = [100, 100]
    # subplots_list = [True, True]
    # property_func_list = [False, False]
    # list_of_arrays_list = [True, True]
    # property_names_list = ["monopolar", "monopolar"]
    # property_args_list = [[], []]
    # property_args_frame_list = [False, False]

    # file_suffix_list = ["_fft"]
    # inbuild_func_list = ["fft_plot"]
    # func_list = [""]
    # steps_per_npz_list = [100]
    # subplots_list = [True]
    # property_func_list = [False]
    # list_of_arrays_list = [True]
    # property_names_list = ["monopolar"]
    # property_args_list = [[]]
    # inbuild_func_args_list = [[1, 62, 1]]
    # property_args_frame_list = [False]

    # file_suffix_list = ["_contrast"]
    # inbuild_func_list = [""]
    # func_list = [""]
    # steps_per_npz_list = [1000]
    # subplots_list = [False]
    # property_func_list = [True]
    # list_of_arrays_list = [False]
    # property_names_list = ["get_contrast"]
    # property_args_list = [[5, 0.3]]
    # inbuild_func_args_list = [[]]
    # property_args_frame_list = [False]


    # steps_per_npz_list = [100]
    # subplots_list = [True]
    # property_func_list = [False]
    # list_of_arrays_list = [True]
    # property_names_list = ["monopolar"]
    # property_args_list = [[]]
    # property_args_frame_list = [False]

    # subplots_list = [False]
    # property_func_list = [True]
    # list_of_arrays_list = [False]
    # property_names_list = ["get_E_explicit"]
    # property_args_list = [[]]
    # property_args_frame_list = [False]


    # property_names = ["get_parity"]
    # property_name = ["get_peak_distances_along"]
    # property_name = ["get_peak_positions"]
    # property_args = [0]

    # property_name = "get_polarization"
    # property_args = [10.0 ** -9, 10.0 ** -9]
    # property_args_frame = True

    # property_name = "get_contrast"
    # property_args = [0.3]

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

    # we want 2S=D, so that the peaks distance equals the distance between max and min of sin
    # delta = s * 2.0

    # move2graphs = True
    move2graphs = False

    for path_anchor_input, movie_start, movie_end in zip(path_anchor_input_list, movie_start_list, movie_end_list, strict=True):
        for (property_func, property_name,
             property_args, property_args_frame, subplots, list_of_arrays, steps_per_npz,
             file_suffix, inbuild_func, inbuild_func_args, func) in zip(
                property_func_list, property_names_list, property_args_list,
                property_args_frame_list, subplots_list, list_of_arrays_list, steps_per_npz_list,
                file_suffix_list, inbuild_func_list, inbuild_func_args_list, func_list, strict=True):
            for i, movie_number in enumerate(range(movie_start, movie_end + 1)):
                # file_suffix = ""
                # file_suffix = "-" + "-".join(map(str, property_args)) + ".png"
                property_filename_suffix = dir_suffix + file_suffix
                property_args_str = False
                # property_args_str = [Path(path_anchor_input,
                                     # f"{dir_name}{counting_format % movie_number}"),
                                     # "pol_", steps_format]
                                     # filename_steps + "pol_", steps_format]
                # property_args = [0.00725, number_of_peaks[i]]
                # command = ["python", "-m", "supersolids.tools.track_property"]
                command = ["python", "/bigwork/dscheier/supersolids/supersolids/tools/track_property.py"]
                flags = [f"-dt={dt}",
                         f"-dir_path={path_anchor_input}",
                         f"-dir_name={dir_name}{counting_format % movie_number}",
                         f"-filename_schroedinger={filename_schroedinger}",
                         f"-filename_steps={filename_steps}",
                         f"-steps_per_npz={steps_per_npz}",
                         f"-steps_format={steps_format}",
                         f"-property_name={property_name}",
                         f"-property_filename_suffix={property_filename_suffix}"]

                if property_func:
                    flags.append("--property_func")
                if property_args:
                    property_args_parsed = list(map(str, property_args))
                    flags.append(f"--property_args")
                    flags += property_args_parsed
                if property_args_str:
                    property_args_str_parsed = list(map(str, property_args_str))
                    flags.append(f"--property_args_str")
                    flags += property_args_str_parsed
                if subplots:
                    flags.append(f"--subplots")
                if property_args_frame:
                    flags.append(f"--property_args_frame")
                if inbuild_func:
                    flags.append(f"-inbuild_func={inbuild_func}")
                if inbuild_func_args:
                    inbuild_func_args_parsed = list(map(str, inbuild_func_args))
                    flags.append(f"--inbuild_func_args")
                    flags += inbuild_func_args_parsed
                if func:
                    flags.append(f"-func={func}")
                if frame_start is None:
                    flags.append(f"-take_last={take_last}")
                else:
                    flags.append(f"-frame_start={frame_start}")
                    if frame_end:
                        flags.append(f"-frame_end={frame_end}")
                if list_of_arrays:
                    flags.append(f"--list_of_arrays")


                # command needs flags to be provided as list
                command.extend(flags)

                print(command)
                p = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, shell=False)

                # communicate needs flags to be provided as string seperated by " ".
                flags_parsed = " ".join(flags)
                out, err = p.communicate(flags_parsed.encode())
                print(f"out:\n{out}\n")
                print(f"err:\n{err}\n")

            # copy results from all movies into one directory (to compare them easier in image slideshows)
            path_anchor_output = Path(path_anchor_input, "graphs", property_name + dir_suffix)

            filename_extensions = [".png", ".npz"]
            for filename_extension in filename_extensions:
                cp_plots(movie_start, (movie_end - movie_start) + 1,
                         path_anchor_input, dir_name, property_name + property_filename_suffix,
                         path_anchor_output, property_name + property_filename_suffix,
                         counting_format, filename_extension=filename_extension, move=move2graphs)
