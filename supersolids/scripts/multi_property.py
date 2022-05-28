#!/usr/bin/env python
import subprocess
from pathlib import Path

from supersolids.helper import get_path
from supersolids.scripts.cp_plots import cp_plots


def string_float(s):
    return s, float(s)


# Script runs, if script is run as main script (called by python *.py)
if __name__ == "__main__":
    # path_anchor_input = Path("/home/dsche/supersolids/results/begin_mixture_13/")
    # path_anchor_input = Path("/run/media/dsche/scr2/begin_mixture_13/")
    # path_anchor_input = Path("/run/media/dsche/ITP Transfer/joseph_injunction2/")
    # path_anchor_input = Path("/run/media/dsche/ITP Transfer/begin_mixture_a12_grid/")
    # path_anchor_input = Path("/bigwork/dscheier/results/begin_gpu/")
    # path_anchor_input = Path("/bigwork/dscheier/results/begin_xN/")
    # path_anchor_input = Path("/run/media/dsche/ITP Transfer/begin_mixture_15_6125/")
    # path_anchor_input = Path("/home/dsche/supersolids/results/")
    # path_anchor_input = Path("/bigwork/dscheier/results/begin_gpu_big/")
    path_anchor_input = Path("/bigwork/dscheier/results/begin_mixture_a12_grid/")
    # path_anchor_input = Path("/bigwork/dscheier/results/begin_mixture_a12_small_grid/")

    mixture = True
    filename_schroedinger: str = "schroedinger.pkl"

    if mixture:
        filename_steps = "mixture_step_"
    else:
        filename_steps = "step_"

    # steps_per_npz = 1
    # steps_per_npz = 10000
    steps_per_npz = 1000

    frame_start = None
    if frame_start is None:
        take_last = 4
    else:
        take_last = None
        # frame_start = 607000
        # frame_start = 50000
        # frame_start = 10000
        # frame_start = 1150000
        # frame_start = 1500000
        # frame_start = 0
        # frame_end = 20000
        # frame_end = 100000

    movie_start = 1
    movie_end = 50
    # movie_end = 45

    dt = 0.0002

    dir_name = "movie"
    counting_format = "%03d"

    steps_format = "%07d"

    if take_last is None:
        dir_suffix = f"_paper_{frame_start}"
        # dir_suffix = "_right"
    else:
        dir_suffix = f"_paper_takelast_{take_last}"

    # file_suffix = "_fft"
    # inbuild_func = "fft_plot"
    inbuild_func = ""
    func = ""
    # func = "lambda x, y: (x, y)"
    # func = "lambda x, y: (x[1:], np.abs(np.diff(y)) / y[1:])"

    subplots = False
    property_func = False
    # property_name = "E"
    # property_name = "mu_arr"

    # subplots = True
    property_func = True
    # property_name = "get_center_of_mass"
    # property_name = "get_parity"
    # property_name = "get_peak_distances_along"
    # property_name = "get_peak_positions"
    # property_args = [0]
    # property_args = []

    property_name = "get_polarization"
    property_args = [0.01, 0.01]
    property_args_frame = True

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

    file_suffix = ""
    # file_suffix = "-" + "-".join(map(str, property_args)) + ".png"
    property_filename_suffix = dir_suffix + file_suffix

    for i, movie_number in enumerate(range(movie_start, movie_end + 1)):
        property_args_str = [Path(path_anchor_input, f"{dir_name}{counting_format % movie_number}"),
                             filename_steps + "pol_", steps_format]
        # property_args = [0.00725, number_of_peaks[i]]
        command = ["python", "-m", "supersolids.tools.track_property"]
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
        if func:
            flags.append(f"-func={func}")
        if frame_start is None:
            flags.append(f"-take_last={take_last}")
        else:
            flags.append(f"-frame_start={frame_start}")
            if frame_end:
                flags.append(f"-frame_end={frame_end}")


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
    cp_plots(movie_start, (movie_end - movie_start) + 1,
             path_anchor_input, dir_name, property_name + property_filename_suffix,
             path_anchor_output, property_name + property_filename_suffix,
             counting_format, filename_extension=".png")
