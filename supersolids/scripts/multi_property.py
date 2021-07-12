#!/usr/bin/env python
import subprocess
from pathlib import Path


def string_float(s):
    return s, float(s)


# Script runs, if script is run as main script (called by python *.py)
if __name__ == "__main__":
    # path_anchor_input = Path("/home/dsche/supersolids/results/jospeh_injunction/real_global/")
    # path_anchor_input = Path("/run/media/dsche/ITP Transfer/joseph_injunction2/y_kick/kick_1.0/")
    path_anchor_input = Path("/run/media/dsche/ITP Transfer/joseph_injunction2/y_kick/kick_0.1/")
    # path_anchor_input = Path("/run/media/dsche/ITP Transfer/joseph_injunction2/y_kick/kick_0.01/")

    frame_start = 1610000
    # frame_start = 1210000
    steps_per_npz = 1000
    # movie_start = 695
    # movie_end = 714
    # movie_start = 720
    # movie_end = 739
    movie_start = 745
    movie_end = 764

    dt = 0.0002

    dir_name = "movie"
    counting_format = "%03d"

    steps_format = "%07d"

    dir_suffix = "_right"

    file_suffix = "_fft"
    inbuild_func = "fft_plot"
    # inbuild_func = ""
    # func = "lambda x, y: (x, y)"
    func = ""

    property_filename_suffix = dir_suffix + file_suffix

    subplots = True
    property_func = True
    property_name = "get_center_of_mass"
    # property_name = "get_parity"
    # property_name = "get_peak_distances_along"
    # property_name = "get_peak_positions"
    # property_args = [0]
    # property_args = []

    # property_name = "get_phase_var"
    # property_args = [0, 256, 0, 128, 0, 32]
    property_args = [0, 128, 0, 128, 0, 32]
    # property_args = [128, 256, 0, 128, 0, 32]

    # property_name = "get_phase_var_neighborhood"
    # [prob_min, amount]
    # property_args = [0.02, 4]

    # we want 2S=D, so that the peaks distance equals the distance between max and min of sin
    # delta = s * 2.0

    property_args_parsed = " ".join(map(str, property_args))
    for i in range(movie_start, movie_end + 1):
        command = ["python", "-m", "supersolids.tools.track_property"]
        flags = f"""-dt={dt} -dir_name={movie_string}{counting_format % i} -frame_start={frame_start} -steps_per_npz={steps_per_npz} -steps_format={steps_format} -property_name={property_name} -property_filename_suffix={property_filename_suffix}"""
        if property_func:
            flags += " --property_func"
        if property_args_parsed:
            flags += f" --property_args {property_args_parsed}"
        if subplots:
            flags += f" --subplots"

        flags_nosplit = f"""-dir_path={dir_path}"""
        command.extend([flags_nosplit] + flags.split(" "))

        print(command)
        p = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, shell=False)
        out, err = p.communicate(flags.encode())
        print(f"out:\n{out}\n")
        print(f"err:\n{err}\n")
