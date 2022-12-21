#!/usr/bin/env python

import subprocess
from pathlib import Path
import numpy as np
import time

from supersolids.helper.dict2str import dic2str


if __name__ == "__main__":
    supersolids_version = "0.1.37rc1"
    dir_path = Path("/home/dscheiermann/results/begin_gpu_11_18/")

    xvfb_display = 970
    movie_string = "movie"
    counting_format = "%03d"

    N = 63000
    a11 = 100.0

    m_list = [163.9, 163.9]
    a_dd = 130.8
    dipol = 9.0
    # dipol = 10.0
    dipol2 = 10.0
    a_dd_list = [a_dd, (dipol/dipol2) * a_dd, (dipol/dipol2) ** 2.0 * a_dd]
    # a_dd_list = [a_dd, 0.0, (dipol/10.0) ** 2.0 * a_dd]

    movie_number = 1
    mem_in_MB = 750
    gpu_index = 0
    Box = {"x0": -12, "x1": 12, "y0": -3, "y1": 3, "z0": -5, "z1": 5}
    # Box = {"x0": -20, "x1": 20, "y0": -3, "y1": 3, "z0": -5, "z1": 5}
    Res = {"x": 256, "y": 64, "z": 64}
    # Res = {"x": 256, "y": 64, "z": 64}
    # Box = {"x0": -20, "x1": 20, "y0": -7, "y1": 7, "z0": -6, "z1": 6}
    # Res = {"x": 128, "y": 64, "z": 64} # 450MB
    # Res = {"x": 256, "y": 128, "z": 32}
    # Res = {"x": 128, "y": 64, "z": 32}
    # Res = {"x": 256, "y": 128, "z": 128} # 1600MB
    # Res = {"x": 128, "y": 128, "z": 128} # 1000MB
    # Res = {"x": 128, "y": 128, "z": 64} # 600MB
    # Res = {"x": 128, "y": 64, "z": 64} # 400MB

    noise = [1.0, 1.0]
    # noise = [0.9, 1.1]
    accuracy = 0.0

    w_x_freq = 33.0
    w_y_freq = 110.0
    w_z_freq = 167.0
    w_x = 2.0 * np.pi * w_x_freq
    w_y = 2.0 * np.pi * w_y_freq
    w_z = 2.0 * np.pi * w_z_freq

    # a_s = 0.000000004656
    # a = {"a_x": 4.5, "a_y": 2.0, "a_z": 1.5}

    # for mixtures
    a = {"a_x": 4.0, "a_y": 0.8, "a_z": 1.8}

    max_timesteps = 350001
    # max_timesteps = 80001
    # max_timesteps = 35001
    # max_timesteps = 101
    # max_timesteps = 11
    dt = 0.0002
    steps_per_npz = 10000
    # steps_per_npz = 1000
    # steps_per_npz = 100
    # steps_per_npz = 1
    steps_format = "%07d"

    N2_part = 0.5

    epsilon_start = 0.0
    epsilon_end = 0.1
    epsilon_step = 0.2

    a12_start = 62.5
    a12_end = 97.6
    # a12_start = 62.5
    # a12_end = 98.1
    a12_step = 5.0
    a12_step = 2.5

    func_filename = "distort.txt"

    skip = 0
    skip_counter = 0
    j_counter = 0
    # j_counter = skip - 1
    end = 0

    for epsilon in np.arange(epsilon_start, epsilon_end, epsilon_step):
        for a12 in np.arange(a12_start, a12_end, a12_step):
            skip_counter += 1
            if skip_counter < skip:
                continue
            if skip_counter == end:
                break
            epsilon_string = round(epsilon, ndigits=5)
            a12_string = round(a12, ndigits=5)
            N2 = int(N * N2_part)
            N_list = [N - N2, N2]
            # tilt = 10 ** epsilon
            tilt = epsilon

            # a_s_list in triu (triangle upper matrix) form: a11, a12, a22
            a_s_list = [a11, a12, a11]

            movie_number_after = movie_number + j_counter
            movie_after = f"{movie_string}{counting_format % movie_number_after}"
            dir_name_result = movie_after

            jobname = f"{supersolids_version}_{mem_in_MB}M_e_{epsilon_string}a12_{a12_string}_m{movie_number_after}"

            heredoc = "\n".join(["#!/bin/bash",
                                f""" 
Xvfb :{xvfb_display - j_counter} &
export DISPLAY=:{xvfb_display - j_counter}

source $(conda info --base)/etc/profile.d/conda.sh
conda activate /home/dscheiermann/miniconda/envs/solids

echo $(hostname -a)
echo $DISPLAY
echo $CONDA_PREFIX
echo "supersolids={supersolids_version}"
echo {jobname}

# export QT_QPA_PLATFORM=xcb

/home/dscheiermann/miniconda/envs/solids/bin/python3.10 -m supersolids \
-Box={dic2str(Box)} \
-Res={dic2str(Res)} \
-max_timesteps={max_timesteps} \
-dt={dt} \
-steps_per_npz={steps_per_npz} \
-steps_format={steps_format} \
-dir_path={dir_path} \
-dir_name_result={dir_name_result} \
-a={dic2str(a)} \
-w_x={w_x} \
-w_y={w_y} \
-w_z={w_z} \
-accuracy={accuracy} \
-noise {' '.join(map(str, noise))} \
--N_list {' '.join(map(str, N_list))} \
--m_list {' '.join(map(str, m_list))} \
--a_dd_list {' '.join(map(str, a_dd_list))} \
--a_s_list {' '.join(map(str, a_s_list))} \
-tilt={tilt} \
--V_interaction \
--offscreen \
--gpu_index={gpu_index} \
--mixture &

"""
            ])

            print(heredoc)
            with open(Path(dir_path, f"sbatch_{dir_name_result}.sh"), "w") as f:
                f.write(f"{heredoc}\n")

            j_counter += 1
