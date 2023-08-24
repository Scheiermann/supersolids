#!/usr/bin/env python

import subprocess
from pathlib import Path
import numpy as np
import time

from supersolids.helper.dict2str import dic2str


if __name__ == "__main__":
    supersolids_version = "0.1.38rc1"
    # dir_path = Path("/home/dscheiermann/results/begin_gpu_02_23_no_dipol_1comp_w_paper/")
    # dir_path = Path("/home/dscheiermann/results/begin_gpu_02_23_no_dipol_no_lhy_1comp_w_paper/")
    # dir_path = Path("/home/dscheiermann/results/begin_gpu_03_15_no_dipol_no_lhy_1comp_w_paper/")
    dir_path = Path("/home/dscheiermann/results/begin_gpu_04_24_2d/")
    # dir_path = Path("/home/dscheiermann/results/begin_gpu_04_03/")
    # dir_path = Path("/home/dscheiermann/results/begin_gpu_02_27_no_dipol_no_lhy_1comp_w_30/")
    # dir_path = Path("/home/dscheiermann/results/begin_gpu_02_06_no_V_1comp/")
    # dir_path = Path("/home/dscheiermann/results/begin_gpu_01_13_dip9/")
    # dir_path = Path("/home/dscheiermann/results/begin_gpu_12_28/")
    # dir_path = Path("/home/dscheiermann/results/begin_gpu_12_28_to_102_dip9/")
    if not dir_path.is_dir():
        dir_path.mkdir(parents=True)

    xvfb_display = 914
    movie_string = "movie"
    counting_format = "%03d"

    # N = 5000
    # N = 30000
    # N = 50000
    N = 80000
    # a11 = 100.0

    m_list = [163.9, 0.0]
    # m_list = [163.9, 163.9]
    a_dd = 0.0
    # a_dd = 130.8
    # dipol = 9.0
    # dipol = 10.0
    # dipol = 0.0
    a_dd_list = [a_dd, 0.0, 0.0]
    # a_dd_list = [a_dd, (dipol/dipol2) * a_dd, (dipol/dipol2) ** 2.0 * a_dd]
    # a_dd_list = [a_dd, 0.0, (dipol/10.0) ** 2.0 * a_dd]

    lhy_factor = 0.0
    # lhy_factor = 1.0

    movie_number = 2
    mem_in_MB = 1400
    gpu_index = 0

    Box = {"x0": -8, "x1": 8, "y0": -8, "y1": 8}
    Res = {"x": 16, "y": 16}

    noise = [0.9, 1.1]
    accuracy = 0.0

    # f_x_freq = 33.0
    # f_y_freq = 110.0
    # f_z_freq = 167.0

    # f_x_freq = 30.0
    # f_y_freq = 110.0
    # f_z_freq = 90.0

    f_x_freq = 30.0
    # f_x_freq = 50.0
    f_y_freq = f_x_freq
    f_z_freq = f_x_freq

    # f_x_freq = 100.0
    # f_y_freq = 100.0
    # f_z_freq = 100.0

    # f_x_freq = 50.0
    # f_y_freq = 50.0
    # f_z_freq = 50.0
    w_x = 2.0 * np.pi * f_x_freq
    w_y = 2.0 * np.pi * f_y_freq

    # for mixtures
    a = {"a_x": 1.0, "a_y": 1.0}

    mu = {"mu_x": 0.0, "mu_y": 0.0}

    # max_timesteps = 350001
    # max_timesteps = 90001
    max_timesteps = 1001
    # max_timesteps = 101
    dt = 0.0002
    # steps_per_npz = 1000
    # steps_per_npz = 1000
    steps_per_npz = 100
    # steps_per_npz = 10
    # steps_per_npz = 1
    steps_format = "%07d"

    N2_part = 0.0
    # N2_part = 0.5

    epsilon_start = 0.0
    epsilon_end = 0.1
    epsilon_step = 0.2

    # a11_start = 0.0
    # a11_end = 0.01
    # a11_step = 0.1

    # a11_end = 20.1
    # a11_start = 70.0
    # a11_end = 70.1
    # a11_start = 80.0
    # a11_end = 80.1
    # a11_step = 0.1

    # a11_start = 0.0
    # a11_end = 0.1
    a11_start = 94.0
    a11_end = 94.1
    # a11_start = 80.0
    # a11_end = 80.1
    # a11_end = 100.1
    # a11_step = 1.0
    # a11_step = 2.0
    a11_step = 0.5

    func_filename = "distort.txt"

    skip = 0
    skip_counter = 0
    j_counter = 0
    # j_counter = skip - 1
    end = 0

    # a12_array = np.array([70.5, 71.0, 71.5, 72.0, 73.0, 73.5, 74.0, 74.5, 75.5, 76.0, 76.5, 77.0, 78.0, 78.5, 79.0, 79.5, 80.5, 81.0, 81.5, 82.0, 83.0, 83.5, 84.0, 84.5, 85.5, 86.0, 86.5, 87.0, 88.0, 88.5, 89.0, 89.5])
    # a12_array = np.array([90.5, 91.0, 91.5, 92.0, 93.0, 93.5, 94.0, 94.5, 95.5, 96.0, 96.5, 97.0, 98.0, 98.5, 99.0, 99.5, 100.0, 100.5, 101.0, 101.5, 102.0])
    a11_array = np.arange(a11_start, a11_end, a11_step)
    for epsilon in np.arange(epsilon_start, epsilon_end, epsilon_step):
        for a11 in a11_array:
            skip_counter += 1
            if skip_counter < skip:
                continue
            if skip_counter == end:
                break
            epsilon_string = round(epsilon, ndigits=5)
            a11_string = round(a11, ndigits=5)
            N2 = int(N * N2_part)
            N_list = [N - N2, N2]
            # tilt = 10 ** epsilon
            tilt = epsilon

            # a_s_list in triu (triangle upper matrix) form: a11, a12, a22
            # a_s_list = [a11, a12, a11]
            a_s_list = [a11, 0.0, 0.0]

            movie_number_after = movie_number + j_counter
            movie_after = f"{movie_string}{counting_format % movie_number_after}"
            dir_name_result = movie_after

            jobname = f"{supersolids_version}_{mem_in_MB}M_e_{epsilon_string}a11_{a11_string}_m{movie_number_after}"

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
-accuracy={accuracy} \
-noise {' '.join(map(str, noise))} \
--N_list {' '.join(map(str, N_list))} \
--m_list {' '.join(map(str, m_list))} \
--a_dd_list {' '.join(map(str, a_dd_list))} \
--a_s_list {' '.join(map(str, a_s_list))} \
-mu={dic2str(mu)} \
-tilt={tilt} \
--offscreen \
-gpu_index={gpu_index} \
-lhy_factor={lhy_factor} \
--V_interaction \
--mixture \\
&

"""
            ])

            print(heredoc)
            with open(Path(dir_path, f"sbatch_{dir_name_result}.sh"), "w") as f:
                f.write(f"{heredoc}\n")

            j_counter += 1
