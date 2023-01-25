#!/usr/bin/env python

# author: Daniel Scheiermann
# email: daniel.scheiermann@itp.uni-hannover.de
# license: MIT
# Please feel free to use and modify this, but keep the above information.

import subprocess
from pathlib import Path
import numpy as np

from supersolids.helper.dict2str import dic2str


if __name__ == "__main__":
    supersolids_version = "0.1.37rc7"
    # dir_path = Path("/bigwork/dscheier/results/begin_gpu_big/")
    # dir_path = Path("/home/dscheiermann/results/begin_gpu_12_28/")
    # dir_path = Path("/home/dscheiermann/results/begin_gpu_12_28_to_102/")
    dir_path = Path("/home/dscheiermann/results/begin_gpu_12_28_to_102_dip9/")

    dir_path_log = Path(f"{dir_path}/log/")
    dir_path_log.mkdir(parents=True, exist_ok=True)

    mem_in_MB = 1400
    xvfb_display = 490
    gpu_index=0

    mixture: bool = True

    Box = {"x0": -12, "x1": 12, "y0": -3, "y1": 3, "z0": -5, "z1": 5}
    Res = {"x": 256, "y": 64, "z": 64}

    max_timesteps = 1000001
    dt = 0.0002
    steps_per_npz = 10000
    accuracy = 0.0

    f_y = 110.0
    f_z = 167.0
    w_y = 2.0 * np.pi * f_y
    w_z = 2.0 * np.pi * f_z

    f_x_open = 33.0
    # delta_f_x_start = -4.5
    # delta_f_x_start = -1.5
    delta_f_x_start = -1.0
    delta_f_x_end = -0.4
    delta_f_x_step = 1.0

    # a12_start = 62.5
    # a12_end = 97.6
    # a12_end = 62.6
    # a12_step = 2.5

    if mixture:
        file_start = "step_"
    else:
        file_start = "step_"

    file_number = 350000
    file_format = "%07d"
    file_pattern = ".npz"
    file_name = f"{file_start}{file_format % file_number}{file_pattern}"

    movie_string = "movie"
    counting_format = "%03d"
    movie_number = 1
    files2last = 40
    load_from_multi = True
    load_outer_loop = True

    func_filename = "distort.txt"

    j_counter = 0
    k_counter = 0
    skip_counter = 0
    skip = j_counter

    func_list = []
    func_path_list = []
    dir_path_func_list = []
    # a12_array = np.array([70.5, 71.0, 71.5, 72.0, 73.0, 73.5, 74.0, 74.5, 75.5, 76.0, 76.5, 77.0, 78.0, 78.5, 79.0, 79.5, 80.5, 81.0, 81.5, 82.0, 83.0, 83.5, 84.0, 84.5, 85.5, 86.0, 86.5, 87.0, 88.0, 88.5, 89.0, 89.5])
    a12_array = np.array([90.5, 91.0, 91.5, 92.0, 93.0, 93.5, 94.0, 94.5, 95.5, 96.0, 96.5, 97.0, 98.0, 98.5, 99.0, 99.5, 100.0, 100.5, 101.0, 101.5, 102.0])
    # a12_array = np.arange(a12_start, a12_end, a12_step)
    for a12 in a12_array:
        for delta_f_x in np.arange(delta_f_x_start, delta_f_x_end, delta_f_x_step):
            skip_counter += 1
            if skip_counter < skip:
                continue
            func_list.append([])
            f_x = f_x_open + delta_f_x
            f_x_string = round(f_x, ndigits=5)
            a12_string = round(a12, ndigits=5)
            # d_string = 0.0001 * 10.0 ** round(d, ndigits=5)

            w_x = 2.0 * np.pi * f_x
            w = {"w_x": eval(f"{w_x}"), "w_y": eval(f"{w_y}"), "w_z": eval(f"{w_z}")}

            # V = f"lambda x, y, z: {v_string} * np.sin(np.pi*x/{d_string}) ** 2"
            # V = f"lambda x, y, z: {v_string} * np.sin( (np.pi/4.0) + (np.pi*x/{d_string}) )"
            # V = f"lambda x, y, z: {v_string} * np.sin( (np.pi*x/{d_string}) )"
            # V = f"lambda x, y, z: {v_string} * np.exp(-((x ** 2.0) /{d_string} ** 2.0) )"
            V = None
            # func_list[j_counter].append(f"-V='{V}' ")
            func_list[j_counter].append(f"f_x={f_x_string} ")
            # func_list[j_counter].append(f"a12={a12_string} ")
            # func_list[j_counter].append(f"--real_time ")

            noise_func = None
            # noise_func = f"lambda gauss, k: 1.0"
            # noise_func=f"lambda gauss, k: np.concatenate(\
            #     (np.exp(-1.0j * np.mgrid[-10: 10: complex(0, 256), -5: 5: complex(0, 128), -4: 4: complex(0, 32)][1][:128, :128, :] * {d_string} * (1.0 + 2.0 * k * np.pi /4.0)),\
            #      np.exp(1.0j * np.mgrid[-10: 10: complex(0, 256), -5: 5: complex(0, 128), -4: 4: complex(0, 32)][1][128:, :, :] * {d_string} * (1.0 + 2.0 * k * np.pi /4.0))),\
            #     axis=0) * gauss"
            # func_list[j_counter].append(f"-noise_func='{noise_func}' ")

            if load_from_multi:
                if load_outer_loop:
                    movie_number_now = movie_number + k_counter
                else:
                    movie_number_now = movie_number + j_counter
            else:
                movie_number_now = movie_number

            movie_now = f"{movie_string}{counting_format % movie_number_now}"
            movie_number_after = movie_number + files2last + j_counter
            movie_after = f"{movie_string}{counting_format % movie_number_after}"

            dir_path_func = Path(dir_path, movie_after)
            dir_path_func_list.append(dir_path_func)
            func_path = Path(dir_path_func, func_filename)
            func_path_list.append(func_path)

            jobname = f"{supersolids_version}_fx_{f_x_string}m{movie_number_now}m{movie_number_after}"

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
        
/home/dscheiermann/miniconda/envs/solids/bin/python3.10 -m supersolids.tools.simulate_npz \
-Box={dic2str(Box)} \
-Res={dic2str(Res)} \
-max_timesteps={max_timesteps} \
-dt={dt} \
-steps_per_npz={steps_per_npz} \
-accuracy={accuracy} \
-dir_name_load={movie_now} \
-dir_name_result={movie_after} \
-filename_npz={file_name} \
-dir_path={dir_path} \
--V_reload \
-w={dic2str(w)} \
-gpu_index={gpu_index} \
--real_time \
--offscreen

# -V={V} \
# -noise_func='{noise_func}'\
# -neighborhood 0.02 4
"""
            ])

            print(heredoc)
            with open(Path(dir_path, f"sbatch_{movie_after}.sh"), "w") as f:
                f.write(f"{heredoc}\n")

            j_counter += 1
        k_counter += 1


    j_counter = 0
    # put distort.txt with the used V for every movie
    for i, f_x in enumerate(np.arange(delta_f_x_start, delta_f_x_end, delta_f_x_step)):
        for j, a12 in enumerate(a12_array):
            func = func_list[j_counter]
            func_path = func_path_list[j_counter]
            dir_path_func = dir_path_func_list[j_counter]
            if func_path.is_file():
                print(f"File {func_path} already exists!")
            else:
                if not dir_path_func.is_dir():
                    dir_path_func.mkdir(mode=0o751)

                with open(func_path, "a") as func_file:
                    func_string = '\n'.join(func)
                    func_file.write(f"{func_string}")

            j_counter += 1
