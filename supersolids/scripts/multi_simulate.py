#!/usr/bin/env python

# author: Daniel Scheiermann
# email: daniel.scheiermann@itp.uni-hannover.de
# license: MIT
# Please feel free to use and modify this, but keep the above information.

import subprocess
from pathlib import Path
import numpy as np


def dic2str(dic):
    dic_str = str(dic).replace("\'", "\"")
    dic_str_single_quote_wrapped = f"'{dic_str}'"

    return dic_str_single_quote_wrapped


supersolids_version = "0.1.34rc7"
dir_path = Path("/bigwork/dscheier/supersolids/supersolids/results/begin_alpha/")
# dir_path = Path("/home/dsche/supersolids/results/")

xvfb_display = 50

max_timesteps = 700001
dt = 0.0002
steps_per_npz = 1000
accuracy = 0.0

w_x_freq = 33.0
w_z_freq = 167.0

v_start = 60000
v_end = 61000
v_step = 5000

d_start = 0.35
d_end = 0.40
d_step = 0.05

file_start = "step_"
file_number = 1130000
file_format = "%07d"
file_pattern = ".npz"
file_name = f"{file_start}{file_format % file_number}{file_pattern}"

movie_string = "movie"
counting_format = "%03d"
movie_number = 34
files2last = 55
load_from_multi = True

func_filename = "distort.txt"

j_counter = 0

func_list = []
func_path_list = []
dir_path_func_list = []
for v in np.arange(v_start, v_end, v_step):
    for d in np.arange(d_start, d_end, d_step)[::-1]:
        func_list.append([])
        v_string = round(v, ndigits=5)
        d_string = round(d, ndigits=5)
        # d_string = 0.0001 * 10.0 ** round(d, ndigits=5)

        w_x = 2.0 * np.pi * w_x_freq
        w_y = 2.0 * np.pi * (w_x_freq / d_string)
        w_z = 2.0 * np.pi * w_z_freq
        w={"w_x": eval(f"{w_x}"), "w_y": eval(f"{w_y}"), "w_z": eval(f"{w_z}")}

        # V = f"lambda x, y, z: {v_string} * np.sin(np.pi*x/{d_string}) ** 2"
        # V = f"lambda x, y, z: {v_string} * np.sin( (np.pi/4.0) + (np.pi*x/{d_string}) )"
        # V = f"lambda x, y, z: {v_string} * np.sin( (np.pi*x/{d_string}) )"
        # V = f"lambda x, y, z: {v_string} * np.exp(-((x ** 2.0) /{d_string} ** 2.0) )"
        V = None
        # func_list[j_counter].append(f"-V='{V}' ")
        func_list[j_counter].append(f"N={v_string} ")
        func_list[j_counter].append(f"alpha={d_string} ")
        # func_list[j_counter].append(f"--real_time ")

        noise_func = None
        # noise_func = f"lambda gauss, k: 1.0"
        # noise_func=f"lambda gauss, k: np.concatenate(\
        #     (np.exp(-1.0j * np.mgrid[-10: 10: complex(0, 256), -5: 5: complex(0, 128), -4: 4: complex(0, 32)][1][:128, :128, :] * {d_string} * (1.0 + 2.0 * k * np.pi /4.0)),\
        #      np.exp(1.0j * np.mgrid[-10: 10: complex(0, 256), -5: 5: complex(0, 128), -4: 4: complex(0, 32)][1][128:, :, :] * {d_string} * (1.0 + 2.0 * k * np.pi /4.0))),\
        #     axis=0) * gauss"
        # func_list[j_counter].append(f"-noise_func='{noise_func}' ")

        if load_from_multi:
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

        heredoc = f"""#!/bin/bash
#==================================================
#PBS -N {supersolids_version}_v{v_string}dx{d_string}
#PBS -d /bigwork/dscheier/
#PBS -e /bigwork/dscheier/supersolids/supersolids/results/error_$PBS_JOBID.txt
#PBS -o /bigwork/dscheier/supersolids/supersolids/results/output_$PBS_JOBID.txt
#PBS -l nodes=1:ppn=1:ws
#PBS -l walltime=250:00:00
#PBS -l mem=4GB
#PBS -l vmem=4GB

# >>> conda initialize >>>
# !! Contents within this block are managed by 'conda init' !!
__conda_setup="$('/bigwork/dscheier/miniconda3/bin/conda' 'shell.zsh' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/bigwork/dscheier/miniconda3/etc/profile.d/conda.sh" ]; then
        . "/bigwork/dscheier/miniconda3/etc/profile.d/conda.sh"
    else
        export PATH="/bigwork/dscheier/miniconda3/bin:$PATH"
    fi
fi
unset __conda_setup
# <<< conda initialize <<<

Xvfb :{xvfb_display - j_counter} &
export DISPLAY=:{xvfb_display - j_counter}

conda activate /bigwork/dscheier/miniconda3/envs/solids
echo $DISPLAY
echo $CONDA_PREFIX
echo $(which python3)
echo $(which pip3)

/bigwork/dscheier/miniconda3/bin/pip3 install -i https://test.pypi.org/simple/ supersolids=={supersolids_version}
# /bigwork/dscheier/miniconda3/bin/pip3 install -i https://pypi.org/simple/supersolids=={supersolids_version}

/bigwork/dscheier/miniconda3/bin/python3.8 -m supersolids.tools.simulate_npz \
-Res='{{"x": 256, "y": 128, "z": 32}}' \
-Box='{{"x0": -10, "x1": 10, "y0": -5, "y1": 5, "z0": -4, "z1": 4}}' \
-max_timesteps={max_timesteps} \
-dt={dt} \
-steps_per_npz={steps_per_npz} \
-accuracy={accuracy} \
-dir_name_load={movie_now} \
-dir_name_result={movie_after} \
-filename_npz={file_name} \
-dir_path={dir_path} \
--V_reload \
--real_time \
--offscreen

# -w={dic2str(w)} \
# -V={V} \
# -noise_func='{noise_func}'\
# -neighborhood 0.02 4
"""
        print(heredoc)

        p = subprocess.Popen(["qsub"], stdin=subprocess.PIPE, stdout=subprocess.PIPE, shell=False)
        out, err = p.communicate(heredoc.encode())
        p.wait()

        j_counter += 1


j_counter = 0
# put distort.txt with the used V for every movie
for i, v_0 in enumerate(np.arange(v_start, v_end, v_step)):
    for j, delta in enumerate(np.arange(d_start, d_end, d_step)):
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

