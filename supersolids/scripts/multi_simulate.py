#!/usr/bin/env python
import subprocess
from pathlib import Path
import numpy as np

supersolids_version = "0.1.32rc12"
dir_path = Path("/bigwork/dscheier/supersolids/results/")
# dir_path = Path("/home/dsche/supersolids/results/")

xvfb_display = 98

max_timesteps = 700001
dt = 0.0002
steps_per_npz = 1000
accuracy = 0.0

# 0.12 in radian is 7°, for small-angle approximation
v_start = 2.0
v_end = 33.0
v_step = 10.0

d_start = 0.2
d_end = 1.1
d_step = 0.2

file_start = "step_"
file_number = 1150000
file_pattern = ".npz"
file_name = f"{file_start}{file_number}{file_pattern}"

movie_string = "movie"
counting_format = "%03d"
movie_number = 151
files2last = 324
movie_now = f"{movie_string}{counting_format % movie_number}"

func_filename = "distort.txt"

j_counter = 0

func_list = []
func_path_list = []
dir_path_func_list = []
for v in np.arange(v_start, v_end, v_step):
    for d in np.arange(d_start, d_end, d_step):
        func_list.append([])
        v_string = round(v, ndigits=5)
        d_string = round(d, ndigits=5)

        movie_number_after = movie_number + files2last + j_counter

        # func = f"lambda x, y, z: {v_string} * np.sin(np.pi*x/{d_string}) ** 2"
        # func = f"lambda x, y, z: {v_string} * np.sin( (np.pi/4.0) + (np.pi*x/{d_string}) )"
        # func = f"lambda x, y, z: {v_string} * np.sin( (np.pi*x/{d_string}) )"
        func = f"lambda x, y, z: {v_string} * np.exp(-((x ** 2.0) /{d_string} ** 2.0) )"

        func_list[j_counter].append(func)

        movie_after = f"{movie_string}{counting_format % movie_number_after}"

        dir_path_func = Path(dir_path, movie_after)
        dir_path_func_list.append(dir_path_func)
        func_path = Path(dir_path_func, func_filename)
        func_path_list.append(func_path)

        heredoc = f"""#!/bin/bash
#==================================================
#PBS -N {supersolids_version}_v{v_string}dx{d_string}
#PBS -d /bigwork/dscheier/
#PBS -e /bigwork/dscheier/error.txt
#PBS -o /bigwork/dscheier/output.txt
#PBS -l nodes=1:ppn=1:ws
#PBS -l walltime=24:00:00
#PBS -l mem=5GB
#PBS -l vmem=5GB

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
-max_timesteps={max_timesteps} -dt={dt} -steps_per_npz={steps_per_npz} -accuracy={accuracy} \
-dir_name_load={movie_now} -dir_name_result={movie_after} -filename_npz={file_name} \
-dir_path={dir_path} -V='{func}' \
--offscreen \
--V_reload \
-noise_func=lambda gauss, k: np.concatenate(
(np.exp(-1.0j * np.mgrid[-10: 10: complex(0, 256), -5: 5: complex(0, 128), -4: 4: complex(0, 32)][1][:128, :128, :] * (1.0 + 2.0 * k * np.pi /4.0)),
np.exp(1.0j * np.mgrid[-10: 10: complex(0, 256), -5: 5: complex(0, 128), -4: 4: complex(0, 32)][1][128:, :, :] * (1.0 + 2.0 * k * np.pi /4.0))),
axis=0) * gauss

# -noise_func=lambda gauss, k: np.concatenate((np.ones(shape=(128,128,32)) * np.exp(-1.0j * (1.0 + 2.0 * k * np.pi /4.0)), np.ones(shape=(128,128,32))), axis=0) * gauss
# -noise -0.1 0.1 -noise_func='lambda x: np.exp(1.0j * x)' --real_time
# -noise_func=lambda gauss, k: np.exp(-1.0j * (4.0 + k * 2.0 * np.pi /4.0) * gauss) -neighborhood 0.02 4
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
        if func_path.is_dir():
            print(f"File {func_path} already exists!")
        else:
            if not dir_path_func.is_dir():
                dir_path_func.mkdir(mode=0o751)

            with open(func_path, "a") as func_file:
                func_string = '\n'.join(func)
                func_file.write(f"{func_string}")

        j_counter += 1

