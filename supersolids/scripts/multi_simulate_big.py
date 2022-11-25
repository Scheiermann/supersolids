#!/usr/bin/env python

# author: Daniel Scheiermann
# email: daniel.scheiermann@itp.uni-hannover.de
# license: MIT
# Please feel free to use and modify this, but keep the above information.

import subprocess
from pathlib import Path
import numpy as np

from supersolids.helper.dict2str import dic2str

supersolids_version = "0.1.36rc4"
dir_path = Path("/bigwork/dscheier/results/begin_gpu_big/")

dir_path_log = Path(f"{dir_path}/log/")
dir_path_log.mkdir(parents=True, exist_ok=True)

slurm: bool = True
mem_in_GB = 8
xvfb_display = 990

mixture: bool = True

Box = {"x0": -20, "x1": 20, "y0": -4, "y1": 4, "z0": -4, "z1": 4}
Res = {"x": 256, "y": 64, "z": 64}

max_timesteps = 1001
dt = 0.0002
steps_per_npz = 10
accuracy = 0.0

f_z = 167.0
w_z = 2.0 * np.pi * f_z

f_x_start = 12.0
f_x_end = 17.0
f_x_step = 2.0

f_y_start = 100.0
f_y_end = 100.1
f_y_step = 0.5

if mixture:
    file_start = "mixture_step_"
else:
    file_start = "step_"

file_number = 150000
file_format = "%07d"
file_pattern = ".npz"
file_name = f"{file_start}{file_format % file_number}{file_pattern}"

movie_string = "movie"
counting_format = "%03d"
movie_number = 1
files2last = 10
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
for f_x in np.arange(f_x_start, f_x_end, f_x_step):
    for f_y in np.arange(f_y_start, f_y_end, f_y_step):
        skip_counter += 1
        if skip_counter < skip:
            continue
        func_list.append([])
        f_x_string = round(f_x, ndigits=5)
        f_y_string = round(f_y, ndigits=5)
        # d_string = 0.0001 * 10.0 ** round(d, ndigits=5)

        w_x = 2.0 * np.pi * f_x
        w_y = 2.0 * np.pi * f_y
        w = {"w_x": eval(f"{w_x}"), "w_y": eval(f"{w_y}"), "w_z": eval(f"{w_z}")}

        # V = f"lambda x, y, z: {v_string} * np.sin(np.pi*x/{d_string}) ** 2"
        # V = f"lambda x, y, z: {v_string} * np.sin( (np.pi/4.0) + (np.pi*x/{d_string}) )"
        # V = f"lambda x, y, z: {v_string} * np.sin( (np.pi*x/{d_string}) )"
        # V = f"lambda x, y, z: {v_string} * np.exp(-((x ** 2.0) /{d_string} ** 2.0) )"
        V = None
        # func_list[j_counter].append(f"-V='{V}' ")
        func_list[j_counter].append(f"f_x={f_x_string} ")
        func_list[j_counter].append(f"f_y={f_y_string} ")
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

        jobname = f"{supersolids_version}_fx_{f_x_string}fy_{f_y_string}"

        if slurm:
            cluster_flags = f"""#==================================================
#SBATCH --job-name {jobname}
#SBATCH -D {dir_path}/log/
#SBATCH --mail-user daniel.scheiermann@itp.uni-hannover.de
#SBATCH --mail-type=END,FAIL
#SBATCH -o output-%j.out
#SBATCH -e error-%j.out
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 0-00:30:00
#SBATCH -p gpu
#SBATCH --mem={mem_in_GB}G
"""

        else:
            cluster_flags = f"""#==================================================
#PBS -N {jobname}
#PBS -M daniel.scheiermann@itp.uni-hannover.de
#PBS -m abe
#PBS -d {dir_path}
#PBS -e {dir_path}/log/error-$PBS_JOBID.txt
#PBS -o {dir_path}/log/output-$PBS_JOBID.txt
#PBS -l nodes=1:ppn=1:ws
#PBS -l walltime=24:00:00
#PBS -l mem={mem_in_GB}GB
#PBS -l vmem={mem_in_GB}GB
"""

        heredoc = "\n".join(["#!/bin/bash",
                             cluster_flags,
                             f"""
# >>> conda initialize >>>
# !! Contents within this block are managed by 'conda init' !!
__conda_setup="$('/bigwork/dscheier/miniconda/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/bigwork/dscheier/miniconda/etc/profile.d/conda.sh" ]; then
        . "/bigwork/dscheier/miniconda/etc/profile.d/conda.sh"
    else
        export PATH="/bigwork/dscheier/miniconda/bin:$PATH"
    fi
fi
unset __conda_setup
# <<< conda initialize <<<

export HOME=$BIGWORK

Xvfb :{xvfb_display - j_counter} &
export DISPLAY=:{xvfb_display - j_counter}

conda activate /bigwork/dscheier/miniconda/envs/solids
echo $DISPLAY
echo $CONDA_PREFIX
echo $(which python3)
echo $(which pip3)
echo "supersolids={supersolids_version}"

# conda install -c scheiermann/label/main supersolids={supersolids_version}
# conda install -c scheiermann/label/testing supersolids={supersolids_version}
# conda install numba
# conda install cupy

# /bigwork/dscheier/miniconda/bin/pip install -i https://test.pypi.org/simple/ supersolids=={supersolids_version}
# /bigwork/dscheier/miniconda/bin/pip install -i https://pypi.org/simple/supersolids=={supersolids_version}

# /bigwork/dscheier/miniconda/bin/python -m supersolids.tools.simulate_npz
        
python -m supersolids.tools.simulate_npz \
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
--offscreen

# -w={dic2str(w)} \
# --real_time \
# -V={V} \
# -noise_func='{noise_func}'\
# -neighborhood 0.02 4
"""])

        print(heredoc)

        if slurm:
            p = subprocess.Popen(["sbatch"], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                 shell=False)
        else:
            p = subprocess.Popen(["qsub"], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                 shell=False)

        out, err = p.communicate(heredoc.encode())
        p.wait()

        j_counter += 1
    k_counter += 1


j_counter = 0
# put distort.txt with the used V for every movie
for i, f_x in enumerate(np.arange(f_x_start, f_x_end, f_x_step)):
    for j, f_y in enumerate(np.arange(f_y_start, f_y_end, f_y_step)):
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

