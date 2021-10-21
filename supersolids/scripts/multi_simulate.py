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

supersolids_version = "0.1.34rc10"
dir_path = Path("/bigwork/dscheier/supersolids/supersolids/results/begin_alpha/")
# dir_path = Path("/home/dsche/supersolids/results/")

slurm: bool = True
mem_in_GB = 6
xvfb_display = 50

max_timesteps = 700001
dt = 0.0002
steps_per_npz = 10000
accuracy = 0.0

w_x_freq = 33.0
w_z_freq = 167.0

N_start = 0.05
N_end = 0.51
N_step = 0.05

a12_start = 0.6
a12_end = 0.91
a12_step = 0.02

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
for N2_part in np.arange(N_start, N_end, N_step):
    for a12 in np.arange(a12_start, a12_end, a12_step)[::-1]:
        func_list.append([])
        N2_part_string = round(N2_part, ndigits=5)
        a12_string = round(a12, ndigits=5)
        # d_string = 0.0001 * 10.0 ** round(d, ndigits=5)

        w_x = 2.0 * np.pi * w_x_freq
        w_y = 2.0 * np.pi * (w_x_freq / a12_string)
        w_z = 2.0 * np.pi * w_z_freq
        w = {"w_x": eval(f"{w_x}"), "w_y": eval(f"{w_y}"), "w_z": eval(f"{w_z}")}

        # V = f"lambda x, y, z: {v_string} * np.sin(np.pi*x/{d_string}) ** 2"
        # V = f"lambda x, y, z: {v_string} * np.sin( (np.pi/4.0) + (np.pi*x/{d_string}) )"
        # V = f"lambda x, y, z: {v_string} * np.sin( (np.pi*x/{d_string}) )"
        # V = f"lambda x, y, z: {v_string} * np.exp(-((x ** 2.0) /{d_string} ** 2.0) )"
        V = None
        # func_list[j_counter].append(f"-V='{V}' ")
        func_list[j_counter].append(f"N2/N1={N2_part_string} ")
        func_list[j_counter].append(f"a12={a12_string} ")
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

        jobname = f"{supersolids_version}_N2_{N2_part_string}a12_{a12_string}"

        if slurm:
            cluster_flags = f"""#==================================================
        #SBATCH --job-name {jobname}
        #SBATCH -D /bigwork/dscheier/supersolids/supersolids/results/
        #SBATCH --mail-user daniel.scheiermann@itp.uni-hannover.de
        #SBATCH --mail-type=END,FAIL
        #SBATCH -o output-%j.out
        #SBATCH -e error-%j.out
        #SBATCH -N 1
        #SBATCH -n 1
        #SBATCH -t 14-00:00:00
        #SBATCH --mem={mem_in_GB}G
        """

        else:
            cluster_flags = f"""#==================================================
        # PBS -N {jobname}
        # PBS -M daniel.scheiermann@itp.uni-hannover.de
        # PBS -d /bigwork/dscheier/supersolids/supersolids/results/
        # PBS -e /bigwork/dscheier/supersolids/supersolids/results/log/error_$PBS_JOBID.txt
        # PBS -o /bigwork/dscheier/supersolids/supersolids/results/log/output_$PBS_JOBID.txt
        # PBS -l nodes=1:ppn=1:ws
        # PBS -l walltime=200:00:00
        # PBS -l mem={mem_in_GB}GB
        # PBS -l vmem={mem_in_GB}GB
        """

        heredoc = "\n".join(["#!/bin/bash",
                             cluster_flags,
                             f"""
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


j_counter = 0
# put distort.txt with the used V for every movie
for i, N2_part in enumerate(np.arange(N_start, N_end, N_step)):
    for j, a12 in enumerate(np.arange(a12_start, a12_end, a12_step)):
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

