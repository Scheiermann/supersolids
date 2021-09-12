#!/usr/bin/env python
import subprocess
from pathlib import Path
import numpy as np
import time


def dic2str(dic):
    dic_str = str(dic).replace("\'", "\"")
    dic_str_single_quote_wrapped = f"'{dic_str}'"

    return dic_str_single_quote_wrapped


xvfb_display = 98
supersolids_version = "0.1.33rc9"
dir_path = Path("/bigwork/dscheier/supersolids/supersolids/results/begin_alpha/")
# dir_path = Path("/home/dsche/supersolids/supersolids/results/begin/")

movie_string = "movie"
counting_format = "%03d"
movie_number = 1

Box = {"x0": -10, "x1": 10, "y0": -5, "y1": 5, "z0": -4, "z1": 4}
Res = {"x": 256, "y": 128, "z": 32}
# N = 50000
# w_y = 518.36
noise = [0.8, 1.2]
accuracy = 0.0
w_x_freq = 33.0
a_s = 0.000000004656
a = {"a_x": 4.5, "a_y": 2.0, "a_z": 1.5}

max_timesteps = 1500001
dt = 0.0002
steps_per_npz = 50000
steps_format = "%07d"
steps_per_npz = 1000
accuracy = 0.0

N_start = 40000
N_end = 61000
N_step = 5000

alpha_start = 0.3
alpha_end = 0.66
alpha_step = 0.05

func_filename = "distort.txt"

j_counter = 0

movie_list = []
func_list = []
func_path_list = []
dir_path_func_list = []
for v in np.arange(N_start, N_end, N_step):
    for d in np.arange(alpha_start, alpha_end, alpha_step):
        func_list.append([])
        v_string = round(v, ndigits=5)
        d_string = round(d, ndigits=5)
        N = v_string

        w_y = 2.0 * np.pi * (w_x_freq / d_string)

        # d_string = 0.0001 * 10.0 ** round(d, ndigits=5)

        # V = f"lambda x, y, z: {v_string} * np.sin(np.pi*x/{d_string}) ** 2"
        # V = f"lambda x, y, z: {v_string} * np.sin( (np.pi/4.0) + (np.pi*x/{d_string}) )"
        # V = f"lambda x, y, z: {v_string} * np.sin( (np.pi*x/{d_string}) )"
        # V = f"lambda x, y, z: {v_string} * np.exp(-((x ** 2.0) /{d_string} ** 2.0) )"
        # func_list[j_counter].append(f"-V='{V}' ")

        movie_number_after = movie_number + j_counter
        movie_after = f"{movie_string}{counting_format % movie_number_after}"
        dir_name_result = movie_after
        movie_list.append(movie_after)
        dir_path_func = Path(dir_path, movie_after)
        dir_path_func_list.append(dir_path_func)
        func_path = Path(dir_path_func, func_filename)
        func_path_list.append(func_path)

        heredoc = f"""#!/bin/bash
#==================================================
#PBS -N {supersolids_version}_v{v_string}dx{d_string}
#PBS -M daniel.scheiermann@itp.uni-hannover.de
#PBS -d /bigwork/dscheier/supersolids/supersolids/results/
#PBS -e /bigwork/dscheier/supersolids/supersolids/results/error_$PBS_JOBID.txt
#PBS -o /bigwork/dscheier/supersolids/supersolids/results/output_$PBS_JOBID.txt
#PBS -l nodes=1:ppn=1:ws
#PBS -l walltime=200:00:00
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

/bigwork/dscheier/miniconda3/bin/python3.8 -m supersolids \
-N={N} \
-Box={dic2str(Box)} \
-Res={dic2str(Res)} \
-max_timesteps={max_timesteps} \
-dt={dt} \
-steps_per_npz={steps_per_npz} \
-steps_format={steps_format} \
-dir_path={dir_path} \
-dir_name_result={dir_name_result} \
-a={dic2str(a)} \
-a_s={a_s} \
-w_y={w_y} \
-accuracy={accuracy} \
-noise {' '.join(map(str, noise))} \
--V_interaction \
--offscreen
"""

        func_list[j_counter].append(f"N={N}")
        func_list[j_counter].append(f"Box={Box}")
        func_list[j_counter].append(f"Res={Res}")
        func_list[j_counter].append(f"dt={dt}")
        func_list[j_counter].append(f"a_s={a_s}")
        func_list[j_counter].append(f"a={a}")
        func_list[j_counter].append(f"w_y={w_y}")
        func_list[j_counter].append(f"noise={noise}")
        func_list[j_counter].append(f"steps_per_npz={steps_per_npz}")

        print(heredoc)

        p = subprocess.Popen(["qsub"], stdin=subprocess.PIPE, stdout=subprocess.PIPE, shell=False)
        out, err = p.communicate(heredoc.encode())
        p.wait()

        j_counter += 1

        existing_dirs = sorted([x for x in dir_path.glob(movie_string + "*") if x.is_dir()])
        existing_dirnames = list(map(lambda path: path.name, existing_dirs))
        print(f"{movie_after}")
        while not movie_after in existing_dirnames:
            print(f"{existing_dirs}")
            print(f"{existing_dirnames}")
            print(f"Directory for {movie_after} not created yet. Waiting 3 seconds.")
            time.sleep(3)
            existing_dirs = sorted([x for x in dir_path.glob(movie_string + "*") if x.is_dir()])
            existing_dirnames = list(map(lambda path: path.name, existing_dirs))


movie_dirs = sorted([x for x in dir_path.glob(movie_string + "*") if x.is_dir()])
movie_dirnames = list(map(lambda path: path.name, movie_dirs))
while not all(item in movie_dirnames for item in movie_list):
    print(f"{movie_list}")
    print(f"{movie_dirnames}")
    print(f"Not all directories for movies created.  Waiting 5 seconds.")
    time.sleep(5)
    movie_dirs = sorted([x for x in dir_path.glob(movie_string + "*") if x.is_dir()])
    movie_dirnames = list(map(lambda path: path.name, movie_dirs))

j_counter = 0
# put distort.txt with the used V for every movie
for i, v_0 in enumerate(np.arange(N_start, N_end, N_step)):
    for j, delta in enumerate(np.arange(alpha_start, alpha_end, alpha_step)):
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

