#!/usr/bin/env python

import subprocess
from pathlib import Path
import numpy as np
import time

from supersolids.helper.dict2str import dic2str

slurm = True
mem_in_GB = 4
xvfb_display = 700
supersolids_version = "0.1.35"
# dir_path = Path("/bigwork/dscheier/supersolids/supersolids/results/begin_schroedinger/")
# dir_path = Path("/bigwork/dscheier/supersolids/supersolids/results/begin_mixture_a12_small_grid/")
dir_path = Path("/bigwork/dscheier/results/begin_pretilt/")
# dir_path = Path("/home/dsche/supersolids/supersolids/results/begin/")

dir_path_log = Path(dir_path, "log")
dir_path_log.mkdir(parents=True, exist_ok=True)


movie_string = "movie"
counting_format = "%03d"
movie_number = 1

N = 63000

m_list = [163.9, 163.9]
a_dd = 130.8
# dipol = 9.0
dipol = 10.0
a_dd_list = [a_dd, (dipol/10.0) * a_dd, (dipol/10.0) ** 2.0 * a_dd]

Box = {"x0": -15, "x1": 15, "y0": -7, "y1": 7, "z0": -6, "z1": 6}
# Res = {"x": 256, "y": 128, "z": 32}
Res = {"x": 128, "y": 64, "z": 32}

noise = [0.9, 1.1]
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

max_timesteps = 100001
dt = 0.0002
steps_per_npz = 10000
# steps_per_npz = 1
steps_format = "%07d"
accuracy = 0.0

N2_part = 0.5

a11_start = 85.0
a11_end = 100.1
a11_step = 2.5

a12_start = 85.0
a12_end = 100.1
a12_step = 2.50

func_filename = "distort.txt"

skip = 0
skip_counter = 0
j_counter = 0
# j_counter = skip - 1
end = 0

movie_list = []
func_list = []
func_path_list = []
dir_path_func_list = []
for a11 in np.arange(a11_start, a11_end, a11_step):
    for a12 in np.arange(a12_start, a12_end, a12_step):
        skip_counter += 1
        if skip_counter < skip:
            continue
        if skip_counter == end:
            break
        func_list.append([])
        a12_string = round(a12, ndigits=5)
        N2 = int(N * N2_part)
        N_list = [N - N2, N2]

        # a_s_list in triu (triangle upper matrix) form: a11, a12, a22
        a_s_list = [a11, a12, a11]

        movie_number_after = movie_number + j_counter
        movie_after = f"{movie_string}{counting_format % movie_number_after}"
        dir_name_result = movie_after
        movie_list.append(movie_after)
        dir_path_func = Path(dir_path, movie_after)
        dir_path_func_list.append(dir_path_func)
        func_path = Path(dir_path_func, func_filename)
        func_path_list.append(func_path)

        jobname = f"{supersolids_version}_a11_{a11}a12_{a12_string}_m{movie_number_after}"

        if slurm:
            cluster_flags = f"""#==================================================
#SBATCH --job-name {jobname}
#SBATCH -D {dir_path_log}
#SBATCH --mail-user daniel.scheiermann@itp.uni-hannover.de
#SBATCH --mail-type=END,FAIL
#SBATCH -o output-%j.out
#SBATCH -e error-%j.out
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 1-00:00:00
#SBATCH --mem={mem_in_GB}G
##SBATCH -p gpu
##SBATCH -w atlas
##SBATCH -w altair,atlas,berti,gemini,mirzam,niobe,pegasus,phad,pollux,rana,sargas,weywot
"""

        else:
            cluster_flags = f"""#==================================================
#PBS -N {jobname}
#PBS -M daniel.scheiermann@itp.uni-hannover.de
#PBS -m abe
#PBS -d /bigwork/dscheier/supersolids/supersolids/results/
#PBS -e /bigwork/dscheier/supersolids/supersolids/results/log_s/error_$PBS_JOBID.txt
#PBS -o /bigwork/dscheier/supersolids/supersolids/results/log_s/output_$PBS_JOBID.txt
#PBS -l nodes=1:ppn=1:ws
#PBS -l walltime=200:00:00
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

Xvfb :{xvfb_display - j_counter} &
export DISPLAY=:{xvfb_display - j_counter}


conda activate /bigwork/dscheier/miniconda/envs/solids

echo $DISPLAY
echo $CONDA_PREFIX
echo $(which python3)
echo $(which pip3)
echo "supersolids={supersolids_version}"

# CUPY_CACHE_DIR=/bigwork/dscheier/miniconda/envs/solids/.cupy/kernel_cache
export HOME=$BIGWORK
echo $HOME
# conda install -c scheiermann/label/main supersolids={supersolids_version}
# conda install -c scheiermann/label/testing supersolids={supersolids_version}
# conda install numba
# conda install cupy

# /bigwork/dscheier/miniconda/bin/pip install -i https://test.pypi.org/simple/ supersolids=={supersolids_version}
# /bigwork/dscheier/miniconda/bin/pip install -i https://pypi.org/simple/supersolids=={supersolids_version}

/bigwork/dscheier/miniconda/envs/solids/bin/python3.10 -m supersolids \
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
--V_interaction \
--offscreen \
--mixture

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

        existing_dirs = sorted([x for x in dir_path.glob(movie_string + "*") if x.is_dir()])
        existing_dirnames = list(map(lambda path: path.name, existing_dirs))
        print(f"{movie_after}")
        while not movie_after in existing_dirnames:
            print(f"Found dirnames: {existing_dirnames}")
            print(f"Directory for {movie_after} not created yet. Waiting 20 seconds.")
            time.sleep(20)
            existing_dirs = sorted([x for x in dir_path.glob(movie_string + "*") if x.is_dir()])
            existing_dirnames = list(map(lambda path: path.name, existing_dirs))
            if existing_dirs:
                print(f"First path: {existing_dirs[0]}")

movie_dirs = sorted([x for x in dir_path.glob(movie_string + "*") if x.is_dir()])
movie_dirnames = list(map(lambda path: path.name, movie_dirs))
while not all(item in movie_dirnames for item in movie_list):
    print(f"{movie_list}")
    print(f"{movie_dirnames}")
    print(f"Not all directories for movies created.  Waiting 5 seconds.")
    time.sleep(5)
    movie_dirs = sorted([x for x in dir_path.glob(movie_string + "*") if x.is_dir()])
    movie_dirnames = list(map(lambda path: path.name, movie_dirs))

# j_counter = 0
# # put distort.txt with the used V for every movie
# for i, v_0 in enumerate(np.arange(N_start, N_end, N_step)):
    # for j, delta in enumerate(np.arange(a12_start, a12_end, a12_step)):
        # func = func_list[j_counter]
        # func_path = func_path_list[j_counter]
        # dir_path_func = dir_path_func_list[j_counter]
        # if func_path.is_dir():
            # print(f"File {func_path} already exists!")
        # else:
            # if not dir_path_func.is_dir():
                # dir_path_func.mkdir(mode=0o751)

            # with open(func_path, "a") as func_file:
                # func_string = '\n'.join(func)
                # func_file.write(f"{func_string}")

        # j_counter += 1
