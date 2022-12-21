#!/usr/bin/env python
import subprocess
from pathlib import Path
import numpy as np
import time

from supersolids.helper.dict2str import dic2str
from supersolids.helper import constants


if __name__ == "__main__":
    slurm = True
    mem_in_GB = 2
    xvfb_display = 890
    supersolids_version = "0.1.37rc1"
    dir_path = Path("/bigwork/dscheier/results/begin_paper/")

    movie_string = "movie"
    counting_format = "%03d"
    movie_number = 1

    N = 55000
    # a11 = 88.0
    # a11 = 88.0 * constants.a_0

    # m_list = [163.9, 163.9]
    m_list = [163.9, 0]
    # a_dd = 130.8
    # a_dd_list = [a_dd, 0, 0]
    # a_dd_list = [a_dd, (9.0/10.0) * a_dd, (9.0/10.0) ** 2.0 * a_dd]

    Box = {"x0": -12, "x1": 12, "y0": -4, "y1": 4, "z0": -4, "z1": 4}
    Res = {"x": 128, "y": 64, "z": 32}

    noise = [0.9, 1.1]
    accuracy = 0.0

    # f_x = 14.0
    f_x = 33.0
    f_y = 100.0
    f_z = 167.0
    w_x = 2.0 * np.pi * f_x
    w_y = 2.0 * np.pi * f_y
    w_z = 2.0 * np.pi * f_z

    # a_s = 0.000000004656
    a = {"a_x": 9.0, "a_y": 1.0, "a_z": 1.5}
    # a = {"a_x": 2.0, "a_y": 2.0, "a_z": 1.5}

    # for mixtures
    # a = {"a_x": 4.0, "a_y": 0.8, "a_z": 1.8}

    max_timesteps = 150001
    dt = 0.0002
    steps_per_npz = 10000
    steps_format = "%07d"
    accuracy = 0.0

    a11_start = 95
    a11_end = 105
    a11_step = 2.5

    a_dd_start = 125.0
    a_dd_end = 135.0
    a_dd_step = 2.5

    # N_start = 150000
    # N_end = 151000
    # N_step = 10000

    # f_x_start = 80.0
    # f_x_end = 81.0
    # f_x_step = 2.0

    func_filename = "distort.txt"

    j_counter = 0
    skip_counter = 0
    skip = j_counter

    movie_list = []
    func_list = []
    func_path_list = []
    dir_path_func_list = []
    for a11 in np.arange(a11_start, a11_end, a11_step):
        for a_dd in np.arange(a_dd_start, a_dd_end, a_dd_step):
            skip_counter += 1
            if skip_counter < skip:
                continue
            func_list.append([])
            N_string = round(N, ndigits=0)
            N_list = [N, 0]

            # a_s_list in triu (triangle upper matrix) form: a11, a12, a22
            # a_s_list = [a11, a12 * a11, a11]
            a_s_list = [a11, 0, 0]
            a_s_string = round(a11, ndigits=3)

            a_dd_list = [a_dd, 0, 0]
            a_dd_string = round(a_dd, ndigits=3)

            # w_x = 2.0 * np.pi * f_x
            # w_y_string = round(f_x, ndigits=3)
            # w_y = 2.0 * np.pi * (w_x_freq / d_string)

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

            jobname = f"{supersolids_version}_a_s_{a_s_string}a_dd_{a_dd_string}"

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
#SBATCH -t 2-00:00:00
#SBATCH --mem={mem_in_GB}G
#SBATCH -p gpu
##SBATCH --exclude=alamak,algedi,baten,canopus,cressida,cursa,crux,dorado,gomeisa,kari,mintaka,nunki,oberon,rigel,telesto,tureis,weywot
##SBATCH -w altair,atlas,berti,gemini,mirzam,niobe,pegasus,phad,pollux,rana,sargas,weywot
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
#PBS -l walltime=100:00:00
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

# conda install -c scheiermann/label/main supersolids={supersolids_version}
# conda install -c scheiermann/label/testing supersolids={supersolids_version}
# conda install numba
# conda install cupy

# /bigwork/dscheier/miniconda/envs/solids/bin/pip install -i https://pypi.org/simple/supersolids=={supersolids_version}
# /bigwork/dscheier/miniconda/envs/solids/bin/pip install -i https://test.pypi.org/simple/ supersolids=={supersolids_version}

/bigwork/dscheier/miniconda/envs/solids/bin/python -m supersolids \
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

"""
            ])

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
