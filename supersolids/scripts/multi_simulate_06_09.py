#!/usr/bin/env python

# author: Daniel Scheiermann
# email: daniel.scheiermann@itp.uni-hannover.de
# license: MIT
# Please feel free to use and modify this, but keep the above information.

import subprocess
from pathlib import Path


if __name__ == "__main__":
    supersolids_version = "0.1.37rc5"
    dir_path = Path("/bigwork/dscheier/results/begin_stacked_a11_05_09/")

    slurm: bool = True
    mem_in_GB = 4
    xvfb_display = 990

    max_timesteps = 220000

    file_start = "step_"
    file_number = 130000
    file_format = "%07d"
    file_pattern = ".npz"
    file_name = f"{file_start}{file_format % file_number}{file_pattern}"

    movie_string = "movie"
    counting_format = "%03d"
    movie_number = 6
    files2last = 10

    func_filename = "distort.txt"

    load_script = "script_0001.pkl"

    movie_number_now = movie_number

    movie_now = f"{movie_string}{counting_format % movie_number_now}"
    movie_number_after = movie_number + files2last
    movie_after = f"{movie_string}{counting_format % movie_number_after}"

    jobname = f"{supersolids_version}_m6_simulate"

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
#SBATCH -p gpu_cuda
#SBATCH --mem={mem_in_GB}G
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

Xvfb :{xvfb_display} &
export DISPLAY=:{xvfb_display}

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

# /bigwork/dscheier/miniconda/bin/pip3 install -i https://test.pypi.org/simple/ supersolids=={supersolids_version}
# /bigwork/dscheier/miniconda/bin/pip3 install -i https://pypi.org/simple/supersolids=={supersolids_version}

# /bigwork/dscheier/miniconda/bin/python3.8 -m supersolids.tools.simulate_npz

python -m supersolids.tools.simulate_npz \
-load_script={load_script} \
-dir_path={dir_path} \
-dir_name_load={movie_now} \
-dir_name_result={movie_after} \
-max_timesteps={max_timesteps}

"""
    ])

    print(heredoc)

    p = subprocess.Popen(["sbatch"], stdin=subprocess.PIPE, stdout=subprocess.PIPE, shell=False)

    out, err = p.communicate(heredoc.encode())
    p.wait()