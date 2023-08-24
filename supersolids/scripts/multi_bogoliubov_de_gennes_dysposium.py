#!/usr/bin/env python
from pathlib import Path

import numpy as np


# Script runs, if script is run as main script (called by python *.py)
if __name__ == "__main__":
    supersolids_version = "0.1.38rc1"

    home = "/home/dscheiermann"
    # home = "/bigwork/dscheier"
    # experiment_suffix = "gpu_04_03"
    # experiment_suffix = "gpu_03_29_test1"
    # experiment_suffix = "gpu_04_18_bog"
    # experiment_suffix = "gpu_04_26_add"
    # experiment_suffix = "gpu_04_27_add3"
    # experiment_suffix = "gpu_05_02"
    # experiment_suffix = "gpu_05_03"
    experiment_suffix = "gpu_05_03_test_dys"
    dir_path = Path(f"{home}/results/begin_{experiment_suffix}/")

    movie_string = "movie"
    counting_format = "%03d"
    movie_start = 5
    movie_end = 5
    movie_number_list = np.arange(movie_start, movie_end + 1, 1)

    filename_schroedinger = "schroedinger.pkl"
    filename_steps = "step_"
    steps_format = "%07d"
    frame = None
    mode = "flat"

    nx_start = 12
    nx_end = 31
    nx_step = 4

    # ny = 16
    # nz = 16
    # ny = 14
    # nz = 14
    # ny = 8
    # nz = 8

    graphs_dirname = "graphs"
    label=""

    print_num_eigenvalues = 100

    mem_in_MB = 1400
    xvfb_display = 990

    skip = 0
    skip_counter = 0
    j_counter = 0
    # j_counter = skip - 1
    end = 0
    
    gpu_index = 1

    ######## END OF USER INPUT #####################################################################

    nx_array = np.arange(nx_start, nx_end, nx_step)
    for movie_number in movie_number_list:
        dir_name = f"{movie_string}{counting_format % movie_number}"
        for nx in nx_array:
            nx = int(nx)
            ny = int(nx)
            nz = int(nx)

            movie_string = "movie"
            counting_format = "%03d"

            skip_counter += 1
            if skip_counter < skip:
                continue
            if skip_counter == end:
                break

            jobname = f"{supersolids_version}_{mem_in_MB}M_{dir_name}_{nx}_{ny}_{nz}"

            heredoc = "\n".join(["#!/bin/bash",
                                f"""
Xvfb :{xvfb_display - j_counter} &
export DISPLAY=:{xvfb_display - j_counter}

source $(conda info --base)/etc/profile.d/conda.sh
conda activate {home}/miniconda/envs/solids

echo $(hostname -a)
echo $DISPLAY
echo $CONDA_PREFIX
echo "supersolids={supersolids_version}"
echo {jobname}

# export QT_QPA_PLATFORM=xcb


## to use local version
# ${home}/miniconda/envs/solids/bin/python3.10 ${home}/supersolids/supersolids/tools/bogoliubov_de_gennes.py \
## to use installed package version
{home}/miniconda/envs/solids/bin/python3.10 -m supersolids.tools.bogoliubov_de_gennes \
-dir_path={dir_path} \
-dir_name={dir_name} \
-filename_schroedinger={filename_schroedinger} \
-filename_steps={filename_steps} \
-steps_format={steps_format} \
-nx={nx} \
-ny={ny} \
-nz={nz} \
-print_num_eigenvalues={print_num_eigenvalues} \
-mode={mode} \
-label={label} \
-gpu_index={gpu_index} \
--recalculate \
--ground_state \
--arnoldi \
# --pytorch \
# --dipol \
# -frame={frame} \
-graphs_dirname={graphs_dirname} &

"""
            ])

            print(heredoc)
            with open(Path(dir_path, f"sbatch_bog_{dir_name}_{nx}_{ny}_{nz}.sh"), "w") as f:
                f.write(f"{heredoc}\n")

            j_counter += 1

