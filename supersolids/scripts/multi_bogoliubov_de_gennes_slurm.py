#!/usr/bin/env python
from pathlib import Path

import numpy as np


# Script runs, if script is run as main script (called by python *.py)
if __name__ == "__main__":
    supersolids_version = "0.1.38rc1"

    # home = "/home/dscheiermann"
    home = "/bigwork/dscheier"
    # home = "/mnt/disk2/dscheiermann"
    # experiment_suffix = "gpu_04_03"
    # experiment_suffix = "gpu_03_29_test1"
    # experiment_suffix = "gpu_04_18_bog"
    # experiment_suffix = "gpu_04_26_add"
    # experiment_suffix = "gpu_04_27_add3"
    # experiment_suffix = "gpu_05_02"
    # experiment_suffix = "gpu_05_06"
    # experiment_suffix = "gpu_05_22"
    # experiment_suffix = "gpu_05_22_dys"
    # experiment_suffix = "gpu_05_23_dys"
    # experiment_suffix = "gpu_05_24_dys"
    # experiment_suffix = "gpu_05_24_dys_res"
    # experiment_suffix = "gpu_05_24_dys_res_256"
    # experiment_suffix = "gpu_05_06_res_128"
    # experiment_suffix = "gpu_05_06_res_128_box_4"
    # experiment_suffix = "gpu_05_06_res_256_box_4"
    # experiment_suffix = "gpu_05_09_dt"
    # experiment_suffix = "gpu_05_09_dt2"
    # experiment_suffix = "gpu_06_09_dys"
    # experiment_suffix = "gpu_06_20"
    # experiment_suffix = "gpu_06_23_dys162"
    experiment_suffix = "gpu_06_23_dys162_box"
    # experiment_suffix = "gpu_06_28"
    dir_path = Path(f"{home}/results/begin_{experiment_suffix}/")

    mem_in_GB = 4

    movie_string = "movie"
    counting_format = "%03d"
    # movie_start = 15
    # movie_end = 15
    # movie_start = 1
    # movie_end = 1
    # movie_start = 15
    # movie_end = 15
    movie_start = 1
    movie_end = 13
    # movie_start = 8
    # movie_end = 8
    movie_number_list = np.arange(movie_start, movie_end + 1, 1)

    arnoldi_num_eigs = 30

    filename_schroedinger = "schroedinger.pkl"
    filename_steps = "step_"
    steps_format = "%07d"
    frame = None
    # mode = "flat"
    # mode = "fft"
    mode = "lin_op"
    # mode = "linear"
    # mode = "dask"
    # mode = "smart"

    csr_cut_off_0 = 0.1

    # nx_start = 6
    # nx_start = 6
    nx_start = 128
    # nx_start = 46
    nx_end = 129
    # nx_start = 32
    # nx_end = 53
    # nx_start = 32
    # nx_end = 17
    # nx_end = 33
    # nx_start = 6
    # nx_end = 15
    # nx_step = 1
    nx_step = 2
    # nx_start = 16
    # nx_end = 33
    # nx_step = 4

    ny = 64
    nz = 64

    stepper_x = 1
    stepper_y = 1
    stepper_z = 1

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
    
    gpu_index = 0

    ######## END OF USER INPUT #####################################################################

    nx_array = np.arange(nx_start, nx_end, nx_step)
    for movie_number in movie_number_list:
        dir_name = f"{movie_string}{counting_format % movie_number}"
        for nx in nx_array:
            # nx = int(nx)
            # ny = int(nx)
            # nz = int(nx)

            nx = int(nx)
            ny = int(ny)
            nz = int(nz)

            movie_string = "movie"
            counting_format = "%03d"

            skip_counter += 1
            if skip_counter < skip:
                continue
            if skip_counter == end:
                break

            jobname = f"{supersolids_version}_{mem_in_MB}M_{dir_name}_{nx}_{ny}_{nz}"

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
##SBATCH -p gpu_cuda
##SBATCH --exclude=alamak,algedi,baten,canopus,cressida,cursa,crux,dorado,gomeisa,kari,mintaka,nunki,oberon,rigel,telesto,tureis,weywot
##SBATCH -w altair,atlas,berti,gemini,mirzam,niobe,pegasus,phad,pollux,rana,sargas,weywot
"""

            heredoc = "\n".join(["#!/bin/bash",
                                 cluster_flags,
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
# {home}/miniconda/envs/solids/bin/python3.10 -m supersolids.tools.bogoliubov_de_gennes \

{home}/miniconda/envs/solids/bin/python3.10 {home}/supersolids/supersolids/tools/bogoliubov_de_gennes.py \
-dir_path={dir_path} \
-dir_name={dir_name} \
-filename_schroedinger={filename_schroedinger} \
-filename_steps={filename_steps} \
-steps_format={steps_format} \
-nx={nx} \
-ny={ny} \
-nz={nz} \
-stepper_x={stepper_x} \
-stepper_y={stepper_y} \
-stepper_z={stepper_z} \
-print_num_eigenvalues={print_num_eigenvalues} \
-mode={mode} \
-label={label} \
-gpu_index={gpu_index} \
--dipol \
--recalculate \
--ground_state \
--arnoldi \
-arnoldi_num_eigs={arnoldi_num_eigs} \
-csr_cut_off_0={csr_cut_off_0} \
--get_eigenvalues \
--reduced_version \
--gpu_off \
# --cut_hermite_values \
# --cut_hermite_orders \
# --dask_dipol \
# --pytorch \
# -frame={frame} \
-graphs_dirname={graphs_dirname} &
"""
            ])

            print(heredoc)
            with open(Path(dir_path,
                           f"sbatch_bog_{dir_name}_{nx}_{ny}_{nz}_{mode}_"
                           + f"{stepper_x}_{stepper_y}_{stepper_z}.sh"), "w") as f:
                f.write(f"{heredoc}\n")

            j_counter += 1

