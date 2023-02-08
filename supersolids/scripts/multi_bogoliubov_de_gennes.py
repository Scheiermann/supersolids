#!/usr/bin/env python
from pathlib import Path

import numpy as np


# Script runs, if script is run as main script (called by python *.py)
if __name__ == "__main__":
    supersolids_version = "0.1.37rc7"

    home = "/bigwork/dscheier"
    experiment_suffix = "gpu_02_06_no_V_1comp"
    dir_path = Path(f"{home}/results/begin_{experiment_suffix}/")

    dir_name = "movie027"
    filename_schroedinger = "schroedinger.pkl"
    filename_steps = "step_"
    steps_format = "%07d"
    frame = None

    n_start = 2
    n_end = 5
    n_step = 1
    recalculate = False

    graphs_dirname = "graphs"

    print_num_eigenvalues = 20

    mem_in_MB = 1400
    xvfb_display = 990

    skip = 0
    skip_counter = 0
    j_counter = 0
    # j_counter = skip - 1
    end = 0


    ######## END OF USER INPUT #####################################################################

    n_array = np.arange(n_start, n_end, n_step)
    for n in n_array:
        n = int(n)
        nx, ny, nz = n, n, n

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
conda activate /home/dscheiermann/miniconda/envs/solids

echo $(hostname -a)
echo $DISPLAY
echo $CONDA_PREFIX
echo "supersolids={supersolids_version}"
echo {jobname}

# export QT_QPA_PLATFORM=xcb

# /home/dscheiermann/miniconda/envs/solids/bin/python3.10 -m supersolids.tools.bogoliubov_de_gennes \
/home/dscheiermann/miniconda/envs/solids/bin/python3.10 /home/dscheiermann/supersolids/supersolids/tools/bogoliubov_de_gennes.py \
-dir_path={dir_path} \
-dir_name={dir_name} \
-filename_schroedinger={filename_schroedinger} \
-filename_steps={filename_steps} \
-steps_format={steps_format} \
-frame={frame} \
-nx={nx} \
-ny={ny} \
-nz={nz} \
-recalculate={recalculate} \
-print_num_eigenvalues={print_num_eigenvalues} \
-graphs_dirname={graphs_dirname} &

"""
        ])

        print(heredoc)
        with open(Path(dir_path, f"sbatch_{dir_name}_{nx}_{ny}_{nz}.sh"), "w") as f:
            f.write(f"{heredoc}\n")

        j_counter += 1

