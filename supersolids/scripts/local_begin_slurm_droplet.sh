#!/bin/bash
#==================================================
#SBATCH --job-name 0.1.34rc24_N55000fy_86.0
#SBATCH -D /bigwork/dscheier/supersolids/supersolids/results/begin_droplet/log/
#SBATCH --mail-user daniel.scheiermann@itp.uni-hannover.de
#SBATCH --mail-type=END,FAIL
#SBATCH -o output-%j.out
#SBATCH -e error-%j.out
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 4-00:00:00
#SBATCH --mem=2G


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

Xvfb :990 &
export DISPLAY=:990

conda activate /bigwork/dscheier/miniconda/envs/solids
echo $DISPLAY
echo $CONDA_PREFIX
echo $(which python3)
echo $(which pip3)

# conda install -c scheiermannsean/label/main supersolids={supersolids_version}
conda install -c scheiermannsean/label/testing supersolids={supersolids_version}
conda install numba
conda install cupy

# /bigwork/dscheier/miniconda/envs/solids/bin/pip install -i https://test.pypi.org/simple/ supersolids==0.1.34rc24
# /bigwork/dscheier/miniconda/envs/solids/bin/pip install -i https://pypi.org/simple/supersolids==0.1.34rc24

python -m supersolids -Box='{"x0": -10, "x1": 10, "y0": -7, "y1": 7, "z0": -5, "z1": 5}' -Res='{"x": 128, "y": 64, "z": 32}' -max_timesteps=1500001 -dt=0.0002 -steps_per_npz=1 -steps_format=%07d -dir_path=/bigwork/dscheier/results/begin_droplet -dir_name_result=movie001 -a='{"a_x": 4.5, "a_y": 2.0, "a_z": 1.5}' -w_x=207.34511513692635 -w_y=540.3539364174444 -w_z=1049.291946298991 -accuracy=0.0 -noise 0.9 1.1 --N_list 55000 0 --m_list 163.9 0 --a_dd_list 130.8 0 0 --a_s_list 4.6567594559464e-09 0 0 --V_interaction --offscreen # --mixture

