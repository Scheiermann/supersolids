#!/usr/bin/env bash
supersolids_version=1.33.rc4
#==================================================
#SBATCH --job-name 1.33.rc4
#SBATCH --workdir /bigwork/dscheier/slurm/
#SBATCH -e /bigwork/dscheier/slurm/error-$SLURM_JOBID.txt
#SBATCH -o /bigwork/dscheier/slurm/output-$SLURM_JOBID.txt
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 0-24:00:00
#SBATCH --mem=5GB
#SBATCH --mem-per-cpu=5GB

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

Xvfb :$SLURM_JOBID &
export DISPLAY=:$SLURM_JOBID

conda activate /bigwork/dscheier/miniconda3/envs/solids
echo $DISPLAY
echo $CONDA_PREFIX
echo $(which python3)
echo $(which pip3)

/bigwork/dscheier/miniconda3/bin/pip3 install -i https://test.pypi.org/simple/ supersolids==${supersolids_version}
# /bigwork/dscheier/miniconda3/bin/pip3 install -i https://pypi.org/simple/supersolids==${supersolids_version}

/bigwork/dscheier/miniconda3/bin/python3.8 -h