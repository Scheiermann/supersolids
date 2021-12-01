#!/usr/bin/env bash
#==================================================
 #PBS -N 1.34rc19
 #PBS -M daniel.scheiermann@itp.uni-hannover.de
 #PBS -d /bigwork/dscheier/
 #PBS -e /bigwork/dscheier/error.txt
 #PBS -o /bigwork/dscheier/output.txt
 #PBS -l nodes=1:ppn=1:ws
 #PBS -l walltime=99:00:00
 #PBS -l mem=4GB
 #PBS -l vmem=4GB

supersolids_version=0.1.34rc19

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

Xvfb :4 &
export DISPLAY=:4

conda activate /bigwork/dscheier/miniconda3/envs/solids
echo $CONDA_PREFIX
echo $(which python3)
echo $(which pip3)
/bigwork/dscheier/miniconda3/bin/pip3 install -i https://test.pypi.org/simple/ supersolids==${supersolids_version}

dir_path="/bigwork/dscheier/supersolids/results/"
steps_per_npz=10000
steps_format="%07d"

# file_number=0
# movie_number_after=1
# movie_after_string="movie00"
# movie_after="$movie_after_string$((movie_number_after))"
# dir_path_after="/bigwork/dscheier/supersolids/results/$movie_after/"

/bigwork/dscheier/miniconda3/bin/python3.8 -m supersolids \
-N=50000 \
-Box='{"x0":-10, "x1":10, "y0":-5, "y1":5, "z0":-4, "z1":4}' \
-Res='{"x":256, "y":128, "z":32}' \
-max_timesteps=1500001 \
-dt=0.0002 \
-steps_per_npz=$steps_per_npz \
-steps_format="${steps_format}" \
-a='{"a_x": 4.5, "a_y": 2.0, "a_z": 1.5}' \
-dir_path="${dir_path}" \
-a_s=0.000000004656 \
-w_y=518.36 \
-accuracy=0.0 \
-noise 0.8 1.2 \
--V_interaction \
--offscreen

# -w_y=518.36
# -w_y=518.36 # w_y = 2 * np.pi * 82.50 # alpha_t=0.4 # get some 1D and all 2D, while bigger N
# -w_y=531.62 # w_y = 2 * np.pi * 84.61 # alpha_t=0.39
# -w_y=575.95 # w_y = 2 * np.pi * 91.67 # alpha_t=0.36 # get all 1D and some 2D, while bigger N
# -w_y=592.41 # w_y = 2 * np.pi * 94.28 # alpha_t=0.35
# -w_y = 628.31 # w_y = 2 * np.pi * 100 # alpha_t=0.33
# -w_z=2100
# w_z= 2 * np.pi * 127 * 2 = 2100
# a_s=0.000000004656=4.656 * 10**-9=88*a_0
# a_s=0.000000004498=4.498 * 10**-9=85*a_0

simulate_exit="$?"

# if simulation did not exit with 1, continue with animation creation
if [ $simulate_exit != 1 ]; then
    printf "\nCreate the animation to the simulation\n"
#    /bigwork/dscheier/miniconda3/bin/python3.8 -m supersolids.tools.load_npz \
#    -frame_start=$frame_start \
#    -dir_path="${dir_path_after}" \
#    -dir_name=$movie_after \
#    -steps_per_npz=$steps_per_npz \
#    -slice_indices='{"x":127,"y":63,"z":15}' \
#    --plot_V
else
    printf "\nsimulate_npz ended with sys.exit(1)\n"
fi
