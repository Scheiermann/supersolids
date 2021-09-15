#!/usr/bin/env bash

dir_path="/run/media/dsche/ITP Transfer/test/"
steps_per_npz=1
steps_format="%07d"

frame_start=0
movie_number_result=17
movie_string="movie"
movie_format="%03d"
printf -v movie_number_formatted ${movie_format} ${movie_number_result}
dir_name_result=$movie_string$movie_number_formatted

dir_path_after="/run/media/dsche/ITP Transfer/test/"

python -m supersolids \
-N=50000 \
-Box='{"x0":-10, "x1":10, "y0":-5, "y1":5, "z0":-4, "z1":4}' \
-Res='{"x":256, "y":128, "z":32}' \
-max_timesteps=5 \
-dt=0.0002 \
-steps_per_npz=$steps_per_npz \
-steps_format="${steps_format}" \
-a='{"a_x": 4.5, "a_y": 2.0, "a_z": 1.5}' \
-dir_path="${dir_path}" \
-dir_name_result="${dir_name_result}" \
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
#    python -m supersolids.tools.load_npz \
#    -frame_start=$frame_start \
#    -dir_path="${dir_path_after}" \
#    -dir_name=$movie_after \
#    -steps_per_npz=$steps_per_npz \
#    -slice_indices='{"x":127,"y":63,"z":15}' \
#    --plot_V
else
    printf "\nsimulate_npz ended with sys.exit(1)\n"
fi
