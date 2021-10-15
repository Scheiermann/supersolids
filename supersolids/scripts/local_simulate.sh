#!/usr/bin/env bash

#dir_path="/run/media/dsche/ITP Transfer/test/"
 dir_path="/home/dsche/supersolids/results"
# dir_path="/run/media/dsche/ITP Transfer/begin_alpha/"
# dir_path="/mnt/extern/begin_alpha/"

max_timesteps=11
dt=0.0002
steps_per_npz=1
accuracy=0.0

file_start="mixture_step_"
#file_start="step_"
file2_start="2-step_"
file_number_load=20
file_format="%07d"
file_pattern=".npz"
printf -v file_number_formatted ${file_format} ${file_number_load}
filename_npz="${file_start}${file_number_formatted}${file_pattern}"
filename2_npz="${file2_start}${file_number_formatted}${file_pattern}"

movie_string="movie"
movie_format="%03d"
movie_number=2

files2last=1
movie_number_after=$((movie_number + files2last))

printf -v movie_number_formatted ${movie_format} ${movie_number}
movie_now="${movie_string}${movie_number_formatted}"
printf -v movie_number_after_formatted ${movie_format} ${movie_number_after}
movie_after="${movie_string}${movie_number_after_formatted}"

python -m supersolids.tools.simulate_npz \
-Res='{"x": 256, "y": 128, "z": 32}' \
-Box='{"x0": -10, "x1": 10, "y0": -5, "y1": 5, "z0": -4, "z1": 4}' \
-max_timesteps=${max_timesteps} \
-dt=${dt} \
-steps_per_npz=${steps_per_npz} \
-accuracy=${accuracy} \
-dir_name_load=${movie_now} \
-dir_name_result=${movie_after} \
-filename_npz=${filename_npz} \
-filename2_npz=${filename2_npz} \
-dir_path="${dir_path}" \
--offscreen \
--V_reload
# --real_time
