#!/usr/bin/env bash

dir_path="/mnt/extern/begin_alpha/"

max_timesteps=700001
dt=0.0002
steps_per_npz=1000
accuracy=0.0

file_start="step_"
file_number=850000
file_format="%07d"
file_pattern=".npz"
printf -v file_number_formatted ${file_format} ${file_number}
file_name="${file_start}${file_number_formatted}${file_pattern}"

movie_string="movie"
movie_format="%03d"
movie_number=11

files2last=40
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
-filename_npz=${file_name} \
-dir_path=${dir_path} \
--offscreen \
--V_reload \
--real_time
