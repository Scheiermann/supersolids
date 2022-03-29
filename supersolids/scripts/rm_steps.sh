#!/usr/bin/env bash


path_anchor_input="/bigwork/dscheier/supersolids/supersolids/results/begin_alpha/"

# next_new allows to point to different movies of one batched simulation
next_new=0

# number of movies in batched simulation
number=$((40 - next_new))

# starting number to numerate the output directories
start=$((1 + next_new))
dir_name="movie"

rm_start=$start
rm_dir_name="movie"

regex="step_*[1-9]000.npz"

for ((j = ${start}; j < $((start + number)); j++))
do
    printf -v movie_number_in "%03d" $((j))

    echo $movie_number_in
    rm "${path_anchor_input}${rm_dir_name}${movie_number_in}/"${regex}
done


