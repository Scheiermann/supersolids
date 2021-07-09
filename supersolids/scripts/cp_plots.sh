#!/usr/bin/env bash

path_anchor_input="/mnt/extern/joseph_injunction2/real_phase_half/"
path_anchor_output="/mnt/extern/joseph_injunction2/real_phase_half/graphs/"

filename_in="get_center_of_mass"
filename_out="get_center_of_mass"
filename_extension=".png"

start=631
number=20
dir_name="movie"

for ((j = start; j < $((start + number)); j++))
do
    printf -v movie_number "%03d" $j
    mkdir "${path_anchor_output}"

    echo $movie_number
    cp "${path_anchor_input}$dir_name${movie_number}/${filename_in}${filename_extension}" "${path_anchor_output}${filename_out}${movie_number}${filename_extension}"
done
