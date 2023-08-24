#!/usr/bin/env bash
dir_name="movie"

start_in=1
number=48

start_out=427

#path="/run/media/dsche/ITP Transfer/joseph_injunction2/"
path="/run/media/dsche/ITP Transfer/joseph_injunction/from_step_1510000/"

for ((i = 0; i < ${number}; i++))
do
    printf -v movie_number_in "%03d" $((start_in + i))
    printf -v movie_number_out "%03d" $((start_out + i))
    mv "$path$dir_name$movie_number_in" "$path"$dir_name$movie_number_out
done
