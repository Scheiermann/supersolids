#!/usr/bin/env bash
old="/step_"
new="/step_"
filetype=".npz"
start=999000
steps=100
number=9


# dir='/run/media/dsche/ITP Transfer/scissor/movie027/'
path_anchor="./"

dir_name="movie"
movie_start=26
movie_number=24

for ((i = $movie_start; i < $((movie_start + movie_number)); i++))
do
    printf -v movie_counter "%03d" $i
    for ((j = start; j < $((start + (number + 1) * steps)); j=$((j + steps))))
    do
        printf -v step_counter "%07d" $j
        echo $path_anchor$dir_name$movie_counter$old$j$filetype
        mv $path_anchor$dir_name$movie_counter$old$j$filetype $path_anchor$dir_name$movie_counter$new$step_counter$filetype
    done
done
