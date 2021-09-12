#!/usr/bin/env bash
old="get_parity"
new="get_parity_2"
filetype=".png"

# dir='/run/media/dsche/ITP Transfer/scissor/movie027/'
path_anchor="/run/media/dsche/ITP Transfer/joseph_injunction2/real_phase_half/"

dir_name="movie"
movie_start=631
movie_number=20

for ((i = $movie_start; i < $((movie_start + movie_number)); i++))
do
    printf -v movie_counter "%03d" $i
    echo "$path_anchor$dir_name$movie_counter/$old$j$filetype"
    mv "$path_anchor$dir_name$movie_counter/$old$j$filetype" "$path_anchor$dir_name$movie_counter/$new$j$filetype"
done
