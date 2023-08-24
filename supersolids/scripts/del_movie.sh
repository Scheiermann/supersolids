#!/usr/bin/env bash

movie_start=475
movie_del_number=25
movie_string="movie"
dir_path="/bigwork/dscheier/supersolids/results/"

for ((i = $movie_start; i < $((movie_start + movie_del_number)); i++))
do
	movie="$movie_string$i"
        rm -r $dir_pathi$movie
done


