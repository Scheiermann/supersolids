#!/usr/bin/env bash

# author: Daniel Scheiermann
# email: daniel.scheiermann@itp.uni-hannover.de
# license: MIT
# Please feel free to use and modify this, but keep the above information.

path_anchor_input="itpx:/bigwork/dscheier/supersolids/results/"
path_anchor_output="/run/media/dsche/ITP Transfer/"

# next_new allows to point to different movies of one batched simulation
next_new=0

# number of movies in batched simulation
number=$((20 - next_new))

# starting number to numerate the output directories
start=$((820 + next_new))
dir_name="movie"

# movie152, 4droplets
# starting number of enumerated directories to download
# download_start=$((795 + next_new))
download_start=$start
download_dir_name="movie"

max_parallel_download=2
# depending on the server settings, number of connections is limited
# create array with sequence to batch the number of movies
download_batch=($(seq $max_parallel_download $max_parallel_download $number))
if (( ${#download_batch[@]} ));then
    :
else
    # add number to sequence
    download_batch+=($number)
fi

if [ $number -ne  ${download_batch[-1]} ]
then
    # add number to sequence
    download_batch+=($number)
fi

echo ${download_batch[*]}

k=0
for batch in "${download_batch[@]}"
do
  for ((j = k; j < $batch; j++))
  do
      printf -v movie_number_in "%03d" $((download_start + j))
      printf -v movie_number_output "%03d" $((start + j))
      mkdir "${path_anchor_output}$dir_name$movie_number_output"

      echo $movie_number_output
      rsync -P "${path_anchor_input}$download_dir_name${movie_number_in}/*" "${path_anchor_output}${dir_name}${movie_number_output}" &
  done
  k=$batch
  wait
done
