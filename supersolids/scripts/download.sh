#!/usr/bin/env bash

path_anchor_input="itpx:/bigwork/dscheier/supersolids/results/joseph_injunction2/"
path_anchor_output="/run/media/dsche/ITP Transfer/joseph_injunction2/y_kick/kick_fix_1.0/"

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
      printf -v movie_number "%03d" $((start + j))
      mkdir "${path_anchor_output}$dir_name$movie_number"

      echo $movie_number
      rsync -P "${path_anchor_input}$download_dir_name$((download_start + j))/*" "${path_anchor_output}${dir_name}${movie_number}" &
  done
  k=$batch
  wait
done
