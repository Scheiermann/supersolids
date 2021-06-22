#!/usr/bin/env bash

# next_new allows to point to different movies of one batched simulation
next_new=0

# number of movies in batched simulation
number=$((30 - next_new))

# starting number to numerate the output directories
start=$((1 + next_new))
dir_name="movie"

# movie152, 4droplets
# starting number of enumerated directories to download
download_start=$((516 + next_new))
download_dir_name="movie"

# depending on the server settings, number of connections is limited
max_parallel_download=10
download_batch=($(seq $max_parallel_download $max_parallel_download $number))
if [ $number -ne  ${download_batch[-1]} ]
then
    download_batch+=($number)
fi

path_anchor="/run/media/dsche/ITP Transfer/joseph_injunction2/real2/"


echo ${download_batch[*]}

k=0
for batch in "${download_batch[@]}"
do
  #for movie_number in $(eval "echo {001..$number}")
  for ((j = k; j < $batch; j++))
  do
      printf -v movie_number "%03d" $((start + j))
      mkdir "$path_anchor$dir_name$movie_number"

      echo $movie_number
      rsync -P itpx:/bigwork/dscheier/supersolids/results/$download_dir_name$((download_start + j))/* "${path_anchor}${dir_name}${movie_number}" &
  done
  k=$batch
  wait
done
