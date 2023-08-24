#/usr/bin/bash

dir="movie"
start=13
end=25

path="/bigwork/dscheier/results/begin_pretilt0.05_a11_60to100/"
pkl="SchroedingerMixtureSummary_"
npz="step_"


for (( i=$start; i<=$end; i++ ))
do
    dirname=$dir"0"$i
    echo $dirname
    cd $path$dirname
    last_npz=$(ls -- $npz*.npz | cut -c 6-12 | sort -n | tail -n 1)
    last_pkl=$(ls -- $pkl*.pkl | cut -c 28-34 | sort -n | tail -n 1)
    echo $last_npz
    echo $last_pkl
    if [[ $last_npz > $last_pkl ]]
    then
        echo $last_npz
        echo $path$dirname/*$last_npz.*
        rm $path$dirname/*$last_npz.*
    else
        echo $last_pkl
        echo $path$dirname/*$last_pkl.*
        rm $path$dirname/*$last_pkl.*
    fi
done
