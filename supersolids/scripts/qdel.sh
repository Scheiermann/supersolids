#!/usr/bin/env bash
start=350862
job_number=50

for ((i = $start; i < $((start + job_number)); i++))
do
	qdel $i
done

