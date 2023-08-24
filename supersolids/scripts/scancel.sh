#!/usr/bin/env bash
start=50119
job_number=31

for ((i = $start; i < $((start + job_number)); i++))
do
	scancel $i
done

