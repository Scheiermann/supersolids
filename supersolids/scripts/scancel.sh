#!/usr/bin/env bash
start=9150
job_number=9

for ((i = $start; i < $((start + job_number)); i++))
do
	scancel $i
done

