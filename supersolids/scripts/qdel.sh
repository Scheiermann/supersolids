#!/usr/bin/env bash
start=322466
job_number=25

for ((i = $start; i < $((start + job_number)); i++))
do
	qdel $i
done

