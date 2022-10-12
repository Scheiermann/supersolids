#!/usr/bin/env bash
start=350605
job_number=92

mem=6
vmem=6

for ((i = $start; i < $((start + job_number)); i++))
do
	qalter -l mem=${mem} $i
	qalter -l vmem=${vmem}G $i
done

