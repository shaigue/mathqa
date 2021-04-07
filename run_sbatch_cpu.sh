#!/bin/bash

# Parameters for sbatch
NUM_NODES=1
NUM_CORES=2
NUM_GPUS=0
JOB_NAME="mathqa-cpu"
OUTPUT_FILE="slurm-%j.out"

sbatch \
	-N $NUM_NODES \
	-c $NUM_CORES \
	--gres=gpu:$NUM_GPUS \
	--job-name $JOB_NAME \
	--output $OUTPUT_FILE \
	./run_main.sh
