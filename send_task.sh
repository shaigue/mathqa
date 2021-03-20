#!/bin/bash
# script that runs main python script for sbatch
#https://slurm.schedmd.com/sbatch.html
#https://hpc.cswp.cs.technion.ac.il/newton-computational-cluster/#using-slurm
git pull
sbatch --cpus-per-task=1 --gres=gpu:1 --job-name="mathqa" --output="slurm_out.txt" ./run_main.sh

