#!/bin/bash
###
# Conda parameters
#
CONDA_HOME=$HOME/miniconda3
CONDA_ENV=mathqa

echo "*** SLURM BATCH JOB '$JOB_NAME' STARTING ***"
# Setup the conda env
echo "*** Activating environment $CONDA_ENV ***"
source "$CONDA_HOME/etc/profile.d/conda.sh"
conda activate $CONDA_ENV
# set the python path to the current directory
export PYTHONPATH="$PWD"
# Run python with the args to the script
python main.py
echo "*** SLURM BATCH JOB '$JOB_NAME' DONE ***"