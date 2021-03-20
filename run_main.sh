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

# pull the latest version
git pull

# Run python with the args to the script
python main.py

# update the weights and logs to git
git add checkpoint.pt training_logs.json
git commit -m "automatic weights and logs upload after training"
git push

echo "*** SLURM BATCH JOB '$JOB_NAME' DONE ***"