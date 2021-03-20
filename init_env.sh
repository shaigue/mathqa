# https://hpc.cswp.cs.technion.ac.il/newton-computational-cluster/
# cloning the repo from git
# with a generated rsa key, first generate it with the links below
# https://stackoverflow.com/questions/8588768/how-do-i-avoid-the-specification-of-the-username-and-password-at-every-git-push
# https://www.toolsqa.com/git/clone-repository-using-ssh/#:~:text=Press%20Clone%20or%20download%20and,want%20to%20clone%20the%20repository.

# clone the repo
yes | git clone git@github.com:shaigue/mathqa.git
# now install miniconda
# get the link for the installer and the sha from here
# https://docs.conda.io/en/latest/miniconda.html#linux-installers
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh

hash_value=1314b90489f154602fd794accfc90446111514a5a72fe1f71ab83e07de9504a7
echo "$hash_value Miniconda3-latest-Linux-x86_64.sh" | sha256sum --check --status

if [ $? != 0 ]; then
  echo "bad sha256sum"
  exit 1
fi
echo "good sha256sum"
yes yes | bash Miniconda3-latest-Linux-x86_64.sh

# TODO: figure out how to start anaconda now/ restart the shell

# enter the repo and install the environment
# TODO: install the environment using srun to catch the cpu
cd mathqa
# TODO: installing with env did not work with pytorch, need to install with gpu suppurt
conda env create -n mathqa python=3.9
conda activate mathqa
conda install pytorch cudatoolkit=11 -c pytorch -c conda-forge

conda activate mathqa-torch

export PATH="/usr/local/cuda/bin:$PATH"
export LD_LIBRARY_PATH="/usr/local/cuda/lib64:$LD_LIBRARY_PATH"
