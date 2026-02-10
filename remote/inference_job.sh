#!/bin/bash
#SBATCH --account=def-sonol
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16GB
#SBATCH --time=02:00:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

printf "\nLoading the Python 3.12.4 module."
module load python/3.12.4

printf "\nCreating the environment."
virtualenv --no-download $SLURM_TMPDIR/cdsenv

printf "\nActivating the environment."
source $SLURM_TMPDIR/cdsenv/bin/activate

cd .. # go back to root
cd EcoPerceiver/

printf "\nInstalling EcoPerceiver dependencies."
pip install -e . --no-index

cd .. # go back to root
cd CS-Pipeline/pipeline/

printf "\nInstalling CarbonSense Data Pipeline dependencies."
pip install -e . --no-index

cd .. # go back to CS-Pipeline
cd .. # go back to root
cd EcoPerceiver/inference/

nvidia-smi

printf "\nExecuting the inference script."
python3 inference.py
