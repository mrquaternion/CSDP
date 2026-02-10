#!/bin/bash
#SBATCH --account=def-sonol
#SBATCH --cpus-per-task=1
#SBATCH --mem=16GB
#SBATCH --time=02:00:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

printf "\nLoading required modules."
module load python/3.12 mpi4py

printf "\nCreating the environment."
virtualenv --no-download $SLURM_TMPDIR/cdsenv

printf "\nActivating the environment."
source $SLURM_TMPDIR/cdsenv/bin/activate

cd .. # go back to root
cd CS-Pipeline/pipeline/

printf "\nInstalling CarbonSense Data Pipeline dependencies."
pip install --upgrade pip --no-index
pip install -e . --no-index
pip install scipy --no-index

printf "\nLaunching the database.\n"
cd .. # go back to CS-Pipeline
cd .. # go back to root
cd EcoPerceiver/inference/
python3 launch_db.py
