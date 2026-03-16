#!/bin/bash
#SBATCH --account=def-sonol
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:4
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

printf "\nLoading required modules.\n"
module load proj/9.2.0 python/3.12
REPO_DIR="$HOME/scratch/CarbonCast/EcoPerceiver"

printf "\nCreating the environment."
virtualenv --no-download $SLURM_TMPDIR/ccenv
source $SLURM_TMPDIR/ccenv/bin/activate

pip install --upgrade pip --no-index

printf "\nInstalling EcoPerceiver dependencies."
cd "$REPO_DIR"
pip install -e . --no-cache-dir --no-index
pip install scipy --no-index
pip install h5py h5netcdf --no-index

printf "\nExecuting the inference script."
cd eval/
python3 era5_db_launch.py
python3 test_era5.py
