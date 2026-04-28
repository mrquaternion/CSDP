#!/bin/bash
#SBATCH --account=def-sonol
#SBATCH --mem=256G
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=8
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

set -euo pipefail

: "${CONFIG:?CONFIG is not set}"

printf "\nLoading required modules.\n"
module load proj/9.2.0 python/3.12

VENV_DIR="$SLURM_TMPDIR/ccenv"

printf "\nCreating the environment at %s.\n" "$VENV_DIR"
virtualenv --no-download "$VENV_DIR"
source "$VENV_DIR/bin/activate"
pip install --upgrade pip --no-index

REPO_DIR="$HOME/scratch/CarbonCast/carboncast"
cd "$REPO_DIR"

printf "\nInstalling CarbonCast dependencies.\n"
pip install -e . --no-index

export OUTPUTS_TMP_DIR="$SLURM_TMPDIR/outputs_tmp"
export HDF5_USE_FILE_LOCKING=FALSE

echo "Starting pipeline with config: $CONFIG"
era5dp process --config "$CONFIG"
