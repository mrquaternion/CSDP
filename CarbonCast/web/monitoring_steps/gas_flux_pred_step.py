import textwrap


def gas_flux_pred_job_script(
    slurm_account: str,
    memory: str,
    cpus: int,
    wall_time: str,
    gpus: int,
) -> str:
    return textwrap.dedent(
        f"""\
            #!/bin/bash
            #SBATCH --account={slurm_account}
            #SBATCH --mem={memory}
            #SBATCH --time={wall_time}
            #SBATCH --cpus-per-task={cpus}
            #SBATCH --gres=gpu:{gpus}
            #SBATCH --output=slurm-%j.out
            #SBATCH --error=slurm-%j.err

            printf "\\nLoading required modules.\\n"
            module load proj/9.2.0 python/3.12
            REPO_DIR="$HOME/scratch/CarbonCast/EcoPerceiver"

            printf "\\nCreating the environment."
            virtualenv --no-download $SLURM_TMPDIR/ccenv
            source $SLURM_TMPDIR/ccenv/bin/activate

            pip install --upgrade pip --no-index

            printf "\\nInstalling EcoPerceiver dependencies."
            cd "$REPO_DIR"
            pip install -e . --no-cache-dir --no-index
            pip install scipy --no-index
            pip install h5py h5netcdf --no-index
            pip install timezonefinder --no-index

            printf "\\nExecuting the inference script."
            cd eval/
            python3 era5_db_launch.py
            python3 test_era5.py
        """
    )
