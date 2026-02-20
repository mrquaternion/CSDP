import textwrap


def gas_flux_pred_job_script(slurm_account: str, memory: str, cpus: int, wall_time: str) -> str:
    return textwrap.dedent(
        f"""\
            #!/bin/bash
            #SBATCH --account={slurm_account}
            #SBATCH --mem={memory}
            #SBATCH --time={wall_time}
            #SBATCH --cpus-per-task={cpus}
            #SBATCH --gres=gpu:1
            #SBATCH --output=slurm-%j.out
            #SBATCH --error=slurm-%j.err

            set -euo pipefail

            printf "\\nLoading required modules.\\n"
            module load StdEnv/2023 openmpi/4.1.5 netcdf-mpi/4.9.2 mpi4py/4.0.3 proj/9.2.0 python/3.12
            REPO_DIR="$HOME/scratch/CarbonCast/EcoPerceiver"

            printf "\\nCreating the environment."
            virtualenv --no-download $SLURM_TMPDIR/ccenv
            source $SLURM_TMPDIR/ccenv/bin/activate

            pip install --upgrade pip --no-index

            printf "\\nInstalling EcoPerceiver dependencies."
            cd "$REPO_DIR"
            pip install -e . --no-cache-dir --ignore-installed --no-index
            pip install scipy --no-index

            export LD_PRELOAD=$EBROOTOPENMPI/lib/libmpi.so

            printf "\\nExecuting the inference script."
            cd eval/
            python3 era5_db_launch.py
            python3 test_era5.py
        """
    )
