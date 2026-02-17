import re
import shlex
import textwrap
import time
from pathlib import Path
from typing import Any, Callable

import yaml

from .download_and_sync import CONFIG_FILENAME, LOCAL_CONFIG_DIR, get_ssh_common_options, run

TERMINAL_STATES = {
    "COMPLETED",
    "FAILED",
    "CANCELLED",
    "TIMEOUT",
    "NODE_FAIL",
    "PREEMPTED",
    "OUT_OF_MEMORY",
    "BOOT_FAIL",
    "DEADLINE",
}


def validate_post_processing_payload(payload: dict[str, Any] | None):
    memory = str((payload or {}).get("memory", "")).strip().upper()
    cpus_raw = str((payload or {}).get("cpus", "")).strip()
    wall_time = str((payload or {}).get("time", "")).strip()
    slurm_account = str((payload or {}).get("slurm_account", "")).strip()

    if not re.match(r"^\d+[KMGTP]?$", memory):
        raise ValueError("Invalid memory format. Use values like 8G, 32G, or 16000M.")

    if not cpus_raw.isdigit() or int(cpus_raw) <= 0:
        raise ValueError("CPUs per task must be a positive integer.")
    cpus = int(cpus_raw)

    time_pattern = r"^(\d+-)?\d{1,2}:\d{2}(:\d{2})?$"
    if not re.match(time_pattern, wall_time):
        raise ValueError("Invalid time format. Use HH:MM, HH:MM:SS, or D-HH:MM:SS.")

    if not slurm_account:
        raise ValueError("Slurm account is required.")
    if not re.match(r"^[A-Za-z0-9._-]+$", slurm_account):
        raise ValueError("Invalid Slurm account format.")

    return memory, cpus, wall_time, slurm_account


def render_postprocessing_job_script(slurm_account: str, memory: str, cpus: int, wall_time: str) -> str:
    return textwrap.dedent(
        f"""\
        #!/bin/bash
        #SBATCH --account={slurm_account}
        #SBATCH --mem={memory}
        #SBATCH --time={wall_time}
        #SBATCH --cpus-per-task={cpus}
        #SBATCH --output=slurm-%j.out
        #SBATCH --error=slurm-%j.err

        set -euo pipefail

        printf "\\nLoading required modules.\\n"
        module load python/3.12 mpi4py

        TMPROOT="${{SLURM_TMPDIR:-/tmp}}"
        VENV_DIR="$TMPROOT/csdpenv"

        printf "\\nCreating the environment at %s.\\n" "$VENV_DIR"
        virtualenv --no-download "$VENV_DIR"

        printf "\\nActivating the environment.\\n"
        source "$VENV_DIR/bin/activate"

        REPO_DIR="$HOME/scratch/CSDP/pipeline"
        cd "$REPO_DIR"

        printf "\\nInstalling CarbonSense Data Pipeline dependencies.\\n"
        pip install --upgrade pip --no-index
        pip install -e . --no-index

        CONFIG_PATH="${{1:-$REPO_DIR/config/process_config.yml}}"
        echo "Starting pipeline with config: $CONFIG_PATH"
        carbonpipeline process --config "$CONFIG_PATH"
        """
    )


def write_generated_job_script(
    slurm_account: str,
    memory: str,
    cpus: int,
    wall_time: str,
    output_path: Path | None = None,
) -> Path:
    job_script_path = output_path or (LOCAL_CONFIG_DIR / "postprocess_job.generated.sh")
    job_script_path.parent.mkdir(parents=True, exist_ok=True)
    job_script_path.write_text(
        render_postprocessing_job_script(
            slurm_account=slurm_account,
            memory=memory,
            cpus=cpus,
            wall_time=wall_time,
        ),
        encoding="utf-8",
    )
    job_script_path.chmod(0o755)
    return job_script_path.resolve()


def prepare_remote_postprocessing_assets(
    account: str,
    configuration_data: Any | None,
    local_job_script_path: Path,
    emit_output: Callable[[str, str], None],
):
    del configuration_data  # Keep process config based only on normalized download config.
    config_path = LOCAL_CONFIG_DIR / CONFIG_FILENAME
    if not config_path.exists():
        raise RuntimeError(f"Download config file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as fh:
        config_data = yaml.safe_load(fh) or {}

    config_data["action"] = "process"

    manifest_path_value = config_data.get("manifest")
    manifest_path = Path(manifest_path_value) if manifest_path_value else None
    if not manifest_path or not manifest_path.exists():
        raise RuntimeError(
            "Manifest file from download step was not found. Start download first, then post-processing."
        )

    remote_pipeline_dir = "scratch/CSDP/pipeline"
    remote_config_dir = f"{remote_pipeline_dir}/config"
    remote_manifests_dir = f"{remote_pipeline_dir}/manifests"
    remote_scripts_dir = f"{remote_pipeline_dir}/scripts"
    remote_config_path = f"{remote_config_dir}/process_config.yml"
    remote_job_script_path = f"{remote_scripts_dir}/{local_job_script_path.name}"
    remote_manifest_path = f"{remote_manifests_dir}/{manifest_path.name}"

    # Rewrite local absolute manifest path to the remote cluster path used by the job.
    remote_user = account.split("@", 1)[0]
    remote_home_manifest_path = f"/home/{remote_user}/{remote_manifest_path}"

    config_data["manifest"] = remote_home_manifest_path
    process_config_path = LOCAL_CONFIG_DIR / "process_config.yml"
    with process_config_path.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(config_data, fh, default_flow_style=False)

    repo_root = Path(__file__).resolve().parents[1]  # .../pipeline
    remote_repo_dir = "scratch/CSDP/pipeline"
    cluster_pyproject = repo_root / "pyproject.cluster.toml"
    local_pyproject = cluster_pyproject if cluster_pyproject.exists() else (repo_root / "pyproject.toml")

    ssh_opts = get_ssh_common_options()
    run(
        [
            "ssh",
            *ssh_opts,
            account,
            "mkdir",
            "-p",
            remote_config_dir,
            remote_manifests_dir,
            remote_scripts_dir,
            f"{remote_pipeline_dir}/outputs",
            f"{remote_pipeline_dir}/errors",
        ],
        on_output=lambda text, stream="sync": emit_output(text, stream),
        output_stream="sync",
    )
    run(
        [
            "scp",
            *ssh_opts,
            str(process_config_path),
            f"{account}:{remote_config_path}",
        ],
        on_output=lambda text, stream="sync": emit_output(text, stream),
        output_stream="sync",
    )
    run(
        [
            "scp",
            *ssh_opts,
            str(manifest_path),
            f"{account}:{remote_manifest_path}",
        ],
        on_output=lambda text, stream="sync": emit_output(text, stream),
        output_stream="sync",
    )
    run(
        [
            "scp",
            *ssh_opts,
            str(local_job_script_path),
            f"{account}:{remote_job_script_path}",
        ],
        on_output=lambda text, stream="sync": emit_output(text, stream),
        output_stream="sync",
    )
    run(
        [
            "scp",
            *ssh_opts,
            str(local_pyproject),
            f"{account}:{remote_repo_dir}/pyproject.toml",
        ],
        on_output=lambda text, stream="sync": emit_output(text, stream),
        output_stream="sync",
    )
    run(
        [
            "scp",
            *ssh_opts,
            "-r",
            str(repo_root / "carbonpipeline"),
            f"{account}:{remote_repo_dir}/"
        ],
        on_output=lambda text, stream="sync": emit_output(text, stream),
        output_stream="sync",
    )
    run(
        [
            "ssh",
            *ssh_opts,
            account,
            "chmod",
            "+x",
            remote_job_script_path,
        ],
        on_output=lambda text, stream="sync": emit_output(text, stream),
        output_stream="sync",
    )

    return remote_pipeline_dir, remote_config_path, remote_job_script_path


def submit_postprocessing_job(
    account: str,
    remote_pipeline_dir: str,
    remote_job_script_path: str,
    remote_config_path: str,
    emit_output: Callable[[str, str], None],
):
    sbatch_cmd = (
        f"cd ~/{remote_pipeline_dir} && "
        f"sbatch ~/{remote_job_script_path} ~/{remote_config_path}"
    )
    output = run(
        ["ssh", *get_ssh_common_options(), account, sbatch_cmd],
        on_output=lambda text, stream="sync": emit_output(text, stream),
        output_stream="sync",
    )
    match = re.search(r"Submitted batch job\s+(\d+)", output)
    if not match:
        raise RuntimeError("Could not parse Slurm job ID from sbatch output.")
    job_id = match.group(1)
    emit_output(f"Submitted Slurm job ID: {job_id}\n", "sync")
    return job_id


def _run_remote_command(account: str, command: str) -> str:
    ssh_opts = get_ssh_common_options()
    return run(
        ["ssh", *ssh_opts, account, f"bash -lc {shlex.quote(command)}"],
        on_output=None,
        output_stream="sync",
    )


def _get_job_state(account: str, job_id: str) -> tuple[str | None, str]:
    # Prefer squeue for live status (PENDING/RUNNING), then sacct for terminal accounting info.
    squeue_cmd = f"squeue -h -j {job_id} -o '%T|%M|%l' || true"
    squeue_output = _run_remote_command(account, squeue_cmd).strip()
    if squeue_output:
        state = squeue_output.split("|", 1)[0].strip()
        return state, f"{job_id}|{squeue_output}"

    sacct_cmd = (
        f"sacct -j {job_id} --format=JobID,State,ExitCode,Elapsed -P -n "
        f"| awk -F'|' '$1==\"{job_id}\" {{print; exit}}'"
    )
    sacct_output = _run_remote_command(account, sacct_cmd).strip()
    if sacct_output:
        parts = sacct_output.split("|")
        state = parts[1].strip() if len(parts) > 1 else None
        return state, sacct_output
    return None, ""


def _get_remote_file_size(account: str, remote_path: str) -> int:
    cmd = f"if [ -f ~/{remote_path} ]; then wc -c < ~/{remote_path}; else echo 0; fi"
    output = _run_remote_command(account, cmd).strip()
    try:
        return int(output or "0")
    except ValueError:
        return 0


def _read_remote_file_delta(account: str, remote_path: str, from_byte: int) -> tuple[str, int]:
    size = _get_remote_file_size(account, remote_path)
    if size <= 0:
        return "", 0

    # File may rotate/truncate; restart from beginning if cursor is beyond current size.
    if from_byte >= size:
        from_byte = 0

    if from_byte == size:
        return "", size

    start = from_byte + 1
    cmd = f"if [ -f ~/{remote_path} ]; then tail -c +{start} ~/{remote_path}; fi"
    return _run_remote_command(account, cmd), size


def monitor_postprocessing_job(
    account: str,
    remote_pipeline_dir: str,
    job_id: str,
    emit_output: Callable[[str, str], None],
    poll_interval_seconds: int = 10,
) -> str:
    out_path = f"{remote_pipeline_dir}/slurm-{job_id}.out"
    err_path = f"{remote_pipeline_dir}/slurm-{job_id}.err"
    out_cursor = 0
    err_cursor = 0
    last_status_line = ""
    stable_empty_polls = 0
    poll_count = 0

    while True:
        poll_count += 1
        state, status_line = _get_job_state(account, job_id)

        if status_line and status_line != last_status_line:
            emit_output(f"[sacct] {status_line}\n", "sync")
            last_status_line = status_line
        elif status_line and poll_count % 6 == 0:
            # Keep-alive every ~60s with default 10s polling interval.
            emit_output(f"[status] {status_line}\n", "sync")

        out_delta, out_cursor = _read_remote_file_delta(account, out_path, out_cursor)
        if out_delta:
            emit_output(out_delta, "sync")

        err_delta, err_cursor = _read_remote_file_delta(account, err_path, err_cursor)
        if err_delta:
            emit_output(err_delta, "error")

        normalized_state = (state or "").split("+", 1)[0].strip().upper()
        if normalized_state in TERMINAL_STATES:
            return normalized_state

        # sacct can lag briefly after submit; allow a short grace period when no state is visible.
        if not state:
            stable_empty_polls += 1
            if stable_empty_polls >= 3:
                emit_output(
                    "Job state still unavailable (sacct/squeue). Continuing to poll.\n",
                    "sync",
                )
                stable_empty_polls = 0
        else:
            stable_empty_polls = 0

        time.sleep(poll_interval_seconds)
