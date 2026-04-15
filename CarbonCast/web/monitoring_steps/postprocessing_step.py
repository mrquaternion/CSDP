import re
import shlex
import textwrap
import time
import json
import shutil
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path
from typing import Any, Callable

import yaml

from .gas_flux_pred_step import gas_flux_pred_job_script
from .download_and_sync import (
    CONFIG_FILENAME,
    LOCAL_CONFIG_DIR,
    get_rsync_ssh_command,
    get_ssh_common_options,
    run,
)

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

ECO_REPO_URL = "https://github.com/mjfortier/EcoPerceiver.git"
ECO_REPO_BRANCH = "era5-tweak"
ECO_REPO_DIRNAME = "EcoPerceiver"
ECO_REPO_PARENT_DIR = "scratch/CarbonCast"
ECO_REPO_DIR = f"{ECO_REPO_PARENT_DIR}/{ECO_REPO_DIRNAME}"
PIPELINE_REMOTE_DIR = "scratch/CarbonCast/carboncast"
ZENODO_RECORD_URL = "https://zenodo.org/records/18704871"
LAND_SEA_MASK: list[tuple[str, str]] = [
    (
        "https://confluence.ecmwf.int/download/attachments/140385202/lsm_1279l4_0.1x0.1.grb_v4_unpack.nc?version=1&modificationDate=1591983422208&api=v2",
        "lsm.nc",
    ),
]

def _validate_job_resources(memory: str, cpus_raw: str, wall_time: str) -> tuple[str, int, str]:
    memory = str(memory).strip().upper()
    cpus_raw = str(cpus_raw).strip()
    wall_time = str(wall_time).strip()

    if not re.match(r"^\d+[KMGTP]?$", memory):
        raise ValueError("Invalid memory format. Use values like 8G, 32G, or 16000M.")

    if not cpus_raw.isdigit() or int(cpus_raw) <= 0:
        raise ValueError("CPUs per task must be a positive integer.")
    cpus = int(cpus_raw)

    time_pattern = r"^(\d+-)?\d{1,2}:\d{2}(:\d{2})?$"
    if not re.match(time_pattern, wall_time):
        raise ValueError("Invalid time format. Use HH:MM, HH:MM:SS, or D-HH:MM:SS.")

    return memory, cpus, wall_time


def validate_post_processing_payload(payload: dict[str, Any] | None):
    memory = str((payload or {}).get("memory", "")).strip().upper()
    cpus_raw = str((payload or {}).get("cpus", "")).strip()
    wall_time = str((payload or {}).get("time", "")).strip()
    slurm_account = str((payload or {}).get("slurm_account", "")).strip()

    memory, cpus, wall_time = _validate_job_resources(memory, cpus_raw, wall_time)

    validate_slurm_account(slurm_account)

    return memory, cpus, wall_time, slurm_account


def validate_slurm_account(slurm_account: str):
    slurm_account = str(slurm_account or "").strip()
    if not slurm_account:
        raise ValueError("Slurm account is required.")
    if not re.match(r"^[A-Za-z0-9._-]+$", slurm_account):
        raise ValueError("Invalid Slurm account format.")
    return slurm_account


def validate_optional_job_resources(
    payload: dict[str, Any] | None,
    fallback_memory: str,
    fallback_cpus: int,
    fallback_wall_time: str,
) -> tuple[str, int, str]:
    cfg = payload or {}
    if not cfg:
        return fallback_memory, fallback_cpus, fallback_wall_time

    memory = str(cfg.get("memory", "")).strip() or fallback_memory
    cpus_raw = str(cfg.get("cpus", "")).strip() or str(fallback_cpus)
    wall_time = str(cfg.get("time", "")).strip() or fallback_wall_time
    return _validate_job_resources(memory, cpus_raw, wall_time)


def validate_optional_gas_flux_job_resources(
    payload: dict[str, Any] | None,
    fallback_memory: str,
    fallback_cpus: int,
    fallback_wall_time: str,
    fallback_gpus: int,
) -> tuple[str, int, str, int]:
    memory, cpus, wall_time = validate_optional_job_resources(
        payload=payload,
        fallback_memory=fallback_memory,
        fallback_cpus=fallback_cpus,
        fallback_wall_time=fallback_wall_time,
    )

    cfg = payload or {}
    gpus_raw = str(cfg.get("gpus", "")).strip() or str(fallback_gpus)
    if not gpus_raw.isdigit() or int(gpus_raw) <= 0:
        raise ValueError("GPUs must be a positive integer.")

    return memory, cpus, wall_time, int(gpus_raw)


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
            module load proj/9.2.0 python/3.12 

            VENV_DIR="$SLURM_TMPDIR/ccenv"

            printf "\\nCreating the environment at %s.\\n" "$VENV_DIR"
            virtualenv --no-download "$VENV_DIR"
            source "$VENV_DIR/bin/activate"
            pip install --upgrade pip --no-index

            REPO_DIR="$HOME/scratch/CarbonCast/carboncast"
            cd "$REPO_DIR"

            printf "\\nInstalling CarbonCast dependencies.\\n"
            pip install -e . --no-index

            CONFIG_PATH="${{1:-$REPO_DIR/config/process_config.yml}}"

            echo "Starting pipeline with config: $CONFIG_PATH"
            era5dp process --config "$CONFIG_PATH"
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


def write_generated_gas_flux_job_script(
    slurm_account: str,
    memory: str,
    cpus: int,
    wall_time: str,
    gpus: int,
    output_path: Path | None = None,
) -> Path:
    job_script_path = output_path or (LOCAL_CONFIG_DIR / "gas_flux_pred.generated.sh")
    job_script_path.parent.mkdir(parents=True, exist_ok=True)
    job_script_path.write_text(
        gas_flux_pred_job_script(
            slurm_account=slurm_account,
            memory=memory,
            cpus=cpus,
            wall_time=wall_time,
            gpus=gpus,
        ),
        encoding="utf-8",
    )
    job_script_path.chmod(0o755)
    return job_script_path.resolve()


def prepare_remote_postprocessing_assets(
    account: str,
    configuration_data: Any | None,
    local_job_script_path: Path,
    include_gas_flux_repo: bool,
    emit_output: Callable[[str, str], None],
):
    del configuration_data  # keep process config based only on normalized download config
    config_path = LOCAL_CONFIG_DIR / CONFIG_FILENAME
    if not config_path.exists():
        raise RuntimeError(f"Download config file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as fh:
        config_data = yaml.safe_load(fh) or {}

    aggregation_type = config_data.get("aggregation-type")
    config_data["action"] = "process"
    config_data["geometries-directory"] = None
    config_data["delete-source-after-aggregation"] = (
        isinstance(aggregation_type, str)
        and aggregation_type.strip().lower() not in {"", "none"}
    )

    manifest_path_value = config_data.get("manifest")
    manifest_path = Path(manifest_path_value) if manifest_path_value else None
    if not manifest_path or not manifest_path.exists():
        raise RuntimeError(
            "Manifest file from download step was not found. Start download first, then post-processing."
        )

    remote_pipeline_dir = PIPELINE_REMOTE_DIR
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

    repo_root = Path(__file__).resolve().parents[2]  # .../carboncast
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
            f"{account}:{remote_pipeline_dir}/pyproject.toml",
        ],
        on_output=lambda text, stream="sync": emit_output(text, stream),
        output_stream="sync",
    )
    run(
        [
            "scp",
            *ssh_opts,
            "-r",
            str(repo_root / "era5dp"),
            f"{account}:{remote_pipeline_dir}/"
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
    if include_gas_flux_repo:
        ensure_ecoperceiver_repo(
            account=account,
            remote_pipeline_dir=remote_pipeline_dir,
            emit_output=emit_output,
        )

    return remote_pipeline_dir, remote_config_path, remote_job_script_path


def prepare_remote_gas_flux_assets(
    account: str,
    local_job_script_path: Path,
    emit_output: Callable[[str, str], None],
) -> tuple[str, str]:
    remote_scripts_dir = f"{ECO_REPO_PARENT_DIR}/scripts"
    remote_job_script_path = f"{remote_scripts_dir}/{local_job_script_path.name}"

    run(
        ["ssh", *get_ssh_common_options(), account, "mkdir", "-p", remote_scripts_dir],
        on_output=lambda text, stream="sync": emit_output(text, stream),
        output_stream="sync",
    )
    run(
        [
            "scp",
            *get_ssh_common_options(),
            str(local_job_script_path),
            f"{account}:{remote_job_script_path}",
        ],
        on_output=lambda text, stream="sync": emit_output(text, stream),
        output_stream="sync",
    )
    run(
        ["ssh", *get_ssh_common_options(), account, "chmod", "+x", remote_job_script_path],
        on_output=lambda text, stream="sync": emit_output(text, stream),
        output_stream="sync",
    )

    ensure_ecoperceiver_repo(
        account=account,
        remote_pipeline_dir="",
        emit_output=emit_output,
    )

    local_resnet_dir, local_runs_dir, local_era5_data_dir = download_gas_flux_assets_locally(emit_output=emit_output)
    sync_gas_flux_assets_to_cluster(
        account=account,
        local_resnet_dir=local_resnet_dir,
        local_runs_dir=local_runs_dir,
        local_era5_data_dir=local_era5_data_dir,
        emit_output=emit_output,
    )
    sync_processed_outputs_to_cluster(
        account=account,
        emit_output=emit_output,
    )

    return ECO_REPO_DIR, remote_job_script_path


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


def submit_gas_flux_job(
    account: str,
    remote_repo_dir: str,
    remote_job_script_path: str,
    emit_output: Callable[[str, str], None],
) -> str:
    sbatch_cmd = (
        f"cd ~/{remote_repo_dir} && "
        f"sbatch ~/{remote_job_script_path}"
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


def ensure_ecoperceiver_repo(
    account: str,
    remote_pipeline_dir: str,
    emit_output: Callable[[str, str], None],
):
    del remote_pipeline_dir
    cmd = (
        f"set -euo pipefail; "
        f"mkdir -p ~/{ECO_REPO_PARENT_DIR}; "
        f"if [ ! -d ~/{ECO_REPO_DIR}/.git ]; then "
        f"git clone --branch {ECO_REPO_BRANCH} --single-branch {ECO_REPO_URL} "
        f"~/{ECO_REPO_DIR}; "
        "else "
        f"git -C ~/{ECO_REPO_DIR} fetch origin {ECO_REPO_BRANCH}; "
        f"git -C ~/{ECO_REPO_DIR} checkout {ECO_REPO_BRANCH}; "
        f"git -C ~/{ECO_REPO_DIR} pull --ff-only origin {ECO_REPO_BRANCH}; "
        "fi"
    )
    run(
        ["ssh", *get_ssh_common_options(), account, f"bash -lc {shlex.quote(cmd)}"],
        on_output=lambda text, stream="sync": emit_output(text, stream),
        output_stream="sync",
    )


def download_gas_flux_assets_locally(
    emit_output: Callable[[str, str], None],
) -> tuple[Path, Path, Path]:
    emit_output("Downloading gas-flux assets locally from Zenodo...\n", "sync")
    base_dir = LOCAL_CONFIG_DIR / "gas_flux_assets"
    raw_dir = base_dir / "raw"
    resnet_dir = base_dir / "resnet"
    runs_dir = base_dir / "runs"
    era5_data_dir = base_dir / "era5_data"

    if base_dir.exists():
        shutil.rmtree(base_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    resnet_dir.mkdir(parents=True, exist_ok=True)
    runs_dir.mkdir(parents=True, exist_ok=True)
    era5_data_dir.mkdir(parents=True, exist_ok=True)

    parsed = urllib.parse.urlparse(ZENODO_RECORD_URL)
    parts = [p for p in parsed.path.split("/") if p]
    if len(parts) < 2 or parts[0] != "records":
        raise RuntimeError(f"Invalid Zenodo record URL: {ZENODO_RECORD_URL}")
    record_id = parts[1]

    api_url = f"https://zenodo.org/api/records/{record_id}"
    with urllib.request.urlopen(api_url) as response:
        metadata = json.loads(response.read().decode("utf-8"))
    files = metadata.get("files") or []
    if not files:
        raise RuntimeError("No files found in the Zenodo record.")

    for file_obj in files:
        name = file_obj.get("key") or file_obj.get("filename")
        link = (file_obj.get("links") or {}).get("self")
        if not name or not link:
            continue
        destination = raw_dir / name
        emit_output(f"Downloading {name}\n", "sync")
        with urllib.request.urlopen(link) as src, destination.open("wb") as dst:
            shutil.copyfileobj(src, dst)

    for path in raw_dir.iterdir():
        if path.suffix.lower() == ".zip":
            emit_output(f"Extracting {path.name} into runs/\n", "sync")
            with zipfile.ZipFile(path, "r") as zf:
                zf.extractall(runs_dir)
            continue

        if path.suffix.lower() == ".pth" and "resnet" in path.name.lower():
            shutil.copy2(path, resnet_dir / path.name)
            continue

        shutil.copy2(path, runs_dir / path.name)

    emit_output("Downloading land-sea mask assets locally...\n", "sync")
    for url, target_name in LAND_SEA_MASK:
        destination = era5_data_dir / target_name
        emit_output(f"Downloading {target_name}\n", "sync")
        with urllib.request.urlopen(url) as src, destination.open("wb") as dst:
            shutil.copyfileobj(src, dst)

    return resnet_dir, runs_dir, era5_data_dir


def sync_gas_flux_assets_to_cluster(
    account: str,
    local_resnet_dir: Path,
    local_runs_dir: Path,
    local_era5_data_dir: Path,
    emit_output: Callable[[str, str], None],
):
    rsync = shutil.which("rsync")
    if not rsync:
        raise RuntimeError("rsync is required to upload gas-flux assets to cluster.")

    remote_resnet_dir = f"{ECO_REPO_DIR}/ecoperceiver"
    remote_runs_dir = f"{ECO_REPO_DIR}/experiments/runs"
    remote_era5_data_dir = f"{ECO_REPO_DIR}/experiments/data"
    run(
        [
            "ssh",
            *get_ssh_common_options(),
            account,
            "mkdir",
            "-p",
            f"~/{remote_resnet_dir}",
            f"~/{remote_runs_dir}",
            f"~/{remote_era5_data_dir}",
        ],
        on_output=lambda text, stream="sync": emit_output(text, stream),
        output_stream="sync",
    )

    run(
        [
            rsync,
            "-avh",
            "-e",
            get_rsync_ssh_command(),
            f"{str(local_resnet_dir)}/",
            f"{account}:~/{remote_resnet_dir}/",
        ],
        on_output=lambda text, stream="sync": emit_output(text, stream),
        output_stream="sync",
    )
    run(
        [
            rsync,
            "-avh",
            "-e",
            get_rsync_ssh_command(),
            f"{str(local_runs_dir)}/",
            f"{account}:~/{remote_runs_dir}/",
        ],
        on_output=lambda text, stream="sync": emit_output(text, stream),
        output_stream="sync",
    )
    run(
        [
            rsync,
            "-avh",
            "-e",
            get_rsync_ssh_command(),
            f"{str(local_era5_data_dir)}/",
            f"{account}:~/{remote_era5_data_dir}/",
        ],
        on_output=lambda text, stream="sync": emit_output(text, stream),
        output_stream="sync",
    )


def sync_processed_outputs_to_cluster(
    account: str,
    emit_output: Callable[[str, str], None],
):
    remote_outputs_dir = f"{PIPELINE_REMOTE_DIR}/outputs"
    remote_process_config = f"{PIPELINE_REMOTE_DIR}/config/process_config.yml"
    remote_data_dir = f"{ECO_REPO_DIR}/experiments/data/era5_data"
    remote_cmd = f"""
set -euo pipefail
python3 - <<'PY'
import json
import shutil
from pathlib import Path

home = Path.home()
outputs_dir = home / "{remote_outputs_dir}"
process_cfg = home / "{remote_process_config}"
target_dir = home / "{remote_data_dir}"

if not outputs_dir.is_dir():
    raise RuntimeError(f"Missing source directory: {{outputs_dir}}")
if not process_cfg.exists():
    raise RuntimeError(f"Missing process config: {{process_cfg}}")

manifest_path = None
for raw_line in process_cfg.read_text(encoding="utf-8").splitlines():
    line = raw_line.strip()
    if line.startswith("manifest:"):
        manifest_path = line.split(":", 1)[1].strip().strip("'\"")
        break

if not manifest_path:
    raise RuntimeError("Could not find 'manifest' in process_config.yml")

manifest = Path(manifest_path).expanduser()
if not manifest.exists():
    raise RuntimeError(f"Manifest does not exist: {{manifest}}")

payload = json.loads(manifest.read_text(encoding="utf-8"))
features = payload.get("features") or []
region_ids = [str(f.get("region_id", "")).strip() for f in features if f.get("region_id")]
if not region_ids:
    raise RuntimeError("No region_id values found in manifest features.")

target_dir.mkdir(parents=True, exist_ok=True)
copied = []
for file_path in outputs_dir.iterdir():
    if not file_path.is_file():
        continue
    name = file_path.name
    if any(rid in name for rid in region_ids):
        destination = target_dir / name
        shutil.copy2(file_path, destination)
        copied.append(name)

if not copied:
    raise RuntimeError(
        f"No output files matched manifest region IDs in {{outputs_dir}}. "
        f"Region IDs: {{region_ids}}"
    )

print(f"Copied {{len(copied)}} output files to {{target_dir}}")
for name in copied:
    print(name)
PY
"""
    run(
        [
            "ssh",
            *get_ssh_common_options(),
            account,
            f"bash -lc {shlex.quote(remote_cmd)}",
        ],
        on_output=lambda text, stream="sync": emit_output(text, stream),
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


def _list_remote_files_created_after(
    account: str,
    remote_base_dir: str,
    created_after: str,
    suffixes: tuple[str, ...],
) -> list[str]:
    suffixes_payload = json.dumps([suffix.lower() for suffix in suffixes])
    remote_cmd = textwrap.dedent(
        f"""\
        python3 - <<'PY'
        import json
        from datetime import datetime, timezone
        from pathlib import Path

        base = Path.home() / {json.dumps(remote_base_dir)}
        cutoff = datetime.fromisoformat({json.dumps(created_after)})
        if cutoff.tzinfo is None:
            cutoff = cutoff.replace(tzinfo=timezone.utc)
        else:
            cutoff = cutoff.astimezone(timezone.utc)

        suffixes = tuple(json.loads({json.dumps(suffixes_payload)}))

        if base.is_dir():
            for path in sorted(base.rglob("*")):
                if not path.is_file():
                    continue
                if suffixes and path.suffix.lower() not in suffixes:
                    continue
                mtime = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
                if mtime >= cutoff:
                    print(path.relative_to(base))
        PY
        """
    )
    output = _run_remote_command(account, remote_cmd)
    return [line.strip() for line in output.splitlines() if line.strip()]


def _rsync_remote_files(
    account: str,
    remote_base_dir: str,
    relative_paths: list[str],
    local_out_dir: Path,
    emit_output: Callable[[str, str], None],
) -> str:
    local_out_dir.mkdir(parents=True, exist_ok=True)

    if not relative_paths:
        raise RuntimeError("No remote files matched this job.")

    emit_output(f"Pulling {len(relative_paths)} remote files...\n", "sync")
    for rel_path in relative_paths:
        destination_dir = (local_out_dir / Path(rel_path).parent).resolve()
        destination_dir.mkdir(parents=True, exist_ok=True)
        run(
            [
                "rsync",
                "-e",
                get_rsync_ssh_command(),
                "-avh",
                "--info=progress2",
                f"{account}:~/{remote_base_dir}/{rel_path}",
                str(destination_dir),
            ],
            on_output=lambda text, stream="sync": emit_output(text, stream),
            output_stream="sync",
        )

    return str(local_out_dir.resolve())


def fetch_remote_postprocessing_outputs_for_job(
    account: str,
    remote_pipeline_dir: str,
    created_after: str,
    emit_output: Callable[[str, str], None],
) -> str:
    remote_outputs_dir = f"{remote_pipeline_dir}/outputs"
    relative_paths = _list_remote_files_created_after(
        account=account,
        remote_base_dir=remote_outputs_dir,
        created_after=created_after,
        suffixes=(".nc",),
    )
    return _rsync_remote_files(
        account=account,
        remote_base_dir=remote_outputs_dir,
        relative_paths=relative_paths,
        local_out_dir=LOCAL_CONFIG_DIR.parent / "outputs",
        emit_output=emit_output,
    )


def fetch_remote_gas_flux_csvs_for_job(
    account: str,
    remote_repo_dir: str,
    created_after: str,
    emit_output: Callable[[str, str], None],
) -> str:
    relative_paths = _list_remote_files_created_after(
        account=account,
        remote_base_dir=remote_repo_dir,
        created_after=created_after,
        suffixes=(".csv",),
    )
    return _rsync_remote_files(
        account=account,
        remote_base_dir=remote_repo_dir,
        relative_paths=relative_paths,
        local_out_dir=LOCAL_CONFIG_DIR.parent / "gas_flux_outputs",
        emit_output=emit_output,
    )
