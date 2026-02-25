import os
import platform
import shutil
import shlex
import pexpect
import subprocess
import json
import yaml
import datetime
import threading

from typing import Any
from io import StringIO
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
LOCAL_CONFIG_DIR = PROJECT_ROOT / 'config'
LOCAL_UNZIP_DIR = PROJECT_ROOT / 'datasets' / 'unzip'
REMOTE_DIR = 'scratch/CarbonCast/carboncast/datasets/unzip'
CONFIG_FILENAME = 'config.yml'
SSH_CONTROL_PATH = '/tmp/csdp-ssh-%C'
MANIFESTS_PATH = PROJECT_ROOT / 'manifests'
RSYNC_SLEEP_TIME = 30
DOWNLOAD_REGISTRY_PATH = MANIFESTS_PATH / "cluster_download_registry.json"


def format_yml_data(data: Any | None):
    if data is None: data = {}

    def safe_float(x):
        try:
            return float(x)
        except (TypeError, ValueError):
            return None
    
    data_formatted = { }
    data_formatted['action'] = 'download'

    data_formatted['start-date'] = f"{data.get('start_date')} {data.get('start_time')}".strip()
    data_formatted['end-date'] = f"{data.get('end_date')} {data.get('end_time')}".strip()

    data_formatted['ameriflux-predictors'] = data.get('predictors')

    data_formatted['location-coordinates'] = [
        safe_float(data.get('latitude')), 
        safe_float(data.get('longitude')),
    ]
    data_formatted['bbox-coordinates'] = [
        safe_float(data.get('north')),
        safe_float(data.get('west')),
        safe_float(data.get('south')),
        safe_float(data.get('east')),
    ]

    data_formatted['geometries-directory'] = data.get('geojsons')
    data_formatted['data-file'] = data.get('csv')
    
    data_formatted['aggregation-type'] = data.get('aggregation_type')
    data_formatted['id-field'] = data.get('rid')
    data_formatted['manifest'] = str(MANIFESTS_PATH / 'manifest_{date:%Y-%m-%d_%H:%M:%S}.json'.format(date=datetime.datetime.now()))

    return data_formatted


def emit_output(on_output, text: str, stream_type: str):
    if not on_output or not text:
        return

    try:
        on_output(text, stream_type)
    except TypeError:
        on_output(text)


def download_and_sync(credential: str, data: Any | None, on_output=None):
    config_path = LOCAL_CONFIG_DIR / CONFIG_FILENAME
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_payload = format_yml_data(data)

    with open(config_path, "w") as outfile:
        yaml.dump(config_payload, outfile, default_flow_style=False)

    use_ssh_multiplex = shutil.which('rsync') is not None or shutil.which('scp') is not None
    if use_ssh_multiplex:
        start_ssh_master(credential=credential, on_output=on_output)

    create_remote_directories(credential=credential, on_output=on_output)
    try:
        bootstrap_cluster_download_registry(credential=credential, on_output=on_output)
    except Exception as exc:
        emit_output(
            on_output,
            (
                "Warning: failed to build remote download registry. "
                "Falling back to local-only checks.\n"
                f"Reason: {exc}\n"
            ),
            "sync",
        )

    skip_wtd = False
    wtd_time_window = _compute_wtd_time_window(config_payload)
    if wtd_time_window and _remote_wtd_exists(credential=credential, time_window=wtd_time_window, on_output=on_output):
        skip_wtd = True
        emit_output(
            on_output,
            f"WTD data already exists on cluster for {wtd_time_window}; skipping WTD transfer.\n",
            "sync",
        )
        _mark_wtd_present_in_registry(wtd_time_window)

    download_env = os.environ.copy()
    download_env["ERA5DP_DOWNLOAD_CHECK_MODE"] = "hybrid"
    download_env["ERA5DP_DOWNLOAD_REGISTRY_FILE"] = str(DOWNLOAD_REGISTRY_PATH.resolve())
    download_proc = subprocess.Popen(
        [
            "era5dp", "download", "--config",
            str(config_path),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=download_env,
    )

    stop_sync_event = threading.Event()
    sync_failure: dict[str, Exception | None] = {'error': None}

    def _sync_loop():
        while not stop_sync_event.is_set():
            try:
                sync(
                    credential=credential,
                    on_output=on_output,
                    use_ssh_multiplex=use_ssh_multiplex,
                    finalize_wtd_cleanup=False,
                    skip_wtd_sync=skip_wtd,
                )
            except Exception as exc:
                sync_failure['error'] = exc
                stop_sync_event.set()
                return

            # wait with wake-up support to stop quickly when download ends
            stop_sync_event.wait(RSYNC_SLEEP_TIME)

    sync_thread = threading.Thread(target=_sync_loop, daemon=True)
    sync_thread.start()

    try:
        if download_proc.stdout is None:
            raise RuntimeError("Download process has no stdout stream.")

        # stream download logs continuously without being blocked by rsync
        for line in download_proc.stdout:
            if on_output:
                emit_output(on_output, line, 'download')
            else:
                print(line, end="")

            if sync_failure['error'] is not None:
                download_proc.terminate()
                raise sync_failure['error']

        download_proc.wait()
        if download_proc.returncode not in (0, None):
            raise RuntimeError(f"Download command failed (exit code {download_proc.returncode}).")

        if sync_failure['error'] is not None:
            raise sync_failure['error']

        # Stop periodic sync loop before final sync to avoid concurrent rsync overlap.
        stop_sync_event.set()
        sync_thread.join()

        if sync_failure['error'] is not None:
            raise sync_failure['error']

        # final sync once download has completed
        sync(
            credential=credential,
            on_output=on_output,
            use_ssh_multiplex=use_ssh_multiplex,
            finalize_wtd_cleanup=True,
            skip_wtd_sync=skip_wtd,
        )
    finally:
        stop_sync_event.set()
        if sync_thread.is_alive():
            sync_thread.join(timeout=max(RSYNC_SLEEP_TIME, 1))
        if use_ssh_multiplex:
            stop_ssh_master(credential=credential, on_output=on_output)


def sync(
    credential: str,
    on_output=None,
    use_ssh_multiplex: bool = False,
    finalize_wtd_cleanup: bool = False,
    skip_wtd_sync: bool = False,
):
    LOCAL_UNZIP_DIR.mkdir(parents=True, exist_ok=True)

    commands = get_sync_cmds(
        credential,
        use_ssh_multiplex=use_ssh_multiplex,
        finalize_wtd_cleanup=finalize_wtd_cleanup,
        skip_wtd_sync=skip_wtd_sync,
    )

    output = ""
    used_rsync = False
    for cmd in commands:
        output = run(cmd, on_output=on_output, output_stream='sync')
        used_rsync = used_rsync or (cmd and cmd[0] == "rsync")

    if used_rsync:
        _remove_empty_dirs(LOCAL_UNZIP_DIR)

    if finalize_wtd_cleanup:
        _run_remote_co2_nccopy(credential=credential, on_output=on_output)

    emit_output(on_output, 'Transfer done!\n', 'sync')
    return output


def _remove_empty_dirs(root: Path):
    for dirpath, _, _ in os.walk(root, topdown=False):
        path = Path(dirpath)
        try:
            path.rmdir()
        except OSError:
            continue


def _run_remote_co2_nccopy(credential: str, on_output=None):
    remote_cmd = (
        f"set -euo pipefail; "
        f"module --quiet load netcdf-mpi/4.9.2; "
        f"CO2_DIR=~/{REMOTE_DIR}/CO2; "
        f"[ -d \"$CO2_DIR\" ] || exit 0; "
        f"mapfile -t files < <(find \"$CO2_DIR\" -type f -name '*.nc' | sort); "
        f"[ \"${{#files[@]}}\" -gt 0 ] || exit 0; "
        f"[ \"${{#files[@]}}\" -le 1 ] || echo \"Warning: found ${{#files[@]}} CO2 files; using first.\"; "
        f"src=\"${{files[0]}}\"; "
        f"tmp=\"${{src}}.nccopy.tmp\"; "
        f"echo \"Running remote nccopy on $src\"; "
        f"nccopy -k netCDF-4 \"$src\" \"$tmp\"; "
        f"mv -f \"$tmp\" \"$src\""
    )
    run(
        ["ssh", *get_ssh_common_options(), credential, f"bash -lc {shlex.quote(remote_cmd)}"],
        on_output=on_output,
        output_stream="sync",
    )


def _build_rsync_cmd(
    credential: str,
    use_ssh_multiplex: bool,
    remove_source_files: bool,
    filter_args: list[str] | None = None,
) -> list[str]:
    cmd = ["rsync", "-avh", "--info=progress2", "--prune-empty-dirs"]
    if remove_source_files:
        cmd.append("--remove-source-files")
    if filter_args:
        cmd.extend(filter_args)
    cmd.extend([
        f"{str(LOCAL_UNZIP_DIR)}/",
        f"{credential}:~/{REMOTE_DIR}/",
    ])

    if use_ssh_multiplex:
        cmd[1:1] = ["-e", get_rsync_ssh_command()]
    return cmd


def get_sync_cmds(
    credential: str,
    use_ssh_multiplex: bool = False,
    finalize_wtd_cleanup: bool = False,
    skip_wtd_sync: bool = False,
):
    # tools detection
    rsync = shutil.which('rsync')
    scp = shutil.which('scp')

    if rsync:
        commands = [
            # ERA5/CO2 (and anything outside WTD): move continuously.
            _build_rsync_cmd(
                credential=credential,
                use_ssh_multiplex=use_ssh_multiplex,
                remove_source_files=True,
                filter_args=["--exclude=WTD/***"],
            ),
        ]
        if not skip_wtd_sync:
            # WTD: copy-only during ongoing scrape downloads.
            commands.append(
                _build_rsync_cmd(
                    credential=credential,
                    use_ssh_multiplex=use_ssh_multiplex,
                    remove_source_files=finalize_wtd_cleanup,
                    filter_args=["--include=WTD/***", "--exclude=*"],
                )
            )
        return commands
    
    else:
        if not scp:
            system = platform.system()
            raise RuntimeError(
                f"rsync not found and scp missing.\n"
                f"On {system}, install one of:\n"
                f"- WSL (recommended)\n"
                f"- OpenSSH client\n"
            )

        remote_parent = os.path.dirname(REMOTE_DIR)
        cmd = [
            'scp', *get_ssh_common_options(),
            '-r', str(LOCAL_UNZIP_DIR),
            f"{credential}:~/{remote_parent}/"
        ]
        if use_ssh_multiplex:
            cmd[1:1] = ['-e', get_rsync_ssh_command()]
        return [cmd]


def get_ssh_common_options():
    return [
        '-o', 'ControlMaster=auto',
        '-o', 'ControlPersist=10m',
        '-o', f'ControlPath={SSH_CONTROL_PATH}',
    ]


def get_rsync_ssh_command():
    common = get_ssh_common_options()
    return 'ssh ' + ' '.join(common)


def create_remote_directories(credential: str, on_output=None):
    ssh = shutil.which('ssh')
    if not ssh:
        raise RuntimeError("ssh is required for rsync transfer but was not found.")
    
    cmd = [
        "ssh", *get_ssh_common_options(),
        credential, "mkdir", "-p",
        f"~/{REMOTE_DIR}/"
    ]
    run(cmd, on_output=on_output)


def bootstrap_cluster_download_registry(credential: str, on_output=None):
    emit_output(on_output, "Building remote download registry from cluster files...\n", "sync")

    find_command = (
        f"if [ -d ~/{REMOTE_DIR} ]; then "
        f"find ~/{REMOTE_DIR} -type f \\( -name '*.nc' -o -name '*.tif' \\); "
        "fi"
    )
    output = run(
        ["ssh", *get_ssh_common_options(), credential, find_command],
        on_output=on_output,
        output_stream="sync",
    )

    local_unzip = LOCAL_UNZIP_DIR.resolve()
    entries: dict[str, dict[str, bool]] = {}
    remote_file_count = 0
    token = "/datasets/unzip/"

    for line in output.splitlines():
        normalized = line.strip()
        if token not in normalized or not normalized.startswith("/"):
            continue

        remote_file_count += 1
        relative_file = normalized.split(token, 1)[1].strip()
        local_file = (local_unzip / relative_file).resolve()
        local_dir = str(local_file.parent)

        entry = entries.setdefault(local_dir, {"any": True, "nc": False, "tif": False})
        suffix = local_file.suffix.lower()
        if suffix == ".nc":
            entry["nc"] = True
        elif suffix == ".tif":
            entry["tif"] = True

    payload = {
        "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "source": "cluster_scan",
        "remote_dir": REMOTE_DIR,
        "entries": entries,
    }
    DOWNLOAD_REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
    DOWNLOAD_REGISTRY_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    emit_output(
        on_output,
        (
            f"Registry ready: {len(entries)} folders indexed from "
            f"{remote_file_count} remote files.\n"
        ),
        "sync",
    )


def _has_wtd_predictor(config_payload: dict[str, Any]) -> bool:
    predictors = config_payload.get("ameriflux-predictors")
    if isinstance(predictors, str):
        predictor_values = [p.strip() for p in predictors.split(",") if p.strip()]
    elif isinstance(predictors, list):
        predictor_values = [str(p).strip() for p in predictors if str(p).strip()]
    else:
        predictor_values = []
    return any(p.upper() == "WTD" for p in predictor_values)


def _compute_wtd_time_window(config_payload: dict[str, Any]) -> str | None:
    if not _has_wtd_predictor(config_payload):
        return None

    start = str(config_payload.get("start-date") or "").strip()
    end = str(config_payload.get("end-date") or "").strip()
    if not start or not end:
        return None

    try:
        start_dt = datetime.datetime.fromisoformat(start)
        end_dt = datetime.datetime.fromisoformat(end)
    except ValueError:
        return None

    return f"{start_dt:%Y-%m}_{end_dt:%Y-%m}"


def _remote_wtd_exists(credential: str, time_window: str, on_output=None) -> bool:
    cmd = (
        f"if [ -d ~/{REMOTE_DIR}/WTD/{time_window} ] && "
        f"find ~/{REMOTE_DIR}/WTD/{time_window} -type f -name '*.tif' -print -quit | grep -q .; "
        "then echo 1; else echo 0; fi"
    )
    output = run(
        ["ssh", *get_ssh_common_options(), credential, cmd],
        on_output=on_output,
        output_stream="sync",
    )
    return output.strip().endswith("1")


def _mark_wtd_present_in_registry(time_window: str):
    entry_path = (LOCAL_UNZIP_DIR / "WTD" / time_window).resolve()
    payload: dict[str, Any] = {"entries": {}}
    if DOWNLOAD_REGISTRY_PATH.exists():
        try:
            payload = json.loads(DOWNLOAD_REGISTRY_PATH.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            payload = {"entries": {}}

    entries = payload.setdefault("entries", {})
    entries[str(entry_path)] = {"any": True, "nc": False, "tif": True}
    payload["generated_at"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    DOWNLOAD_REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
    DOWNLOAD_REGISTRY_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def start_ssh_master(credential: str, on_output=None):
    ssh = shutil.which('ssh')
    if not ssh:
        raise RuntimeError("ssh is required for rsync transfer but was not found.")

    cmd = [
        ssh, '-fN',
        *get_ssh_common_options(),
        credential,
    ]
    run(cmd, on_output=on_output, output_stream='sync')


def stop_ssh_master(credential: str, on_output=None):
    ssh = shutil.which('ssh')
    if not ssh:
        return

    cmd = [
        ssh, '-O', 'exit',
        '-o', f'ControlPath={SSH_CONTROL_PATH}',
        credential,
    ]
    try:
        run(cmd, on_output=on_output, output_stream='sync')
    except RuntimeError:
        # connection may already be closed; this should not fail the whole flow
        pass


class OutputTee:
    def __init__(self, buffer, on_output=None, output_stream='download'):
        self.buffer = buffer
        self.on_output = on_output
        self.output_stream = output_stream

    def write(self, data):
        self.buffer.write(data)
        if self.on_output:
            emit_output(self.on_output, data, self.output_stream)

    def flush(self):
        self.buffer.flush()


def run(cmd, env=None, on_output=None, output_stream='download'):
    print(">>", " ".join(cmd))
    emit_output(on_output, ">> " + " ".join(cmd) + "\n", output_stream)

    output_buffer = StringIO()
    child = pexpect.spawn(cmd[0], cmd[1:], env=env, encoding='utf-8')
    child.logfile_read = OutputTee(output_buffer, on_output=on_output, output_stream=output_stream)

    while True:
        try:
            idx = child.expect(['Passcode or option', pexpect.EOF], timeout=120)
        except pexpect.TIMEOUT:
            # Long-running transfers (e.g., rsync with progress output) can exceed the
            # expect timeout without requiring interaction; keep waiting until EOF.
            if not child.isalive():
                break
            continue
        if idx == 0:
            child.sendline('1')
            continue
        break

    child.close()
    output = output_buffer.getvalue()
    if output:
        print(output)

    if child.signalstatus is not None:
        raise RuntimeError(
            f"Transfer command terminated by signal {child.signalstatus}.\n"
            f"Command: {' '.join(cmd)}\n"
            f"Output:\n{output.strip()}"
        )

    if child.exitstatus not in (0, None):
        raise RuntimeError(
            f"Transfer command failed (exit code {child.exitstatus}).\n"
            f"Command: {' '.join(cmd)}\n"
            f"Output:\n{output.strip()}"
        )

    return output
