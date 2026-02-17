import os
import platform
import shutil
import pexpect
import subprocess
import yaml
import time
import datetime
import threading

from typing import Any
from io import StringIO
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOCAL_CONFIG_DIR = PROJECT_ROOT / 'config'
LOCAL_UNZIP_DIR = PROJECT_ROOT / 'datasets' / 'unzip'
REMOTE_DIR = 'scratch/CSDP/pipeline/datasets/unzip'
CONFIG_FILENAME = 'config.yml'
SSH_CONTROL_PATH = '/tmp/csdp-ssh-%C'
MANIFESTS_PATH = PROJECT_ROOT / 'manifests'
RSYNC_SLEEP_TIME = 30


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
        # Backward compatibility for callbacks expecting only one argument.
        on_output(text)


def download_and_sync(credential: str, data: Any | None, on_output=None):
    config_path = LOCAL_CONFIG_DIR / CONFIG_FILENAME
    config_path.parent.mkdir(parents=True, exist_ok=True)

    with open(config_path, "w") as outfile:
        yaml.dump(format_yml_data(data), outfile, default_flow_style=False)

    download_proc = subprocess.Popen(
        [
            "carbonpipeline", "download", "--config",
            str(config_path),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    use_ssh_multiplex = shutil.which('rsync') is not None or shutil.which('scp') is not None
    if use_ssh_multiplex:
        start_ssh_master(credential=credential, on_output=on_output)

    create_remote_directories(credential=credential, on_output=on_output)

    stop_sync_event = threading.Event()
    sync_failure: dict[str, Exception | None] = {'error': None}

    def _sync_loop():
        while not stop_sync_event.is_set():
            try:
                sync(
                    credential=credential,
                    on_output=on_output,
                    use_ssh_multiplex=use_ssh_multiplex,
                )
            except Exception as exc:
                sync_failure['error'] = exc
                stop_sync_event.set()
                return

            # Wait with wake-up support to stop quickly when download ends.
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

        # final sync once download has completed
        sync(
            credential=credential,
            on_output=on_output,
            use_ssh_multiplex=use_ssh_multiplex,
        )
    finally:
        stop_sync_event.set()
        sync_thread.join(timeout=max(RSYNC_SLEEP_TIME, 1))
        if use_ssh_multiplex:
            stop_ssh_master(credential=credential, on_output=on_output)


def sync(credential: str, on_output=None, use_ssh_multiplex: bool = False):
    LOCAL_UNZIP_DIR.mkdir(parents=True, exist_ok=True)

    cmd = get_sync_cmd(credential, use_ssh_multiplex=use_ssh_multiplex)
    output = run(cmd, on_output=on_output, output_stream='sync')

    emit_output(on_output, 'Transfer done!\n', 'sync')
    return output


def get_sync_cmd(credential: str, use_ssh_multiplex: bool = False):
    # tools detection
    rsync = shutil.which('rsync')
    scp = shutil.which('scp')

    if rsync:
        cmd = [
            'rsync', '-avh', '--info=progress2',
            f"{str(LOCAL_UNZIP_DIR)}/",
            f"{credential}:~/{REMOTE_DIR}/"
        ]
        if use_ssh_multiplex:
            cmd[1:1] = ['-e', get_rsync_ssh_command()]

        return cmd
    
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
        return cmd


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
        idx = child.expect(['Passcode or option', pexpect.EOF], timeout=120)
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
