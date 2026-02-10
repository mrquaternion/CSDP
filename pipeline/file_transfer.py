import os
import platform
import shutil
import pexpect
from io import StringIO

from pathlib import Path


LOCAL_DIR = Path(__file__).parent / 'datasets' / 'unzip'
REMOTE_DIR = 'scratch'


def run(cmd, env=None):
    print(">>", " ".join(cmd))

    output_buffer = StringIO()
    child = pexpect.spawn(cmd[0], cmd[1:], env=env, encoding='utf-8')
    child.logfile_read = output_buffer

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


def file_transfer(credential: str):
    if not LOCAL_DIR.exists():
        raise FileNotFoundError(f"LOCAL_DIR does nto exist: {LOCAL_DIR}")

    # tools detection
    rsync = shutil.which('rsync')
    scp = shutil.which('scp')

    if rsync:
        cmd = [
            'rsync', '-avh', '--info=progress2',
            f"{str(LOCAL_DIR)}/",
            f"{credential}:~/{REMOTE_DIR}/"
        ]

        output = run(cmd)

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
            'scp', '-r',
            str(LOCAL_DIR),
            f"{credential}:~/{remote_parent}/"
        ]

        output = run(cmd)

    print('Transfer done!')
    return output
