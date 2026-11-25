"""Gracefully stop only a supervisor belonging to this project."""

import json
import os
from pathlib import Path
import signal
from .scene import ROOT


def main():
    active = ROOT / 'runs/active.json'
    if not active.exists():
        print('No active project run.')
        return
    data = json.loads(active.read_text())
    pid = int(data['supervisor'])
    proc = Path('/proc') / str(pid)
    if not proc.exists():
        print('Supervisor has already exited; no signal sent.')
        return
    argv = (proc / 'cmdline').read_bytes().split(b'\0')
    if b'urban_drone.run' not in argv or (proc / 'cwd').resolve() != ROOT:
        raise SystemExit('PID ownership check failed; no signal sent.')
    os.kill(pid, signal.SIGINT)
    print(f'Landing/cleanup requested for run {data["directory"]}.')


if __name__ == '__main__':
    main()
