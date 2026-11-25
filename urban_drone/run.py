"""Own only this run's processes; preserve all logs and failed experiments."""

import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import signal
import shutil
import subprocess
import sys
import time
from zoneinfo import ZoneInfo

from .scene import ROOT, generate, parameter_overrides
from .camera import configure_camera


def running_px4():
    result = subprocess.run(['pgrep', '-x', 'px4'], capture_output=True, text=True)
    return result.stdout.strip()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--case', default='wall_low')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--jitter', type=float, default=0)
    p.add_argument('--height', type=float)
    p.add_argument('--opening-width', type=float)
    p.add_argument('--timeout', type=float, help='Mission seconds; defaults to 180, or the scene public budget')
    p.add_argument('--baseline', choices=['full', 'side_only', 'global_only'], default='full')
    p.add_argument('--headless', action='store_true')
    p.add_argument('--keep-open', action='store_true', help='Keep the landed Gazebo view until its window closes')
    p.add_argument('--rviz', action='store_true')
    p.add_argument('--camera', choices=['follow', 'view', 'free'], default='follow',
                   help='follow: close rear view (default); view: rear-right view; free: manual; Esc releases tracking')
    p.add_argument('--prepare-only', action='store_true')
    p.add_argument('--px4', type=Path, default=Path.home() / 'PX4-Autopilot')
    args = p.parse_args()
    out = ROOT / 'runs' / (datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%d_%H%M%S_%f') + f'_{args.case}_{args.seed}')
    prior, truth = generate(out, args.case, args.seed, args.jitter,
                            parameter_overrides(args.case, args.height, args.opening_width))
    if args.timeout is None:
        args.timeout = prior.get('mission_timeout', 180)
    snapshot = out / 'source'
    for folder in ('urban_drone', 'config', 'models'):
        shutil.copytree(ROOT / folder, snapshot / folder,
                        ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    sources = [*(snapshot / 'urban_drone').glob('*.py'), *(snapshot / 'config').glob('*'),
               *(snapshot / 'models').rglob('*.sdf'), *(snapshot / 'models').rglob('*.config')]
    manifest = dict(arguments=vars(args).copy(), created=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
                    source_sha256={str(f.relative_to(snapshot)): hashlib.sha256(f.read_bytes()).hexdigest()
                                   for f in sources})
    manifest['arguments']['px4'] = str(args.px4)
    (out / 'run_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(f'Run directory: {out}', flush=True)
    if args.prepare_only:
        return
    if running_px4():
        raise SystemExit('A PX4 instance is already running. Stop the completed old SITL first; no process was killed.')
    build = args.px4 / 'build/px4_sitl_default'
    binary, rootfs = build / 'bin/px4', build / 'rootfs'
    if not binary.exists():
        raise SystemExit(f'PX4 binary missing: {binary}')
    env = os.environ.copy()
    env.update(GZ_IP='127.0.0.1', GZ_PARTITION='urban_delivery',
               GZ_SIM_RESOURCE_PATH=f'{snapshot}/models:{args.px4}/Tools/simulation/gz/models:' + env.get('GZ_SIM_RESOURCE_PATH', ''),
               GZ_SIM_SYSTEM_PLUGIN_PATH=str(build / 'src/modules/simulation/gz_plugins'),
               PX4_SIM_MODEL='gz_x500', PX4_GZ_STANDALONE='1', PX4_GZ_WORLD='urban_delivery',
               PX4_GZ_MODEL_NAME='urban_x500_0', PYTHONNOUSERSITE='1', ROS_DOMAIN_ID='42',
               PYTHONPATH=str(snapshot) + ':' + env.get('PYTHONPATH', ''))
    processes, logs = [], []
    active = ROOT / 'runs/active.json'
    active.write_text(json.dumps(dict(supervisor=os.getpid(), directory=str(out))) + '\n')

    def interrupt(*_):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, interrupt)

    def launch(name, command, cwd=ROOT):
        log = (out / f'{name}.log').open('w')
        process = subprocess.Popen(command, cwd=cwd, env=env, stdout=log, stderr=subprocess.STDOUT,
                                   stdin=subprocess.DEVNULL, start_new_session=True)
        processes.append((name, process))
        logs.append(log)
        return process

    def stop(process, sig=signal.SIGTERM):
        if process.poll() is None:
            os.killpg(process.pid, sig)

    interrupted = False
    trial_finished = False
    try:
        launch('gazebo', ['gz', 'sim', '-r', '-s', '-v', '2', '--seed', str(args.seed),
                          str(out / 'urban_delivery.sdf')])
        evaluator = launch('evaluator', [sys.executable, '-m', 'urban_drone.evaluate',
                                       '--truth', str(out / 'truth.json'), '--output', str(out)], snapshot)
        if not args.headless:
            # Edit camera positions here: -X behind, -Y right, +Z above.
            camera_offsets = {
                'follow': (-2, 0, 2),   # close rear view
                'view': (-5, -5, 8),    # rear-right view
            }
            launch('gui', ['gz', 'sim', '-g'])
        if args.rviz:
            import yaml
            view = yaml.safe_load((ROOT / 'config/view.rviz').read_text())
            if prior.get('local_window'):
                current = view['Visualization Manager']['Views']['Current']
                current['Distance'] = max(prior['bounds'][1]-prior['bounds'][0],prior['bounds'][3]-prior['bounds'][2])*1.1
                current['Focal Point'].update(X=(prior['bounds'][0]+prior['bounds'][1])/2, Y=0, Z=prior['cruise_height'])
            view_file = out/'view.rviz'
            view_file.write_text(yaml.safe_dump(view))
            launch('rviz', ['rviz2', '-d', str(view_file)])
        px4 = launch('px4', [str(binary), '-d'], rootfs)
        # PX4's startup script discovers the standalone world and model.
        for _ in range(60):
            if px4.poll() is not None:
                raise RuntimeError('PX4 exited during startup; see px4.log')
            text = (out / 'px4.log').read_text()
            if 'Startup script returned successfully' in text:
                break
            time.sleep(0.5)
        else:
            raise RuntimeError('PX4 startup timed out')
        # Never change safety parameters here. The existing approved SITL
        # configuration is required and explicitly recorded.
        status = subprocess.run([str(build / 'bin/px4-param'), 'show', 'NAV_DLL_ACT'], cwd=rootfs,
                                env=env, capture_output=True, text=True, timeout=5)
        (out / 'sitl_parameters.txt').write_text(status.stdout + status.stderr)
        launch('mavros', ['ros2', 'launch', 'mavros', 'px4.launch',
                          'fcu_url:=udp://:14540@127.0.0.1:14580'])
        telemetry_command = [sys.executable, '-m', 'urban_drone.telemetry',
                             '--events', str(out / 'events.jsonl')]
        if not args.headless:
            telemetry_command.append('--gazebo-markers')
        launch('telemetry', telemetry_command, snapshot)
        if not args.headless:
            # Native transport in this supervisor must use the same partition
            # as its GUI subprocess. Check real GUI state before claiming success.
            for key in ('GZ_PARTITION','GZ_IP'):
                os.environ[key]=env[key]
            try:
                camera_status=configure_camera(args.camera,camera_offsets.get(args.camera))
            except Exception as error:
                camera_status=dict(confirmed=False,mode=args.camera,reason=str(error))
            (out/'camera_status.json').write_text(json.dumps(camera_status,indent=2)+'\n')
            if camera_status['confirmed']:
                print(f'Camera confirmed: {args.camera}, offset={camera_status.get("offset")}. '
                      'Esc releases tracking for the rest of this run.',flush=True)
            else:
                print(f'Camera not confirmed: {camera_status["reason"]}. '
                      'See camera_status.json; flight planning is unchanged.',flush=True)
        mission = launch('mission', [sys.executable, '-m', 'urban_drone.mission',
                                    '--prior', str(out / 'prior.json'), '--output', str(out),
                                    '--timeout', str(args.timeout), '--baseline', args.baseline], snapshot)
        last_line = ''
        deadline = time.monotonic() + args.timeout + 70
        while mission.poll() is None:
            if time.monotonic() > deadline:
                raise RuntimeError('Mission process exceeded its shutdown budget')
            lines = (out / 'mission.log').read_text().splitlines()
            if lines and lines[-1] != last_line:
                last_line = lines[-1]
                print(last_line, flush=True)
            time.sleep(1)
        time.sleep(1)
        stop(evaluator)
        evaluator.wait(timeout=5)
        subprocess.run([sys.executable, '-m', 'urban_drone.report', str(out)],
                       cwd=snapshot, env=env, check=True)
        result = json.loads((out / 'result.json').read_text())
        trial_finished = True
        (ROOT / 'runs/latest.txt').write_text(str(out) + '\n')
        if not result['success']:
            raise SystemExit(1)
        if args.keep_open and not args.headless:
            print('Flight finished and disarmed. Close Gazebo or run scripts/stop.sh to exit.', flush=True)
            gui = next(proc for name, proc in processes if name == 'gui')
            while gui.poll() is None:
                time.sleep(1)
    except KeyboardInterrupt:
        interrupted = not trial_finished
        print('Stopping this run; requesting AUTO.LAND if still armed.', flush=True)
    finally:
        # On any error while armed, request landing before terminating PX4.
        if any(name == 'px4' and proc.poll() is None for name, proc in processes):
            subprocess.run([str(build / 'bin/px4-commander'), 'land'], cwd=rootfs, env=env,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=5)
            for _ in range(40):
                status = subprocess.run([str(build / 'bin/px4-commander'), 'status'], cwd=rootfs,
                                        env=env, capture_output=True, text=True, timeout=5)
                if 'Disarmed' in status.stdout:
                    break
                time.sleep(0.5)
        for name, proc in reversed(processes):
            stop(proc)
        for name, proc in reversed(processes):
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                stop(proc, signal.SIGKILL)
                proc.wait(timeout=3)
        for log in logs:
            log.close()
        if active.exists() and json.loads(active.read_text()).get('supervisor') == os.getpid():
            active.unlink()
        if interrupted:
            (out / 'interrupted.txt').write_text('Operator interrupted this run. Not a successful trial.\n')


if __name__ == '__main__':
    main()
