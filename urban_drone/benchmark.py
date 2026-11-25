"""Run declared seeded SITL trials and summarize every recorded outcome."""

import argparse
import csv
import json
from pathlib import Path
import subprocess
import sys
from .scene import ROOT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cases', nargs='+', default=['wall_low', 'wall_high', 'building',
                                                      'gate_open', 'gate_narrow'])
    parser.add_argument('--seeds', nargs='+', type=int, default=[0, 1, 2])
    parser.add_argument('--baselines', nargs='+', choices=['full', 'side_only', 'global_only'], default=['full'])
    parser.add_argument('--jitter', type=float, default=0.5)
    parser.add_argument('--timeout', type=float, default=180)
    parser.add_argument('--execute', action='store_true', help='Actually launch each SITL trial')
    args = parser.parse_args()
    trials = [dict(case=c, seed=s, baseline=b) for c in args.cases for s in args.seeds for b in args.baselines]
    print(json.dumps(dict(trial_count=len(trials), trials=trials), indent=2))
    if not args.execute:
        print('Dry run. Add --execute to fly this declared experiment matrix.')
        return
    rows = []
    for trial in trials:
        before = set((ROOT / 'runs').iterdir())
        command = [sys.executable, '-m', 'urban_drone.run', '--case', trial['case'],
                   '--seed', str(trial['seed']), '--baseline', trial['baseline'],
                   '--jitter', str(args.jitter), '--timeout', str(args.timeout), '--headless']
        process = subprocess.run(command, cwd=ROOT)
        created = [f for f in set((ROOT / 'runs').iterdir()) - before if f.is_dir()]
        directory = max(created, key=lambda f: f.stat().st_mtime) if created else None
        result_file = directory / 'result.json' if directory else None
        result = json.loads(result_file.read_text()) if result_file and result_file.exists() else {}
        row = dict(**trial, success=bool(result.get('success', False)), exit_code=process.returncode,
                   run=str(directory), failure=result.get('mission', {}).get('failure') or
                   ('invalid_or_intrusion' if not result.get('success', False) else ''),
                   elapsed=result.get('mission', {}).get('elapsed'),
                   clearance=result.get('evaluation', {}).get('minimum_clearance'))
        rows.append(row)
        # Incremental ledger preserves failed and interrupted experiments.
        with (ROOT / 'runs/benchmark.csv').open('w') as file:
            writer = csv.DictWriter(file, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    successes = sum(row['success'] for row in rows)
    print(f'Actual delivery rate: {successes}/{len(rows)} = {successes/len(rows):.1%}')


if __name__ == '__main__':
    main()
