"""Produce an offline evidence plot and combined result from an actual run."""

import argparse
import csv
import json
from pathlib import Path
import numpy as np


def report(directory):
    directory = Path(directory)
    truth = json.loads((directory / 'truth.json').read_text())
    mission_file, evaluation_file = directory / 'mission_result.json', directory / 'evaluation.json'
    mission = json.loads(mission_file.read_text()) if mission_file.exists() else {'success': False, 'failure': 'no_result'}
    evaluation = json.loads(evaluation_file.read_text()) if evaluation_file.exists() else {'valid': False}
    result = dict(case=truth['case'], seed=truth['seed'], baseline=mission.get('baseline', 'unknown'),
                  success=bool(mission['success'] and evaluation.get('valid') and
                               not evaluation.get('envelope_intrusion', True) and
                               evaluation.get('goal_xy_error', 999) <= 1.0),
                  mission=mission, evaluation=evaluation)
    (directory / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
    ledger = directory.parent / 'trials.csv'
    row = dict(run=directory.name, case=result['case'], seed=result['seed'],
               baseline=result['baseline'], success=result['success'],
               elapsed=mission.get('elapsed'), failure=mission.get('failure'),
               minimum_clearance=evaluation.get('minimum_clearance'),
               goal_xy_error=evaluation.get('goal_xy_error'))
    old = []
    if ledger.exists():
        with ledger.open() as file:
            old = [r for r in csv.DictReader(file) if r['run'] != directory.name]
    with ledger.open('w') as file:
        writer = csv.DictWriter(file, fieldnames=list(row))
        writer.writeheader()
        writer.writerows([*old, row])
    render_plot(directory, result)
    print(json.dumps(result, indent=2))
    return result


def render_plot(directory, result, output=None):
    """Render recorded data without changing evaluation records or the trial ledger."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle, Rectangle, Polygon
    directory = Path(directory)
    truth = json.loads((directory / 'truth.json').read_text())
    mission = result['mission']
    fig, (ax, altitude) = plt.subplots(1, 2, figsize=(14, 5.5), constrained_layout=True)
    for road in truth.get('roads',[]):
        x,y=road['center']; dx,dy=road['size']
        ax.add_patch(Rectangle((x-dx/2,y-dy/2),dx,dy,color='#e2e4e6',zorder=0))
    for b in truth['objects']:
        x, y = b['center'][:2]
        dx, dy = b['size'][:2]
        color = '#a6b0bc' if b['known'] else '#eb8a35'
        if truth.get('city_style') and not b['known']:
            color='#db1919'
        corners=np.array([[-dx,-dy],[dx,-dy],[dx,dy],[-dx,dy]])/2
        yaw=b.get('yaw',0)
        rot=np.array([[np.cos(yaw),-np.sin(yaw)],[np.sin(yaw),np.cos(yaw)]])
        patch=Circle((x,y),dx/2,color=color,alpha=.65) if b.get('shape')=='cylinder' else Polygon(corners@rot.T+[x,y],color=color,alpha=.65)
        ax.add_patch(patch)
    if (directory / 'global_smooth.csv').exists():
        p = np.loadtxt(directory / 'global_smooth.csv', delimiter=',')
        ax.plot(p[:, 0], p[:, 1], '--', color='#e62d2d', linewidth=2,
                label='Initial global plan (RED)', zorder=3)
        altitude.axhline(float(p[0, 2])+truth.get('spawn_body_height', 0.24),
                         color='#e62d2d', linestyle='--', label='Planned cruise height')
    if (directory / 'ground_truth_trajectory.csv').exists():
        p = np.loadtxt(directory / 'ground_truth_trajectory.csv', delimiter=',')
        if p.ndim == 2:
            ax.plot(p[:, 1], p[:, 2], color='#235aeb', linewidth=2,
                    label='Actual flight (BLUE)', zorder=4)
            altitude.plot(p[:, 0], p[:, 3], color='#235aeb', label='Actual body height')
    ax.scatter(*truth['start'], color='#344154', marker='o', s=60, label='Pickup', zorder=5)
    ax.scatter(*truth['goal'], color='#344154', marker='*', s=110, label='Delivery', zorder=5)
    ax.set(xlabel='East / m', ylabel='North / m', title=f'{truth["case"]} | seed {truth["seed"]}')
    ax.set_aspect('equal',adjustable='box')
    if truth.get('city_style'):
        ax.set_xlim(truth['start'][0]-15,truth['goal'][0]+15)
        ax.set_ylim(-65,65)
    ax.grid(alpha=0.2)
    ax.legend(fontsize=8)
    altitude.set(xlabel='Elapsed / s', ylabel='World body height / m', title='Actual altitude')
    altitude.grid(alpha=0.2)
    altitude.legend(fontsize=8)
    status = 'SUCCESS' if result['success'] else 'FAILED / INCOMPLETE'
    fig.suptitle(f'PX4 SITL | {status} | '+', '.join(mission.get('strategies', [])))
    output = Path(output) if output is not None else directory / 'flight_report.png'
    fig.savefig(output, dpi=150)
    plt.close(fig)
    return output


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('directory')
    p.add_argument('--plot-only', action='store_true', help='Re-render without rewriting result files')
    p.add_argument('--output', type=Path)
    args = p.parse_args()
    if args.plot_only:
        directory = Path(args.directory)
        result = json.loads((directory / 'result.json').read_text())
        print(render_plot(directory, result, args.output))
    else:
        if args.output is not None:
            p.error('--output requires --plot-only')
        report(args.directory)


if __name__ == '__main__':
    main()

