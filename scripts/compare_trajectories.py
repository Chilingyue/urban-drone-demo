"""Offline comparison only: retained Gazebo truth never enters flight control."""
import argparse
import json
from pathlib import Path
import numpy as np
from scipy.signal import savgol_filter


def analyse(directory):
    directory=Path(directory)
    result=json.loads((directory/'result.json').read_text())
    prior=json.loads((directory/'prior.json').read_text())
    data=np.loadtxt(directory/'ground_truth_trajectory.csv',delimiter=',')
    airborne=np.where((data[:,3]>prior['cruise_height']-.4)&(data[:,1]<prior['goal'][0]-2.))[0]
    data=data[airborne[0]:airborne[-1]+1]
    t=np.arange(data[0,0],data[-1,0],.1)
    xyz=np.column_stack([np.interp(t,data[:,0],data[:,i]) for i in (1,2,3)])
    velocity=savgol_filter(xyz,11,3,deriv=1,delta=.1,axis=0)
    accel=savgol_filter(xyz,11,3,deriv=2,delta=.1,axis=0)
    jerk=savgol_filter(xyz,11,3,deriv=3,delta=.1,axis=0)
    speed=np.linalg.norm(velocity,axis=1)
    yaw=np.arctan2(velocity[:,1],velocity[:,0]);delta=np.arctan2(np.sin(np.diff(yaw)),np.cos(np.diff(yaw)))
    moving=np.linalg.norm(velocity[:,:2],axis=1)>.3
    variation=np.degrees(abs(delta[moving[:-1]&moving[1:]])).sum()
    events=[json.loads(l) for l in(directory/'events.jsonl').read_text().splitlines()]
    summary=dict(run=directory.name,success=result['success'],elapsed_s=result['mission'].get('elapsed'),
                 path_length_m=result['evaluation']['path_length'],
                 minimum_clearance_m=result['evaluation']['minimum_clearance'],
                 goal_error_m=result['evaluation']['goal_xy_error'],
                 decision_count=sum(e['event']=='decision' for e in events),
                 execution_hold_count=sum(e['event']=='execution_hold' for e in events),
                 hover_seconds=float(np.sum(speed<.2)*.1),
                 backwards_east_m=float(np.maximum(-velocity[:,0],0).sum()*.1),
                 heading_variation_deg=float(variation),
                 acceleration_p95_m_s2=float(np.percentile(np.linalg.norm(accel,axis=1),95)),
                 jerk_p95_m_s3=float(np.percentile(np.linalg.norm(jerk,axis=1),95)))
    return summary,xyz,speed,accel


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('before',type=Path);parser.add_argument('after',type=Path)
    parser.add_argument('--output',type=Path,default=Path('docs/trajectory_comparison.png'))
    args=parser.parse_args()
    before=analyse(args.before);after=analyse(args.after)
    before_world=json.loads((args.before/'truth.json').read_text())
    after_world=json.loads((args.after/'truth.json').read_text())
    same_geometry=all(before_world[k]==after_world[k] for k in ['objects','start','goal'])
    output=dict(before=before[0],after=after[0],method='Ground-truth cruise segment z > cruise-0.4 m, before goal X-2 m; '
                'resample 10 Hz; position derivatives via 11-sample (1.1 s) cubic Savitzky-Golay; '
                'heading variation only where XY speed >0.3 m/s. Read-only offline evaluation.',
                same_geometry=same_geometry)
    args.output.parent.mkdir(exist_ok=True,parents=True)
    args.output.with_suffix('.json').write_text(json.dumps(output,indent=2)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon,Circle
    truth=json.loads((args.after/'truth.json').read_text());prior=json.loads((args.after/'prior.json').read_text())
    fig,axes=plt.subplots(2,2,figsize=(14,8),constrained_layout=True)
    ax=axes[0,0]
    for b in truth['objects']:
        x,y,_=b['center'];dx,dy,_=b['size'];yaw=b.get('yaw',0)
        color='#b9c0c8' if b['known'] else '#df7777'
        corners=np.array([[-dx,-dy],[dx,-dy],[dx,dy],[-dx,dy]])/2
        rot=np.array([[np.cos(yaw),-np.sin(yaw)],[np.sin(yaw),np.cos(yaw)]])
        ax.add_patch(Circle((x,y),dx/2,color=color,alpha=.5) if b.get('shape')=='cylinder' else Polygon(corners@rot.T+[x,y],color=color,alpha=.5))
    plan=np.loadtxt(args.after/'global_smooth.csv',delimiter=',')
    ax.plot(plan[:,0],plan[:,1],'--',color='#dc3333',label='Initial global plan')
    for item,color,label in [(before,'#777777','Before'),(after,'#235aeb','After')]:
        info,p,v,a=item
        ax.plot(p[:,0],p[:,1],color=color,label=label,linewidth=1.8)
        axes[0,1].plot(p[:,0],p[:,2],color=color,label=label)
        axes[1,0].plot(p[:,0],v,color=color,label=label,alpha=.8)
        axes[1,1].plot(p[:,0],np.linalg.norm(a,axis=1),color=color,label=label,alpha=.8)
    ax.set(xlim=(-5,245),ylim=(-30,25),title='Actual route comparison',xlabel='East / m',ylabel='North / m')
    ax.set_aspect('equal',adjustable='box')
    axes[0,1].axhline(prior['cruise_height']+.24,linestyle='--',color='#dc3333')
    axes[0,1].set(title='Actual altitude',ylabel='World body height / m',xlabel='East / m')
    axes[1,0].set(title='Cruise speed',ylabel='m/s',xlabel='East / m')
    axes[1,1].set(title='Cruise acceleration (same derivative filter)',ylabel='m/s²',xlabel='East / m')
    for ax in axes.flat:ax.legend(fontsize=8);ax.grid(alpha=.2)
    fig.suptitle('Measured flight comparison | fixed scenario seed | circles, online costs and command continuity')
    fig.savefig(args.output,dpi=150);plt.close(fig)
    print(json.dumps(output,indent=2))


if __name__=='__main__':main()
