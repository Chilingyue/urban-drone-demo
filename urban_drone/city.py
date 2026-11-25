"""Author urban scenes offline; no simulator geometry enters control."""

import json
import math
from pathlib import Path
import random
import numpy as np

from .planning import PriorMap, length
from .scene import ROOT, box, expand_obstacle


def build_city(seed=0, jitter=0.0, overrides=None, case='city_1km'):
    cfg = json.loads((ROOT / f'config/{case}.json').read_text())
    rng = random.Random(seed)
    x0, x1, y0, y1 = cfg['bounds']
    lx0,lx1,ly0,ly1=cfg.get('layout_bounds',cfg['bounds'])
    spacing, width = cfg['road_spacing'], cfg['road_width']
    xs = list(np.arange(lx0+min(20,spacing/3), lx1, spacing, dtype=float))
    ys = list(np.arange(math.ceil(ly0/spacing)*spacing, ly1+1, spacing, dtype=float))
    roads = [dict(center=[(x0+x1)/2, y], size=[x1-x0, width]) for y in ys]
    roads += [dict(center=[x, (y0+y1)/2], size=[width, y1-y0]) for x in xs]
    # A central office island forces the public A* route to bend around a
    # known building. Its paved square connects to the surrounding streets.
    plaza = cfg.get('plaza', dict(center=[400,0],size=[144,108]))
    roads.append(plaza)
    office = cfg.get('central_office', dict(center=[400,0,45],size=[48,42,90]))
    objects, unknown_sites = [], []
    index = 0
    for xa, xb in zip(xs[:-1], xs[1:]):
        for ya, yb in zip(ys[:-1], ys[1:]):
            for fx in (0.30, 0.70):
                for fy in (0.30, 0.70):
                    x, y = xa+(xb-xa)*fx, ya+(yb-ya)*fy
                    if (abs(x-plaza['center'][0]) < plaza['size'][0]/2+spacing*.15 and
                            abs(y-plaza['center'][1]) < plaza['size'][1]/2+spacing*.15):
                        continue
                    dx, dy = [rng.uniform(*cfg.get('building_size',[25,33])) for _ in range(2)]
                    height = rng.randint(*cfg.get('high_heights',[48,90])) if index % 6 == 0 else rng.randint(*cfg.get('low_heights',[8,30]))
                    height = max(3,height-cfg.get('height_reduction',0))
                    # A few red L-shaped buildings elsewhere in the city are
                    # also absent from the prior, not just the three test sites.
                    if cfg.get('extra_unknown_buildings',False) and index % 43 == 20 and abs(y) > 100:
                        unknown_sites.append((index, x, y, dx, dy, height))
                    else:
                        b = box(f'city_building_{index:03d}', [x, y, height/2], [dx, dy, height], True)
                        shade = (index % 5) * 0.035
                        b['color'] = [0.42+shade, 0.48+shade, 0.56+shade, 1]
                        objects.append(b)
                    index += 1
    objects.append(box('central_office_island', office['center'], office['size'], True))
    known = [b for b in objects if b['known']]
    heights = sorted(b['size'][2] for b in known)
    rank = math.ceil(cfg['building_height_quantile']*len(heights))-1
    cruise = float(heights[rank]+cfg['roof_clearance'])
    prior = {key: cfg[key] for key in ('bounds', 'start', 'goal', 'grid_resolution',
             'vehicle_radius', 'safety_margin', 'tracking_margin', 'local_window',
             'local_lateral_limit', 'tracking_lookahead', 'landing_timeout', 'mission_timeout')}
    prior.update(known_buildings=known, cruise_height=cruise, max_height=cruise+10,
                 height_aware=True, building_height_quantile=cfg['building_height_quantile'],
                 roof_clearance=cfg['roof_clearance'])
    prior.update({key:cfg[key] for key in ('arrival_hold_seconds','arrival_tolerance',
                  'aperture_lateral_limit','aperture_center_limit') if key in cfg})
    # Scene authors may place obstacles along a public plan. This is done
    # offline, never in the mission, and their positions stay out of prior.json.
    _, route = PriorMap(prior).route()
    # Keep only the flight corridor, rather than rendering distant city blocks.
    # The reference used for pruning contains public geometry only.
    corridor = cfg.get('building_corridor_half_width')
    if corridor is not None:
        probes = route[::max(1,int(5/.15)),:2]
        def near_route(xy):
            xy=np.asarray(xy)
            return float(np.linalg.norm(probes-xy,axis=1).min()) <= corridor
        objects = [b for b in objects if near_route(b['center'][:2]) and
                   x0 < b['center'][0]-b['size'][0]/2 and b['center'][0]+b['size'][0]/2 < x1 and
                   y0 < b['center'][1]-b['size'][1]/2 and b['center'][1]+b['size'][1]/2 < y1]
        unknown_sites = [site for site in unknown_sites if near_route(site[1:3])]
        known = [b for b in objects if b['known']]
        heights = sorted(b['size'][2] for b in known)
        cruise = float(heights[math.ceil(cfg['building_height_quantile']*len(heights))-1]+cfg['roof_clearance'])
        prior.update(known_buildings=known, cruise_height=cruise,max_height=cruise+10)
        _, route = PriorMap(prior).route()
    distances = np.r_[0, np.cumsum(np.linalg.norm(np.diff(route, axis=0), axis=1))]
    sites = []
    for i, source in enumerate(cfg['obstacles']):
        spec = dict(source)
        at = min(int(np.searchsorted(distances, source['progress']*distances[-1])), len(route)-2)
        tangent = route[at+1, :2]-route[max(0, at-1), :2]
        tangent /= np.linalg.norm(tangent)
        normal = np.array([-tangent[1], tangent[0]])
        xy = route[at, :2] + normal*rng.uniform(-jitter, jitter)
        spec['center'] = [0, 0]
        if spec['kind'] == 'gate':
            spec.update(height=cruise+spec['height_above_cruise'],
                        opening_bottom=cruise-spec['opening_below_cruise'],
                        opening_top=cruise+spec['opening_above_cruise'])
        elif spec['kind'] == 'pillar':
            spec['height'] = cruise+spec['height_above_cruise']
        else:
            spec['size'] = [*spec['size'][:2], cruise+spec['size'][2]]
        if overrides:
            spec.update(overrides)
        yaw = math.atan2(tangent[1], tangent[0])
        for b in expand_obstacle(spec, i):
            local = b['center'][:2]
            b['center'][:2] = (xy+tangent*local[0]+normal*local[1]).tolist()
            b['yaw'], b['color'] = yaw, [0.86, 0.055, 0.045, 1]
            objects.append(b)
        sites.append(dict(kind=spec['kind'], center=xy.tolist(), yaw=yaw,
                          progress=source['progress'], opening_offset=spec.get('opening_offset'),
                          opening_width=spec.get('opening_width')))
    for i, x, y, dx, dy, height in unknown_sites:
        for suffix, center, size in [
            ('a', [x-dx/3, y, height/2], [dx/3, dy, height]),
            ('b', [x+dx/6, y-dy/3, height/2], [2*dx/3, dy/3, height])]:
            b = box(f'unknown_L_{i}_{suffix}', center, size)
            b['color'] = [0.86, 0.055, 0.045, 1]
            objects.append(b)
    clipped_roads=[]
    for road in roads:
        c=np.array(road['center']); half=np.array(road['size'])/2
        lo=np.maximum(c-half,[x0,y0]); hi=np.minimum(c+half,[x1,y1])
        if np.all(hi>lo):
            clipped_roads.append(dict(center=((lo+hi)/2).tolist(),size=(hi-lo).tolist()))
    truth = dict(case=case, seed=seed, jitter=jitter, bounds=cfg['bounds'],
                 start=cfg['start'], goal=cfg['goal'], objects=objects, roads=clipped_roads,
                 road_width=cfg['road_width'],
                 demonstration_sites=sites, spawn_body_height=0.24,
                 vehicle_radius=cfg['vehicle_radius'], city_style=True,
                 public_building_fraction_below_cruise=sum(h < cruise for h in heights)/len(heights),
                 initial_route_length=length(route))
    return prior, truth


def preview(out, prior, truth):
    """A labelled scene plan, never presented as a completed flight."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle, Polygon, Rectangle
    _, route = PriorMap(prior).route()
    out = Path(out)
    np.savetxt(out/'city_plan.csv', route, delimiter=',', header='x,y,z (plan only; no flight)')
    fig, axes = plt.subplots(2, 1, figsize=(14, 10), constrained_layout=True,
                             gridspec_kw={'height_ratios': [3, 1]})
    for ax in axes:
        ax.set_facecolor('#87977c')
        for road in truth['roads']:
            x, y = road['center']; dx, dy = road['size']
            ax.add_patch(Rectangle((x-dx/2, y-dy/2), dx, dy, color='#51575b'))
        for b in truth['objects']:
            x, y, _ = b['center']; dx, dy, _ = b['size']
            corners = np.array([[-dx,-dy],[dx,-dy],[dx,dy],[-dx,dy]])/2
            yaw = b.get('yaw', 0)
            rot = np.array([[math.cos(yaw),-math.sin(yaw)],[math.sin(yaw),math.cos(yaw)]])
            color = '#aeb8c3' if b['known'] else '#db1919'
            patch = Circle((x,y),dx/2,facecolor=color,edgecolor='#30373d',linewidth=.25) if b.get('shape')=='cylinder' else Polygon(corners@rot.T+[x,y],facecolor=color,edgecolor='#30373d',linewidth=.25)
            ax.add_patch(patch)
        ax.plot(route[:,0],route[:,1], color='white', linewidth=3)
        ax.plot(route[:,0],route[:,1], '--', color='#e62d2d', linewidth=1.6, label='Initial A* plan (fixed altitude)')
        ax.scatter(*prior['start'], color='#40ee82', edgecolor='white', s=80, zorder=5, label='Start')
        ax.scatter(*prior['goal'], color='#4389ff', edgecolor='white', marker='*', s=160, zorder=5, label='Delivery')
        for i, site in enumerate(truth['demonstration_sites']):
            ax.annotate(f'{i+1}: {site["kind"]}', site['center'], xytext=(0,22), textcoords='offset points',
                        ha='center', color='#fff1e9', fontsize=9, arrowprops=dict(arrowstyle='-',color='white'),
                        bbox=dict(facecolor='#302e2e', alpha=.85, edgecolor='none', pad=3))
        ax.set_aspect('equal'); ax.set(xlabel='East / m', ylabel='North / m')
    x0,x1,y0,y1=prior['bounds']; axes[0].set(xlim=(x0,x1),ylim=(y0,y1))
    axes[0].set_title(f'{x1-x0} m x {y1-y0} m city | grey: known buildings; red: unknown geometry')
    axes[1].set(xlim=(prior['start'][0]-10,prior['goal'][0]+10),ylim=(-50,50),title='Delivery corridor detail (scene design, not a flown trajectory)')
    axes[0].legend(loc='upper left', fontsize=8)
    fig.suptitle(f'Cruise {prior["cruise_height"]:.1f} m | above {truth["public_building_fraction_below_cruise"]:.1%} of public buildings | '
                 f'route {truth["initial_route_length"]:.0f} m')
    fig.savefig(out/'scene_overview.png', dpi=150)
    plt.close(fig)
    # An elevation makes offset openings visible: a footprint projection alone
    # cannot show whether the cruise line intersects an opening or solid wall.
    sites=truth['demonstration_sites']
    fig,axes=plt.subplots(1,len(sites),figsize=(17,5.5),squeeze=False)
    fig.subplots_adjust(left=.04,right=.99,bottom=.18,top=.78,wspace=.30)
    profile_height=max(b['center'][2]+b['size'][2]/2 for b in truth['objects'] if not b['known'])+2
    profile_width=max(source.get('width',source.get('size',[0,0])[1])/2 for source in json.loads((ROOT/f'config/{truth["case"]}.json').read_text())['obstacles'])+2
    for i,(ax,site) in enumerate(zip(axes[0],sites)):
        normal=np.array([-math.sin(site['yaw']),math.cos(site['yaw'])])
        parts=[b for b in truth['objects'] if b['name'].startswith(f'unknown_{i}_')]
        half_width=0
        height=0
        for b in parts:
            lateral=float((np.array(b['center'][:2])-site['center'])@normal)
            w,h=b['size'][1:]
            ax.add_patch(Rectangle((lateral-w/2,b['center'][2]-h/2),w,h,facecolor='#c62424',edgecolor='#702020'))
            half_width=max(half_width,abs(lateral)+w/2);height=max(height,b['center'][2]+h/2)
        ax.axvline(0,color='#666',linestyle=':',linewidth=1)
        ax.axhline(prior['cruise_height']+truth['spawn_body_height'],color='#999',linestyle='--',linewidth=1)
        ax.scatter([0],[prior['cruise_height']+truth['spawn_body_height']],color='#f02020',edgecolor='white',marker='X',s=110,zorder=5,label='Initial plan')
        title=f'{i+1}: {site["kind"]}\n{2*half_width:.0f} m wide, {height:.1f} m high'
        if site['kind']=='gate': title+=f'\nOpening offset {site["opening_offset"]:+.1f} m'
        ax.set(xlim=(-profile_width,profile_width),ylim=(0,profile_height),xlabel='Lateral offset / m',title=title)
        ax.set_aspect('equal',adjustable='box');ax.grid(alpha=.15)
        if i==0: ax.set_ylabel('World height / m')
    fig.suptitle('Obstacle elevations | red X = initial cruise plan; these are geometry views, not flight traces')
    fig.text(.5,.04,'Both gate openings are offset: the original cruise line crosses solid wall, not the opening.',ha='center')
    fig.savefig(out/'obstacle_profiles.png',dpi=150)
    plt.close(fig)
