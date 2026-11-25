"""Generate physical SDF and independent public/prior and evaluator manifests."""

import argparse
import copy
import json
from pathlib import Path
import random
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]


def box(name, center, size, known=False):
    if len(size) != 3 or any(float(v) <= 0 for v in size):
        raise ValueError('box dimensions must be positive')
    return dict(name=name, center=list(center), size=list(size), known=known)


def expand_obstacle(spec, index):
    x, y = spec['center']
    if spec['kind'] == 'pillar':
        radius,height=spec['radius'],spec['height']
        b=box(f'unknown_{index}_pillar',[x,y,height/2],[2*radius,2*radius,height])
        b['shape']='cylinder'
        return [b]
    if spec['kind'] != 'gate':
        dx, dy, dz = spec['size']
        return [box(f'unknown_{index}_{spec["kind"]}', [x, y, dz / 2], [dx, dy, dz])]
    w, h, d = spec['width'], spec['height'], spec['depth']
    gap, bottom, top = spec['opening_width'], spec['opening_bottom'], spec['opening_top']
    offset=spec.get('opening_offset',0)
    if not 0 < gap < w or not abs(offset)+gap/2 < w/2 or not 0 <= bottom < top < h:
        raise ValueError('gate opening must be inside its outer frame')
    lower,upper=offset-gap/2,offset+gap/2
    left,right=lower+w/2,w/2-upper
    result = [
        box(f'unknown_{index}_gate_left', [x, y+(-w/2+lower)/2, h / 2], [d, left, h]),
        box(f'unknown_{index}_gate_right', [x, y+(upper+w/2)/2, h / 2], [d, right, h]),
        box(f'unknown_{index}_gate_lintel', [x, y+offset, (h + top) / 2], [d, gap, h - top]),
    ]
    if bottom > 0:
        result.append(box(f'unknown_{index}_gate_sill', [x, y+offset, bottom / 2], [d, gap, bottom]))
    return result


def build(case='wall_low', seed=0, jitter=0.0, overrides=None):
    if case in ('city_1km', 'city_300'):
        from .city import build_city
        return build_city(seed, jitter, overrides, case)
    config = json.loads((ROOT / 'config/scenes.json').read_text())
    if case not in config['cases']:
        raise ValueError(f'unknown case: {case}')
    spec = copy.deepcopy(config['cases'][case])
    rng = random.Random(seed)
    for obstacle in spec['obstacles']:
        obstacle['center'][1] += rng.uniform(-jitter, jitter)
        if overrides:
            obstacle.update(overrides)
    objects = []
    for building in config['known_buildings']:
        x, y = building['center']
        dx, dy, dz = building['size']
        # Short single-obstacle trials end before the office footprint.
        if spec.get('goal', config['goal'])[0] <= 20 and building['name'] == 'office':
            continue
        objects.append(box(building['name'], [x, y, dz / 2], [dx, dy, dz], True))
    for i, obstacle in enumerate(spec['obstacles']):
        objects.extend(expand_obstacle(obstacle, i))
    prior = {key: config[key] for key in (
        'bounds', 'start', 'cruise_height', 'max_height', 'grid_resolution',
        'vehicle_radius', 'safety_margin', 'tracking_margin')}
    prior['goal'] = spec.get('goal', config['goal'])
    prior['known_buildings'] = [b for b in objects if b['known']]
    truth = dict(case=case, seed=seed, jitter=jitter, objects=objects,
                 start=config['start'], goal=prior['goal'],
                 spawn_body_height=0.24, vehicle_radius=config['vehicle_radius'])
    return prior, truth


def add_box(world, b):
    model = ET.SubElement(world, 'model', name=b['name'])
    ET.SubElement(model, 'static').text = 'true'
    ET.SubElement(model, 'pose').text = ' '.join(map(str, b['center'] + [0, 0, b.get('yaw', 0)]))
    link = ET.SubElement(model, 'link', name='link')
    for tag in ('collision', 'visual'):
        part = ET.SubElement(link, tag, name=tag)
        geom = ET.SubElement(part, 'geometry')
        if b.get('shape') == 'cylinder':
            cylinder=ET.SubElement(geom,'cylinder')
            ET.SubElement(cylinder,'radius').text=str(b['size'][0]/2)
            ET.SubElement(cylinder,'length').text=str(b['size'][2])
        else:
            ET.SubElement(ET.SubElement(geom, 'box'), 'size').text = ' '.join(map(str, b['size']))
        if tag == 'visual':
            color = '0.50 0.58 0.68 1' if b['known'] else '0.95 0.48 0.12 1'
            if 'color' in b:
                color = ' '.join(map(str, b['color']))
            material = ET.SubElement(part, 'material')
            ET.SubElement(material, 'ambient').text = color
            ET.SubElement(material, 'diffuse').text = color


def add_roads(world, truth):
    """Flat visual scenery: road markings are not collision obstacles."""
    model = ET.SubElement(world, 'model', name='city_roads')
    ET.SubElement(model, 'static').text = 'true'
    link = ET.SubElement(model, 'link', name='link')
    count = 0

    def surface(center, size, color, height):
        nonlocal count
        x0,x1,y0,y1=truth['bounds']
        lo=[max(center[0]-size[0]/2,x0),max(center[1]-size[1]/2,y0)]
        hi=[min(center[0]+size[0]/2,x1),min(center[1]+size[1]/2,y1)]
        if hi[0]<=lo[0] or hi[1]<=lo[1]:
            return
        center=[(lo[0]+hi[0])/2,(lo[1]+hi[1])/2]
        size=[hi[0]-lo[0],hi[1]-lo[1]]
        count += 1
        visual = ET.SubElement(link, 'visual', name=f'paving_{count}')
        ET.SubElement(visual, 'pose').text = f'{center[0]} {center[1]} {height} 0 0 0'
        geom = ET.SubElement(visual, 'geometry')
        ET.SubElement(ET.SubElement(geom, 'box'), 'size').text = f'{size[0]} {size[1]} 0.005'
        material = ET.SubElement(visual, 'material')
        ET.SubElement(material, 'ambient').text = color
        ET.SubElement(material, 'diffuse').text = color
    for road in truth.get('roads', []):
        x, y = road['center']; dx, dy = road['size']
        surface([x,y], [dx+5,dy+5], '0.55 0.55 0.52 1', .006)
        surface([x,y], [dx,dy], '0.16 0.18 0.20 1', .014)
        if max(dx,dy)/min(dx,dy) > 8:
            line_size = [dx,.20] if dx > dy else [.20,dy]
            surface([x,y], line_size, '0.88 0.74 0.30 1', .020)
    x0,x1,_,_ = truth['bounds']
    road_width = truth.get('road_width',truth['roads'][0]['size'][1])
    for x in range(int(x0+8),int(x1),16):
        for y in (-road_width/4,road_width/4):
            surface([x,y], [7,.18], '0.9 0.9 0.86 1', .024)


def world_sdf(truth):
    sdf = ET.Element('sdf', version='1.9')
    world = ET.SubElement(sdf, 'world', name='urban_delivery')
    physics = ET.SubElement(world, 'physics', name='physics', type='ignored')
    ET.SubElement(physics, 'max_step_size').text = '0.004'
    ET.SubElement(physics, 'real_time_factor').text = '1.0'
    ET.SubElement(world, 'gravity').text = '0 0 -9.80665'
    # Zurich, matching the existing PX4 SITL world and estimator magnetic model.
    ET.SubElement(world, 'magnetic_field').text = '6e-06 2.3e-05 -4.2e-05'
    geo = ET.SubElement(world, 'spherical_coordinates')
    for tag, value in [('surface_model', 'EARTH_WGS84'), ('world_frame_orientation', 'ENU'),
                       ('latitude_deg', '47.397971057728974'), ('longitude_deg', '8.546163739800146'),
                       ('elevation', '488'), ('heading_deg', '0')]:
        ET.SubElement(geo, tag).text = value
    for lib, cls in [('physics', 'Physics'), ('user-commands', 'UserCommands'),
                     ('scene-broadcaster', 'SceneBroadcaster'), ('imu', 'Imu'),
                     ('air-pressure', 'AirPressure'), ('navsat', 'NavSat'),
                     ('magnetometer', 'Magnetometer'), ('sensors', 'Sensors')]:
        plugin = ET.SubElement(world, 'plugin', filename=f'gz-sim-{lib}-system',
                               name=f'gz::sim::systems::{cls}')
        if cls == 'Magnetometer':
            ET.SubElement(plugin, 'use_units_gauss').text = 'false'
            ET.SubElement(plugin, 'use_earth_frame_ned').text = 'false'
        if cls == 'Sensors':
            ET.SubElement(plugin, 'render_engine').text = 'ogre2'
    scene = ET.SubElement(world, 'scene')
    ET.SubElement(scene, 'ambient').text = '0.75 0.75 0.75 1'
    ET.SubElement(scene, 'background').text = '0.62 0.74 0.85 1'
    sun = ET.SubElement(world, 'light', name='sun', type='directional')
    ET.SubElement(sun, 'pose').text = '0 0 30 0 0 0'
    ET.SubElement(sun, 'direction').text = '-0.4 0.2 -0.9'
    ET.SubElement(sun, 'diffuse').text = '0.9 0.9 0.9 1'
    ET.SubElement(sun, 'cast_shadows').text = 'true'
    if truth.get('city_style'):
        x0,x1,y0,y1 = truth['bounds']
        ground = box('ground', [(x0+x1)/2,(y0+y1)/2,-.1], [x1-x0,y1-y0,.2], True)
        ground['color'] = [.31,.39,.28,1]
        add_box(world, ground)
        add_roads(world, truth)
    else:
        add_box(world, box('ground', [20, 0, -0.1], [100, 80, 0.2], True))
    for b in truth['objects']:
        add_box(world, b)
    for name, xy, color in [('pickup', truth['start'], '0.1 0.7 0.3 1'),
                            ('delivery', truth['goal'], '0.1 0.4 0.9 1')]:
        marker = ET.SubElement(world, 'model', name=name + '_pad')
        ET.SubElement(marker, 'static').text = 'true'
        ET.SubElement(marker, 'pose').text = f'{xy[0]} {xy[1]} 0.005 0 0 0'
        vis = ET.SubElement(ET.SubElement(marker, 'link', name='link'), 'visual', name='pad')
        geom = ET.SubElement(vis, 'geometry')
        ET.SubElement(ET.SubElement(geom, 'box'), 'size').text = '2 2 0.01'
        material = ET.SubElement(vis, 'material')
        ET.SubElement(material, 'ambient').text = color
        ET.SubElement(material, 'diffuse').text = color
    include = ET.SubElement(world, 'include')
    ET.SubElement(include, 'uri').text = 'model://urban_x500'
    ET.SubElement(include, 'name').text = 'urban_x500_0'
    ET.SubElement(include, 'pose').text = f'{truth["start"][0]} {truth["start"][1]} 0 0 0 0'
    ET.indent(sdf)
    return ET.tostring(sdf, encoding='unicode')


def generate(out, case='wall_low', seed=0, jitter=0.0, overrides=None):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    prior, truth = build(case, seed, jitter, overrides)
    (out / 'prior.json').write_text(json.dumps(prior, indent=2) + '\n')
    (out / 'truth.json').write_text(json.dumps(truth, indent=2) + '\n')
    (out / 'urban_delivery.sdf').write_text(world_sdf(truth))
    if case in ('city_1km','city_300'):
        from .city import preview
        preview(out, prior, truth)
    return prior, truth


def parameter_overrides(case, height=None, opening_width=None):
    overrides = {}
    if height is not None:
        if case in ('city_1km','city_300'):
            raise ValueError(f'Set city obstacle heights in config/{case}.json')
        cfg = json.loads((ROOT / 'config/scenes.json').read_text())
        spec = cfg['cases'][case]['obstacles'][0]
        if 'size' not in spec:
            raise ValueError('--height is for box/wall cases; edit gate height in config')
        size = spec['size'][:]
        size[2] = height
        overrides['size'] = size
    if opening_width is not None:
        overrides['opening_width'] = opening_width
    return overrides


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--case', default='wall_low')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--jitter', type=float, default=0)
    p.add_argument('--out', type=Path, default=ROOT / 'generated/current')
    p.add_argument('--height', type=float)
    p.add_argument('--opening-width', type=float)
    args = p.parse_args()
    overrides = parameter_overrides(args.case, args.height, args.opening_width)
    prior, truth = generate(args.out, args.case, args.seed, args.jitter, overrides)
    print(f'Generated {args.case}: {len(prior["known_buildings"])} prior objects, '
          f'{len(truth["objects"])} physical objects; {args.out}')


if __name__ == '__main__':
    main()

