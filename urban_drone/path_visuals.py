"""Read-only Gazebo GUI overlays; no flight commands or simulator truth."""

import threading

PLAN_RED = (0.90, 0.18, 0.18)
FLIGHT_BLUE = (0.14, 0.35, 0.92)


def path_marker(ident, points, color, z_offset=0.24):
    from gz.msgs10.marker_pb2 import Marker
    marker = Marker()
    marker.ns, marker.id = 'urban_drone_paths', ident
    marker.action, marker.type = Marker.ADD_MODIFY, Marker.LINE_STRIP
    marker.visibility = Marker.GUI
    marker.pose.orientation.w = 1.0
    marker.scale.x = marker.scale.y = marker.scale.z = 1.0
    marker.material.lighting = False
    for field in ('ambient', 'diffuse', 'emissive'):
        value = getattr(marker.material, field)
        value.r, value.g, value.b = color
        value.a = 1.0
    # MAVROS height is relative to the initial body position, 0.24 m above
    # the world origin in this model. The same offset applies to both paths.
    for x, y, z in points:
        point = marker.point.add()
        point.x, point.y, point.z = float(x), float(y), float(z)+z_offset
    return marker


class GazeboPathOverlay:
    def __init__(self):
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.paths = ((), ())
        self.worker = threading.Thread(target=self.work, daemon=True)
        self.worker.start()

    def update(self, planned, flown):
        with self.lock:
            self.paths = planned, flown

    def work(self):
        from gz.transport13 import Node
        from gz.msgs10.marker_v_pb2 import Marker_V
        from gz.msgs10.boolean_pb2 import Boolean
        node = Node()
        announced = False
        while not self.stop_event.wait(0.4):
            with self.lock:
                planned, flown = self.paths
            message = Marker_V()
            for ident, points, color in [(1, planned, PLAN_RED), (2, flown, FLIGHT_BLUE)]:
                if len(points) < 2:
                    continue
                message.marker.append(path_marker(ident, points, color))
            if not message.marker:
                continue
            # /marker is one-way; /marker_array supplies an acknowledgement.
            # This worker never blocks ROS pose reception or PX4 setpoints.
            ok, response = node.request('/marker_array', message, Marker_V, Boolean, 200)
            if ok and response.data and not announced:
                print('Gazebo path overlays connected: initial plan RED, flown path BLUE.', flush=True)
                announced = True

    def close(self):
        self.stop_event.set()
        self.worker.join(timeout=2)
