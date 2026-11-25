"""Read-only live visualization of estimated flight and public decision events."""

import argparse
from collections import deque
import json
from pathlib import Path
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path as RosPath
from visualization_msgs.msg import Marker


class Telemetry(Node):
    def __init__(self, events, gazebo_markers=False):
        super().__init__('urban_delivery_visualization')
        self.events = Path(events)
        self.event_offset = 0
        self.remainder = ''
        self.strategy, self.phase = 'TRACK', 'WAIT'
        self.trace = deque()
        self.latest = None
        self.trace_time = 0
        self.initial_plan = ()
        self.overlay = None
        if gazebo_markers:
            from .path_visuals import GazeboPathOverlay
            self.overlay = GazeboPathOverlay()
        self.path_pub = self.create_publisher(RosPath, '/urban/actual_path', 10)
        self.vehicle_pub = self.create_publisher(Marker, '/urban/vehicle_marker', 10)
        self.create_subscription(PoseStamped, '/mavros/local_position/pose', self.pose,
                                 qos_profile_sensor_data)
        self.create_subscription(RosPath, '/urban/global_path', self.plan, 10)
        self.create_timer(0.5, self.publish)

    def pose(self, message):
        self.latest = message
        now = self.get_clock().now().nanoseconds
        if now - self.trace_time > 200_000_000:
            self.trace_time = now
            self.trace.append(message)

    def plan(self, message):
        if not self.initial_plan and len(message.poses) >= 2:
            self.initial_plan = tuple((p.pose.position.x, p.pose.position.y, p.pose.position.z)
                                      for p in message.poses)

    def publish(self):
        if self.events.exists():
            with self.events.open() as file:
                file.seek(self.event_offset)
                text = self.remainder + file.read()
                self.event_offset = file.tell()
            lines = text.split('\n')
            self.remainder = lines.pop()
            for line in lines:
                if not line:
                    continue
                record = json.loads(line)
                if record['event'] == 'decision':
                    self.strategy = record['strategy']
                elif record['event'] == 'rejoin':
                    self.strategy = 'TRACK'
                elif record['event'] == 'phase':
                    self.phase = record['phase']
                elif record['event'] == 'mission_finished':
                    self.phase = 'DONE'
        if self.latest is None:
            return
        path = RosPath()
        path.header.frame_id = 'map'
        path.header.stamp = self.get_clock().now().to_msg()
        path.poses = list(self.trace)
        self.path_pub.publish(path)
        if self.overlay is not None:
            flown = tuple((p.pose.position.x, p.pose.position.y, p.pose.position.z) for p in self.trace)
            self.overlay.update(self.initial_plan, flown)
        for ident, kind in [(0, Marker.SPHERE), (1, Marker.TEXT_VIEW_FACING)]:
            marker = Marker()
            marker.header = path.header
            marker.ns, marker.id, marker.type, marker.action = 'vehicle', ident, kind, Marker.ADD
            marker.pose.position.x = self.latest.pose.position.x
            marker.pose.position.y = self.latest.pose.position.y
            marker.pose.position.z = self.latest.pose.position.z + (1.0 if ident else 0)
            marker.pose.orientation.w = 1.0
            marker.scale.x = marker.scale.y = marker.scale.z = 0.7 if not ident else 0.5
            marker.color.r, marker.color.g, marker.color.b, marker.color.a = 0.2, 0.9, 0.95, 1.0
            if ident:
                marker.text = f'{self.phase} | {self.strategy}'
                marker.color.r = marker.color.g = marker.color.b = 1.0
            self.vehicle_pub.publish(marker)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--events', required=True)
    parser.add_argument('--gazebo-markers', action='store_true')
    args = parser.parse_args()
    rclpy.init()
    node = Telemetry(args.events, args.gazebo_markers)
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        pass
    finally:
        if node.overlay is not None:
            node.overlay.close()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()

