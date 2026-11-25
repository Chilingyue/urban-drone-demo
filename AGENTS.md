# Urban drone demonstration

- Keep the existing `/mnt/d/px4_sim` project unchanged.
- Fly only in PX4 SITL. Do not connect this demonstration to hardware.
- The mission/planning process may read only `prior.json`, never `truth.json`,
  scene configuration, Gazebo entity poses, or `LaserScan.world_pose`.
- Transform range observations with timestamped MAVROS estimated poses and the
  documented lidar extrinsic. Ground truth belongs only to the evaluator.
- Unknown space is not verified free space. Execution requires recent sensing.
- Record actual trials, including failures; never invent delivery rates.
- Keep stage status in README honest. Validate flight changes in SITL.

