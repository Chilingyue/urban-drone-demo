# Side-switch regression fixture

`side_corridor_scan.npz` is the estimated occupied-point snapshot at map update
163 in real SITL trial `20261117_131437_592726_delivery_0`. It contains only the
observed point cloud, current estimated position and reference rejoin point.
`side_corridor_prior.json` is that run's public building prior.

No simulator obstacle truth is included. The geometry regression assumes the
aperture probes are visible, so that the old aperture exclusion is exercised.
The old planner rejects the right corridor when aperture detection is enabled;
the same map without virtual aperture exclusions has a cheaper, valid right route.

# 柱子右侧停滞回放

`column_clearance_prior.json` 是运行 `20261117_144435_854242_city_300_0` 的公开先验。
`column_clearance_222.npz`、`230.npz`、`234.npz` 仅保存当轮传感器融合得到的占据点
和 MAVROS 估计位置，不含未知障碍几何真值或 Gazebo 实体位姿。
距离分别约为 1.042 m、0.792 m、0.744 m，用于复现膨胀起点不可用，以及验证
恢复连接线不降低 0.75 m 执行阈值、不向未观测空间执行、不修改占据地图。
夹具没有完整原始扫描，回放几何检查将可见性单独设定；真实可见性必须经 SITL 验证。

