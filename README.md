# 楼宇配送：A* 与在线局部避障

Python + PX4 SITL + Gazebo Harmonic + ROS2 Jazzy/MAVROS。下面按日常使用频率排列操作，仅用于仿真。

仓库附带两份精选的完整飞行记录压缩包，选择依据、大小和解压方法见 [示例记录](examples/recordings/README.md)。
其他历史 `runs/` 飞行记录、`replays/` 会话、`generated/` 场景产物和旧备份不上传。
首次运行会重新生成所需场景和本轮数据；历史回放/报告示例需要先解压对应记录。
测试所需的小型数据保留在 `tests/data/`，整理后的验证图表和检查记录保留在 `docs/`。

## 常用操作

先进入项目目录；下列命令均在这个目录的终端中执行：

```bash
cd ~/urban_drone_demo
```

### 1. 启动城区飞行

```bash
bash scripts/demo.sh --case city_300 --keep-open
```

启动一轮新的自主飞行，默认预算 480 秒；任务成功落地后保留 Gazebo 窗口。
每次都会根据本轮观测重新规划，不保证重复上一次的蓝线。查看同一次历史路径请用第 3 项回放。

### 2. 停止飞行、重新运行
在另一个项目终端执行：

```bash
bash scripts/stop.sh
```
正常飞完后也可以关闭 Gazebo，等待原启动终端退出，再执行第 1 项重新飞行。

### 3. 历史轨迹回放

首次使用仓库附带的两份记录时，在项目根目录解压（每份只需一次；已有同名目录会跳过）：

```bash
for recording in examples/recordings/*.tar.gz; do
  target="runs/$(basename "$recording" .tar.gz)"
  if [ ! -e "$target" ]; then
    tar -xzf "$recording"
  fi
done
```

较新的 11 月 18 日 13:36 那轮（265.4 秒送达，无恢复动作、无全局重规划）：

```bash
bash scripts/replay.sh runs/20261118_133653_949261_city_300_0
```

回放之前讨论的 20:36 那轮（第 5 个障碍实际穿洞）：

```bash
bash scripts/replay.sh runs/20261117_203650_676851_city_300_0
```

等待 Gazebo 加载，再在回放控制面板点击“播放”。支持播放/暂停、倍速、拖动进度、
从头和镜头切换；蓝线随进度增长，拖回较早时刻时蓝线同步截断。关闭任一回放窗口即可结束。

直接定位第 4、5 个障碍附近，分别运行需要的命令：

```bash
# 第 4 个障碍附近
bash scripts/replay.sh runs/20261117_203650_676851_city_300_0 --start 225
# 第 5 个障碍附近，半速
bash scripts/replay.sh runs/20261117_203650_676851_city_300_0 --start 265 --speed 0.5
# 对比之前讨论的 21:09 那轮（未附带，需本机原始记录）
bash scripts/replay.sh runs/20261117_210947_206077_city_300_0 --start 255
```

回放只读原实验数据，不启动 PX4、不重新规划；使用估计航迹复现历史蓝线与策略，
不还原完整姿态和动力学。回放检查已通过，真实图形窗口联调仍待用户测试。
详细说明见 [REPLAY.md](REPLAY.md)。

### 4. 调整镜头

观看飞行时，点击 Gazebo 三维画面后按 **Esc** 解除跟随；本轮不会自动锁回。
**Shift + 左键拖动**或按住滚轮拖动旋转，左键拖动平移，滚轮滚动缩放。
鼠标需放在三维画面内，避免在侧栏操作。

从启动时选择镜头，下面两条分别运行：

```bash
# 自由视角
bash scripts/demo.sh --case city_300 --camera free --keep-open
# 较高的巡视角度
bash scripts/demo.sh --case city_300 --camera view --keep-open
```

当前飞行镜头设置与 [run.py](urban_drone/run.py) 一致：

| 参数 | 相对机体偏移 (x, y, z)，单位 m | 效果 |
|---|---|---|
| `--camera follow`（默认） | `(-2, 0, 2)` | 后上方近距离跟随 |
| `--camera view` | `(-5, -5, 8)` | 右后上方较高视角 |
| `--camera free` | 不设置跟随 | 手动观察 |

在 `run.py` 中搜索 `camera_offsets` 可调整机位：负 X 为后方，负 Y 为右侧，正 Z 为上方。
保存后下次启动生效，无需重新编译。固定的是相对机位，镜头会随无人机移动。
启动时会等待 GUI 连接和目标模型加载；目标被 GUI 暂时清空时，仅在启动阶段重试绑定。
结果保存在本轮 `camera_status.json`（包含尝试次数）；
只有确认后才输出 `Camera confirmed`。目标和跟随状态稳定确认后停止重试，按 Esc 后保持手动。
已在独立 Gazebo GUI 实测 follow/view 随目标移动、free 保持原位，
记录见 [相机验证](docs/camera_follow_validation.json)；未启动 PX4 或飞行。
回放镜头由独立面板控制，回放命令的高视角参数是 `--camera overview`，与飞行的 `view` 不同。
鼠标操作参见 [Gazebo Harmonic GUI](https://gazebosim.org/docs/harmonic/gui/)。

### 5. 查看飞行结果与日志

```bash
# 查看最近生成完整报告的运行目录
cat runs/latest.txt
# 打开该轮完整轨迹图
xdg-open "$(cat runs/latest.txt)/flight_report.png"
# 查看该轮任务和评测结果
cat "$(cat runs/latest.txt)/result.json"
```

每轮文件放在 `runs/时间_场景_种子/`；中途停止的运行可能没有完整报告，也可能不更新 `latest.txt`。

| 文件 | 查看内容 |
|---|---|
| `flight_report.png` | 初始红线、完整实际轨迹与高度 |
| `result.json` | 任务是否送达、净空、落点误差等 |
| `events.jsonl` | 策略切换、候选评分、淘汰原因与制动事件 |
| `estimated_trajectory.csv` | 估计航迹、策略与阶段；回放数据源 |
| `ground_truth_trajectory.csv` | 独立评测真值轨迹 |
| `command_trace.csv` | 实际发送的位置目标及制动状态 |
| `mission.log` / `px4.log` / `gazebo.log` | 规划任务、飞控、仿真器日志 |
| `source/` / `run_manifest.json` | 本轮冻结源码、模型、参数与摘要 |

### 6. 修改后的算法与接口检查

```bash
bash scripts/check.sh
```

当前检查集共119项，另含Python编译与脚本语法检查；新增从实时位置向前接续、减速停止、后撤授权的回归，详见 [FORWARD_ADJUSTMENT_FIX.md](FORWARD_ADJUSTMENT_FIX.md)。
这些检查不启动飞行，也不能替代完整 SITL 验收。

针对“单侧墙距 1.1 米”的离线检查：

```bash
source scripts/env.sh
python3 scripts/probe_wall_clearance.py
```

分别检查参考点距离与外壳净空，包含合成激光可见性、方向制动、新规划接出，
以及真实控制方法在理想位置响应下连续前进。结果写入
[wall_1p1_probe.json](docs/wall_1p1_probe.json)，不启动 PX4/Gazebo，不验证真实飞行动力学。

针对“穿洞时左壁 1.1 米”的四面门洞检查：

```bash
source scripts/env.sh
python3 scripts/probe_gate_clearance.py
```

洞宽 4.5 m、高 4 m、墙厚 1.5 m；包含左右壁、洞顶和洞底。
分别使用表面点云、合成激光经真实 0.30 m 体素建图，按 5 Hz 扫描/20 Hz 控制验证。
结果见 [gate_left_1p1_probe.json](docs/gate_left_1p1_probe.json)，仍是理想跟踪的离线测试。

### 7. 其他场景、RViz 与无界面运行

下列每条命令分别启动一次实验，结束上一轮后再选择下一条：

```bash
bash scripts/demo.sh --case wall_low
bash scripts/demo.sh --case wall_high
bash scripts/demo.sh --case gate_open --rviz
bash scripts/demo.sh --case gate_narrow
bash scripts/demo.sh --case delivery --timeout 240 --rviz --keep-open
bash scripts/demo.sh --case city_300 --headless
```

| 常用参数 | 用途 |
|---|---|
| `--rviz` | 显示 RViz 地图、估计位置和路径 |
| `--headless` | 不开 GUI，仍执行真实 PX4 SITL |
| `--timeout 240` | 设置任务预算，单位秒 |
| `--seed 7 --jitter 1` | 固定种子并调整未知障碍横向位置 |
| `--height 7` | 小场景单墙高度覆盖 |
| `--opening-width 1` | 小场景门洞宽度覆盖 |

小场景参数在 [config/scenes.json](config/scenes.json)，支持 `empty`、`wall_low`、`wall_high`、
`building`、`gate_open`、`gate_narrow`、`delivery`；城区参数在 [config/city_300.json](config/city_300.json)。

### 8. 只生成场景预览，不飞行

```bash
source scripts/env.sh
python3 -m urban_drone.scene --case city_300 --out generated/city_300
```

小场景可将 `--case` 换成 `wall_low` 等名称。预览图不代表已经完成实飞。
回放数据准备也可以单独检查，不启动 Gazebo：

```bash
bash scripts/replay.sh runs/20261117_203650_676851_city_300_0 --prepare-only
```

### 9. 批量实验与重新生成报告图

```bash
source scripts/env.sh
# 只打印矩阵，不飞行
python3 -m urban_drone.benchmark --cases wall_low wall_high gate_open gate_narrow --seeds 0 1
# 显式执行矩阵，会依次启动多轮 SITL
python3 -m urban_drone.benchmark --cases wall_low wall_high gate_open gate_narrow \
  --seeds 0 1 --baselines full side_only global_only --execute
```

只重画历史报告图，不覆盖原结果或试验账本：

```bash
python3 -m urban_drone.report runs/20261117_203650_676851_city_300_0 --plot-only
```

## 项目与当前版本

独立的 Python + PX4 SITL + Gazebo Harmonic + ROS2 Jazzy/MAVROS 项目。
原有 `/mnt/d/px4_sim` 演示保留。

**当前为从当前位置调整后续路径的修正版，119项检查通过，等待用户新一轮SITL测试**。
最新用户飞行已送达，但有墙前和柱旁回退，原因见 [BACKTRACK_REVIEW.md](BACKTRACK_REVIEW.md)。
本次改为安全时先向前减速、再停止，跳过身后旧节点，确认路径堵塞且无非后撤连接后才允许后撤，
详见 [FORWARD_ADJUSTMENT_FIX.md](FORWARD_ADJUSTMENT_FIX.md)。原始 [对照图](docs/latest_backtrack_review.png) 保留。
修正同一轮后评OVER可能因快照过期而被淘汰的问题，规划使用固定评估时刻，提交/执行仍实时验证，
详见 [PLANNING_CLOCK_FIX.md](PLANNING_CLOCK_FIX.md)。尚未切换为纯几何代价与主动探索模型。
全局路线仅在确认不可行，或计入换路代价后至少改善 15% 且不少于 5 加权秒、经过多次确认时替换，
详见 [GLOBAL_REPLAN_POLICY.md](GLOBAL_REPLAN_POLICY.md)。备选搜索不等于已替换工作路线，红线仍保留初始规划。
三维方向威胁、实际速度的停止路径、可行旧路线持续执行与紧急恢复见 [DIRECTIONAL_SAFETY.md](DIRECTIONAL_SAFETY.md)。
第一道墙顶停住的原因与前阶段修正见 [ESCAPE_POLICY.md](ESCAPE_POLICY.md)。
墙顶/洞口重复后撤的记录与修正范围见 [RECOVERY_RESUME_FIX.md](RECOVERY_RESUME_FIX.md)。
当前普通执行最低参考点距离 **0.50 m**（外壳净空 **0.15 m**），新路径 **1.20 m**，偏好 **1.80 m**；
参数、机体尺寸折算与限制见 [CLEARANCE_POLICY.md](CLEARANCE_POLICY.md)。方向限速与原航线合流版
在完整实飞中结果变差（18 次全局重规划、落点误差 145.65 m）已回退；其参数与
回退记录见 [MOTION_RETURN_UPDATE.md](MOTION_RETURN_UPDATE.md)。

目标：同一套算法依据真实仿真传感器观测，比较越过、左右侧绕、穿洞，
安全回归主航线并配送降落。未知障碍不会提前写进航点。

## 分阶段验收

1. **已验收**：参数化楼宇/围挡/门架，先验与真值分离；A* 和碰撞检查后的轨迹平滑。
2. **已验收**：真实三维激光 → MAVROS 位姿变换 → 局部占据地图 → 在线选择 → PX4 实飞。
3. **已验收**：高低围挡、宽窄门洞与楼形障碍；46 m 组合配送完成越过、侧绕、穿洞和落地。
4. **已实现并小规模实测**：固定种子批量运行、对照组、失败归因与结果统计。
   冻结控制代码的 5 个扰动场景全部通过，完整统计见 [VALIDATION.md](VALIDATION.md)。

正式大样本送达率实验仍需另行运行；当前小规模回归不作为 95% 送达率证明。
实机 Loco/Flow 硬件验证属于独立实验，本项目不会伪造实机结果。
实现细节、假设和后续扩展见 [DESIGN.md](DESIGN.md)。

已修复黄色柱子前不必要的换边：探测到开口后，不再把该通道从侧绕搜索中人为删除；
同一次规划使用一致地图快照；同侧机动以成本和切换门槛保持稳定，不再排除更优的越障候选。该阶段 41 项飞行算法检查通过，
定位过程与实际轨迹对照见 [SIDE_SWITCH_FIX.md](SIDE_SWITCH_FIX.md)。

最新迭代加入局部相切圆弧、曲率限速、连续指令过渡，以及对 OVER/PASS/SIDE 的
统一预测耗时与风险评分。三个固定点的圆曲率唯一，圆弧半径通过切入/切出位置调整，
且必须满足观测净空。最新城区实飞已送达：加速度与 jerk 降低，耗时增加，
仍有局部换路与等待；完整权衡、算法说明和验证结果见 [TRAJECTORY_REVIEW.md](TRAJECTORY_REVIEW.md)。

## 城区场景：300×120 米长条区域

新增 `city_300`，参数见 [config/city_300.json](config/city_300.json)。
只保留航线附近的街区建筑，种子 0 有 **23 栋已知建筑、5 处红色未知障碍**：
2 道墙、1 根圆柱、2 道带偏置洞口的宽墙。带洞墙由多个碰撞盒组成，合计仍算一处障碍。
道路、路面标线和起终点平台只用于展示；红色材质不会作为算法输入。
裁去远离航线的空白地面；建筑整体降低约 10 米，最低保留 3 米。

全局 A* 在固定高度规划：取已知建筑高度的第 80 百分位，再加 2 米净空。
种子 0 的巡航高度为 **13 米**，高于 **82.6%** 的已知建筑，初始航线约 **241 米**。
低于巡航平面且满足净空的屋顶允许飞越，高楼仍需要全局绕行。
局部规划根据真实激光观测自主选择升高、侧绕或穿洞，随后回归巡航高度；
五处障碍只在场景生成阶段布置，不预先写入飞行节点的地图或策略。
位姿仍在执行净空内时，局部规划先用前进/侧向短连接接到正常规划起点；
侧向与越障搜索只允许前进和同层横移，代价同时惩罚平均与最紧净空，穿洞候选
优先走测得开口中心。净空恢复只在所有普通候选失败后作为最后手段，并优先
初始红线前方/前上方，再考虑横向和竖直；必要时仍可安全后撤。
已经过近时优先验证向外脱离，恢复后保留机动上下文；不能保证所有夹角均有可行出口。
到达目标后，估计位置需连续 4 秒保持在目标 0.2 米内才请求自动降落，
避免巡航尚未减速就切换降落模式；最终仍由独立真值评测检查 1 米落点误差。

| 障碍 | 几何尺寸 | 洞口与初始航线 |
|---|---|---|
| 第 1、4 处墙 | 宽 14 m，高 14.2 m | 无洞 |
| 第 2 处圆柱 | 直径 5 m，高 38 m | 无洞 |
| 第 3 处较宽高墙 | 宽 20 m，高 25 m | 洞宽 4.5 m、高 6 m，中心侧移 +4.2 m |
| 第 5 处低矮宽墙 | 宽 38 m，高 15.5 m | 洞宽 4.5 m、高 4 m，中心侧移 −4.2 m |

两个洞口都不包含初始航线；初始路线在该处穿过实体墙面，必须在线调整。
第一道带洞墙提供穿洞/侧绕选择，最后一道低矮宽墙提供越过/穿洞选择；
具体策略由观测、可行性与代价决定，不能把几何设计当作预设飞行动作。



预览：[城区与初始航线](generated/city_300/scene_overview.png)、
[五处障碍正面与初始巡航位置](generated/city_300/obstacle_profiles.png)。这些图只表示场景设计，
不是已完成飞行的记录。实时蓝线仍从空白开始随飞行延长。
当前版本的实际飞行验收状态见 [CITY_VALIDATION.md](CITY_VALIDATION.md)。
原有 `delivery` 等小场景继续保留；`city_1km` 作为未验收的大区域扩展配置保留。

## 运行与可视化说明

每条飞行命令独立管理本轮 Gazebo、PX4、MAVROS、任务与评测进程。
发现已有 PX4 时会拒绝启动，不会使用宽泛 pkill。
飞行节点运行本轮 `source/` 快照，编辑项目源文件不会改变已运行的实验。
使用 ROS_DOMAIN_ID=42 和 GZ_PARTITION=urban_delivery，避免接入旧任务节点。

Gazebo 和 RViz 中，**红线是起飞前的全局规划路线，蓝线是实际飞行的估计轨迹**。
红线保持初始规划，后续重规划不会覆盖它；蓝线根据 MAVROS 实测位姿逐段延长。
每次飞行蓝线从空白开始，不提前显示后续航段，也不加载历史轨迹。
RViz 的当前局部候选路线默认隐藏，可在 Displays 中启用黄色的 Active local route。
球形标记与文字显示飞机位置、任务阶段和当前策略。可视化节点只读观测与事件，
Gazebo 轨迹绘制在独立工作线程运行，不等待或修改飞控设定点。
导出的 `flight_report.png` 使用红色初始规划线、蓝色 Gazebo 真值轨迹线，展示记录的完整轨迹。

## 证据与验收标准

每次尝试保存在 `runs/时间_场景_种子/`，失败也保留。
包含原始日志、估计和真值轨迹、候选策略代价、实际安全间距、到点误差、
`result.json` 与 `flight_report.png`。`runs/trials.csv` 汇总实际记录。
每次运行冻结 `source/` 源码与模型快照并记录哈希，后续编辑不会改变该次运行。



`global_only` 仍保留感知制动，遇障碍会停止并超时，不能为了对照而主动撞机。
`side_only` 禁用越障候选，保持巡航高度；开口直接可通过时允许正常跟踪。
成功要求：到达目标、落地锁定、独立真值有效、无球形飞机包络侵入障碍、
最终平面误差不超过 1 m。碰撞指标是保守几何包络检查，不冒充接触传感器。
开发调试样本与最终固定版本的统计实验应分开报告。

生成目录中的 `prior.json` 是算法唯一可读取的环境文件；
`truth.json` 与 SDF 只供仿真器、评测器使用。

## 数据边界

- 定位来自 MAVROS 估计位姿；激光数据由 Gazebo Transport Python 接收。
- 不使用 `LaserScan.world_pose`、实体真值或场景中的未知障碍尺寸控制飞行。
- 接收时刻将扫描与最近位姿配对，超过 0.2 秒不融合；这是初版时间同步近似，
  并非已完成的硬件级时间同步。需要记录实际观测延迟。
- 未知空间可用于候选轨迹搜索，但执行前必须被最近扫描确认，不能当作已知自由空间。
- 用球形飞机包络加安全余量做局部碰撞检查，全球地图保守膨胀楼宇占地。
- 主城区与小场景规划另留 0.70 m 跟踪余量，执行检查与规划余量分开，避免跟踪切角后卡在缓冲区。
- 当前范围传感器未注入额外测距噪声；PX4 模型原有传感器噪声保留。
  固定种子同时传给场景生成器与 Gazebo，不承诺 ROS2 调度的逐位确定性。
- 欧氏启发函数仅保证所定义栅格代价下的 A* 最优性；平滑和局部候选不宣称全局最优。

## 环境

首次获取源码：

```bash
git clone https://github.com/Chilingyue/urban-drone-demo.git ~/urban_drone_demo
cd ~/urban_drone_demo
```

本仓库提供演示代码；ROS2 Jazzy、MAVROS、Gazebo Harmonic、PX4 SITL及相关Python依赖需在本机准备好。
本机验证环境的版本记录见 [ENVIRONMENT.json](ENVIRONMENT.json)。

使用已有 `~/PX4-Autopilot/build/px4_sitl_default/bin/px4`，无需重新编译固件。
世界文件继承已验证的 Zurich 地理原点、地磁场及磁力计 ENU/tesla 选项。
已有 SITL 的 `NAV_DLL_ACT=0` 是用户明确授权的无地面站演示配置；
本项目不会自动更改飞控安全参数，且仅供 SITL。

官方参考：

- https://gazebosim.org/api/transport/13/python.html
- https://gazebosim.org/docs/harmonic/sensors/
