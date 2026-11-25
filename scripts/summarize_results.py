"""Summarize retained real-run artifacts without dropping failed attempts."""

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / 'runs'
CONTROL_FILES = ['urban_drone/mission.py', 'urban_drone/local.py',
                 'urban_drone/perception.py', 'urban_drone/planning.py']


def main():
    attempts = []
    for path in sorted(RUNS.glob('*/result.json')):
        result = json.loads(path.read_text())
        attempts.append((path.parent, result))
    regression = RUNS / 'regression_5cases.csv'
    rows = list(csv.DictReader(regression.open())) if regression.exists() else []
    text = ['# 实际飞行验收记录', '', '所有数值来自保留的真实 PX4 SITL 运行文件。', '',
            '## 固定版本回归', '',
            '种子 1、未知障碍横向扰动范围 ±0.4 m；以下是小规模回归，不是 95% 送达率的统计证明。', '',
            '| 场景 | 结果 | 在线策略 | 最小包络间距/m | 最终误差/m |',
            '|---|---|---|---:|---:|']
    versions = []
    for row in rows:
        directory = Path(row['run'])
        result = json.loads((directory / 'result.json').read_text())
        versions.append(json.loads((directory / 'run_manifest.json').read_text())['source_sha256'])
        strategies = ', '.join(dict.fromkeys(result['mission']['strategies']))
        text.append(f'| {row["case"]} | {"通过" if result["success"] else "失败"} | {strategies} | '
                    f'{result["evaluation"]["minimum_clearance"]:.3f} | '
                    f'{result["evaluation"]["goal_xy_error"]:.3f} |')
    if versions:
        same = all(all(v[k] == versions[0][k] for k in CONTROL_FILES) for v in versions)
        text += ['', f'控制模块源码哈希一致：{same}。']
    delivery = [(p, r) for p, r in attempts if r['case'] == 'delivery' and r['success']]
    if delivery:
        directory, result = delivery[-1]
        text += ['', '## 组合配送', '',
                 f'- 完成真实 46 m 配送、在线越过/侧绕/穿洞、AUTO.LAND 和落地锁定。',
                 f'- 耗时 {result["mission"]["elapsed"]:.1f} s，实际飞行长度 '
                 f'{result["evaluation"]["path_length"]:.2f} m。',
                 f'- 最小包络间距 {result["evaluation"]["minimum_clearance"]:.3f} m，'
                 f'最终平面误差 {result["evaluation"]["goal_xy_error"]:.3f} m。',
                 f'- 记录：`{directory.relative_to(ROOT)}`。',
                 f'- [飞行轨迹报告]({directory.relative_to(ROOT)}/flight_report.png)。']
        margin = json.loads((directory / 'prior.json').read_text())['safety_margin']
        if result['evaluation']['minimum_clearance'] < margin:
            text += ['', f'该次组合任务虽然无包络侵入并完成送达，但最小间距低于 {margin} m '
                     '的配置目标，不能把规划余量等同于实际净空。后续需针对狭窄通道的感知误差、'
                     '跟踪切角收紧验证标准；这条不达间距目标的记录完整保留。']
    comparable = {}
    for directory, result in attempts:
        if result['case'] == 'wall_low' and result['seed'] == 1:
            truth = json.loads((directory / 'truth.json').read_text())
            if truth['jitter'] == 0.4:
                comparable[result['baseline']] = (directory, result)
    if len(comparable) == 3:
        values = list(comparable.values())
        reference = json.loads((values[0][0] / 'truth.json').read_text())['objects']
        for directory, _ in values:
            assert json.loads((directory / 'truth.json').read_text())['objects'] == reference
        text += ['', '## 相同低墙场景的基线对照', '',
                 '| 方法 | 送达 | 实际路程/m | 耗时/s | 说明 |',
                 '|---|---|---:|---:|---|']
        for baseline in ['full', 'side_only', 'global_only']:
            directory, result = comparable[baseline]
            text.append(f'| {baseline} | {result["success"]} | '
                        f'{result["evaluation"]["path_length"]:.2f} | '
                        f'{result["mission"]["elapsed"]:.1f} | '
                        f'{result["mission"]["failure"] or ", ".join(dict.fromkeys(result["mission"]["strategies"]))} |')
        text += ['', 'global_only 保留传感器制动；停滞后安全降落，失败不被隐去。',
                 '三组的场景真值完全相同；单次对照只能说明该测试用例中的表现。']
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, (plan, altitude) = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
        colors = dict(full='#16824b', side_only='#3166b4', global_only='#c05243')
        for baseline, (directory, result) in comparable.items():
            p = __import__('numpy').loadtxt(directory / 'ground_truth_trajectory.csv', delimiter=',')
            plan.plot(p[:, 1], p[:, 2], label=baseline, color=colors[baseline])
            altitude.plot(p[:, 0], p[:, 3], label=baseline, color=colors[baseline])
        wall = next(b for b in reference if not b['known'])
        from matplotlib.patches import Rectangle
        x, y, _ = wall['center']; dx, dy, _ = wall['size']
        plan.add_patch(Rectangle((x-dx/2, y-dy/2), dx, dy, color='#e89641', alpha=0.6))
        plan.set(xlabel='East / m', ylabel='North / m', title='Same unknown wall, different planners')
        plan.axis('equal'); plan.grid(alpha=0.2); plan.legend()
        altitude.set(xlabel='Elapsed / s', ylabel='World body height / m', title='Actual flight altitude')
        altitude.grid(alpha=0.2); altitude.legend()
        fig.savefig(RUNS / 'baseline_comparison.png', dpi=150)
        plt.close(fig)
        text += ['', '[基线飞行对照图](runs/baseline_comparison.png)。']
    failures = sum(not result['success'] for _, result in attempts)
    text += ['', '## 开发过程与统计口径', '',
             f'当前保留 {len(attempts)} 次有评测结果的运行，其中 {failures} 次未送达；'
             '包含早期调试与预期失败的基线，不能把不同算法版本混在一起估计送达率。',
             '完整尝试见 `runs/trials.csv`，每次失败原因见对应 `result.json`。',
             '早期问题包括起飞等待误计入无进展超时、激光机身自遮挡、楼角跟踪余量不足；'
             '相关失败原始日志保留，修复后另行试飞验收。',
             '95% 仿真送达率和 85% 实机送达率均未作为本项目已验证结论。', '']
    (ROOT / 'VALIDATION.md').write_text('\n'.join(text))
    print(f'Wrote {ROOT / "VALIDATION.md"}')


if __name__ == '__main__':
    main()
