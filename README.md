# BadmintonSight

YOLO + MOT-style tracker for shuttlecock tracking.

> 项目结构、算法细节、实测数据与改进路线见 `docs/PROJECT_ANALYSIS.md`（该文档与 `_probe/` 被 `.gitignore` 排除，仅本地保留）。

## Features

- Static noise suppression for fixed reflection points
- Adaptive λ motion/appearance cost fusion
- Linear velocity prediction association
- Dynamic frame distance threshold
- RANSAC parabola trajectory cleaning
- Fading historical point visualization

## Setup

```bash
pip install -r requirements.txt
```

## 下载权重与演示视频

本仓库**不把二进制素材放进 git**（保持克隆轻量），模型权重与示例视频通过 **GitHub Releases** 分发：

| 文件 | 大小 | 放到 | SHA256 |
|---|---|---|---|
| [`yolo11s-ball.pt`](https://github.com/codezhangxiong/BadmintonSight/releases/download/v0.1.0-assets/yolo11s-ball.pt) | 18.3 MB | `weights/yolo11s-ball.pt` | `C21113960FADCE7F96B9F70F86802142482775405F60A9761328491868EDF31B` |
| [`badminton.mp4`](https://github.com/codezhangxiong/BadmintonSight/releases/download/v0.1.0-assets/badminton.mp4) | 2.8 MB | `videos/badminton.mp4` | `435E42F42C27D1BEDE9B1690BA230386EB066FA2072188197B84F9AF5A2E87EE` |
| [`badminton01.mp4`](https://github.com/codezhangxiong/BadmintonSight/releases/download/v0.1.0-assets/badminton01.mp4) | 8.4 MB | `videos/badminton01.mp4` | `390980C1A96400936F92E84821F6B0424B56712ACF6CB60772AC84C7D2205E99` |

> 跑通默认示例只需要 `yolo11s-ball.pt` + `badminton.mp4`；`badminton01.mp4` 是备用素材。
> 若 Release 尚未发布、或你使用了别的 tag，把链接里的 `v0.1.0-assets` 换成实际 tag，或直接到仓库 **Releases** 页面下载。

下载后放到对应目录，并校验（PowerShell）：

```powershell
Get-FileHash weights\yolo11s-ball.pt, videos\badminton.mp4, videos\badminton01.mp4 -Algorithm SHA256
```

> 这些文件都被 `.gitignore` 排除，`git status` 里不会出现；`weights/`、`videos/` 目录需要自己创建。

## 运行

```bash
# 最简：输出到 outputs/track_<时间戳>.mp4，并写出同名轨迹 CSV
python scripts/run_track.py --video videos/badminton.mp4 --weights weights/yolo11s-ball.pt

# 无窗口 / 批处理（服务器、CI 必用）
python scripts/run_track.py --video videos/badminton.mp4 --weights weights/yolo11s-ball.pt --no-gui

# 只跑前 100 帧做调试
python scripts/run_track.py --video videos/badminton.mp4 --weights weights/yolo11s-ball.pt --no-gui --limit 100

# 三段式局部运行（采集 → 段处理 → 渲染）
# ① 只采集 + 段处理：产出 3 个 CSV + capture.json，不生成视频（约 18s）
python scripts/run_track.py --video videos/badminton.mp4 --weights weights/yolo11s-ball.pt \
    --output outputs/run.mp4 --capture-only --no-gui
# ② 只重做段处理：调门控阈值 / max_miss / ransac 参数后重算决策与分段（约 0.6s，不加载 YOLO）
python scripts/run_track.py --segments-from outputs/run.capture.json \
    --output outputs/run.mp4 --config configs/track_cfg.yaml
# ③ 只渲染：复用 capture.json 重放视觉层（约 24s，不加载 YOLO）
python scripts/run_track.py --video videos/badminton.mp4 \
    --output outputs/run.mp4 --render-from outputs/run.capture.json --no-gui
```

**三段式结构**（默认行为不变，一条命令仍是"采集 → 段处理 → 渲染"）：

| 阶段 | 模块 | 做什么 | 产物 |
|---|---|---|---|
| 采集 | `src/capture.py` `capture_video()` | 逐帧 YOLO 检测、候选过滤、静态标记；**只记录每帧全部候选，不做跟踪决策** | `CaptureResult`（可存 JSON） |
| 段处理 | `src/segments.py` `resolve_tracks()` | 重放**关联 → 预测 → 门控 → 平滑**，再切段（连续丢帧规则）、RANSAC 清洗 | `traj.csv` + `segments.csv` + `clean.csv` |
| 渲染 | `src/render.py` `render_video()` | 只按帧快照重放视觉层并写 mp4 | 视频 |

好处：**跟踪决策（含门控）全部在段处理阶段**。采集结果里存了每帧全部候选，因此调门控阈值、`max_miss`、RANSAC 参数、甚至更换关联策略都**不必重跑 YOLO**（实测 0.6s 重算全片决策与分段；把 `max_miss` 从 5 改成 2 只需 0.63s，段数 15→21）；改渲染（配色、标注、慢动作）同样只需重跑渲染阶段；后续接入 Kalman/批优化时，可以在段处理阶段替换"预测+门控+平滑"，采集与渲染都不用动。

启动耗时：首次跑检测约 **5 秒**才出第一帧（`import torch` 装载 CUDA 库约 2.2s + 首次 CUDA/cuDNN 初始化约 2.4s，与权重大小无关），期间会打印 `[采集] ...` 进度提示。`--render-from` / `--segments-from` 都**不加载 YOLO**，启动约 0.2 秒。

- 可从**任意工作目录**调用（脚本会自动定位项目根），不再需要设置 `PYTHONPATH`。
- 相对路径一律以**项目根**为基准解析。
- `--output` 的父目录会自动创建；若输出无法创建（路径不可写、编码器缺失），会**直接报错退出**，不会静默产生空结果。

| 参数 | 默认 | 说明 |
|---|---|---|
| `--config` | `configs/track_cfg.yaml` | 配置文件 |
| `--video` | 必填 | 输入视频（`--segments-from` 模式可省略） |
| `--weights` | 必填 | YOLO 权重（`--render-from` / `--segments-from` 模式可省略） |
| `--output` | `outputs/track_<时间戳>.mp4` | 输出视频（`--capture-only` 时只用作 CSV/JSON 命名基准） |
| `--no-gui` | 关 | 不弹 OpenCV 窗口 |
| `--no-traj` | 关 | 不写轨迹 CSV |
| `--limit` | 全部 | 只处理前 N 帧 |
| `--capture-only` | 关 | 只跑采集 + 段处理，写 CSV + capture.json |
| `--segments-from` | 空 | 从 capture.json 重做跟踪决策 + 分段（只写 CSV，跳过 YOLO） |
| `--render-from` | 空 | 从已有 capture.json 渲染（跳过 YOLO） |
| `--capture-json` | 空 | 另存采集+段处理结果 JSON；`--capture-only` 默认 `<输出主干>.capture.json` |

## 输出

一次运行产出 1 个视频 + 3 个 CSV（与视频同目录、同文件名主干）：

```
outputs/track_20260101_120000.mp4          可视化结果
outputs/track_20260101_120000.traj.csv     逐帧事件表
outputs/track_20260101_120000.segments.csv 轨迹段汇总
outputs/track_20260101_120000.clean.csv    清洗后的轨迹点
```

### `<stem>.traj.csv` — 逐帧事件

每帧一行，`event` 取值 `obs`（观测被接受）/ `rejected`（判为离群丢弃）/ `miss`（无观测）。

| 列 | 含义 |
|---|---|
| `frame` | 视频帧号（1-based，与日志一致） |
| `segment_id` | 当前**活跃**轨迹段编号；段外帧（首个观测之前、清段之后的丢帧）为空 |
| `event` | `obs` / `rejected` / `miss` |
| `obs_x`,`obs_y` | 原始观测坐标（像素，被拒帧也会记录，便于复盘门控） |
| `smooth_x`,`smooth_y` | 平滑后轨迹点（仅 `obs` 行有值） |
| `conf` | 候选置信度 |
| `bbox_x1..y2` | 候选框像素坐标 |
| `is_static` | 是否被静态光斑判定标记（1/0） |
| `dist_to_prev` | 与上一个平滑点的距离（离群判定依据） |
| `threshold` | 当帧使用的离群阈值 |

### `<stem>.segments.csv` — 段汇总

| 列 | 含义 |
|---|---|
| `segment_id` | 段编号（0 起） |
| `start_frame`,`end_frame` | 该段**首个/最后一个有观测的帧** |
| `n_raw` | 段内原始（平滑后）点数 |
| `n_cleaned` | RANSAC 清洗后保留点数 |
| `first_x..last_y` | 段首尾坐标 |

`n_raw <= ransac.min_samples` 的短段**不做清洗**，按原样保留（`n_cleaned = n_raw`），仍会写入；
更长段才跑 RANSAC 抛物线清洗。旧版本对短段也会清洗、且清洗结果不落盘（短段可能被清成 0 点）。

### `<stem>.clean.csv` — 清洗后轨迹

`segment_id, frame, x, y`，RANSAC 内点按原始时间顺序输出，可直接用于后续坐标变换/事件分析。

## 测试

```bash
python -m unittest discover -s tests -t .
```

覆盖 `detector` 的过滤/空输入、`cleaner` 的 mask 契约、`pipeline._clean_segment` 的短段清洗契约、`export` 的 CSV 落盘格式。
测试框架只用标准库 `unittest`（不依赖 pytest），测试代码会用到项目已有的 numpy。
端到端视频流程需要权重与 GPU，未纳入单元测试，按上面的"运行"章节手动验证。

## 配置

`configs/track_cfg.yaml` 分四块：`gates`（门控开关）、`tracker`（平滑/关联/门控参数/阈值）、`ransac`（抛物线清洗）、`detector`（检测置信度）。

### 门控开关 `gates`

每个门控都可以单独开关（`true`=启用、`false`=关闭并放行），用于消融实验：

| 开关 | 控制什么 | 关闭后的行为 | 需要重跑 YOLO |
|---|---|---|---|
| `yolo_conf` | YOLO 推理置信度阈值（`detector.yolo_conf`） | 传 `conf=0.0`，保留全部检测 | 是 |
| `detector_conf` | 候选置信度下限（`detector.conf_min`） | 不再过滤低置信候选 | 是 |
| `static_suppression` | 静态候选标记 + 120 惩罚 | 跳过 `mark_static_candidates` | 是 |
| `association` | 关联层区分度裁决（`tracker.assoc_*`） | 退化为纯代价最小（旧行为） | 否（`--segments-from` 即可） |
| `distance` | 距离门（`tracker.dynamic_*`） | 距离门放行全部观测 | 否（`--segments-from` 即可） |
| `motion_model` | 运动模型门（`tracker.motion_model.*`） | 跳过二次曲线残差判定 | 否 |
| `ransac` | 段后 RANSAC 清洗（`ransac.*`） | `n_cleaned == n_raw` | 否 |

实测消融（固定采集结果，仅 replay 层门控）：

| 配置 | 接受 | 拒绝(距离/运动) | 段数 | clean 保留率 |
|---|---|---|---|---|
| 全开 | 644 | 67 (25/42) | 7 | 52.5% |
| `association` OFF（旧行为） | 655 | 56 (26/30) | 5 | 46.0% |
| `distance` OFF | 668 | 43 (0/43) | 4 | 45.5% |
| `motion_model` OFF | 691 | 20 (20/0) | 2 | 27.8% |
| 两者 OFF | 711 | 0 / 0 | 1 | 23.8% |
| `ransac` OFF | 644 | 67 | 7 | 100% |

### 关联层（方案 C）

`src/tracker/association.py::select_best_candidate` 在代价最小之外增加"区分度裁决"：

- 当 top-1 与 top-2 的代价差 **< `tracker.assoc_margin`（默认 80）** 时，说明两个候选都说得通
  （典型：球与反光点在两组位置交替出现，预测点落在两者之间），按顺序裁决：
  1. **连续性优先**：次优候选到"上一个已接受观测"的距离若明显更近（近出 `assoc_cont_margin`，默认 30px），选它；
  2. 距离接近时用**置信度**决胜；
  3. 否则保持代价最小。
- 可选 `tracker.assoc_max_dist > 0`：剔除离预测点超过该距离的候选；若全部超限则保留全部（不硬拒绝，避免快球误丢）。
- **不要改成"conf 高者恒胜"**：实测高 conf 假阳性（球拍/反光点，conf≥0.8 但偏 100+px）会污染平滑窗口并引发连锁拒绝。

实测（全片 730 帧）：交替跳变帧 **10 → 3**，接受 655→644、段数 5→7。

### 门控链

**门控**由 `src/tracker/gating.py` 提供，串联两道（任何一道拒绝即丢弃该帧观测）：
1. `DistanceGate`：观测到上一个平滑点的距离超过动态阈值（`dynamic_gain/min/max_thresh`）。
2. `MotionModelGate`：对最近 `window`（默认 6）个已接受观测按时间分别拟合 `x(t)`、`y(t)` 二次曲线，外推当前帧，**残差 > `residual_thresh`（默认 80px）即拒绝**。击球/折返时（窗口拟合 RMS 过大，或残差大但方向反转）**重置拟合窗口并放行**，避免误杀击球点。参数在 `tracker.motion_model` 下。

候选提取不做外形（面积/长宽比）与 ROI 过滤——实测这两类门控在本数据上从未触发，已删除。

## 已知限制

- 关联仍是单目标启发式：代价函数量纲失配（conf 权重只有 1-λ≈0.15），已用"区分度裁决"补偿，但仍是启发式；λ 自适应分支在 50fps 素材上不会触发。
- 距离门控的 `dynamic_gain` / `dynamic_max_thresh` 实测无效（阈值几乎总贴 `dynamic_min_thresh`，当前取 200px，依据见下）。
- 运动模型门的残差阈值（80px）基于本机数据统计：正常接受帧残差 p50=1.6 / p90=54 / p95=90px；30px 会误杀、150px 接近失效。
- 抛物线清洗假设 `y=f(x)`，对左右往返的整段轨迹不成立；门控放宽后段变长，清洗保留率进一步下降（当前约 40~90% 不等），`clean.csv` 需谨慎使用。
- 单目标、无轨迹 ID、无球员/球场坐标、无击球事件。
- 详细数据与改进优先级见 `docs/PROJECT_ANALYSIS.md` §5、§8。
