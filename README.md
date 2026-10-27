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

需要自备两类大文件（均被 `.gitignore` 排除，不随仓库分发）：

| 位置 | 内容 |
|---|---|
| `weights/*.pt` | YOLO 权重；本项目使用 `weights/yolo11s-ball.pt`（单类 `badminton`） |
| `videos/*.mp4` | 待处理视频；示例为 `videos/badminton.mp4`、`videos/badminton01.mp4` |

## 运行

```bash
# 最简：输出到 outputs/track_<时间戳>.mp4，并写出同名轨迹 CSV
python scripts/run_track.py --video videos/badminton.mp4 --weights weights/yolo11s-ball.pt

# 无窗口 / 批处理（服务器、CI 必用）
python scripts/run_track.py --video videos/badminton.mp4 --weights weights/yolo11s-ball.pt --no-gui

# 只跑前 100 帧做调试
python scripts/run_track.py --video videos/badminton.mp4 --weights weights/yolo11s-ball.pt --no-gui --limit 100

# 两阶段用法（跟踪已拆成"先分析、再渲染"两遍扫描）
# ① 只分析：产出 3 个 CSV + analysis.json，不生成视频（约 16s）
python scripts/run_track.py --video videos/badminton.mp4 --weights weights/yolo11s-ball.pt \
    --output outputs/run.mp4 --analyze-only --no-gui
# ② 只渲染：复用上一步的 analysis.json，跳过 YOLO（约 24s，可反复重跑调试视觉）
python scripts/run_track.py --video videos/badminton.mp4 \
    --output outputs/run.mp4 --render-from outputs/run.analysis.json --no-gui
```

**两阶段结构**（默认行为不变，一条命令仍是"分析 → 渲染"）：

| 阶段 | 模块 | 做什么 | 产物 |
|---|---|---|---|
| 分析 | `src/analysis.py` `analyze_video()` | YOLO 检测、候选/静态标记、关联、门控、分段、RANSAC | CSV + `AnalysisResult`（可存 JSON） |
| 渲染 | `src/render.py` `render_video()` | 只按分析快照重放视觉层并写 mp4 | 视频 |

好处：改渲染（配色、标注、慢动作）不用重跑 YOLO；后续接入 Kalman/批优化时，分析阶段可以直接变成"两遍扫描"的第一遍。

- 可从**任意工作目录**调用（脚本会自动定位项目根），不再需要设置 `PYTHONPATH`。
- 相对路径一律以**项目根**为基准解析。
- `--output` 的父目录会自动创建；若输出无法创建（路径不可写、编码器缺失），会**直接报错退出**，不会静默产生空结果。

| 参数 | 默认 | 说明 |
|---|---|---|
| `--config` | `configs/track_cfg.yaml` | 配置文件 |
| `--video` | 必填 | 输入视频 |
| `--weights` | 必填 | YOLO 权重（`--render-from` 模式可省略） |
| `--output` | `outputs/track_<时间戳>.mp4` | 输出视频（`--analyze-only` 时只用作 CSV/JSON 命名基准） |
| `--no-gui` | 关 | 不弹 OpenCV 窗口 |
| `--no-traj` | 关 | 不写轨迹 CSV |
| `--limit` | 全部 | 只处理前 N 帧 |
| `--analyze-only` | 关 | 只跑分析阶段，写 CSV + analysis.json |
| `--render-from` | 空 | 从已有 analysis.json 渲染（跳过 YOLO） |
| `--analysis-json` | 空 | 另存分析结果 JSON；`--analyze-only` 默认 `<输出主干>.analysis.json` |

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

`configs/track_cfg.yaml` 分三组：`tracker`（平滑/关联/阈值）、`ransac`（抛物线清洗）、`detector`（候选过滤）。参数含义、实测灵敏度与已知失效项见 `docs/PROJECT_ANALYSIS.md` §3、§5。

## 已知限制

- 关联仍是单目标启发式：代价函数量纲失配，实际近似最近邻；λ 自适应分支在 50fps 素材上不会触发。
- 离群门控基于平滑点计速，阈值实际退化为常数 60px，实测约 20% 帧的观测被丢弃。
- 抛物线清洗假设 `y=f(x)`，对左右往返的整段轨迹不成立（保留率可低至 44%）。
- 单目标、无轨迹 ID、无球员/球场坐标、无击球事件。
- 详细数据与改进优先级见 `docs/PROJECT_ANALYSIS.md` §5、§8。
