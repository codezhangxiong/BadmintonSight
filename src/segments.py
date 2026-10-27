"""段处理阶段：跟踪决策 + 切段 + 清洗 + CSV 落盘。

三段式结构中的第二段：

    采集    analyze_video()      逐帧检测 + 候选记录（无状态、不判断）
    段处理  resolve_tracks()     本文件：关联 → 预测 → 门控 → 平滑 → 切段 → RANSAC 清洗
    渲染    render_video()       读原视频重放视觉层并写 mp4

为什么门控（以及关联/平滑）必须放在这里而不是采集阶段：
关联需要"历史轨迹"做预测（predict_linear），而历史轨迹只由"通过门控的观测"构成；
如果把门控单独留在采集端，它依赖平滑状态、平滑又依赖门控结果，形成循环依赖。
因此正确的职责划分是：**采集=感知（无状态），段处理=决策（有状态）**。

这样带来两个实际收益：
- 调门控阈值 / max_miss / ransac 参数、甚至更换关联策略，都**不需要重跑 YOLO**
  （--segments-from）：分析结果里已经存了每帧全部候选；
- 后续接入 Kalman / 批优化时，可以直接在 replay_decisions 里替换"预测+门控+平滑"，
  采集与渲染都不用动。
"""

from collections import deque

import numpy as np

from src.export import TrajectoryRecorder
from src.tracker.association import select_best_candidate
from src.tracker.cleaner import ransac_clean_parabola
from src.tracker.predictor import dynamic_step_threshold, predict_linear
from utils.common import distance

# 渲染阶段的历史点上限，必须与画图端 draw_fade_points(max_keep=30) 一致
TRAIL_MAX_KEEP = 30


# --------------------------------------------------------------------------- 候选还原
def _candidate_dict(row):
    """把采集记录的候选行还原成关联/导出所需的 dict。

    row = [x, y, is_static, confidence, area_ratio, aspect_ratio, x1, y1, x2, y2]
    area_ratio / aspect_ratio 还原成 np.float32：采集时它们来自 YOLO 的 float32，
    这样重放时 compute_cost 的浮点运算与采集阶段逐位一致，保证结果可复现。
    """
    return {
        "point": (int(row[0]), int(row[1])),
        "is_static": bool(row[2]),
        "confidence": float(row[3]),
        "area_ratio": np.float32(row[4]),
        "aspect_ratio": np.float32(row[5]),
        "bbox": (int(row[6]), int(row[7]), int(row[8]), int(row[9])),
    }


def _chosen_candidate(record):
    """找到本帧被选中候选的原始行（用于写 traj.csv 的 conf/bbox 列）。"""
    if record.chosen is None:
        return None
    for row in record.candidates:
        if int(row[0]) == int(record.chosen[0]) and int(row[1]) == int(record.chosen[1]):
            return _candidate_dict(row)
    return None


# --------------------------------------------------------------------------- RANSAC 清洗
def _clean_segment(points, frames, ransac_cfg):
    """对一段轨迹做 RANSAC 清洗。

    返回 (cleaned_points, cleaned_frames, mask)。mask 与输入等长且保序，
    因此可以直接用布尔掩码把内点映射回原始帧号。

    点数 <= min_samples 的短段**不做清洗**：RANSAC 至少需要 min_samples 个点，
    对恰好等量的输入没有冗余度，可能把整段判成离群而清空（实测 5 点含 1 个离群 → 0 点）。
    短段按原样返回，保证 segments.csv 的 n_cleaned == n_raw 契约成立。
    """
    if len(points) <= int(ransac_cfg["min_samples"]):
        return list(points), list(frames), np.ones(len(points), dtype=bool)
    cleaned, mask = ransac_clean_parabola(
        points,
        min_samples=ransac_cfg["min_samples"],
        max_iter=ransac_cfg["max_iter"],
        residual_thresh=ransac_cfg["residual_thresh"],
        return_mask=True,
    )
    cleaned_frames = [frames[i] for i in range(min(len(mask), len(frames))) if mask[i]]
    return cleaned, cleaned_frames, mask


def _build_segment(segment_id, current, ransac_cfg):
    """把一段 (frame, point) 序列清洗成 segments 行 + 清洗后点。"""
    points = [point for _, point in current]
    frames = [frame for frame, _ in current]
    cleaned, cleaned_frames, _ = _clean_segment(points, frames, ransac_cfg)
    segment = {
        "segment_id": segment_id,
        # 起止帧取"实际有观测的帧"，与轨迹点范围一致，便于和视频帧对齐
        "start_frame": frames[0],
        "end_frame": frames[-1],
        "n_raw": len(points),
        "n_cleaned": len(cleaned),
        "first_x": points[0][0],
        "first_y": points[0][1],
        "last_x": points[-1][0],
        "last_y": points[-1][1],
    }
    return segment, cleaned, cleaned_frames


# --------------------------------------------------------------------------- 决策重放
def replay_decisions(frames, cfg, verbose=True):
    """重放关联 → 预测 → 门控 → 平滑，把决策结果写回 FrameRecord。

    这是原采集循环里"决策部分"的等价实现（旧版逻辑逐行保留），区别只是
    观测输入来自 frames 里已记录的候选，而不是现场跑 YOLO。

    返回统计 dict：accepted / rejected / miss。
    """
    track_cfg = cfg["tracker"]
    smooth_win = deque(maxlen=track_cfg["smooth_window_len"])
    full_traj = []
    missing = 0
    MAX_MISS = track_cfg["max_miss"]
    accepted = 0
    rejected = 0
    missed = 0

    for record in frames:
        candidates = [_candidate_dict(row) for row in record.candidates]
        ref_pt = predict_linear(full_traj, n_lookback=4)
        best = select_best_candidate(candidates, ref_pt, full_traj, cfg)

        if best is not None:
            obs_pt = best["point"]
            record.chosen = [int(obs_pt[0]), int(obs_pt[1])]
            outlier = False
            d = None
            th = None
            if len(smooth_win) > 0:
                last_p = smooth_win[-1]
                d = distance(obs_pt, last_p)
                th = dynamic_step_threshold(
                    full_traj,
                    base_gain=track_cfg["dynamic_gain"],
                    min_thresh=track_cfg["dynamic_min_thresh"],
                    max_thresh=track_cfg["dynamic_max_thresh"],
                )
                if d > th:
                    outlier = True
            record.dist_to_prev = d
            record.threshold = th
            if outlier:
                missing += 1
                rejected += 1
                record.rejected = True
                record.event = "rejected"
                if verbose:
                    print(f"[Frame {record.frame}] OBS {obs_pt} outlier drop")
            else:
                missing = 0
                accepted += 1
                smooth_win.append(np.array(obs_pt))
                smooth_pt = tuple(np.mean(np.array(list(smooth_win)), axis=0))
                full_traj.append(smooth_pt)
                record.smooth = [float(smooth_pt[0]), float(smooth_pt[1])]
                record.event = "obs"
                if verbose:
                    print(f"[Frame {record.frame}] obs {obs_pt}, smooth {np.round(smooth_pt, 1)}")
        else:
            missing += 1
            missed += 1
            record.event = "miss"
            if verbose:
                print(f"[Frame {record.frame}] no obs, missing={missing}")

        record.trail = [[float(x), float(y)] for x, y in list(full_traj)[-TRAIL_MAX_KEEP:]]

        if missing > MAX_MISS:
            # 段结束：清空平滑/轨迹缓冲（分段与清洗在 collect_segments 里做）
            full_traj.clear()
            smooth_win.clear()
            missing = 0

    return {"accepted": accepted, "rejected": rejected, "miss": missed}


# --------------------------------------------------------------------------- 切段 + 清洗
def collect_segments(frames, cfg):
    """按 event 序列切段、分配 segment_id、RANSAC 清洗。

    分段规则与旧版"循环内 missing > max_miss 即触发"严格等价：
    连续丢帧（rejected 或 miss）计数超过 max_miss 时结束当前段；视频结束时收尾最后一段。

    返回 (segments, clean_map)：
    segments   —— 每段汇总（写入 segments.csv，也会存进 AnalysisResult）
    clean_map  —— {segment_id: (cleaned_points, cleaned_frames)}，用于写 clean.csv
    """
    max_miss = int(cfg["tracker"]["max_miss"])
    ransac_cfg = cfg["ransac"]

    segments = []
    clean_map = {}
    seg_id = 0
    active = None
    current = []
    missing = 0

    def flush():
        nonlocal seg_id, active, current
        if not current:
            active = None
            return
        segment, cleaned, cleaned_frames = _build_segment(seg_id, current, ransac_cfg)
        segments.append(segment)
        clean_map[seg_id] = (cleaned, cleaned_frames)
        seg_id += 1
        current = []
        active = None

    for record in frames:
        if record.event == "obs":
            missing = 0
            if active is None:
                active = seg_id
            record.segment_id = active
            current.append((record.frame, tuple(record.smooth)))
        else:
            record.segment_id = active
            missing += 1
            if missing > max_miss:
                flush()
                missing = 0

    # 视频结束：收尾最后一段
    if current:
        flush()

    return segments, clean_map


# --------------------------------------------------------------------------- 编排
def resolve_tracks(analysis, cfg, out_path=None, save_traj=True, verbose=True):
    """段处理阶段总入口：决策重放 → 切段清洗 → 写三份 CSV。

    参数
    ----
    analysis  采集结果（frames 里已记录候选；本函数填充决策字段）
    cfg       配置 dict（tracker / ransac / detector）
    out_path  输出视频路径；CSV sidecar 按它的目录/主干命名
    save_traj 是否写 CSV
    verbose   是否打印逐帧决策与段结束日志

    返回 analysis（原地更新并返回，便于链式调用）。
    """
    stats = replay_decisions(analysis.frames, cfg, verbose=verbose)
    segments, clean_map = collect_segments(analysis.frames, cfg)

    if verbose:
        for segment in segments:
            if segment["n_raw"] > int(cfg["ransac"]["min_samples"]):
                print(f"\n==== trajectory end, raw len: {segment['n_raw']} ====")
                print(f"==== cleaned len: {segment['n_cleaned']} ====\n")

    csv_paths = {}
    if save_traj and out_path:
        recorder = TrajectoryRecorder(out_path, enabled=True)
        try:
            for record in analysis.frames:
                recorder.add_frame(
                    record.frame,
                    record.segment_id,
                    record.event,
                    obs=record.chosen,
                    smooth=record.smooth,
                    candidate=_chosen_candidate(record),
                    dist_to_prev=record.dist_to_prev,
                    threshold=record.threshold,
                )
            for segment in segments:
                recorder.add_segment(segment)
                cleaned, cleaned_frames = clean_map[segment["segment_id"]]
                if cleaned_frames:
                    recorder.add_clean_points(
                        segment["segment_id"],
                        [(f, p[0], p[1]) for f, p in zip(cleaned_frames, cleaned)],
                    )
                else:
                    recorder.add_clean_flat(segment["segment_id"], cleaned)
        finally:
            recorder.close()
        csv_paths = recorder.summary()

    analysis.accepted = stats["accepted"]
    analysis.rejected = stats["rejected"]
    analysis.miss = stats["miss"]
    analysis.segments = segments
    analysis.csv = csv_paths
    return analysis
