"""段处理阶段：把逐帧采集结果离线切成"飞行段"，清洗并落盘。

三段式结构中的第二段：

    采集    analyze_video()      逐帧流式：检测 → 关联 → 门控 → 平滑，只记录，不做段处理
    段处理  process_segments()    离线批：按"连续丢帧"切段 → RANSAC 清洗 → 写 CSV
    渲染    render_video()        离线批：读原视频重放视觉层

为什么单独拆出来：
- 段处理只依赖采集结果 frames（event / frame / smooth），与 YOLO、视频解码无关，
  因此可以"分析一次、调参重跑段处理多次"（max_miss / ransac 参数）；
- 后续升级成按飞行弧切段、时间参数化拟合、Kalman/RTS 平滑、Viterbi 重优化时，
  只需要改本模块，不碰采集循环；
- 分段规则与旧版逐帧触发完全等价（由 tests/test_segments.py 固定，输出 CSV 逐字节一致）。
"""

import numpy as np

from src.export import SegmentRecorder
from src.tracker.cleaner import ransac_clean_parabola


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


def process_segments(frames, cfg, out_path=None, save_csv=True, verbose=True):
    """对采集结果做离线分段 + RANSAC 清洗。

    参数
    ----
    frames    采集阶段的 FrameRecord 列表（只用到 .frame / .event / .smooth）
    cfg       配置 dict（需要 tracker.max_miss 与 ransac.*）
    out_path  输出视频路径；CSV sidecar 按它的目录/主干命名。None 则只计算不落盘
    save_csv  是否写 segments.csv / clean.csv
    verbose   是否打印段结束日志

    返回
    ----
    (segments, csv_summary)

    分段规则（与旧版逐帧触发严格等价）：
    连续丢帧数 > max_miss 时结束当前段；视频结束时若还有未结束的段，同样收尾。
    """
    max_miss = int(cfg["tracker"]["max_miss"])
    ransac_cfg = cfg["ransac"]
    enabled = bool(save_csv) and out_path is not None
    recorder = SegmentRecorder(out_path, enabled=enabled) if enabled else None

    segments = []
    seg_id = 0
    current = []
    missing = 0

    def flush():
        nonlocal seg_id, current
        if not current:
            return
        segment, cleaned, cleaned_frames = _build_segment(seg_id, current, ransac_cfg)
        segments.append(segment)
        if recorder is not None:
            recorder.add_segment(segment)
            if cleaned_frames:
                recorder.add_clean_points(
                    seg_id, [(f, p[0], p[1]) for f, p in zip(cleaned_frames, cleaned)]
                )
            else:
                recorder.add_clean_flat(seg_id, cleaned)
        if verbose and segment["n_raw"] > int(ransac_cfg["min_samples"]):
            print(f"\n==== trajectory end, raw len: {segment['n_raw']} ====")
            print(f"==== cleaned len: {segment['n_cleaned']} ====\n")
        seg_id += 1
        current = []

    try:
        for record in frames:
            if record.event == "obs":
                missing = 0
                current.append(
                    (record.frame, (float(record.smooth[0]), float(record.smooth[1])))
                )
            else:
                missing += 1
                if missing > max_miss:
                    flush()
                    missing = 0
        flush()
    finally:
        if recorder is not None:
            recorder.close()

    return segments, (recorder.summary() if recorder is not None else {})
