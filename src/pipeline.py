"""pipeline 编排层：采集 → 段处理 → 渲染，三段式。

    采集    src/capture.py    capture_video()     逐帧检测 + 候选记录（无状态、不判断）
    段处理  src/segments.py   resolve_tracks()    关联/预测/门控/平滑 + 切段 + RANSAC 清洗
    渲染    src/render.py     render_video()      只画图，读原视频重放视觉层并写 mp4

关键设计：**跟踪决策（含门控）全部在段处理阶段**。采集结果里存了每帧的全部候选，
因此换关联策略、调门控阈值/分段参数都不需要重跑 YOLO（--segments-from）。

默认行为（run_pipeline）仍然是"一条命令得到视频 + CSV"，与旧版 CLI 兼容；
额外支持只采集+段处理、只重做段处理、只渲染，见 scripts/run_track.py。
"""

import os

from src.capture import (  # noqa: F401  (对外保留数据结构与采集入口)
    CaptureResult,
    FrameRecord,
    capture_video,
    load_capture,
    save_capture,
)
from src.render import draw_frame, render_video  # noqa: F401
from src.segments import _clean_segment, collect_segments, replay_decisions, resolve_tracks  # noqa: F401


def _summary(out_path, capture, rendered_frames, capture_json=None):
    summary = {
        "out_path": out_path,
        "frames": capture.processed,
        "expected_frames": capture.total_frames,
        "accepted": capture.accepted,
        "rejected": capture.rejected,
        "miss": capture.miss,
        "segments": capture.segments,
        "rendered_frames": rendered_frames,
    }
    summary.update(capture.csv)
    if capture_json:
        summary["capture_json"] = capture_json
    return summary


def run_pipeline(video_path, out_path, cfg, show_gui=True, limit=None, save_traj=True, capture_json=None):
    """三段式跑完整视频：采集 → 段处理 → 渲染。

    参数
    ----
    video_path      输入视频
    out_path        输出 mp4（父目录自动创建；CSV sidecar 与它同名同目录）
    cfg             configs/track_cfg.yaml 加载后的 dict（需含 tracker/ransac/detector，并注入 model_path）
    show_gui        渲染时是否弹 OpenCV 窗口
    limit           只处理前 N 帧（调试用），None 表示整片
    save_traj       是否写三份 CSV
    capture_json    若给出路径，则把"采集 + 决策"结果落盘为 JSON（便于之后只渲染/只重做段处理）

    返回 summary dict（字段与旧版兼容，另加 rendered_frames / capture_json）。
    """
    capture = capture_video(video_path, cfg, limit=limit, verbose=True)
    resolve_tracks(capture, cfg, out_path=out_path, save_traj=save_traj, verbose=True)
    if capture_json:
        save_capture(capture, capture_json)
    rendered = render_video(video_path, out_path, capture, show_gui=show_gui)
    return _summary(out_path, capture, rendered["frames"], capture_json=capture_json)


def run_capture_only(video_path, cfg, out_path, limit=None, save_traj=True, capture_json=None):
    """只跑采集 + 段处理：写 CSV（可选）与采集 JSON，不生成视频。"""
    capture = capture_video(video_path, cfg, limit=limit, verbose=True)
    resolve_tracks(capture, cfg, out_path=out_path, save_traj=save_traj, verbose=True)
    if not capture_json:
        capture_json = os.path.splitext(out_path)[0] + ".capture.json"
    save_capture(capture, capture_json)
    return _summary(out_path, capture, 0, capture_json=capture_json)


def run_segments_only(capture_json, cfg, out_path, save_traj=True, verbose=True):
    """只重做"跟踪决策 + 分段"：从已有 JSON 重放关联/门控/平滑/切段/清洗并写 CSV。

    不重跑 YOLO、不解码视频。用于调门控阈值、max_miss、ransac 参数后快速重算。
    """
    capture = load_capture(capture_json)
    resolve_tracks(capture, cfg, out_path=out_path, save_traj=save_traj, verbose=verbose)
    return _summary(out_path, capture, 0, capture_json=capture_json)


def run_render_only(video_path, out_path, capture_json, show_gui=False):
    """只做渲染：从已有采集 JSON 生成视频（不加载 YOLO，也不重做段处理）。"""
    capture = load_capture(capture_json)
    rendered = render_video(video_path, out_path, capture, show_gui=show_gui)
    return _summary(out_path, capture, rendered["frames"], capture_json=capture_json)
