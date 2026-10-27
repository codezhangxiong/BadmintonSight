"""pipeline 编排层：采集 → 段处理 → 渲染，三段式。

    采集    src/analysis.py   analyze_video()     逐帧检测 + 候选记录（无状态、不判断）
    段处理  src/segments.py   resolve_tracks()    关联/预测/门控/平滑 + 切段 + RANSAC 清洗
    渲染    src/render.py     render_video()      只画图，读原视频重放视觉层并写 mp4

关键设计：**跟踪决策（含门控）全部在段处理阶段**。采集结果里存了每帧的全部候选，
因此换关联策略、调门控阈值/分段参数都不需要重跑 YOLO（--segments-from）。

默认行为（run_pipeline）仍然是"一条命令得到视频 + CSV"，与旧版 CLI 兼容；
额外支持只采集+段处理、只重做段处理、只渲染，见 scripts/run_track.py。
"""

import os

from src.analysis import (  # noqa: F401  (对外保留数据结构与采集入口)
    AnalysisResult,
    FrameRecord,
    analyze_video,
    load_analysis,
    save_analysis,
)
from src.render import draw_frame, render_video  # noqa: F401
from src.segments import _clean_segment, collect_segments, replay_decisions, resolve_tracks  # noqa: F401


def _summary(out_path, analysis, rendered_frames, analysis_json=None):
    summary = {
        "out_path": out_path,
        "frames": analysis.processed,
        "expected_frames": analysis.total_frames,
        "accepted": analysis.accepted,
        "rejected": analysis.rejected,
        "miss": analysis.miss,
        "segments": analysis.segments,
        "rendered_frames": rendered_frames,
    }
    summary.update(analysis.csv)
    if analysis_json:
        summary["analysis_json"] = analysis_json
    return summary


def run_pipeline(video_path, out_path, cfg, show_gui=True, limit=None, save_traj=True, analysis_json=None):
    """三段式跑完整视频：采集 → 段处理 → 渲染。

    参数
    ----
    video_path      输入视频
    out_path        输出 mp4（父目录自动创建；CSV sidecar 与它同名同目录）
    cfg             configs/track_cfg.yaml 加载后的 dict（需含 tracker/ransac/detector，并注入 model_path）
    show_gui        渲染时是否弹 OpenCV 窗口
    limit           只处理前 N 帧（调试用），None 表示整片
    save_traj       是否写三份 CSV
    analysis_json   若给出路径，则把"采集 + 决策"结果落盘为 JSON（便于之后只渲染/只重做段处理）

    返回 summary dict（字段与旧版兼容，另加 rendered_frames / analysis_json）。
    """
    analysis = analyze_video(video_path, cfg, limit=limit, verbose=True)
    resolve_tracks(analysis, cfg, out_path=out_path, save_traj=save_traj, verbose=True)
    if analysis_json:
        save_analysis(analysis, analysis_json)
    rendered = render_video(video_path, out_path, analysis, show_gui=show_gui)
    return _summary(out_path, analysis, rendered["frames"], analysis_json=analysis_json)


def run_analysis_only(video_path, cfg, out_path, limit=None, save_traj=True, analysis_json=None):
    """只跑采集 + 段处理：写 CSV（可选）与分析 JSON，不生成视频。"""
    analysis = analyze_video(video_path, cfg, limit=limit, verbose=True)
    resolve_tracks(analysis, cfg, out_path=out_path, save_traj=save_traj, verbose=True)
    if not analysis_json:
        analysis_json = os.path.splitext(out_path)[0] + ".analysis.json"
    save_analysis(analysis, analysis_json)
    return _summary(out_path, analysis, 0, analysis_json=analysis_json)


def run_segments_only(analysis_json, cfg, out_path, save_traj=True, verbose=True):
    """只重做"跟踪决策 + 分段"：从已有 JSON 重放关联/门控/平滑/切段/清洗并写 CSV。

    不重跑 YOLO、不解码视频。用于调门控阈值、max_miss、ransac 参数后快速重算。
    """
    analysis = load_analysis(analysis_json)
    resolve_tracks(analysis, cfg, out_path=out_path, save_traj=save_traj, verbose=verbose)
    return _summary(out_path, analysis, 0, analysis_json=analysis_json)


def run_render_only(video_path, out_path, analysis_json, show_gui=False):
    """只做渲染：从已有分析 JSON 生成视频（不加载 YOLO，也不重做段处理）。"""
    analysis = load_analysis(analysis_json)
    rendered = render_video(video_path, out_path, analysis, show_gui=show_gui)
    return _summary(out_path, analysis, rendered["frames"], analysis_json=analysis_json)
