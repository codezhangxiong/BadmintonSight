"""pipeline 编排层：把跟踪拆成"先分析、再渲染"两遍扫描。

历史：旧版是一个 while 循环里同时做"检测/关联/平滑/门控/画图/写视频"，
无法二次渲染，也无法插入需要回看整段的算法。现在拆成：

    src/analysis.py  analyze_video()   第一遍：只计算，产出 AnalysisResult（可落盘 JSON）
    src/render.py    render_video()    第二遍：只画图，读原视频重放视觉层并写 mp4

默认行为（run_pipeline）仍然是"一条命令得到视频 + CSV"，与旧版 CLI 兼容；
额外支持"只分析"和"只渲染"，见 scripts/run_track.py 的 --analyze-only / --render-from。
"""

from src.analysis import (  # noqa: F401  (_clean_segment 对外保留，测试与旧调用方可能引用)
    AnalysisResult,
    FrameRecord,
    _clean_segment,
    analyze_video,
    load_analysis,
    save_analysis,
)
from src.render import draw_frame, render_video  # noqa: F401

import os


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
    """两阶段跑完整视频：分析 → 渲染。

    参数
    ----
    video_path      输入视频
    out_path        输出 mp4（父目录自动创建；CSV sidecar 与它同名同目录）
    cfg             configs/track_cfg.yaml 加载后的 dict（需含 tracker/ransac/detector，并注入 model_path）
    show_gui        渲染时是否弹 OpenCV 窗口
    limit           只处理前 N 帧（调试用），None 表示整片
    save_traj       是否写轨迹 CSV
    analysis_json   若给出路径，则把分析结果落盘为 JSON（便于之后 --render-from 复用）

    返回 summary dict（字段与旧版兼容，另加 rendered_frames / analysis_json）。
    """
    analysis = analyze_video(
        video_path, cfg, out_path=out_path, limit=limit, save_traj=save_traj, verbose=True
    )
    if analysis_json:
        save_analysis(analysis, analysis_json)
    rendered = render_video(video_path, out_path, analysis, show_gui=show_gui)
    return _summary(out_path, analysis, rendered["frames"], analysis_json=analysis_json)


def run_analysis_only(video_path, cfg, out_path, limit=None, save_traj=True, analysis_json=None):
    """只做第一遍扫描：写 CSV（可选）与分析 JSON，不生成视频。"""
    analysis = analyze_video(
        video_path, cfg, out_path=out_path, limit=limit, save_traj=save_traj, verbose=True
    )
    if not analysis_json:
        analysis_json = os.path.splitext(out_path)[0] + ".analysis.json"
    save_analysis(analysis, analysis_json)
    return _summary(out_path, analysis, 0, analysis_json=analysis_json)


def run_render_only(video_path, out_path, analysis_json, show_gui=False):
    """只做第二遍渲染：从已有分析 JSON 生成视频（不加载 YOLO）。"""
    analysis = load_analysis(analysis_json)
    rendered = render_video(video_path, out_path, analysis, show_gui=show_gui)
    return _summary(out_path, analysis, rendered["frames"], analysis_json=analysis_json)
