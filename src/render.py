"""渲染阶段：只画图、不决策。

输入是 analysis.AnalysisResult（第一遍扫描的帧快照），输出是可视化 mp4。
本模块不加载 YOLO、不做关联/门控，因此渲染逻辑可以独立修改
（换配色、按速度上色、加标注、慢动作）而不影响跟踪结果。

绘制顺序必须与旧版单遍实现严格一致，否则两阶段输出会与基线不逐字节相同：
    原始红框 → 候选点(蓝/紫) → 选中黄点 → 平滑红点 → 渐隐历史轨迹
"""

import os

import cv2

from src.visualizer import draw_candidates, draw_fade_points


def _draw_raw_boxes(frame, raw_boxes):
    """复刻旧版调试红框：xywh + conf → 矩形 + 置信度文本。"""
    for cx, cy, bw, bh, conf in raw_boxes:
        x1 = int(cx - bw / 2)
        y1 = int(cy - bh / 2)
        x2 = int(cx + bw / 2)
        y2 = int(cy + bh / 2)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 1)
        cv2.putText(
            frame,
            f"{conf:.2f}",
            (x1, y1 - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            (0, 0, 255),
            1,
        )


def draw_frame(frame, record):
    """把一条 FrameRecord 的视觉层画到 frame 上（原地修改）。

    单独抽出来便于渲染层单测：不需要视频和模型，只要一条记录 + 一张图。
    """
    _draw_raw_boxes(frame, record.raw_boxes)

    # 候选点：draw_candidates 需要 dict 形式的候选（point + is_static）
    draw_candidates(
        frame,
        [
            {"point": (int(x), int(y)), "is_static": bool(is_static)}
            for x, y, is_static in record.candidates
        ],
    )

    if record.chosen is not None:
        cv2.circle(frame, (int(record.chosen[0]), int(record.chosen[1])), 7, (0, 255, 255), -1)

    if record.smooth is not None:
        cv2.circle(
            frame,
            (int(record.smooth[0]), int(record.smooth[1])),
            6,
            (0, 0, 255),
            -1,
        )

    draw_fade_points(frame, record.trail, max_keep=30)
    return frame


def render_video(video_path, out_path, analysis, show_gui=False):
    """第二遍扫描：读原视频，按分析快照逐帧重放视觉层并写 mp4。

    返回 {"out_path", "frames"}。
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"无法打开视频: {video_path}")

    out_dir = os.path.dirname(os.path.abspath(out_path))
    os.makedirs(out_dir, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(out_path, fourcc, analysis.fps, (analysis.width, analysis.height))
    if not out.isOpened():
        cap.release()
        raise RuntimeError(
            f"无法创建输出视频（检查编码器 mp4v 是否可用、路径是否有写权限）: {out_path}"
        )

    rendered = 0
    try:
        for record in analysis.frames:
            ret, frame = cap.read()
            if not ret:
                break
            draw_frame(frame, record)
            out.write(frame)
            rendered += 1
            if show_gui:
                cv2.imshow("debug", frame)
                if cv2.waitKey(20) & 0xFF == ord("q"):
                    break
    finally:
        out.release()
        cap.release()
        if show_gui:
            cv2.destroyAllWindows()

    return {"out_path": out_path, "frames": rendered}
