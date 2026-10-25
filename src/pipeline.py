import os
from collections import deque

import cv2
import numpy as np
from ultralytics import YOLO

from src.detector import extract_shuttlecock_candidates
from src.export import TrajectoryRecorder
from src.tracker.association import mark_static_candidates, select_best_candidate
from src.tracker.cleaner import ransac_clean_parabola
from src.tracker.predictor import dynamic_step_threshold, predict_linear
from src.visualizer import draw_candidates, draw_fade_points
from utils.common import distance


def _clean_segment(points, frames, ransac_cfg):
    """对一段轨迹做 RANSAC 清洗。

    返回 (cleaned_points, cleaned_frames, mask)。mask 与输入等长且保序，
    因此可以直接用布尔掩码把内点映射回原始帧号。
    """
    cleaned, mask = ransac_clean_parabola(
        points,
        min_samples=ransac_cfg["min_samples"],
        max_iter=ransac_cfg["max_iter"],
        residual_thresh=ransac_cfg["residual_thresh"],
        return_mask=True,
    )
    cleaned_frames = [frames[i] for i in range(min(len(mask), len(frames))) if mask[i]]
    return cleaned, cleaned_frames, mask


def run_pipeline(video_path, out_path, cfg, show_gui=True, limit=None, save_traj=True):
    """跑完整条视频。

    参数
    ----
    video_path  输入视频
    out_path    输出 mp4（父目录会自动创建）
    cfg         configs/track_cfg.yaml 加载后的 dict，需含 tracker/ransac/detector 段，
                并由调用方注入 model_path
    show_gui    是否弹 OpenCV 调试窗口（服务器/CI 传 False）
    limit       只处理前 N 帧（调试用），None 表示整片
    save_traj   是否写出轨迹 CSV sidecar

    返回
    ----
    dict：输出路径、CSV 路径、帧数统计与分段列表
    """
    model = YOLO(cfg["model_path"])

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"无法打开视频: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        fps = 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if w <= 0 or h <= 0:
        cap.release()
        raise ValueError(f"无法读取视频尺寸: {video_path}")

    out_dir = os.path.dirname(os.path.abspath(out_path))
    os.makedirs(out_dir, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(out_path, fourcc, fps, (w, h))
    if not out.isOpened():
        cap.release()
        raise RuntimeError(
            f"无法创建输出视频（检查编码器 mp4v 是否可用、路径是否有写权限）: {out_path}"
        )

    recorder = TrajectoryRecorder(out_path, enabled=save_traj)

    track_cfg = cfg["tracker"]
    ransac_cfg = cfg["ransac"]
    smooth_win = deque(maxlen=track_cfg["smooth_window_len"])
    full_traj = []
    full_traj_frames = []
    missing = 0
    MAX_MISS = track_cfg["max_miss"]
    roi = [(50, 50), (w - 50, h - 50)]

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    processed = 0
    accepted = 0
    rejected = 0
    missed = 0
    last_frame = 0
    segment_id = 0
    segments = []

    def finalize_segment(end_frame):
        """结束当前轨迹段：清洗、落盘、清空缓冲。"""
        nonlocal segment_id
        if full_traj:
            raw = list(full_traj)
            raw_frames = list(full_traj_frames)
            cleaned, cleaned_frames, _ = _clean_segment(raw, raw_frames, ransac_cfg)
            if len(raw) > ransac_cfg["min_samples"]:
                print(f"\n==== trajectory end, raw len: {len(raw)} ====")
                print(f"==== cleaned len: {len(cleaned)} ====\n")
            segments.append(
                {
                    "segment_id": segment_id,
                    # 起止帧取"实际有观测的帧"，与轨迹点范围一致，便于和视频帧对齐
                    "start_frame": raw_frames[0] if raw_frames else end_frame,
                    "end_frame": raw_frames[-1] if raw_frames else end_frame,
                    "n_raw": len(raw),
                    "n_cleaned": len(cleaned),
                    "first_x": raw[0][0],
                    "first_y": raw[0][1],
                    "last_x": raw[-1][0],
                    "last_y": raw[-1][1],
                }
            )
            recorder.add_segment(segments[-1])
            if cleaned_frames:
                recorder.add_clean_points(
                    segment_id, [(f, p[0], p[1]) for f, p in zip(cleaned_frames, cleaned)]
                )
            else:
                recorder.add_clean_flat(segment_id, cleaned)
            segment_id += 1
        full_traj.clear()
        full_traj_frames.clear()
        smooth_win.clear()

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if limit is not None and processed >= limit:
                break
            processed += 1
            fid = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            last_frame = fid
            results = model(frame, conf=cfg["detector"]["yolo_conf"], verbose=False)[0]

            # draw raw boxes for debug
            raw = results.boxes
            if raw is not None:
                xywh = raw.xywh.detach().cpu().numpy()
                confs = raw.conf.detach().cpu().numpy()
                for cx, cy, bw, bh, conf in zip(xywh[:, 0], xywh[:, 1], xywh[:, 2], xywh[:, 3], confs):
                    x1 = int(cx - bw / 2)
                    y1 = int(cy - bh / 2)
                    x2 = int(cx + bw / 2)
                    y2 = int(cy + bh / 2)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 1)
                    cv2.putText(frame, f"{conf:.2f}", (x1, y1 - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)

            candidates = extract_shuttlecock_candidates(results.boxes, frame.shape, cfg, roi)
            candidates = mark_static_candidates(
                candidates,
                fid,
                grid_size=track_cfg["grid_size"],
                static_frame_count=track_cfg["static_frame_count"],
            )
            draw_candidates(frame, candidates)

            ref_pt = predict_linear(full_traj, n_lookback=4)
            best = select_best_candidate(candidates, ref_pt, full_traj, cfg)

            smooth_pt = None
            d = None
            th = None
            if best is not None:
                obs_pt = best["point"]
                cv2.circle(frame, obs_pt, 7, (0, 255, 255), -1)
                outlier = False
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
                if outlier:
                    missing += 1
                    rejected += 1
                    print(f"[Frame {fid}] OBS {obs_pt} outlier drop")
                    recorder.add_frame(
                        fid,
                        segment_id,
                        "rejected",
                        obs=obs_pt,
                        candidate=best,
                        dist_to_prev=d,
                        threshold=th,
                    )
                else:
                    missing = 0
                    accepted += 1
                    smooth_win.append(np.array(obs_pt))
                    arr = np.array(list(smooth_win))
                    smooth_pt = tuple(np.mean(arr, axis=0))
                    full_traj.append(smooth_pt)
                    full_traj_frames.append(fid)
                    print(f"[Frame {fid}] obs {obs_pt}, smooth {np.round(smooth_pt, 1)}")
                    recorder.add_frame(
                        fid,
                        segment_id,
                        "obs",
                        obs=obs_pt,
                        smooth=smooth_pt,
                        candidate=best,
                        dist_to_prev=d,
                        threshold=th,
                    )
            else:
                missing += 1
                missed += 1
                print(f"[Frame {fid}] no obs, missing={missing}")
                recorder.add_frame(fid, segment_id, "miss")

            if smooth_pt is not None:
                cv2.circle(frame, (int(smooth_pt[0]), int(smooth_pt[1])), 6, (0, 0, 255), -1)
            draw_fade_points(frame, full_traj, max_keep=30)

            out.write(frame)
            if show_gui:
                cv2.imshow("debug", frame)
                if cv2.waitKey(20) & 0xFF == ord("q"):
                    break

            if missing > MAX_MISS:
                finalize_segment(fid)
                missing = 0
    finally:
        # 视频结束 / 按 q 退出时，最后一段此前不会触发清洗，这里补上，避免轨迹丢失
        if full_traj:
            finalize_segment(last_frame)
        recorder.close()
        cap.release()
        out.release()
        if show_gui:
            cv2.destroyAllWindows()

    summary = {
        "out_path": out_path,
        "frames": processed,
        "expected_frames": total_frames,
        "accepted": accepted,
        "rejected": rejected,
        "miss": missed,
        "segments": segments,
    }
    summary.update(recorder.summary())
    return summary
