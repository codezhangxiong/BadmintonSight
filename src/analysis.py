"""采集阶段：只做逐帧计算与记录，不做段处理、不画图、不写视频。

三段式结构中的第一段：

    采集    analyze_video()      本文件：检测 → 关联 → 门控 → 平滑，逐帧记录
    段处理  process_segments()   src/segments.py：离线切段 + RANSAC 清洗 + 写 CSV
    渲染    render_video()       src/render.py：读原视频重放视觉层并写 mp4

采集阶段只输出两样东西：
- FrameRecord 列表（每帧的渲染快照 + 决策结果），供渲染与段处理使用；
- <stem>.traj.csv（逐帧事件表，流式写出）。

一致性约束（重要）：
FrameRecord 必须记录**复刻原单遍渲染所需的全部数据**；分段所需的 (frame, event, smooth)
也要完整，否则段处理无法离线重算出与旧版一致的段。回归验证方式：
全片输出 mp4 与基线 MD5 逐字节比对、三个 CSV 与基线 SHA256 比对。
"""

import json
import os
import time
from collections import deque
from dataclasses import dataclass, field

import cv2
import numpy as np

from src.detector import extract_shuttlecock_candidates
from src.export import FrameRecorder
from src.tracker import association
from src.tracker.association import mark_static_candidates, select_best_candidate
from src.tracker.predictor import dynamic_step_threshold, predict_linear
from utils.common import distance

# 渲染阶段固定使用的历史点上限，必须与画图端 draw_fade_points(max_keep=30) 一致
TRAIL_MAX_KEEP = 30


@dataclass
class FrameRecord:
    """一帧的采集结果（渲染快照 + 决策），是段处理与渲染的唯一输入。"""

    frame: int
    raw_boxes: list = field(default_factory=list)  # [[cx,cy,bw,bh,conf], ...]
    candidates: list = field(default_factory=list)  # [[x,y,is_static], ...]
    chosen: list = None  # [x,y] 门控前选中的候选（被拒也会画黄点）
    rejected: bool = False
    smooth: list = None  # [x,y] 平滑点
    trail: list = field(default_factory=list)  # 该帧绘制用的历史点（最多 30 个）
    segment_id: int = None
    event: str = "miss"  # obs / rejected / miss

    def to_dict(self):
        return {
            "frame": self.frame,
            "raw_boxes": self.raw_boxes,
            "candidates": self.candidates,
            "chosen": self.chosen,
            "rejected": self.rejected,
            "smooth": self.smooth,
            "trail": self.trail,
            "segment_id": self.segment_id,
            "event": self.event,
        }

    @classmethod
    def from_dict(cls, data):
        return cls(
            frame=int(data["frame"]),
            raw_boxes=data.get("raw_boxes") or [],
            candidates=data.get("candidates") or [],
            chosen=data.get("chosen"),
            rejected=bool(data.get("rejected", False)),
            smooth=data.get("smooth"),
            trail=data.get("trail") or [],
            segment_id=data.get("segment_id"),
            event=data.get("event", "miss"),
        )


@dataclass
class AnalysisResult:
    """采集结果：逐帧快照 + 统计。segments 由段处理阶段填充。"""

    video_path: str
    fps: float
    width: int
    height: int
    total_frames: int
    processed: int
    accepted: int
    rejected: int
    miss: int
    segments: list
    frames: list
    csv: dict
    limit: int = None

    def to_dict(self):
        return {
            "video_path": self.video_path,
            "fps": self.fps,
            "width": self.width,
            "height": self.height,
            "total_frames": self.total_frames,
            "processed": self.processed,
            "accepted": self.accepted,
            "rejected": self.rejected,
            "miss": self.miss,
            "segments": self.segments,
            "frames": [f.to_dict() for f in self.frames],
            "csv": self.csv,
            "limit": self.limit,
        }

    @classmethod
    def from_dict(cls, data):
        return cls(
            video_path=data["video_path"],
            fps=float(data["fps"]),
            width=int(data["width"]),
            height=int(data["height"]),
            total_frames=int(data.get("total_frames") or 0),
            processed=int(data.get("processed") or 0),
            accepted=int(data.get("accepted") or 0),
            rejected=int(data.get("rejected") or 0),
            miss=int(data.get("miss") or 0),
            segments=data.get("segments") or [],
            frames=[FrameRecord.from_dict(f) for f in data.get("frames") or []],
            csv=data.get("csv") or {},
            limit=data.get("limit"),
        )

    @property
    def rendered_frames(self):
        return len(self.frames)


def save_analysis(result, path):
    """把分析结果写成 JSON（紧凑格式，避免体积膨胀）。"""
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(result.to_dict(), fh, ensure_ascii=False, separators=(",", ":"))
    os.replace(tmp, path)
    return path


def load_analysis(path):
    with open(path, "r", encoding="utf-8") as fh:
        return AnalysisResult.from_dict(json.load(fh))


def _collect_raw_boxes(boxes):
    """把 YOLO 原始框转成可序列化的列表，供渲染阶段复刻调试红框。"""
    if boxes is None:
        return []
    xywh = boxes.xywh.detach().cpu().numpy()
    confs = boxes.conf.detach().cpu().numpy()
    return [
        [float(cx), float(cy), float(bw), float(bh), float(conf)]
        for cx, cy, bw, bh, conf in zip(xywh[:, 0], xywh[:, 1], xywh[:, 2], xywh[:, 3], confs)
    ]


def _collect_candidates(candidates):
    """把候选点转成可序列化列表（渲染只需 point + is_static）。"""
    return [
        [int(cand["point"][0]), int(cand["point"][1]), bool(cand.get("is_static", False))]
        for cand in candidates
    ]


def analyze_video(video_path, cfg, out_path, limit=None, save_traj=True, verbose=True):
    """第一遍扫描：逐帧采集，不写视频、不画图、不做段处理。

    参数
    ----
    video_path  输入视频
    out_path    最终输出 mp4 的路径（本阶段只用它的目录/主干命名 traj.csv）
    limit       只处理前 N 帧（调试用）
    save_traj   是否写逐帧事件 CSV（traj.csv）
    verbose     是否打印逐帧日志与启动进度

    返回 AnalysisResult（segments 为空，由 process_segments 填充）。
    """
    # 延迟导入 ultralytics：它会连锁加载 torch 及 CUDA 动态库（本机实测约 2.2s）。
    # 只有真正要跑检测时才需要这笔开销；--render-from 走渲染路径，不应为它付启动成本。
    from ultralytics import YOLO

    t_start = time.perf_counter()
    if verbose:
        # 进度提示走 stdout（数据输出本来就是逐帧日志；stderr 在 Windows PowerShell
        # 下会被渲染成红色错误块，容易误解为报错）
        print(f"[采集] 加载 YOLO 权重: {cfg['model_path']}", flush=True)
    model = YOLO(cfg["model_path"])
    if verbose:
        print(
            f"[采集] 权重加载完成（{time.perf_counter() - t_start:.2f}s）；"
            "正在初始化 CUDA 并执行首帧推理（首次约 3~5 秒）...",
            flush=True,
        )

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

    # 模块级静态记忆是全局状态；每次采集都从空记忆开始，保证可复现
    association.static_candidate_memory.clear()

    recorder = None
    try:
        recorder = FrameRecorder(out_path, enabled=save_traj)

        track_cfg = cfg["tracker"]
        smooth_win = deque(maxlen=track_cfg["smooth_window_len"])
        full_traj = []
        missing = 0
        MAX_MISS = track_cfg["max_miss"]
        roi = [(50, 50), (w - 50, h - 50)]

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        processed = 0
        accepted = 0
        rejected = 0
        missed = 0
        segment_id = 0
        active_segment_id = None
        frames = []

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if limit is not None and processed >= limit:
                break
            processed += 1
            fid = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            results = model(frame, conf=cfg["detector"]["yolo_conf"], verbose=False)[0]
            if verbose and processed == 1:
                print(
                    f"[采集] 初始化完成（累计 {time.perf_counter() - t_start:.2f}s），开始逐帧处理",
                    flush=True,
                )

            raw_boxes = _collect_raw_boxes(results.boxes)

            candidates = extract_shuttlecock_candidates(results.boxes, frame.shape, cfg, roi)
            candidates = mark_static_candidates(
                candidates,
                fid,
                grid_size=track_cfg["grid_size"],
                static_frame_count=track_cfg["static_frame_count"],
            )

            ref_pt = predict_linear(full_traj, n_lookback=4)
            best = select_best_candidate(candidates, ref_pt, full_traj, cfg)

            record = FrameRecord(frame=fid, raw_boxes=raw_boxes)
            smooth_pt = None
            d = None
            th = None
            if best is not None:
                obs_pt = best["point"]
                record.chosen = [int(obs_pt[0]), int(obs_pt[1])]
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
                    record.rejected = True
                    record.event = "rejected"
                    if verbose:
                        print(f"[Frame {fid}] OBS {obs_pt} outlier drop")
                    recorder.add_frame(
                        fid,
                        active_segment_id,
                        "rejected",
                        obs=obs_pt,
                        candidate=best,
                        dist_to_prev=d,
                        threshold=th,
                    )
                else:
                    missing = 0
                    accepted += 1
                    if active_segment_id is None:
                        active_segment_id = segment_id
                    smooth_win.append(np.array(obs_pt))
                    arr = np.array(list(smooth_win))
                    smooth_pt = tuple(np.mean(arr, axis=0))
                    full_traj.append(smooth_pt)
                    record.smooth = [float(smooth_pt[0]), float(smooth_pt[1])]
                    record.event = "obs"
                    if verbose:
                        print(f"[Frame {fid}] obs {obs_pt}, smooth {np.round(smooth_pt, 1)}")
                    recorder.add_frame(
                        fid,
                        active_segment_id,
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
                record.event = "miss"
                if verbose:
                    print(f"[Frame {fid}] no obs, missing={missing}")
                recorder.add_frame(fid, active_segment_id, "miss")

            # 记录渲染快照：此时 full_traj 的状态与旧版 draw_fade_points 的输入完全一致
            record.candidates = _collect_candidates(candidates)
            record.trail = [[float(x), float(y)] for x, y in list(full_traj)[-TRAIL_MAX_KEEP:]]
            record.segment_id = active_segment_id
            frames.append(record)

            if missing > MAX_MISS:
                # 段结束：推进段号、清空平滑/轨迹缓冲（只影响后续帧的 segment_id 列；
                # 真正的分段与清洗在 process_segments 里离线完成）
                if full_traj:
                    segment_id += 1
                    active_segment_id = None
                full_traj.clear()
                smooth_win.clear()
                missing = 0

        return AnalysisResult(
            video_path=video_path,
            fps=fps,
            width=w,
            height=h,
            total_frames=total_frames,
            processed=processed,
            accepted=accepted,
            rejected=rejected,
            miss=missed,
            segments=[],
            frames=frames,
            csv=recorder.summary(),
            limit=limit,
        )
    finally:
        if recorder is not None:
            recorder.close()
        cap.release()
