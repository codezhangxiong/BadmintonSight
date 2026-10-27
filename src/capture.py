"""采集阶段：纯感知，只记录"看到了什么"，不做任何跟踪决策。

三段式结构中的第一段：

    采集    capture_video()      本文件：检测 → 候选过滤 → 静态标记，逐帧记录候选
    段处理  resolve_tracks()     src/segments.py：关联 + 预测 + 门控 + 平滑 + 切段 + 清洗
    渲染    render_video()       src/render.py：读原视频重放视觉层并写 mp4

职责边界（重要）：
- 采集阶段**不判断哪个候选是球、不做门控、不丢弃观测**，只把每帧的候选完整记录下来
  （point / is_static / conf / area_ratio / aspect_ratio / bbox）。因此更换关联策略、
  门控阈值、平滑方式、分段规则都不需要重跑 YOLO。
- 决策链（关联→预测→门控→平滑→分段）全部在段处理阶段重放，见 src/segments.py。

为什么门控必须和关联/平滑一起放在段处理：
关联需要"历史轨迹"做预测（predict_linear），而历史轨迹只由"通过门控的观测"构成。
如果把门控单独留在采集端，它就得依赖平滑状态、而平滑又依赖门控结果，形成环。
所以正确的切分是：采集=感知（无状态），段处理=决策（有状态）。
"""

import json
import os
import time
from dataclasses import dataclass, field

import cv2

from src.detector import extract_shuttlecock_candidates
from src.tracker import association
from src.tracker.association import mark_static_candidates


@dataclass
class FrameRecord:
    """一帧的采集结果 + 决策结果。

    采集阶段只填 frame / raw_boxes / candidates；其余字段由段处理阶段（resolve_tracks）填充。
    """

    frame: int
    raw_boxes: list = field(default_factory=list)  # [[cx,cy,bw,bh,conf], ...]
    # 候选：[x, y, is_static, confidence, area_ratio, aspect_ratio, x1, y1, x2, y2]
    # 前 3 项供渲染使用，中间 3 项供段处理重放关联代价计算，后 4 项供 traj.csv 导出
    candidates: list = field(default_factory=list)
    chosen: list = None  # [x,y] 门控前选中的候选（被拒也会画黄点）
    rejected: bool = False
    smooth: list = None  # [x,y] 平滑点
    trail: list = field(default_factory=list)  # 该帧绘制用的历史点（最多 30 个）
    segment_id: int = None
    event: str = "miss"  # obs / rejected / miss
    # 门控决策细节（段处理阶段填充；导出 traj.csv 的 dist_to_prev / threshold 列）
    dist_to_prev: float = None
    threshold: float = None

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
            "dist_to_prev": self.dist_to_prev,
            "threshold": self.threshold,
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
            dist_to_prev=data.get("dist_to_prev"),
            threshold=data.get("threshold"),
        )


@dataclass
class CaptureResult:
    """采集结果：逐帧候选 + 决策字段（段处理阶段填充）。"""

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


def save_capture(result, path):
    """把采集结果写成 JSON（紧凑格式，避免体积膨胀）。"""
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(result.to_dict(), fh, ensure_ascii=False, separators=(",", ":"))
    os.replace(tmp, path)
    return path


def load_capture(path):
    with open(path, "r", encoding="utf-8") as fh:
        return CaptureResult.from_dict(json.load(fh))


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
    """记录候选中"感知 + 代价计算 + 导出"所需的全部字段。

    每行 10 项：[x, y, is_static, confidence, area_ratio, aspect_ratio, x1, y1, x2, y2]
    - 前 3 项供渲染使用；
    - 中间 3 项供段处理重放关联代价计算；
    - 后 4 项（bbox）供 traj.csv 导出。
    段处理阶段要重放关联，因此必须把代价计算所需字段一起存下来，否则无法离线重算。
    """
    rows = []
    for cand in candidates:
        bbox = cand.get("bbox") or (0, 0, 0, 0)
        rows.append(
            [
                int(cand["point"][0]),
                int(cand["point"][1]),
                bool(cand.get("is_static", False)),
                float(cand["confidence"]),
                float(cand["area_ratio"]),
                float(cand["aspect_ratio"]),
                int(bbox[0]),
                int(bbox[1]),
                int(bbox[2]),
                int(bbox[3]),
            ]
        )
    return rows


def capture_video(video_path, cfg, limit=None, verbose=True):
    """第一遍扫描：只做检测与候选记录（无状态），不判断、不丢弃、不写 CSV。

    参数
    ----
    video_path  输入视频
    limit       只处理前 N 帧（调试用）
    verbose     是否打印逐帧日志与启动进度

    返回 CaptureResult（决策字段为空，由 resolve_tracks 填充）。
    """
    # 延迟导入 ultralytics：它会连锁加载 torch 及 CUDA 动态库（本机实测约 2.2s）。
    # 只有真正要跑检测时才需要这笔开销；只渲染/只重做段处理时不应付这笔启动成本。
    from ultralytics import YOLO

    t_start = time.perf_counter()
    if verbose:
        # 进度提示走 stdout（Windows PowerShell 会把 stderr 渲染成红色错误块）
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

    track_cfg = cfg["tracker"]
    roi = [(50, 50), (w - 50, h - 50)]

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    processed = 0
    frames = []

    try:
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
                    f"[采集] 初始化完成（累计 {time.perf_counter() - t_start:.2f}s），开始逐帧检测",
                    flush=True,
                )

            candidates = extract_shuttlecock_candidates(results.boxes, frame.shape, cfg, roi)
            candidates = mark_static_candidates(
                candidates,
                fid,
                grid_size=track_cfg["grid_size"],
                static_frame_count=track_cfg["static_frame_count"],
            )

            frames.append(
                FrameRecord(
                    frame=fid,
                    raw_boxes=_collect_raw_boxes(results.boxes),
                    candidates=_collect_candidates(candidates),
                )
            )
            if verbose:
                print(f"[采集] Frame {fid}: 候选 {len(candidates)} 个")

        return CaptureResult(
            video_path=video_path,
            fps=fps,
            width=w,
            height=h,
            total_frames=total_frames,
            processed=processed,
            accepted=0,
            rejected=0,
            miss=0,
            segments=[],
            frames=frames,
            csv={},
            limit=limit,
        )
    finally:
        cap.release()
