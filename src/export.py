"""轨迹与分段结果的 CSV 导出。

pipeline 跑完一段视频后，除可视化 mp4 之外还会写出三份 sidecar CSV（与输出视频同名 stem）：

    <stem>.traj.csv      逐帧事件表：obs（接受）/ rejected（被判离群）/ miss（无观测）
    <stem>.segments.csv  每段轨迹汇总：起止帧、清洗前后点数、首尾点
    <stem>.clean.csv     每段清洗后的 (frame, x, y)，RANSAC 内点按原始顺序输出

设计约束：
- 只依赖标准库，不 import cv2 / ultralytics，方便单元测试与复用；
- enabled=False 时所有方法都是空操作，不创建任何文件；
- 列名保持稳定，便于下游 pandas / Excel 直接读取。
"""

import csv
import os

TRAJ_FIELDS = [
    "frame",
    "segment_id",
    "event",
    "obs_x",
    "obs_y",
    "smooth_x",
    "smooth_y",
    "conf",
    "bbox_x1",
    "bbox_y1",
    "bbox_x2",
    "bbox_y2",
    "is_static",
    "dist_to_prev",
    "threshold",
]

SEGMENT_FIELDS = [
    "segment_id",
    "start_frame",
    "end_frame",
    "n_raw",
    "n_cleaned",
    "first_x",
    "first_y",
    "last_x",
    "last_y",
]

CLEAN_FIELDS = ["segment_id", "frame", "x", "y"]


def _fmt(value, digits=2):
    """把数字/布尔/None 统一格式化成 CSV 单元格，避免科学计数法。"""
    if value is None:
        return ""
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, int):
        return str(value)
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def _as_xy(point):
    if point is None:
        return None, None
    return point[0], point[1]


class TrajectoryRecorder:
    """逐帧事件 + 分段汇总 + 清洗后轨迹的 CSV 写入器。"""

    def __init__(self, out_path, enabled=True):
        self.enabled = enabled
        stem = os.path.splitext(os.path.abspath(out_path))[0]
        self.traj_path = f"{stem}.traj.csv"
        self.segments_path = f"{stem}.segments.csv"
        self.clean_path = f"{stem}.clean.csv"
        self._traj_file = None
        self._seg_file = None
        self._clean_file = None
        self._traj_writer = None
        self._seg_writer = None
        self._clean_writer = None
        self._closed = False
        if self.enabled:
            self._open()

    # ------------------------------------------------------------------ open/close
    def _open(self):
        os.makedirs(os.path.dirname(self.traj_path) or ".", exist_ok=True)
        self._traj_file = open(self.traj_path, "w", newline="", encoding="utf-8")
        self._seg_file = open(self.segments_path, "w", newline="", encoding="utf-8")
        self._clean_file = open(self.clean_path, "w", newline="", encoding="utf-8")
        self._traj_writer = csv.DictWriter(self._traj_file, fieldnames=TRAJ_FIELDS)
        self._seg_writer = csv.DictWriter(self._seg_file, fieldnames=SEGMENT_FIELDS)
        self._clean_writer = csv.DictWriter(self._clean_file, fieldnames=CLEAN_FIELDS)
        self._traj_writer.writeheader()
        self._seg_writer.writeheader()
        self._clean_writer.writeheader()

    def close(self):
        if self._closed:
            return
        self._closed = True
        for fh in (self._traj_file, self._seg_file, self._clean_file):
            if fh is not None:
                fh.close()
        self._traj_file = self._seg_file = self._clean_file = None

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        self.close()
        return False

    # ------------------------------------------------------------------ rows
    def add_frame(
        self,
        frame,
        segment_id,
        event,
        obs=None,
        smooth=None,
        candidate=None,
        dist_to_prev=None,
        threshold=None,
    ):
        """记录一帧事件。

        frame       视频帧号（与日志一致，1-based）
        segment_id  当前轨迹段编号；无段时为最后一次分配的编号
        event       "obs" | "rejected" | "miss"
        obs         原始观测坐标 (x, y)，无观测传 None
        smooth      平滑后轨迹点 (x, y)，仅 event="obs" 时有值
        candidate   被选中的候选 dict，用于取 conf / bbox / is_static
        """
        if not self.enabled:
            return
        candidate = candidate or {}
        obs_x, obs_y = _as_xy(obs)
        smooth_x, smooth_y = _as_xy(smooth)
        bbox = candidate.get("bbox") or (None, None, None, None)
        self._traj_writer.writerow(
            {
                "frame": frame,
                "segment_id": segment_id,
                "event": event,
                "obs_x": _fmt(obs_x),
                "obs_y": _fmt(obs_y),
                "smooth_x": _fmt(smooth_x),
                "smooth_y": _fmt(smooth_y),
                "conf": _fmt(candidate.get("confidence"), digits=4),
                "bbox_x1": _fmt(bbox[0]),
                "bbox_y1": _fmt(bbox[1]),
                "bbox_x2": _fmt(bbox[2]),
                "bbox_y2": _fmt(bbox[3]),
                "is_static": _fmt(bool(candidate.get("is_static", False))),
                "dist_to_prev": _fmt(dist_to_prev),
                "threshold": _fmt(threshold),
            }
        )

    def add_segment(self, segment):
        """记录一段轨迹的汇总信息。"""
        if not self.enabled:
            return
        self._seg_writer.writerow(
            {
                "segment_id": segment.get("segment_id"),
                "start_frame": segment.get("start_frame"),
                "end_frame": segment.get("end_frame"),
                "n_raw": segment.get("n_raw"),
                "n_cleaned": segment.get("n_cleaned"),
                "first_x": _fmt(segment.get("first_x")),
                "first_y": _fmt(segment.get("first_y")),
                "last_x": _fmt(segment.get("last_x")),
                "last_y": _fmt(segment.get("last_y")),
            }
        )

    def add_clean_points(self, segment_id, points):
        """记录某段清洗后的点；points 为 (frame, x, y) 的可迭代对象。"""
        if not self.enabled:
            return
        for frame, x, y in points:
            self._clean_writer.writerow(
                {
                    "segment_id": segment_id,
                    "frame": frame,
                    "x": _fmt(x),
                    "y": _fmt(y),
                }
            )

    def add_clean_flat(self, segment_id, points):
        """记录某段清洗后的点；points 为 (x, y) 的可迭代对象（无帧号时使用）。"""
        if not self.enabled:
            return
        for x, y in points:
            self._clean_writer.writerow(
                {
                    "segment_id": segment_id,
                    "frame": "",
                    "x": _fmt(x),
                    "y": _fmt(y),
                }
            )

    # ------------------------------------------------------------------ info
    def summary(self):
        return {
            "traj_csv": self.traj_path if self.enabled else "",
            "segments_csv": self.segments_path if self.enabled else "",
            "clean_csv": self.clean_path if self.enabled else "",
        }
