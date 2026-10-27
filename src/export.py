"""轨迹与分段结果的 CSV 导出。

三段式流程里，三份 sidecar CSV（与输出视频同名 stem）由不同阶段写出：

    <stem>.traj.csv      逐帧事件表（采集阶段写：obs / rejected / miss）
    <stem>.segments.csv  每段轨迹汇总（段处理阶段写）
    <stem>.clean.csv     每段清洗后的 (frame, x, y)（段处理阶段写）

对应三个写入器：
    FrameRecorder     → traj.csv
    SegmentRecorder   → segments.csv + clean.csv
    TrajectoryRecorder→ 三份一起写（向后兼容旧调用方，内部组合上面两个）

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


def _sidecar_path(out_path, kind):
    """按输出视频的目录 + 主干生成 sidecar 路径，例如 foo.mp4 → foo.traj.csv。"""
    stem = os.path.splitext(os.path.abspath(out_path))[0]
    return f"{stem}.{kind}.csv"


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


class FrameRecorder:
    """逐帧事件表写入器（<stem>.traj.csv），采集阶段使用。"""

    def __init__(self, out_path, enabled=True):
        self.enabled = enabled
        self.traj_path = _sidecar_path(out_path, "traj")
        self._file = None
        self._writer = None
        self._closed = False
        if self.enabled:
            self._open()

    def _open(self):
        os.makedirs(os.path.dirname(self.traj_path) or ".", exist_ok=True)
        self._file = open(self.traj_path, "w", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._file, fieldnames=TRAJ_FIELDS)
        self._writer.writeheader()

    def close(self):
        if self._closed:
            return
        fh = self._file
        self._file = None
        self._closed = True
        if fh is not None:
            try:
                fh.close()
            except Exception:
                pass

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        self.close()
        return False

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
        segment_id  当前活跃段编号；段外帧传 None → CSV 留空
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
        self._writer.writerow(
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

    def summary(self):
        return {"traj_csv": self.traj_path if self.enabled else ""}


class SegmentRecorder:
    """分段汇总 + 清洗轨迹写入器（<stem>.segments.csv / <stem>.clean.csv），段处理阶段使用。"""

    def __init__(self, out_path, enabled=True):
        self.enabled = enabled
        self.segments_path = _sidecar_path(out_path, "segments")
        self.clean_path = _sidecar_path(out_path, "clean")
        self._seg_file = None
        self._clean_file = None
        self._seg_writer = None
        self._clean_writer = None
        self._closed = False
        if self.enabled:
            self._open()

    def _open(self):
        os.makedirs(os.path.dirname(self.segments_path) or ".", exist_ok=True)
        opened = []
        try:
            self._seg_file = open(self.segments_path, "w", newline="", encoding="utf-8")
            opened.append(self._seg_file)
            self._clean_file = open(self.clean_path, "w", newline="", encoding="utf-8")
            opened.append(self._clean_file)
        except Exception:
            # 部分打开失败时把已打开的句柄关掉，避免泄漏
            for fh in opened:
                try:
                    fh.close()
                except Exception:
                    pass
            self._seg_file = self._clean_file = None
            raise
        self._seg_writer = csv.DictWriter(self._seg_file, fieldnames=SEGMENT_FIELDS)
        self._clean_writer = csv.DictWriter(self._clean_file, fieldnames=CLEAN_FIELDS)
        self._seg_writer.writeheader()
        self._clean_writer.writeheader()

    def close(self):
        if self._closed:
            return
        files = (self._seg_file, self._clean_file)
        self._seg_file = self._clean_file = None
        self._closed = True
        for fh in files:
            if fh is not None:
                try:
                    fh.close()
                except Exception:
                    pass

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        self.close()
        return False

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

    def summary(self):
        return {
            "segments_csv": self.segments_path if self.enabled else "",
            "clean_csv": self.clean_path if self.enabled else "",
        }


class TrajectoryRecorder:
    """三份 CSV 一起写的组合写入器（向后兼容旧调用方）。"""

    def __init__(self, out_path, enabled=True):
        self.enabled = enabled
        self._frames = FrameRecorder(out_path, enabled)
        self._segments = SegmentRecorder(out_path, enabled)

    @property
    def traj_path(self):
        return self._frames.traj_path

    @property
    def segments_path(self):
        return self._segments.segments_path

    @property
    def clean_path(self):
        return self._segments.clean_path

    def add_frame(self, *args, **kwargs):
        self._frames.add_frame(*args, **kwargs)

    def add_segment(self, *args, **kwargs):
        self._segments.add_segment(*args, **kwargs)

    def add_clean_points(self, *args, **kwargs):
        self._segments.add_clean_points(*args, **kwargs)

    def add_clean_flat(self, *args, **kwargs):
        self._segments.add_clean_flat(*args, **kwargs)

    def close(self):
        self._frames.close()
        self._segments.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        self.close()
        return False

    def summary(self):
        out = {}
        out.update(self._frames.summary())
        out.update(self._segments.summary())
        return out
