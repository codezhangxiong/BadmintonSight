"""BadmintonSight 单帧决策诊断工具。

用途
----
给定视频、权重和目标帧号，脚本从第 1 帧开始，按 ``src/pipeline.py`` 中
``run_pipeline`` 的同一顺序逐帧推进跟踪状态；只在目标帧把
“检测 → 过滤 → 静止标记 → 线性预测 → λ 加权选点 → 门控 → 平滑/丢弃/漏检”
的完整决策链打印成人类可读的数字，并保存一张诊断图。

用法示例（Windows；运行时必须加 -B，禁止写 pyc）
------------------------------------------------
.. code-block:: text

    E:\\development\\BadmintonSight\\.venv\\Scripts\\python.exe -B ^
        E:\\development\\BadmintonSight\\scripts\\explain_frame.py ^
        --video videos\\badminton.mp4 ^
        --weights weights\\yolo11s-ball.pt ^
        --frame 120

    # 从任意目录运行（相对路径仍按项目根解析）
    cd /d E:\\development\\BadmintonSight\\.tmp-dev
    E:\\development\\BadmintonSight\\.venv\\Scripts\\python.exe -B ^
        E:\\development\\BadmintonSight\\scripts\\explain_frame.py ^
        --video E:\\development\\BadmintonSight\\videos\\badminton.mp4 ^
        --weights E:\\development\\BadmintonSight\\weights\\yolo11s-ball.pt ^
        --frame 20

相对路径统一按**项目根**（本文件上一级目录）解析，和 ``scripts/run_track.py``
的 ``_resolve`` 思路一致，因此不依赖当前工作目录。

重要说明
--------
这是**诊断工具**，不是第二套跟踪实现：检测、静止标记、预测、代价、选点、
门控、平滑全部直接调用 ``src/`` 里的函数，状态机的分支顺序逐行对齐
``src/pipeline.py``。最终行为以 ``src/pipeline.py`` 为准。

脚本不写输出视频、不弹 OpenCV 窗口、不修改 ``src/`` 下任何文件，运行产物只写
``--out-dir``（默认 ``.tmp-dev/explain``）。
"""

import argparse
import sys
import unicodedata
from collections import deque
from pathlib import Path

# 双保险：即使调用方忘了 -B，也绝不在 src/ 下生成 .pyc（这是只读诊断工具）。
sys.dont_write_bytecode = True

# 允许从任意工作目录直接运行：把项目根加入 sys.path 后 src/ 与 utils/ 才可导入。
# 必须在 import src.* 之前执行，写法参考 scripts/run_track.py。
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import yaml  # noqa: E402
from ultralytics import YOLO  # noqa: E402

from src.detector import extract_shuttlecock_candidates  # noqa: E402
from src.tracker import association  # noqa: E402
from src.tracker.association import compute_cost, select_best_candidate  # noqa: E402
from src.tracker.predictor import (  # noqa: E402
    dynamic_step_threshold,
    get_current_speed,
    predict_linear,
)
from utils.common import distance  # noqa: E402

DEFAULT_CONFIG = "configs/track_cfg.yaml"
DEFAULT_OUT_DIR = ".tmp-dev/explain"
DEFAULT_FRAME = 120
SECTION_WIDTH = 86

# 诊断图的 BGR 配色（OpenCV 是 BGR，不是 RGB）
COLOR_RAW = (160, 160, 160)       # 灰：YOLO 原始框
COLOR_CAND = (255, 0, 0)          # 蓝：普通候选
COLOR_STATIC = (255, 0, 255)      # 紫：静止候选
COLOR_SELECTED = (0, 255, 0)      # 绿：本帧选中
COLOR_REF = (0, 255, 255)         # 黄：线性预测参考点
COLOR_GATE = (0, 165, 255)        # 橙：门控圆
COLOR_SMOOTH = (0, 0, 255)        # 红：最终平滑点


# ---------------------------------------------------------------------------
# 控制台与文本对齐
# ---------------------------------------------------------------------------
def configure_console():
    """强制 stdout/stderr 用 UTF-8，避免 Windows 控制台把中文打成乱码。"""
    for stream in (sys.stdout, sys.stderr):
        if stream is None or not hasattr(stream, "reconfigure"):
            continue
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, OSError):
            pass


def _text_width(text):
    """按终端显示宽度算字符串长度：中文/全角字符占 2 列，ASCII 占 1 列。"""
    return sum(2 if unicodedata.east_asian_width(ch) in ("W", "F") else 1 for ch in str(text))


def _pad(text, width, align="l"):
    """按显示宽度补空格，保证含中文的列也能对齐（不会截断超长内容）。"""
    text = str(text)
    gap = max(0, width - _text_width(text))
    if align == "r":
        return " " * gap + text
    return text + " " * gap


def _field(label, value, width=30):
    """打印一行 ``标签：值``，标签按显示宽度对齐。"""
    print(f"  {_pad(label, width)}：{value}")


def _fmt_point(pt, digits=1):
    """把点格式化成 ``(x, y)``；候选点是整数时 digits=0。"""
    if pt is None:
        return "None"
    return f"({float(pt[0]):.{digits}f}, {float(pt[1]):.{digits}f})"


def _fmt_recent(points):
    """按 pipeline 的箭头风格展示最近几个轨迹点。"""
    if not points:
        return "（空）"
    return " → ".join(_fmt_point(p) for p in points)


# ---------------------------------------------------------------------------
# 参数、配置与路径
# ---------------------------------------------------------------------------
def _resolve(path_str, must_exist=False):
    """相对路径统一按项目根解析，避免依赖当前工作目录（同 run_track.py）。"""
    p = Path(path_str)
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    p = p.resolve()
    if must_exist and not p.exists():
        raise FileNotFoundError(f"找不到文件：{p}")
    return p


def parse_args(argv=None):
    """解析命令行参数。--frame 是 1-based 帧号，默认 120。"""
    parser = argparse.ArgumentParser(
        description="BadmintonSight 单帧决策诊断（行为以 src/pipeline.py 为准）"
    )
    parser.add_argument("--video", required=True, help="输入视频路径（相对路径按项目根解析）")
    parser.add_argument("--weights", required=True, help="YOLO .pt 权重路径（相对路径按项目根解析）")
    parser.add_argument(
        "--config",
        default=DEFAULT_CONFIG,
        help=f"配置 yaml（相对路径按项目根解析；默认 {DEFAULT_CONFIG}）",
    )
    parser.add_argument(
        "--frame",
        type=int,
        default=DEFAULT_FRAME,
        help=f"目标帧号，1-based（默认 {DEFAULT_FRAME}）",
    )
    parser.add_argument(
        "--out-dir",
        default=DEFAULT_OUT_DIR,
        help=f"诊断图输出目录（相对路径按项目根解析；默认 {DEFAULT_OUT_DIR}）",
    )
    args = parser.parse_args(argv)
    if args.frame <= 0:
        parser.error("--frame 必须是正整数（帧号从 1 开始）")
    return args


def load_config(config_path):
    """读取 yaml 并检查 run_pipeline 必需的 tracker/detector/ransac 三段。"""
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"配置文件内容为空或不是字典：{config_path}")
    for key in ("tracker", "detector", "ransac"):
        if key not in cfg:
            raise ValueError(f"配置文件缺少 '{key}' 段：{config_path}")
    return cfg


# ---------------------------------------------------------------------------
# 状态机：与 src/pipeline.py 的 run_pipeline 一一对应
# ---------------------------------------------------------------------------
class TrackingState:
    """run_pipeline 里那组逐帧可变状态的最小复刻。

    - ``smooth_win`` 用 ``deque(maxlen=smooth_window_len)``，和 pipeline 一样
      超过窗口长度时自动丢最旧的点；
    - ``full_traj`` 存的是**平滑后**的点，不是原始观测点；
    - ``roi`` 只在 detector 过滤时使用，视频尺寸确定后不再变化。
    """

    def __init__(self, cfg, width, height):
        track_cfg = cfg["tracker"]
        self.smooth_win = deque(maxlen=track_cfg["smooth_window_len"])
        self.full_traj = []
        self.full_traj_frames = []
        self.missing = 0
        self.max_miss = track_cfg["max_miss"]
        self.roi = [(50, 50), (width - 50, height - 50)]

    def reset_segment(self):
        """复刻 pipeline.finalize_segment 里的清缓冲动作（不涉及 RANSAC/落盘）。"""
        self.full_traj.clear()
        self.full_traj_frames.clear()
        self.smooth_win.clear()


def advance_state(state, best, fid, cfg):
    """按 run_pipeline 的同一顺序推进一帧状态，并返回本帧动作明细。

    为什么要这样算（顺序不能换）：
    1. 只有 ``best`` 存在且 ``smooth_win`` 非空时才门控；本段第一个点直接接受。
    2. 接受时先把**原始观测点** append 进平滑窗口，再对窗口求均值，均值才写进
       ``full_traj`` —— 所以预测用的是平滑点，不是抖动更大的原始点。
    3. ``missing`` 在接受时归零、拒收/漏检时 +1；``missing > max_miss`` 时
       pipeline 会在帧末清空整段轨迹缓冲，这里同样清空并把 missing 归零。
    """
    track_cfg = cfg["tracker"]
    detail = {
        "action": "miss",
        "obs": None,
        "smooth": None,
        "d": None,
        "th": None,
        "missing": state.missing,
        "missing_final": state.missing,
        "traj_len": len(state.full_traj),
        "window_len": len(state.smooth_win),
        "reset": False,
    }

    if best is not None:
        obs_pt = best["point"]
        d = None
        th = None
        outlier = False
        if len(state.smooth_win) > 0:
            last_p = state.smooth_win[-1]
            d = distance(obs_pt, last_p)
            th = dynamic_step_threshold(
                state.full_traj,
                base_gain=track_cfg["dynamic_gain"],
                min_thresh=track_cfg["dynamic_min_thresh"],
                max_thresh=track_cfg["dynamic_max_thresh"],
            )
            if d > th:
                outlier = True

        if outlier:
            state.missing += 1
            detail.update(action="rejected", obs=obs_pt, d=d, th=th)
        else:
            state.missing = 0
            state.smooth_win.append(np.array(obs_pt))
            arr = np.array(list(state.smooth_win))
            smooth_pt = tuple(np.mean(arr, axis=0))
            state.full_traj.append(smooth_pt)
            state.full_traj_frames.append(fid)
            detail.update(action="obs", obs=obs_pt, smooth=smooth_pt, d=d, th=th)
    else:
        state.missing += 1

    detail["missing"] = state.missing
    detail["traj_len"] = len(state.full_traj)
    detail["window_len"] = len(state.smooth_win)

    # pipeline 在每帧末尾检查 missing > MAX_MISS，然后 finalize_segment 清缓冲。
    if state.missing > state.max_miss:
        state.reset_segment()
        state.missing = 0
        detail["reset"] = True
    detail["missing_final"] = state.missing
    return detail


# ---------------------------------------------------------------------------
# 取数：目标帧报告
# ---------------------------------------------------------------------------
def _raw_boxes_for_draw(boxes):
    """把 Ultralytics Boxes 转成绘图用的 (x1, y1, x2, y2, conf) 列表。

    坐标取整方式与 run_pipeline 里画原始框的代码一致（int(cx - bw/2)），
    这样诊断图上的灰框和 pipeline 调试视频里的红框是同一批框。
    """
    if boxes is None:
        return []
    xywh = boxes.xywh.detach().cpu().numpy()
    if len(xywh) == 0:
        return []
    conf_tensor = getattr(boxes, "conf", None)
    if conf_tensor is None:
        confs = np.ones(len(xywh), dtype=np.float32)
    else:
        confs = conf_tensor.detach().cpu().numpy()
    rows = []
    for (cx, cy, bw, bh), conf in zip(xywh, confs):
        rows.append(
            (
                int(cx - bw / 2),
                int(cy - bh / 2),
                int(cx + bw / 2),
                int(cy + bh / 2),
                float(conf),
            )
        )
    return rows


def _candidate_rows(candidates, ref_pt, lam, static_penalty, best):
    """逐候选复算代价，返回 (序号, cand, cost_mot, cost_app, total, 是否选中)。

    选中的判定用 ``cand is best`` 的**对象身份**，而不是重新比大小：
    best 本身来自 association.select_best_candidate（权威实现），这里只负责
    把它挑中的那个候选和它的分数并排展示出来。
    """
    rows = []
    for idx, cand in enumerate(candidates):
        total, cost_mot, cost_app = compute_cost(
            cand, ref_pt, lam, static_penalty=static_penalty
        )
        rows.append(
            (
                idx,
                cand,
                float(cost_mot),
                float(cost_app),
                float(total),
                cand is best,
            )
        )
    return rows


def _build_story(report):
    """把本帧决策翻译成 3~6 行大白话，少用术语，数字和表格保持一致。"""
    detail = report["action_detail"]
    action = detail["action"]
    raw = report["raw_count"]
    n = report["candidate_count"]
    lines = []

    if n == 0:
        if raw == 0:
            lines.append("这一帧 YOLO 一个框都没给出，所以没有任何候选。")
        else:
            lines.append(
                f"这一帧 YOLO 给出 {raw} 个框，但按置信度、面积、长宽比和 ROI "
                "筛完，一个候选都没留下。"
            )
        if report["ref_pt"] is not None:
            lines.append(
                f"按之前的轨迹，球本来预计会出现在 {_fmt_point(report['ref_pt'])} 附近，"
                "但这一帧没有候选可用。"
            )
        else:
            lines.append("轨迹还不到 4 个点，本来也没法预测球会去哪；这一帧又没有候选。")
        missing_line = (
            f"程序把这一帧记为漏检，missing 从 {report['missing_before']} "
            f"变成 {detail['missing']}"
        )
        if detail["reset"]:
            missing_line += "；清空发生在帧末，见下一句。"
        else:
            missing_line += "，轨迹点和平滑窗口都不动。"
        lines.append(missing_line)
        if detail["reset"]:
            if report["len_before"] > 0:
                lines.append(
                    f"这一帧之后 missing 超过上限 {report['max_miss']}，整段轨迹被清空，"
                    "下一帧会重新开始。"
                )
            else:
                lines.append(
                    f"这一帧之后 missing 超过上限 {report['max_miss']}，程序把计数归零；"
                    "本来也没有轨迹可清，下一帧继续等候选。"
                )
        return _ensure_story_lines(lines)

    lines.append(f"这一帧 YOLO 给出 {raw} 个框，筛完留下 {n} 个候选。")
    if report["ref_pt"] is None:
        lines.append("轨迹还不到 4 个点，程序没法预测球的位置，只能先看“长得像球”的外观分。")
    else:
        lines.append(
            f"程序用最近 4 个平滑点估计方向，预测球这一帧会出现在 "
            f"{_fmt_point(report['ref_pt'])} 附近。"
        )

    best_idx = report["best_index"]
    only_one = n == 1
    if report["ref_pt"] is None:
        if only_one:
            select_line = f"唯一的 {best_idx} 号候选，外观总分 {report['best_total']:.2f}"
        else:
            select_line = (
                f"比较外观分后，{best_idx} 号候选的总分 {report['best_total']:.2f} 最低"
            )
    else:
        if only_one:
            select_line = f"唯一的 {best_idx} 号候选离预测点 {report['best_dist_ref']:.1f} 像素"
        else:
            select_line = f"{best_idx} 号候选离预测点 {report['best_dist_ref']:.1f} 像素"
        if report["best_static"]:
            select_line += (
                f"（它连续几帧待在同一小格，被标成静止候选，代价额外加 "
                f"{report['static_penalty']:.0f}）"
            )
        if only_one:
            select_line += f"，加权总分 {report['best_total']:.2f}"
        else:
            select_line += f"，加权总分 {report['best_total']:.2f} 最低"
    select_line += "，被选为这一帧的观测点。"
    lines.append(select_line)

    if action == "rejected":
        lines.append(
            f"但它离上一个平滑点 {detail['d']:.1f} 像素，超过允许的 "
            f"{detail['th']:.1f} 像素，像是“瞬移”，被丢掉。"
        )
        lines.append(
            f"丢掉的点不会进平滑窗口，轨迹保持原样，missing 变成 {detail['missing']}。"
        )
        if detail["reset"]:
            lines.append(
                f"恰好这一帧之后 missing 超过上限 {report['max_miss']}，整段轨迹被清空，"
                "下一帧会重新开始。"
            )
        return _ensure_story_lines(lines)

    # action == "obs"：本帧接受
    if report["smooth_prev"] is None:
        lines.append("这是本段收下的第一个点，没有上一平滑点可以比较，不做门控，直接接受。")
    else:
        lines.append(
            f"它离上一个平滑点 {detail['d']:.1f} 像素，没有超过允许的 "
            f"{detail['th']:.1f} 像素，所以判定为正常移动。"
        )
    lines.append(
        f"把它放进平滑窗口（现在 {detail['window_len']} 个点）求平均，"
        f"得到新的平滑点 {_fmt_point(detail['smooth'])}，missing 归零。"
    )
    return _ensure_story_lines(lines)


def _ensure_story_lines(lines):
    """兜底：保证“这一段发生了什么”始终是 3~6 行。"""
    if len(lines) < 3:
        lines.append("这一帧的状态变化见上面的数字。")
    return lines[:6]


# ---------------------------------------------------------------------------
# 诊断图
# ---------------------------------------------------------------------------
def _ipt(pt):
    """把 float 点转成 OpenCV 需要的整数像素坐标。"""
    return (int(round(float(pt[0]))), int(round(float(pt[1]))))


def _label(canvas, text, org, color, scale=0.45, thickness=1):
    """带黑色描边的文字，保证在浅色/深色场地上都能看清。"""
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(canvas, text, org, font, scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
    cv2.putText(canvas, text, org, font, scale, color, thickness, cv2.LINE_AA)


def draw_diagnostic(frame, report):
    """在原始帧副本上画出本帧所有决策元素，返回可直接 imwrite 的 BGR 图。

    绘制顺序有讲究：先画门控圆和轨迹参考，再画候选点，最后画选中的绿点和
    平滑点红点，避免大圆/框盖住真正关键的点。
    """
    canvas = frame.copy()
    height, width = canvas.shape[:2]

    # 1) YOLO 原始框（灰）—— detector 过滤前的全量输入
    for x1, y1, x2, y2, conf in report["raw_boxes"]:
        cv2.rectangle(canvas, (x1, y1), (x2, y2), COLOR_RAW, 1, cv2.LINE_AA)
        _label(canvas, f"{conf:.2f}", (x1, max(12, y1 - 3)), COLOR_RAW, scale=0.35)

    # 2) 门控圆（橙，细线）：圆心=上一平滑点，半径=th
    smooth_prev = report["smooth_prev"]
    th = report["th_used"] if report["th_used"] is not None else report["th_preview"]
    if smooth_prev is not None and th is not None:
        center = _ipt(smooth_prev)
        radius = max(1, int(round(float(th))))
        cv2.circle(canvas, center, radius, COLOR_GATE, 1, cv2.LINE_AA)
        _label(
            canvas,
            f"gate r={float(th):.1f}",
            (max(4, center[0] - 55), max(14, center[1] - radius - 6)),
            COLOR_GATE,
            scale=0.42,
        )

    # 3) 线性预测参考点（黄色空心圆 + 十字）
    if report["ref_pt"] is not None:
        ref = _ipt(report["ref_pt"])
        cv2.circle(canvas, ref, 9, COLOR_REF, 1, cv2.LINE_AA)
        cv2.line(canvas, (ref[0] - 13, ref[1]), (ref[0] + 13, ref[1]), COLOR_REF, 1, cv2.LINE_AA)
        cv2.line(canvas, (ref[0], ref[1] - 13), (ref[0], ref[1] + 13), COLOR_REF, 1, cv2.LINE_AA)
        _label(canvas, "ref_pt", (ref[0] + 14, ref[1] - 10), COLOR_REF, scale=0.42)

    # 4) 候选点：普通=蓝，静止=紫，选中=绿且更大更粗
    best_index = report["best_index"]
    for idx, cand, _cost_mot, _cost_app, _total, selected in report["rows"]:
        pt = _ipt(cand["point"])
        if selected:
            cv2.circle(canvas, pt, 10, COLOR_SELECTED, 2, cv2.LINE_AA)
            cv2.circle(canvas, pt, 5, COLOR_SELECTED, -1, cv2.LINE_AA)
            _label(canvas, f"#{idx} selected", (pt[0] + 11, pt[1] + 5), COLOR_SELECTED, scale=0.48)
        else:
            color = COLOR_STATIC if cand.get("is_static") else COLOR_CAND
            cv2.circle(canvas, pt, 4, color, -1, cv2.LINE_AA)
            _label(canvas, f"#{idx}", (pt[0] + 7, pt[1] + 4), color, scale=0.45)

    # 5) 被门控丢弃的观测点：红色叉，和“新平滑点红点”区分开
    detail = report["action_detail"]
    if detail["action"] == "rejected" and detail["obs"] is not None:
        obs = _ipt(detail["obs"])
        cv2.line(canvas, (obs[0] - 8, obs[1] - 8), (obs[0] + 8, obs[1] + 8), COLOR_SMOOTH, 2, cv2.LINE_AA)
        cv2.line(canvas, (obs[0] - 8, obs[1] + 8), (obs[0] + 8, obs[1] - 8), COLOR_SMOOTH, 2, cv2.LINE_AA)
        _label(canvas, "rejected", (obs[0] + 10, obs[1] - 8), COLOR_SMOOTH, scale=0.45)

    # 6) 最终平滑点（红点，只在本帧接受时存在）
    if detail["smooth"] is not None:
        smooth = _ipt(detail["smooth"])
        cv2.circle(canvas, smooth, 7, COLOR_SMOOTH, -1, cv2.LINE_AA)
        _label(canvas, "smooth", (smooth[0] + 9, smooth[1] + 16), COLOR_SMOOTH, scale=0.5)

    # 7) 左上角图例 + 右上角帧号
    legend = [
        ("gray  : YOLO raw box", COLOR_RAW),
        ("blue  : candidate", COLOR_CAND),
        ("purple: static candidate", COLOR_STATIC),
        ("green : selected candidate", COLOR_SELECTED),
        ("yellow: ref_pt (linear prediction)", COLOR_REF),
        ("orange: gate circle radius = th", COLOR_GATE),
        ("red   : smoothed point", COLOR_SMOOTH),
    ]
    box_w, box_h = 330, 16 * len(legend) + 14
    overlay = canvas.copy()
    cv2.rectangle(overlay, (8, 8), (8 + box_w, 8 + box_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, canvas, 0.45, 0, canvas)
    for i, (text, color) in enumerate(legend):
        cv2.putText(
            canvas,
            text,
            (16, 26 + i * 16),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            color,
            1,
            cv2.LINE_AA,
        )
    _label(
        canvas,
        f"frame {report['frame']}  lambda={report['lam']:.2f}",
        (width - 290, 26),
        (255, 255, 255),
        scale=0.55,
    )
    return canvas


def _save_png(canvas, png_path):
    """用 imencode 写 PNG，避开 cv2.imwrite 在非 ASCII 路径上的编码问题。"""
    ok, buffer = cv2.imencode(".png", canvas)
    if not ok:
        raise RuntimeError(f"OpenCV 编码 PNG 失败：{png_path}")
    try:
        Path(png_path).write_bytes(buffer.tobytes())
    except OSError as exc:
        raise RuntimeError(f"无法写入诊断图 {png_path}：{exc}") from exc


# ---------------------------------------------------------------------------
# 报告输出
# ---------------------------------------------------------------------------
def _print_candidate_table(rows):
    """打印候选代价表；列宽用显示宽度计算，中文表头也能和数字对齐。"""
    columns = [
        ("序号", 5, "r"),
        ("point", 16, "l"),
        ("conf", 7, "r"),
        ("area_ratio", 11, "r"),
        ("is_static", 10, "l"),
        ("cost_mot", 9, "r"),
        ("cost_app", 9, "r"),
        ("total", 9, "r"),
        ("选中", 5, "l"),
    ]
    print("  " + " ".join(_pad(name, width, align) for name, width, align in columns))
    for idx, cand, cost_mot, cost_app, total, selected in rows:
        cells = [
            (idx, 5, "r"),
            (_fmt_point(cand["point"], digits=0), 16, "l"),
            (f"{float(cand['confidence']):.3f}", 7, "r"),
            (f"{float(cand['area_ratio']):.6f}", 11, "r"),
            ("是" if cand.get("is_static") else "否", 10, "l"),
            (f"{cost_mot:.2f}", 9, "r"),
            (f"{cost_app:.3f}", 9, "r"),
            (f"{total:.3f}", 9, "r"),
            ("是" if selected else "", 5, "l"),
        ]
        print("  " + " ".join(_pad(value, width, align) for value, width, align in cells))


def print_report(report):
    """按“检测 → 权重 → 候选 → 状态 → 门控 → 动作 → 帧后 → 总结”的顺序打印。"""
    detail = report["action_detail"]
    action = detail["action"]

    print("=" * SECTION_WIDTH)
    print("BadmintonSight 单帧决策诊断")
    print("=" * SECTION_WIDTH)
    _field("视频", report["video"], width=14)
    _field("权重", report["weights"], width=14)
    _field("配置", report["config"], width=14)
    _field(
        "视频信息",
        f"{report['width']}x{report['height']}，{report['fps']:.2f} fps，"
        f"总帧数 {report['total_frames']}",
        width=14,
    )
    _field(
        "目标帧",
        f"第 {report['frame']} 帧（1-based；已从第 1 帧顺序推进到本帧）",
        width=14,
    )
    for warning in report["warnings"]:
        print(f"[警告] {warning}")

    print()
    print("[1] 检测结果")
    _field("YOLO 原始框数", report["raw_count"])
    _field("detector 过滤后候选数", report["candidate_count"])
    raw = report["raw_count"]
    n = report["candidate_count"]
    if raw == 0:
        filter_note = "YOLO 没有给出任何框"
    elif raw == n:
        filter_note = "原始框全部通过过滤"
    else:
        filter_note = f"筛掉 {raw - n} 个（置信度/面积/长宽比/ROI 任一不合格）"
    _field("过滤说明", filter_note)

    print()
    print("[2] 关联权重 λ（运动预测 vs 外观的取舍）")
    _field("当前速度", f"{report['speed']:.2f} px/帧{report['speed_note']}")
    _field("fast_speed_thresh", f"{report['fast_th']:.2f} px/帧")
    _field("权重规则", report["lam_rule"])
    _field("λ", f"{report['lam']:.3f}")
    _field("代价公式", "total = λ*cost_mot + (1-λ)*cost_app")
    if report["ref_pt"] is None:
        _field(
            "补充",
            "ref_pt 不存在时 compute_cost 直接返回 total=cost_app、cost_mot=0，"
            "λ 不参与实际排序",
        )

    print()
    print("[3] 候选与代价（同一帧内横向比较）")
    if not report["rows"]:
        print("  （本帧 0 个候选；select_best_candidate 返回 None）")
    else:
        _print_candidate_table(report["rows"])
        print(
            "  说明：cost_mot = 到 ref_pt 的距离 + 静止惩罚；"
            "cost_app = -conf + 3000 * area_ratio"
        )
        print(
            f"        静止候选（is_static=是）的 cost_mot 额外 +"
            f"{report['static_penalty']:.0f}；total = λ*cost_mot + (1-λ)*cost_app。"
        )
        if report["ref_pt"] is None:
            print(
                "        本帧 ref_pt=None，compute_cost 返回 cost_mot=0、total=cost_app；"
                "λ 不参与实际排序。"
            )

    print()
    print("[4] 本帧决策前的状态")
    _field("full_traj 点数", report["len_before"])
    _field("最近 4 个点", _fmt_recent(report["recent"]))
    if report["ref_pt"] is None:
        ref_text = "None（轨迹不足 4 点，无法外推）"
    else:
        ref_text = f"{_fmt_point(report['ref_pt'])}（最近 4 点平均速度外推 1 步）"
    _field("predict_linear", ref_text)
    _field(
        "dynamic_step_threshold",
        f"{report['th_preview']:.2f} px（{report['th_rule']}）",
    )
    _field(
        "上一平滑点",
        _fmt_point(report["smooth_prev"]) if report["smooth_prev"] is not None else "（空）",
    )

    print()
    print("[5] 门控判定（是否把这个观测点并入轨迹）")
    if detail["action"] == "miss":
        _field("obs", "无（本帧没有候选）")
        _field("判定", "未执行：没有观测点，按 miss 处理")
    elif report["smooth_prev"] is None:
        _field("obs", _fmt_point(detail["obs"], digits=0))
        _field("上一平滑点", "（空）")
        _field("判定", "未执行：smooth_win 为空，本段第一个观测点直接接受")
    else:
        _field("obs", _fmt_point(detail["obs"], digits=0))
        _field("上一平滑点", _fmt_point(report["smooth_prev"]))
        _field("d = 距离(obs, 上一平滑点)", f"{detail['d']:.2f} px")
        _field("th = dynamic_step_threshold", f"{detail['th']:.2f} px")
        verdict = (
            "d > th → 判为离群，丢弃"
            if action == "rejected"
            else "d <= th → 不离群，接受"
        )
        _field("判定", verdict)

    print()
    print("[6] 最终动作")
    if action == "obs":
        _field("动作", "obs（接受）")
        _field("obs", _fmt_point(detail["obs"], digits=0))
        _field(
            "smooth_pt",
            f"{_fmt_point(detail['smooth'])}"
            f"（smooth_win {detail['window_len']} 个点的均值）",
        )
        _field("missing", "0（接受后归零）")
    elif action == "rejected":
        _field("动作", "rejected（离群丢弃）")
        _field("obs", _fmt_point(detail["obs"], digits=0))
        _field("d / th", f"{detail['d']:.2f} / {detail['th']:.2f} px")
        _field("missing", f"{detail['missing']}（+1）")
        reset_note = "；帧末清空见 [7]" if detail["reset"] else ""
        _field("平滑窗口/轨迹", "不变（这个点不会进入 full_traj）" + reset_note)
    else:
        _field("动作", "miss（无候选）")
        _field("missing", f"{detail['missing']}（+1）")
        reset_note = "（帧末清空见 [7]）" if detail["reset"] else ""
        _field("平滑窗口/轨迹", "不变" + reset_note)

    print()
    print("[7] 帧后状态")
    if detail["reset"]:
        had_state = report["len_before"] > 0 or report["smooth_prev"] is not None
        _field("full_traj 点数", "0（已清空）" if had_state else "0（保持为空）")
        _field("smooth_win 长度", "0（已清空）" if had_state else "0（保持为空）")
        _field("missing", f"0（清空时归零；触发前为 {detail['missing']}）")
        if had_state:
            _field(
                "轨迹段",
                f"已清空：missing {detail['missing']} > max_miss {report['max_miss']}，"
                "下一帧重新开段",
            )
        else:
            _field(
                "轨迹段",
                f"计数归零：missing {detail['missing']} > max_miss {report['max_miss']}；"
                "本来就没有轨迹，下一帧继续等候选",
            )
    else:
        traj_note = "（本帧 +1）" if action == "obs" else "（不变）"
        window_note = "（本帧接受了 1 个点）" if action == "obs" else "（不变）"
        _field("full_traj 点数", f"{report['state_after']['full_traj']}{traj_note}")
        _field("smooth_win 长度", f"{report['state_after']['smooth_win']}{window_note}")
        _field("missing", report["state_after"]["missing"])
        _field(
            "轨迹段",
            f"保留（missing {report['state_after']['missing']} "
            f"<= max_miss {report['max_miss']}）",
        )

    print()
    print("[8] 这一段发生了什么")
    for line in report["story"]:
        print(f"  {line}")

    print()
    print("=" * SECTION_WIDTH)
    print(f"诊断图    : {report['png_path']}（{report['png_size']} 字节）")
    print("说明      : 这是诊断工具，不写视频、不弹窗；逐帧行为以 src/pipeline.py 的 run_pipeline 为准。")


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def explain_frame(video_path, weights_path, config_path, cfg, frame_no, out_dir):
    """顺序推进到第 frame_no 帧，收集决策数字、画图并保存，返回报告 dict。

    注意：为了让“第 N 帧之前”的状态与 run_pipeline 完全一致，前面每一帧都必须
    真实跑 YOLO、真实调用 mark_static_candidates（它有跨帧记忆），不能跳帧。
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"无法用 OpenCV 打开视频：{video_path}")

    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        if not fps or fps <= 0:
            fps = 30.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if width <= 0 or height <= 0:
            raise RuntimeError(f"无法读取视频尺寸：{video_path}")
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if total_frames > 0 and frame_no > total_frames:
            raise ValueError(
                f"目标帧 {frame_no} 超出视频总帧数 {total_frames}（视频：{video_path}）"
            )

        track_cfg = cfg["tracker"]
        state = TrackingState(cfg, width, height)

        # association.static_candidate_memory 是模块级全局变量。run_track.py 每次
        # 都在新进程里跑，等价于空记忆；这里显式清空，保证重复调用也从头开始。
        association.static_candidate_memory.clear()

        try:
            model = YOLO(str(weights_path))
        except Exception as exc:
            raise RuntimeError(f"无法加载 YOLO 权重：{weights_path}（{exc}）") from exc

        processed = 0
        target_image = None
        report = None

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            processed += 1
            # run_pipeline 也是读帧后用 CAP_PROP_POS_FRAMES 作为 1-based 帧号。
            fid = int(cap.get(cv2.CAP_PROP_POS_FRAMES))

            results = model(frame, conf=cfg["detector"]["yolo_conf"], verbose=False)[0]
            boxes = results.boxes
            raw_count = 0 if boxes is None else int(boxes.xywh.shape[0])

            candidates = extract_shuttlecock_candidates(boxes, frame.shape, cfg, state.roi)
            candidates = association.mark_static_candidates(
                candidates,
                fid,
                grid_size=track_cfg["grid_size"],
                static_frame_count=track_cfg["static_frame_count"],
            )

            ref_pt = predict_linear(state.full_traj, n_lookback=4)
            best = select_best_candidate(candidates, ref_pt, state.full_traj, cfg)

            if fid == frame_no:
                # ---- 决策前快照：这些数字必须在本帧 advance_state 之前取 ----
                len_before = len(state.full_traj)
                missing_before = state.missing
                recent = [tuple(float(v) for v in p) for p in state.full_traj[-4:]]
                smooth_prev = (
                    tuple(float(v) for v in state.smooth_win[-1])
                    if len(state.smooth_win) > 0
                    else None
                )
                speed = get_current_speed(state.full_traj)
                fast_th = float(track_cfg["fast_speed_thresh"])

                # λ 规则照抄 association.select_best_candidate，只为显式展示。
                if ref_pt is None:
                    lam = 0.5
                    lam_rule = "ref_pt 不存在 → 固定 λ=0.5（select_best_candidate 的分支）"
                elif speed > fast_th:
                    lam = float(track_cfg["lambda_fast"])
                    lam_rule = f"{speed:.2f} > {fast_th:.2f} → 快速，λ=lambda_fast={lam:.3f}"
                else:
                    lam = float(track_cfg["lambda_slow"])
                    lam_rule = f"{speed:.2f} <= {fast_th:.2f} → 慢速，λ=lambda_slow={lam:.3f}"

                if len_before < 4:
                    speed_note = (
                        f"（full_traj 只有 {len_before} 点，不足 4 点，函数返回 0.0）"
                    )
                else:
                    speed_note = "（最近 4 个平滑点的平均步长）"

                th_preview = dynamic_step_threshold(
                    state.full_traj,
                    base_gain=track_cfg["dynamic_gain"],
                    min_thresh=track_cfg["dynamic_min_thresh"],
                    max_thresh=track_cfg["dynamic_max_thresh"],
                )
                if len_before < 5:
                    th_rule = (
                        "轨迹不足 5 点，函数固定返回 180.0，且不使用配置的 min/max"
                    )
                else:
                    th_rule = (
                        f"最近 5 点平均步长 × {track_cfg['dynamic_gain']:g}，"
                        f"再裁剪到 [{track_cfg['dynamic_min_thresh']:g}, "
                        f"{track_cfg['dynamic_max_thresh']:g}]"
                    )

                rows = _candidate_rows(
                    candidates,
                    ref_pt,
                    lam,
                    float(track_cfg["static_penalty"]),
                    best,
                )
                best_row = next((row for row in rows if row[5]), None)
                best_index = best_row[0] if best_row is not None else None
                best_total = best_row[4] if best_row is not None else None
                best_dist_ref = (
                    distance(best["point"], ref_pt)
                    if best is not None and ref_pt is not None
                    else None
                )

                warnings = []
                if rows and best is not None:
                    argmin_row = min(rows, key=lambda row: row[4])
                    if argmin_row[1] is not best:
                        warnings.append(
                            f"复算代价的最优候选是 {argmin_row[0]} 号，但 "
                            f"select_best_candidate 返回 {best_index} 号；"
                            "请检查 λ/代价参数是否与 src 不一致。"
                        )

                # 快照取完后再推进状态，保证展示的“决策前状态”真的是决策前。
                detail = advance_state(state, best, fid, cfg)

                report = {
                    "video": str(video_path),
                    "weights": str(weights_path),
                    "config": str(config_path),
                    "fps": float(fps),
                    "total_frames": total_frames,
                    "width": width,
                    "height": height,
                    "frame": frame_no,
                    "processed": processed,
                    "raw_count": raw_count,
                    "candidate_count": len(candidates),
                    "raw_boxes": _raw_boxes_for_draw(boxes),
                    "rows": rows,
                    "best_index": best_index,
                    "best_total": best_total,
                    "best_dist_ref": best_dist_ref,
                    "best_static": bool(best.get("is_static", False)) if best else False,
                    "len_before": len_before,
                    "missing_before": missing_before,
                    "recent": recent,
                    "ref_pt": ref_pt,
                    "speed": float(speed),
                    "fast_th": fast_th,
                    "lam": float(lam),
                    "lam_rule": lam_rule,
                    "speed_note": speed_note,
                    "th_preview": float(th_preview),
                    "th_rule": th_rule,
                    "th_used": detail["th"],
                    "smooth_prev": smooth_prev,
                    "action_detail": detail,
                    "state_after": {
                        "full_traj": len(state.full_traj),
                        "smooth_win": len(state.smooth_win),
                        "missing": state.missing,
                    },
                    "max_miss": state.max_miss,
                    "static_penalty": float(track_cfg["static_penalty"]),
                    "warnings": warnings,
                }
                target_image = frame.copy()
                break

            if fid > frame_no:
                raise ValueError(
                    f"帧号异常：期望停在 {frame_no}，CAP_PROP_POS_FRAMES 却返回 {fid}"
                )

            # 非目标帧：静默推进，完全复用同一个状态机函数。
            advance_state(state, best, fid, cfg)
    finally:
        cap.release()

    if report is None or target_image is None:
        raise ValueError(
            f"没有读到第 {frame_no} 帧：视频实际只读出 {processed} 帧"
            f"（元信息总帧数 {total_frames}）。请换一个更小的 --frame。"
        )

    report["story"] = _build_story(report)

    # 画图、落盘。输出目录在 main 里已做过预检；这里再建一次以防函数被单独调用。
    out_dir = Path(out_dir)
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise RuntimeError(f"无法创建输出目录 {out_dir}：{exc}") from exc

    png_path = out_dir / f"frame_{frame_no:04d}.png"
    canvas = draw_diagnostic(target_image, report)
    _save_png(canvas, png_path)
    report["png_path"] = str(png_path)
    report["png_size"] = int(png_path.stat().st_size)
    return report


def main(argv=None):
    """CLI 入口：参数/路径校验失败一律中文报错并返回退出码 1。"""
    configure_console()
    args = parse_args(argv)

    try:
        config_path = _resolve(args.config, must_exist=True)
        video_path = _resolve(args.video, must_exist=True)
        weights_path = _resolve(args.weights, must_exist=True)
    except FileNotFoundError as exc:
        print(f"错误：{exc}", file=sys.stderr)
        print(
            "提示：--video / --weights / --config 的相对路径都按项目根解析；"
            "也可以直接传绝对路径。",
            file=sys.stderr,
        )
        return 1

    try:
        cfg = load_config(config_path)
    except (OSError, ValueError, yaml.YAMLError) as exc:
        print(f"错误：无法加载配置 {config_path}：{exc}", file=sys.stderr)
        return 1
    cfg["model_path"] = str(weights_path)

    out_dir = _resolve(args.out_dir)
    if out_dir.exists() and not out_dir.is_dir():
        print(f"错误：--out-dir 不是目录：{out_dir}", file=sys.stderr)
        return 1
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        print(f"错误：无法创建输出目录 {out_dir}：{exc}", file=sys.stderr)
        return 1

    try:
        report = explain_frame(
            video_path,
            weights_path,
            config_path,
            cfg,
            args.frame,
            out_dir,
        )
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"错误：{exc}", file=sys.stderr)
        return 1

    print_report(report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
