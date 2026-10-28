"""门控模块：把"这一帧观测是否接受"的判定从 replay 循环里拆出来。

门控链（按顺序执行，任何一门拒绝即拒绝）：

1. **DistanceGate（距离门）**
   观测到"上一个平滑点"的距离 > 动态阈值 → 拒绝。阈值 = 最近 N 步平均步长 × gain，
   再裁剪到 [min_thresh, max_thresh]（实测几乎总是贴 min_thresh）。

2. **MotionModelGate（运动模型门）**
   对最近 `window` 个**已接受观测**按时间 t 分别拟合 x(t)、y(t) 二次曲线
   （时间参数化，允许 x 非单调；比图像平面 y=f(x) 更适合往返轨迹），
   预测当前帧位置，计算候选观测到预测位置的残差。残差过大 → 拒绝。

   **击球/折返处理**（关键）：击球瞬间速度方向突变，二次模型必然失配。
   若直接按残差拒绝，会把正确的击球点当离群点误杀。因此本门在两处做特殊处理：
   - 窗口自身拟合 RMS 过大 → 说明窗口已经跨越了击球点（或轨迹折返）：
     重置窗口（只保留最后一个观测）并**放行**；
   - 残差过大但方向明显反转（观测位移与模型预测速度夹角 > 90°）→ 判定为击球：
     重置窗口并**放行**。
   只有"残差大且方向不反转"才判为真正的离群观测。

设计约束：
- 门控是纯判定（给决策），不修改轨迹状态；窗口由门控自己维护（只记录已接受观测）。
- 所有阈值来自 config，缺省值在 `DEFAULTS` 里，缺配置时也能跑。
"""

from collections import deque

import numpy as np

from src.tracker.predictor import dynamic_step_threshold
from utils.common import distance

# 运动模型门默认参数（config 里 tracker.motion_model 可覆盖）
DEFAULTS = {
    "enabled": True,
    "window": 6,            # 拟合窗口长度（最近 N 个已接受观测）
    "min_points": 5,        # 至少这么多点才拟合（二次 3 参数 + 冗余）
    "residual_thresh": 30.0,  # 候选到曲线预测位置的距离阈值（px）
    "rms_thresh": 20.0,     # 窗口自身拟合 RMS 阈值（px），超过视为跨越击球点
    "reverse_cos": -0.2,    # 方向反转判定：cos(预测速度, 实测位移) < 该值视为击球
}


def _fit_quadratic(times, values):
    """对 (t, v) 做二次拟合，返回 (coeff, t0, scale, rms, 在 t 处的预测函数所需参数)。

    t 先归一化到窗口内 [0, 1]，避免 t² 在 50fps 帧号下病态。
    """
    t = np.asarray(times, dtype=float)
    v = np.asarray(values, dtype=float)
    t0 = float(t[0])
    scale = float(max(1.0, t[-1] - t[0]))
    tn = (t - t0) / scale
    A = np.vstack([tn * tn, tn, np.ones(len(tn))]).T
    coeff, *_ = np.linalg.lstsq(A, v, rcond=None)
    pred = A @ coeff
    rms = float(np.sqrt(np.mean((pred - v) ** 2)))
    return coeff, t0, scale, rms


def _predict(coeff, t0, scale, t):
    tn = (t - t0) / scale
    return float(coeff[0] * tn * tn + coeff[1] * tn + coeff[2])


def _velocity(coeff, t0, scale, t):
    """二次曲线在 t 处的导数（速度，单位 值/帧）。"""
    tn = (t - t0) / scale
    return float((2.0 * coeff[0] * tn + coeff[1]) / scale)


class GateDecision:
    """一次门控判定的结果。"""

    __slots__ = ("accepted", "reason", "distance", "distance_thresh", "residual", "motion_reset")

    def __init__(
        self,
        accepted,
        reason,
        distance=None,
        distance_thresh=None,
        residual=None,
        motion_reset=False,
    ):
        self.accepted = accepted
        self.reason = reason
        self.distance = distance
        self.distance_thresh = distance_thresh
        self.residual = residual
        self.motion_reset = motion_reset

    def __repr__(self):
        return (
            f"GateDecision(accepted={self.accepted}, reason={self.reason}, "
            f"d={self.distance}, th={self.distance_thresh}, "
            f"residual={self.residual}, reset={self.motion_reset})"
        )


class DistanceGate:
    """距离门：到上一个平滑点的距离超过动态阈值 → 拒绝。"""

    def __init__(self, tracker_cfg, enabled=True):
        self.enabled = bool(enabled)
        self._cfg = {
            "base_gain": tracker_cfg.get("dynamic_gain", 2.0),
            "min_thresh": tracker_cfg.get("dynamic_min_thresh", 60.0),
            "max_thresh": tracker_cfg.get("dynamic_max_thresh", 320.0),
        }

    def evaluate(self, obs, smooth_last, full_traj):
        """smooth_last 为 None（段内第一帧）或门被关闭时直接放行。"""
        if not self.enabled:
            return GateDecision(True, "disabled")
        if smooth_last is None:
            return GateDecision(True, "ok")
        d = distance(obs, smooth_last)
        th = dynamic_step_threshold(
            full_traj,
            base_gain=self._cfg["base_gain"],
            min_thresh=self._cfg["min_thresh"],
            max_thresh=self._cfg["max_thresh"],
        )
        if d > th:
            return GateDecision(False, "distance", distance=d, distance_thresh=th)
        return GateDecision(True, "ok", distance=d, distance_thresh=th)


class MotionModelGate:
    """运动模型门：二次曲线（x(t)、y(t)）残差判定，带击球/折返重置。"""

    def __init__(self, tracker_cfg):
        raw = dict(DEFAULTS)
        raw.update(tracker_cfg.get("motion_model") or {})
        self.enabled = bool(raw["enabled"])
        self.window = int(raw["window"])
        self.min_points = int(raw["min_points"])
        self.residual_thresh = float(raw["residual_thresh"])
        self.rms_thresh = float(raw["rms_thresh"])
        self.reverse_cos = float(raw["reverse_cos"])
        self._win = deque(maxlen=max(3, self.window))

    # ------------------------------------------------------------------ state
    def reset(self, keep_last=0):
        """清空拟合窗口；keep_last=1 时保留最后一个已接受观测作为新窗口起点。"""
        last = self._win[-1] if self._win else None
        self._win.clear()
        if keep_last and last is not None:
            self._win.append(last)

    def accept(self, obs, frame):
        """记录一个已接受的观测（窗口只含已接受观测）。"""
        if self.enabled:
            self._win.append((int(frame), float(obs[0]), float(obs[1])))

    # ------------------------------------------------------------------ judgement
    def evaluate(self, obs, frame):
        if not self.enabled:
            return GateDecision(True, "ok")
        if len(self._win) < self.min_points:
            return GateDecision(True, "insufficient")

        frames = [w[0] for w in self._win]
        xs = [w[1] for w in self._win]
        ys = [w[2] for w in self._win]
        cx, t0x, scx, rms_x = _fit_quadratic(frames, xs)
        cy, t0y, scy, rms_y = _fit_quadratic(frames, ys)
        rms = max(rms_x, rms_y)

        # 窗口自身不吻合二次模型 → 已跨越击球/折返：重置并放行
        if rms > self.rms_thresh:
            self.reset(keep_last=1)
            return GateDecision(True, "motion_reset_rms", residual=None, motion_reset=True)

        pred = (
            _predict(cx, t0x, scx, frame),
            _predict(cy, t0y, scy, frame),
        )
        residual = distance(obs, pred)

        if residual > self.residual_thresh:
            # 方向反转 → 判定为击球（速度突变），放行并重置窗口
            vx = _velocity(cx, t0x, scx, frame)
            vy = _velocity(cy, t0y, scy, frame)
            dvx = float(obs[0]) - self._win[-1][1]
            dvy = float(obs[1]) - self._win[-1][2]
            v_norm = float(np.hypot(vx, vy))
            d_norm = float(np.hypot(dvx, dvy))
            cos = (vx * dvx + vy * dvy) / (v_norm * d_norm) if v_norm > 1e-6 and d_norm > 1e-6 else 1.0
            if cos < self.reverse_cos:
                self.reset(keep_last=1)
                return GateDecision(
                    True, "motion_reset_reverse", residual=residual, motion_reset=True
                )
            return GateDecision(False, "motion_model", residual=residual)

        return GateDecision(True, "ok", residual=residual)


class GatingChain:
    """门控链：距离门 → 运动模型门；任何一门拒绝即拒绝。

    gates 是 config 里的开关字典（true=启用，false=关闭该门并放行全部观测）：
        gates = {"distance": True, "motion_model": True}
    motion_model 的开关优先级：gates.motion_model > tracker.motion_model.enabled。
    """

    def __init__(self, tracker_cfg, gates=None):
        gates = gates or {}
        self.gates = {
            "distance": bool(gates.get("distance", True)),
            "motion_model": bool(
                gates.get("motion_model", (tracker_cfg.get("motion_model") or {}).get("enabled", True))
            ),
        }
        self.distance_gate = DistanceGate(tracker_cfg, enabled=self.gates["distance"])
        self.motion_gate = MotionModelGate(tracker_cfg)
        self.motion_gate.enabled = self.gates["motion_model"]

    def reset(self):
        self.motion_gate.reset()

    def evaluate(self, obs, frame, smooth_last, full_traj):
        """返回 GateDecision；调用方在 accepted 后应调用 self.accept(obs, frame)。"""
        d = self.distance_gate.evaluate(obs, smooth_last, full_traj)
        if not d.accepted:
            return d
        m = self.motion_gate.evaluate(obs, frame)
        if not m.accepted:
            return m
        # 两门都通过：保留距离门的 d/th 供日志/导出，运动门的残差一起带上
        m.distance = d.distance
        m.distance_thresh = d.distance_thresh
        return m

    def accept(self, obs, frame):
        self.motion_gate.accept(obs, frame)
