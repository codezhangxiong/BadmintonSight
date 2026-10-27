import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.tracker.gating import (  # noqa: E402
    DEFAULTS,
    DistanceGate,
    GatingChain,
    MotionModelGate,
)

TRACKER = {
    "dynamic_gain": 2.0,
    "dynamic_min_thresh": 60.0,
    "dynamic_max_thresh": 320.0,
    "motion_model": {
        "enabled": True,
        "window": 6,
        "min_points": 5,
        "residual_thresh": 30.0,
        "rms_thresh": 20.0,
        "reverse_cos": -0.2,
    },
}

# dynamic_step_threshold 需要 >=5 点；步长约 14px → 阈值 clip(28,60,320)=60
FULL_TRAJ = [(0.0, 0.0), (10.0, 10.0), (20.0, 20.0), (30.0, 30.0), (40.0, 40.0)]


def parabola_points(n=6, t_start=0):
    """x(t)=0.5t²+10t, y(t)=-0.3t²+5t+100，帧号连续。"""
    pts = []
    for i in range(n):
        t = t_start + i
        pts.append((i + 1, (0.5 * t * t + 10 * t, -0.3 * t * t + 5 * t + 100.0)))
    return pts


def parabola_xy(t):
    return (0.5 * t * t + 10 * t, -0.3 * t * t + 5 * t + 100.0)


class DistanceGateTest(unittest.TestCase):
    def test_first_frame_passes_without_last_point(self):
        decision = DistanceGate(TRACKER).evaluate((5, 5), None, FULL_TRAJ)
        self.assertTrue(decision.accepted)
        self.assertEqual(decision.reason, "ok")

    def test_small_step_passes(self):
        decision = DistanceGate(TRACKER).evaluate((45, 45), (40, 40), FULL_TRAJ)
        self.assertTrue(decision.accepted)
        self.assertAlmostEqual(decision.distance, 7.071, places=2)
        self.assertEqual(decision.distance_thresh, 60.0)

    def test_large_step_rejected(self):
        decision = DistanceGate(TRACKER).evaluate((200, 200), (40, 40), FULL_TRAJ)
        self.assertFalse(decision.accepted)
        self.assertEqual(decision.reason, "distance")
        self.assertGreater(decision.distance, decision.distance_thresh)


class MotionModelGateTest(unittest.TestCase):
    def _filled(self, points=None):
        gate = MotionModelGate(TRACKER)
        for frame, point in points or parabola_points():
            gate.accept(point, frame)
        return gate

    def test_insufficient_points_passes(self):
        gate = MotionModelGate(TRACKER)
        for frame, point in parabola_points()[:4]:
            gate.accept(point, frame)
        decision = gate.evaluate((200, 200), 5)
        self.assertTrue(decision.accepted)
        self.assertEqual(decision.reason, "insufficient")

    def test_point_on_curve_passes_with_small_residual(self):
        gate = self._filled()
        decision = gate.evaluate(parabola_xy(6), 7)
        self.assertTrue(decision.accepted)
        self.assertEqual(decision.reason, "ok")
        self.assertLess(decision.residual, 1.0)

    def test_point_far_off_curve_is_rejected(self):
        gate = self._filled()
        decision = gate.evaluate((500, 300), 7)
        self.assertFalse(decision.accepted)
        self.assertEqual(decision.reason, "motion_model")
        self.assertGreater(decision.residual, 30.0)

    def test_direction_reversal_is_treated_as_hit_and_resets(self):
        """击球瞬间：残差大但方向反转 → 放行并重置窗口，而不是当离群点误杀。"""
        gate = self._filled()
        last = parabola_points()[-1][1]
        # 向 -x 方向大幅折返（与窗口末端 +x 速度反向）
        decision = gate.evaluate((last[0] - 150.0, last[1] - 60.0), 7)
        self.assertTrue(decision.accepted)
        self.assertEqual(decision.reason, "motion_reset_reverse")
        self.assertTrue(decision.motion_reset)

    def test_bad_window_resets_instead_of_rejecting(self):
        """窗口本身不吻合二次模型（已跨越击球/折返）→ 重置并放行。"""
        gate = MotionModelGate(TRACKER)
        zigzag = [(1, (0, 0)), (2, (100, 300)), (3, (0, 0)), (4, (100, 300)), (5, (0, 0))]
        for frame, point in zigzag:
            gate.accept(point, frame)
        decision = gate.evaluate((10, 10), 6)
        self.assertTrue(decision.accepted)
        self.assertEqual(decision.reason, "motion_reset_rms")
        self.assertTrue(decision.motion_reset)

    def test_reset_clears_window(self):
        gate = self._filled()
        gate.reset()
        decision = gate.evaluate((200, 200), 7)
        self.assertEqual(decision.reason, "insufficient")

    def test_disabled_gate_always_passes(self):
        gate = MotionModelGate({"motion_model": {"enabled": False}})
        decision = gate.evaluate((999, 999), 1)
        self.assertTrue(decision.accepted)
        self.assertTrue(MotionModelGate({"motion_model": {"enabled": False}}).enabled is False)


class GatingChainTest(unittest.TestCase):
    def test_distance_gate_runs_before_motion_gate(self):
        chain = GatingChain(TRACKER)
        decision = chain.evaluate((300, 300), 1, (0, 0), FULL_TRAJ)
        self.assertFalse(decision.accepted)
        self.assertEqual(decision.reason, "distance")

    def test_chain_accept_updates_motion_window(self):
        chain = GatingChain(TRACKER)
        last = None
        for i in range(6):
            obs = (i * 2.0, i * 2.0)
            decision = chain.evaluate(obs, i + 1, last, FULL_TRAJ)
            self.assertTrue(decision.accepted)
            chain.accept(obs, i + 1)
            last = obs
        # 窗口已满，下一个沿同方向且贴曲线的点应通过
        decision = chain.evaluate((12.0, 12.0), 7, last, FULL_TRAJ)
        self.assertTrue(decision.accepted)

    def test_motion_gate_can_reject_when_distance_passes(self):
        chain = GatingChain(TRACKER)
        for frame, point in parabola_points():
            chain.accept(point, frame)  # 直接塞满运动窗口
        last = parabola_points()[-1][1]
        # 距离门：到 last 只有 50px（< 60 阈值）；但远离曲线 → 运动门拒绝
        obs = (last[0] + 40.0, last[1] + 30.0)
        decision = chain.evaluate(obs, 7, last, FULL_TRAJ)
        self.assertFalse(decision.accepted)
        self.assertEqual(decision.reason, "motion_model")


class DefaultsTest(unittest.TestCase):
    def test_defaults_are_sane(self):
        self.assertGreaterEqual(DEFAULTS["window"], DEFAULTS["min_points"])
        self.assertGreater(DEFAULTS["residual_thresh"], 0)
        self.assertGreater(DEFAULTS["rms_thresh"], 0)
        self.assertLess(DEFAULTS["reverse_cos"], 0)


if __name__ == "__main__":
    unittest.main()
