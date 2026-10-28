import sys
import unittest
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.capture import FrameRecord  # noqa: E402
from src.detector import extract_shuttlecock_candidates  # noqa: E402
from src.segments import _clean_segment, collect_segments  # noqa: E402
from src.tracker.gating import GatingChain  # noqa: E402

FRAME_SHAPE = (720, 1280, 3)
TRACKER = {
    "smooth_window_len": 5,
    "max_miss": 5,
    "dynamic_gain": 2.0,
    "dynamic_min_thresh": 60.0,
    "dynamic_max_thresh": 320.0,
    "motion_model": {
        "enabled": True,
        "window": 6,
        "min_points": 5,
        "residual_thresh": 80.0,
        "rms_thresh": 20.0,
        "reverse_cos": -0.2,
    },
}
FULL_TRAJ = [(0.0, 0.0), (10.0, 10.0), (20.0, 20.0), (30.0, 30.0), (40.0, 40.0)]
RANSAC_CFG = {"min_samples": 5, "max_iter": 80, "residual_thresh": 12}


class _FakeTensor:
    def __init__(self, values):
        self._arr = np.asarray(values, dtype=np.float32)

    @property
    def shape(self):
        return self._arr.shape

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self._arr


class _FakeBoxes:
    def __init__(self, xywh, conf=None):
        self.xywh = _FakeTensor(xywh)
        self.conf = None if conf is None else _FakeTensor(conf)
        self.cls = None


class GatingSwitchTest(unittest.TestCase):
    def test_distance_gate_disabled_passes_far_point(self):
        from src.tracker.gating import DistanceGate

        gate = DistanceGate(TRACKER, enabled=False)
        decision = gate.evaluate((999, 999), (0, 0), FULL_TRAJ)
        self.assertTrue(decision.accepted)
        self.assertEqual(decision.reason, "disabled")

    def test_chain_with_both_gates_disabled_passes_everything(self):
        chain = GatingChain(TRACKER, {"distance": False, "motion_model": False})
        decision = chain.evaluate((999, 999), 1, (0, 0), FULL_TRAJ)
        self.assertTrue(decision.accepted)

    def test_chain_only_distance_enabled(self):
        chain = GatingChain(TRACKER, {"distance": True, "motion_model": False})
        decision = chain.evaluate((300, 300), 1, (0, 0), FULL_TRAJ)
        self.assertFalse(decision.accepted)
        self.assertEqual(decision.reason, "distance")

    def test_chain_only_motion_enabled(self):
        chain = GatingChain(TRACKER, {"distance": False, "motion_model": True})
        # 距离门关闭 → 进入运动门；窗口点数不足 → 放行
        decision = chain.evaluate((300, 300), 1, (0, 0), FULL_TRAJ)
        self.assertTrue(decision.accepted)
        self.assertEqual(decision.reason, "insufficient")

    def test_gates_override_tracker_motion_model_enabled(self):
        tracker_off = dict(TRACKER)
        tracker_off["motion_model"] = dict(TRACKER["motion_model"], enabled=False)

        forced_on = GatingChain(tracker_off, {"motion_model": True})
        self.assertTrue(forced_on.motion_gate.enabled)

        fallback = GatingChain(tracker_off, None)
        self.assertFalse(fallback.motion_gate.enabled)


class DetectorConfSwitchTest(unittest.TestCase):
    def test_conf_min_disabled_keeps_low_confidence_box(self):
        boxes = _FakeBoxes([[640.0, 360.0, 18.0, 18.0]], conf=[0.05])
        cfg_on = {"detector": {"conf_min": 0.12}, "gates": {"detector_conf": True}}
        cfg_off = {"detector": {"conf_min": 0.12}, "gates": {"detector_conf": False}}
        self.assertEqual(extract_shuttlecock_candidates(boxes, FRAME_SHAPE, cfg_on), [])
        self.assertEqual(len(extract_shuttlecock_candidates(boxes, FRAME_SHAPE, cfg_off)), 1)


class RansacSwitchTest(unittest.TestCase):
    def test_clean_segment_disabled_returns_identity(self):
        points = [(float(i), float(i * 3)) for i in range(30)]
        frames = list(range(1, 31))
        cleaned, cleaned_frames, mask = _clean_segment(
            points, frames, RANSAC_CFG, enabled=False
        )
        self.assertEqual(len(cleaned), 30)
        self.assertEqual(cleaned_frames, frames)
        self.assertTrue(mask.all())

    def test_collect_segments_with_ransac_disabled_keeps_all_points(self):
        cfg = {
            "tracker": {"max_miss": 5},
            "ransac": RANSAC_CFG,
            "gates": {"ransac": False},
        }
        frames = [
            FrameRecord(frame=f, event="obs", smooth=[float(f * 3), float(f)])
            for f in range(1, 9)
        ]
        segments, clean_map = collect_segments(frames, cfg)
        self.assertEqual(len(segments), 1)
        self.assertEqual(segments[0]["n_raw"], segments[0]["n_cleaned"])
        self.assertEqual(len(clean_map[0][0]), 8)


class GateDefaultsTest(unittest.TestCase):
    def test_missing_gates_section_enables_all(self):
        chain = GatingChain(TRACKER, None)
        self.assertTrue(chain.gates["distance"])
        self.assertTrue(chain.gates["motion_model"])

    def test_missing_ransac_gate_defaults_to_enabled(self):
        cfg = {"tracker": {"max_miss": 5}, "ransac": RANSAC_CFG}
        frames = [FrameRecord(frame=f, event="obs", smooth=[float(f), float(f)]) for f in range(1, 6)]
        segments, _ = collect_segments(frames, cfg)
        self.assertEqual(len(segments), 1)
        # 短段本来就不清洗，这里只验证不抛异常且契约成立
        self.assertEqual(segments[0]["n_cleaned"], segments[0]["n_raw"])


if __name__ == "__main__":
    unittest.main()
