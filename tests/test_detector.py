import sys
import unittest
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.detector import extract_shuttlecock_candidates  # noqa: E402

FRAME_SHAPE = (720, 1280, 3)
CFG = {
    "detector": {
        "conf_min": 0.12,
        "max_box_area_ratio": 0.004,
        "max_aspect_ratio": 4.0,
        "roi_padding_ratio": 0.08,
    }
}


class _FakeTensor:
    """模拟 torch.Tensor 的 detach().cpu().numpy() 链。"""

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
    def __init__(self, xywh, conf=None, cls=None):
        self.xywh = _FakeTensor(xywh)
        self.conf = None if conf is None else _FakeTensor(conf)
        self.cls = None if cls is None else _FakeTensor(cls)


class DetectorTest(unittest.TestCase):
    def test_none_boxes_returns_empty(self):
        self.assertEqual(extract_shuttlecock_candidates(None, FRAME_SHAPE, CFG), [])

    def test_empty_boxes_returns_empty(self):
        boxes = _FakeBoxes(np.zeros((0, 4)))
        self.assertEqual(extract_shuttlecock_candidates(boxes, FRAME_SHAPE, CFG), [])

    def test_missing_conf_does_not_raise_name_error(self):
        """conf 为 None 时曾因未导入 numpy 而 NameError，这里回归验证。"""
        boxes = _FakeBoxes([[640.0, 360.0, 18.0, 18.0]], conf=None)
        candidates = extract_shuttlecock_candidates(boxes, FRAME_SHAPE, CFG)
        self.assertEqual(len(candidates), 1)
        self.assertAlmostEqual(candidates[0]["confidence"], 1.0)

    def test_conf_length_mismatch_falls_back_instead_of_silently_dropping(self):
        boxes = _FakeBoxes(
            [[640.0, 360.0, 18.0, 18.0], [100.0, 100.0, 18.0, 18.0]],
            conf=[0.5],  # 长度不匹配
        )
        candidates = extract_shuttlecock_candidates(boxes, FRAME_SHAPE, CFG)
        self.assertEqual(len(candidates), 2)
        self.assertAlmostEqual(candidates[0]["confidence"], 1.0)

    def test_candidate_exposes_point_bbox_and_cls(self):
        boxes = _FakeBoxes([[640.4, 359.6, 18.0, 18.0]], conf=[0.8], cls=[0])
        candidate = extract_shuttlecock_candidates(boxes, FRAME_SHAPE, CFG)[0]
        self.assertEqual(candidate["point"], (640, 359))
        self.assertEqual(candidate["bbox"], (631, 350, 649, 368))
        self.assertEqual(candidate["cls"], 0)
        self.assertAlmostEqual(candidate["confidence"], 0.8, places=5)

    def test_candidate_without_cls_has_none(self):
        boxes = _FakeBoxes([[640.0, 360.0, 18.0, 18.0]], conf=[0.8])
        self.assertIsNone(extract_shuttlecock_candidates(boxes, FRAME_SHAPE, CFG)[0]["cls"])

    def test_filters_low_confidence(self):
        boxes = _FakeBoxes([[640.0, 360.0, 18.0, 18.0]], conf=[0.05])
        self.assertEqual(extract_shuttlecock_candidates(boxes, FRAME_SHAPE, CFG), [])

    def test_filters_oversized_box(self):
        boxes = _FakeBoxes([[640.0, 360.0, 200.0, 200.0]], conf=[0.9])
        self.assertEqual(extract_shuttlecock_candidates(boxes, FRAME_SHAPE, CFG), [])

    def test_filters_extreme_aspect_ratio(self):
        boxes = _FakeBoxes([[640.0, 360.0, 8.0, 40.0]], conf=[0.9])
        self.assertEqual(extract_shuttlecock_candidates(boxes, FRAME_SHAPE, CFG), [])

    def test_roi_filter_currently_accepts_everything(self):
        """记录当前 ROI 语义的已知行为（外扩后覆盖整幅画面）。

        这不是期望语义：ROI 过滤器实际失效，属分析报告 P1-11 的已知问题。
        测试在此固定现状，等修 ROI 时应同步更新此用例。
        """
        corners = [(50, 50), (1230, 670)]
        boxes = _FakeBoxes([[2.0, 2.0, 18.0, 18.0]], conf=[0.9])
        self.assertEqual(len(extract_shuttlecock_candidates(boxes, FRAME_SHAPE, CFG, corners)), 1)


if __name__ == "__main__":
    unittest.main()
