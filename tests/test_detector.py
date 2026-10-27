import sys
import unittest
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.detector import extract_shuttlecock_candidates  # noqa: E402

FRAME_SHAPE = (720, 1280, 3)
CFG = {"detector": {"conf_min": 0.12}}


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
        """置信度下限是保留的唯一候选级门控。"""
        boxes = _FakeBoxes([[640.0, 360.0, 18.0, 18.0]], conf=[0.05])
        self.assertEqual(extract_shuttlecock_candidates(boxes, FRAME_SHAPE, CFG), [])

    def test_oversized_box_is_no_longer_filtered(self):
        """已删除外形门控：大框不再被丢弃（改由关联/门控阶段处理）。"""
        boxes = _FakeBoxes([[640.0, 360.0, 200.0, 200.0]], conf=[0.9])
        candidates = extract_shuttlecock_candidates(boxes, FRAME_SHAPE, CFG)
        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0]["point"], (640, 360))

    def test_extreme_aspect_ratio_is_no_longer_filtered(self):
        """已删除外形门控：极端长宽比不再被丢弃。"""
        boxes = _FakeBoxes([[640.0, 360.0, 8.0, 40.0]], conf=[0.9])
        candidates = extract_shuttlecock_candidates(boxes, FRAME_SHAPE, CFG)
        self.assertEqual(len(candidates), 1)

    def test_position_is_no_longer_filtered(self):
        """已删除 ROI 门控：画面边缘的点也是合法候选。"""
        boxes = _FakeBoxes([[2.0, 2.0, 18.0, 18.0]], conf=[0.9])
        candidates = extract_shuttlecock_candidates(boxes, FRAME_SHAPE, CFG)
        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0]["point"], (2, 2))

    def test_area_and_aspect_are_kept_as_metadata(self):
        """外形字段仍作为元数据返回（关联代价使用），只是不再用于过滤。"""
        boxes = _FakeBoxes([[640.0, 360.0, 200.0, 200.0]], conf=[0.9])
        candidate = extract_shuttlecock_candidates(boxes, FRAME_SHAPE, CFG)[0]
        self.assertGreater(candidate["area_ratio"], 0)
        self.assertAlmostEqual(candidate["aspect_ratio"], 1.0, places=3)


if __name__ == "__main__":
    unittest.main()
