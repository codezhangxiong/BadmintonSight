import sys
import unittest
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.pipeline import _clean_segment  # noqa: E402

RANSAC_CFG = {"min_samples": 5, "max_iter": 80, "residual_thresh": 12}


def _parabola(n=60):
    xs = np.linspace(150.0, 1750.0, n)
    ys = -0.0002 * xs**2 + 1.2 * xs + 100.0
    return [(float(x), float(y)) for x, y in zip(xs, ys)], list(range(1, n + 1))


class CleanSegmentTest(unittest.TestCase):
    def test_exactly_min_samples_with_outlier_is_kept(self):
        """恰好 min_samples 个点（含离群）必须原样返回。

        回归测试：RANSAC 对恰好等量的输入没有冗余度，曾把 5 点段清洗成 0 点，
        与 segments.csv 的 n_cleaned == n_raw（短段不清洗）契约不符。
        """
        points = [
            (100.0, 100.0),
            (110.0, 105.0),
            (120.0, 112.0),
            (130.0, 120.0),
            (500.0, 600.0),  # 明显离群
        ]
        frames = [1, 2, 3, 4, 5]
        cleaned, cleaned_frames, mask = _clean_segment(points, frames, RANSAC_CFG)
        self.assertEqual(len(cleaned), 5)
        self.assertEqual(cleaned_frames, frames)
        self.assertTrue(mask.all())

    def test_below_min_samples_is_identity(self):
        points = [(1.0, 2.0), (3.0, 4.0)]
        cleaned, cleaned_frames, mask = _clean_segment(points, [7, 8], RANSAC_CFG)
        self.assertEqual(cleaned, points)
        self.assertEqual(cleaned_frames, [7, 8])
        self.assertEqual(len(mask), 2)

    def test_empty_segment(self):
        cleaned, cleaned_frames, mask = _clean_segment([], [], RANSAC_CFG)
        self.assertEqual(cleaned, [])
        self.assertEqual(cleaned_frames, [])
        self.assertEqual(len(mask), 0)

    def test_frames_align_with_mask_on_normal_segment(self):
        points, frames = _parabola()
        cleaned, cleaned_frames, mask = _clean_segment(points, frames, RANSAC_CFG)
        self.assertEqual(len(mask), len(points))
        self.assertEqual(len(cleaned), len(cleaned_frames))
        self.assertEqual(int(mask.sum()), len(cleaned))
        expected = [frames[i] for i in np.flatnonzero(mask)]
        self.assertEqual(cleaned_frames, expected)
        self.assertGreater(len(cleaned), 0)


if __name__ == "__main__":
    unittest.main()
