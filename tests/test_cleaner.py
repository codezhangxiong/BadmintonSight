import sys
import unittest
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.tracker.cleaner import ransac_clean_parabola  # noqa: E402


def _parabola_points(n=60, x0=150.0, x1=1750.0, a=-0.0002, b=1.2, c=100.0):
    xs = np.linspace(x0, x1, n)
    ys = a * xs**2 + b * xs + c
    return [(float(x), float(y)) for x, y in zip(xs, ys)]


class CleanerMaskTest(unittest.TestCase):
    def test_short_sequence_returns_all_true_mask(self):
        traj = [(100.0, 100.0), (110.0, 120.0), (120.0, 150.0)]
        cleaned, mask = ransac_clean_parabola(traj, min_samples=5, return_mask=True)
        self.assertEqual(len(mask), len(traj))
        self.assertTrue(mask.all())
        self.assertEqual(list(cleaned), list(traj))

    def test_short_sequence_without_mask_keeps_backward_compatible_return(self):
        traj = [(1.0, 2.0), (3.0, 4.0)]
        cleaned = ransac_clean_parabola(traj, min_samples=5)
        self.assertEqual(cleaned, traj)
        self.assertFalse(hasattr(cleaned, "all"))

    def test_mask_matches_cleaned_points_on_clean_parabola(self):
        traj = _parabola_points()
        cleaned, mask = ransac_clean_parabola(traj, min_samples=5, return_mask=True)
        self.assertEqual(len(mask), len(traj))
        self.assertEqual(int(mask.sum()), len(cleaned))
        # 保序子序列：内点按原顺序出现在 cleaned 中
        expected = [traj[i] for i in np.flatnonzero(mask)]
        for (cx, cy), (ex, ey) in zip(cleaned, expected):
            self.assertAlmostEqual(cx, ex, places=3)
            self.assertAlmostEqual(cy, ey, places=3)

    def test_default_call_returns_points_only(self):
        cleaned = ransac_clean_parabola(_parabola_points(), min_samples=5)
        self.assertIsInstance(cleaned, list)
        self.assertTrue(all(isinstance(p, tuple) and len(p) == 2 for p in cleaned))
        self.assertGreater(len(cleaned), 0)

    def test_mask_type_is_boolean_array(self):
        _, mask = ransac_clean_parabola(_parabola_points(), min_samples=5, return_mask=True)
        self.assertEqual(mask.dtype, np.bool_)


if __name__ == "__main__":
    unittest.main()
