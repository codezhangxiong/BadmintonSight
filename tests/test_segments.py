import csv
import os
import shutil
import sys
import unittest
import uuid
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.analysis import FrameRecord  # noqa: E402
from src.segments import process_segments  # noqa: E402

CFG = {
    "tracker": {"max_miss": 5},
    "ransac": {"min_samples": 5, "max_iter": 80, "residual_thresh": 12},
}
TMP_ROOT = PROJECT_ROOT / ".tmp-tests"


def rec(frame, event, smooth=None):
    return FrameRecord(frame=frame, event=event, smooth=smooth)


def obs(frame, x, y):
    return rec(frame, "obs", [float(x), float(y)])


class SegmentRuleTest(unittest.TestCase):
    def test_single_segment_flushed_at_video_end(self):
        frames = [obs(1, 100, 100), obs(2, 110, 105), obs(3, 120, 112)]
        segments, _ = process_segments(frames, CFG, verbose=False)
        self.assertEqual(len(segments), 1)
        self.assertEqual(segments[0]["start_frame"], 1)
        self.assertEqual(segments[0]["end_frame"], 3)
        self.assertEqual(segments[0]["n_raw"], 3)

    def test_five_missing_frames_do_not_close_segment(self):
        """max_miss=5：连续 5 帧丢失不切段，之后来的观测仍属于同一段。"""
        frames = [obs(1, 100, 100), obs(2, 110, 105)]
        frames += [rec(f, "miss") for f in range(3, 8)]  # 3..7 = 5 帧
        frames += [obs(8, 200, 200), obs(9, 210, 205)]
        segments, _ = process_segments(frames, CFG, verbose=False)
        self.assertEqual(len(segments), 1)
        self.assertEqual(segments[0]["n_raw"], 4)  # 1,2,8,9 同属一段
        self.assertEqual(segments[0]["end_frame"], 9)

    def test_sixth_missing_frame_closes_segment(self):
        """第 6 帧连续丢失（missing > max_miss）才切段。"""
        frames = [obs(1, 100, 100), obs(2, 110, 105)]
        frames += [rec(f, "miss") for f in range(3, 9)]  # 3..8 = 6 帧
        frames += [obs(9, 300, 300), obs(10, 310, 305)]
        segments, _ = process_segments(frames, CFG, verbose=False)
        self.assertEqual(len(segments), 2)
        self.assertEqual(segments[0]["start_frame"], 1)
        self.assertEqual(segments[0]["end_frame"], 2)
        self.assertEqual(segments[1]["start_frame"], 9)
        self.assertEqual(segments[1]["end_frame"], 10)

    def test_rejected_frames_count_toward_missing(self):
        """rejected 与 miss 一样计入连续丢失计数。"""
        frames = [obs(1, 0, 0), obs(2, 1, 1)]
        frames += [rec(f, "rejected") for f in range(3, 9)]  # 6 帧 rejected
        frames += [obs(9, 10, 10), obs(10, 11, 11)]
        segments, _ = process_segments(frames, CFG, verbose=False)
        self.assertEqual(len(segments), 2)

    def test_empty_trails_are_not_recorded(self):
        frames = [rec(f, "miss") for f in range(1, 40)]
        segments, _ = process_segments(frames, CFG, verbose=False)
        self.assertEqual(segments, [])

    def test_segment_ids_are_sequential(self):
        frames = [obs(1, 0, 0), obs(2, 1, 1)]
        frames += [rec(f, "miss") for f in range(3, 9)]
        frames += [obs(9, 10, 10), obs(10, 11, 11)]
        frames += [rec(f, "miss") for f in range(11, 17)]
        frames += [obs(17, 20, 20)]
        segments, _ = process_segments(frames, CFG, verbose=False)
        self.assertEqual([s["segment_id"] for s in segments], [0, 1, 2])

    def test_short_segment_is_not_cleaned(self):
        """点数 <= min_samples 的短段原样保留（n_cleaned == n_raw）。"""
        frames = [obs(f, 100 + f, 100 + 2 * f) for f in range(1, 5)]  # 4 点
        segments, _ = process_segments(frames, CFG, verbose=False)
        self.assertEqual(len(segments), 1)
        self.assertEqual(segments[0]["n_raw"], 4)
        self.assertEqual(segments[0]["n_cleaned"], 4)

    def test_long_segment_may_drop_outliers(self):
        """长段会经过 RANSAC：一段纯噪声点，清洗后点数应少于原始（或至少不超过）。"""
        frames = [obs(f, 600 + (f * 37) % 700, 400 + (f * 91) % 300) for f in range(1, 21)]
        segments, _ = process_segments(frames, CFG, verbose=False)
        self.assertEqual(len(segments), 1)
        self.assertLessEqual(segments[0]["n_cleaned"], segments[0]["n_raw"])

    def test_segment_bounds_use_observation_frames(self):
        """段的起止帧取实际观测帧，而不是"第 6 次丢失"的触发帧。"""
        frames = [obs(10, 0, 0), obs(11, 1, 1)]
        frames += [rec(f, "miss") for f in range(12, 18)]  # 12..17 = 6 帧
        frames += [obs(18, 5, 5), obs(19, 6, 6)]
        segments, _ = process_segments(frames, CFG, verbose=False)
        self.assertEqual(segments[0]["start_frame"], 10)
        self.assertEqual(segments[0]["end_frame"], 11)
        self.assertEqual(segments[1]["start_frame"], 18)


class SegmentCsvTest(unittest.TestCase):
    def setUp(self):
        TMP_ROOT.mkdir(parents=True, exist_ok=True)
        self.tmp_dir = os.path.join(str(TMP_ROOT), f"case_{uuid.uuid4().hex[:8]}")
        os.makedirs(self.tmp_dir)
        self.out_path = os.path.join(self.tmp_dir, "clip.mp4")

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    @staticmethod
    def _read(path):
        with open(path, newline="", encoding="utf-8") as fh:
            return list(csv.DictReader(fh))

    def _two_segment_frames(self):
        frames = [obs(1, 100, 100), obs(2, 110, 105)]
        frames += [rec(f, "miss") for f in range(3, 9)]
        frames += [obs(9, 300, 300), obs(10, 310, 305)]
        return frames

    def test_writes_segments_and_clean_csv(self):
        frames = self._two_segment_frames()
        segments, csv_paths = process_segments(frames, CFG, out_path=self.out_path, verbose=False)

        self.assertTrue(os.path.isfile(csv_paths["segments_csv"]))
        self.assertTrue(os.path.isfile(csv_paths["clean_csv"]))
        self.assertTrue(csv_paths["segments_csv"].endswith("clip.segments.csv"))

        seg_rows = self._read(csv_paths["segments_csv"])
        clean_rows = self._read(csv_paths["clean_csv"])
        self.assertEqual(len(seg_rows), len(segments))
        self.assertEqual(len(seg_rows), 2)
        self.assertEqual(
            len(clean_rows), sum(int(s["n_cleaned"]) for s in segments)
        )
        self.assertEqual(
            {r["segment_id"] for r in clean_rows}, {str(s["segment_id"]) for s in segments}
        )

    def test_save_csv_false_writes_nothing(self):
        frames = self._two_segment_frames()
        segments, csv_paths = process_segments(
            frames, CFG, out_path=self.out_path, save_csv=False, verbose=False
        )
        self.assertEqual(len(segments), 2)
        self.assertEqual(csv_paths, {})
        self.assertFalse(os.path.exists(os.path.join(self.tmp_dir, "clip.segments.csv")))

    def test_no_out_path_computes_without_writing(self):
        frames = self._two_segment_frames()
        segments, csv_paths = process_segments(frames, CFG, out_path=None, verbose=False)
        self.assertEqual(len(segments), 2)
        self.assertEqual(csv_paths, {})
        self.assertEqual(os.listdir(self.tmp_dir), [])


if __name__ == "__main__":
    unittest.main()
