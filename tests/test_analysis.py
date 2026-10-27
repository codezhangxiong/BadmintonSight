import json
import os
import shutil
import sys
import unittest
import uuid
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.analysis import AnalysisResult, FrameRecord  # noqa: E402
from src.pipeline import _clean_segment  # noqa: E402,F401  (确认向后兼容的 re-export)
from src.render import draw_frame  # noqa: E402

TMP_ROOT = PROJECT_ROOT / ".tmp-tests"


def _make_record(**kwargs):
    defaults = dict(
        frame=7,
        raw_boxes=[[640.0, 360.0, 18.0, 18.0, 0.71]],
        candidates=[[640, 360, False], [100, 100, True]],
        chosen=[640, 360],
        rejected=False,
        smooth=[639.5, 360.5],
        trail=[[600.0, 340.0], [620.0, 350.0], [639.5, 360.5]],
        segment_id=0,
        event="obs",
    )
    defaults.update(kwargs)
    return FrameRecord(**defaults)


def _make_analysis():
    return AnalysisResult(
        video_path="videos/x.mp4",
        fps=50.0,
        width=1280,
        height=720,
        total_frames=730,
        processed=2,
        accepted=1,
        rejected=0,
        miss=1,
        segments=[{"segment_id": 0, "start_frame": 1, "end_frame": 1, "n_raw": 1, "n_cleaned": 1}],
        frames=[_make_record(frame=1), _make_record(frame=2, chosen=None, smooth=None, event="miss")],
        csv={"traj_csv": "/tmp/x.traj.csv"},
        limit=None,
    )


class FrameRecordTest(unittest.TestCase):
    def test_round_trip_preserves_all_fields(self):
        record = _make_record()
        restored = FrameRecord.from_dict(json.loads(json.dumps(record.to_dict())))
        self.assertEqual(restored.frame, record.frame)
        self.assertEqual(restored.raw_boxes, record.raw_boxes)
        self.assertEqual(restored.candidates, record.candidates)
        self.assertEqual(restored.chosen, record.chosen)
        self.assertEqual(restored.smooth, record.smooth)
        self.assertEqual(restored.trail, record.trail)
        self.assertEqual(restored.segment_id, record.segment_id)
        self.assertEqual(restored.event, record.event)

    def test_from_dict_handles_missing_optional_fields(self):
        restored = FrameRecord.from_dict({"frame": 3})
        self.assertEqual(restored.frame, 3)
        self.assertEqual(restored.raw_boxes, [])
        self.assertEqual(restored.candidates, [])
        self.assertIsNone(restored.chosen)
        self.assertFalse(restored.rejected)
        self.assertEqual(restored.event, "miss")


class AnalysisResultTest(unittest.TestCase):
    def test_round_trip(self):
        analysis = _make_analysis()
        restored = AnalysisResult.from_dict(json.loads(json.dumps(analysis.to_dict())))
        self.assertEqual(restored.video_path, analysis.video_path)
        self.assertEqual(restored.fps, analysis.fps)
        self.assertEqual(restored.width, analysis.width)
        self.assertEqual(restored.height, analysis.height)
        self.assertEqual(restored.processed, analysis.processed)
        self.assertEqual(restored.accepted, analysis.accepted)
        self.assertEqual(len(restored.frames), 2)
        self.assertEqual(restored.frames[1].event, "miss")
        self.assertEqual(restored.segments, analysis.segments)
        self.assertEqual(restored.csv, analysis.csv)
        self.assertEqual(restored.rendered_frames, 2)


class SaveLoadTest(unittest.TestCase):
    def setUp(self):
        TMP_ROOT.mkdir(parents=True, exist_ok=True)
        self.tmp_dir = os.path.join(str(TMP_ROOT), f"case_{uuid.uuid4().hex[:8]}")
        os.makedirs(self.tmp_dir)

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_save_and_load(self):
        from src.analysis import load_analysis, save_analysis

        path = os.path.join(self.tmp_dir, "a.analysis.json")
        save_analysis(_make_analysis(), path)
        self.assertTrue(os.path.isfile(path))
        self.assertFalse(os.path.exists(path + ".tmp"))
        loaded = load_analysis(path)
        self.assertEqual(loaded.processed, 2)
        self.assertEqual(len(loaded.frames), 2)

    def test_save_creates_parent_dir(self):
        from src.analysis import save_analysis

        path = os.path.join(self.tmp_dir, "nested", "deep", "a.json")
        save_analysis(_make_analysis(), path)
        self.assertTrue(os.path.isfile(path))


class DrawFrameTest(unittest.TestCase):
    def test_empty_record_is_noop(self):
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        out = draw_frame(frame.copy(), FrameRecord(frame=1))
        self.assertEqual(out.shape, (720, 1280, 3))
        self.assertEqual(int(out.sum()), 0)

    def test_draw_changes_pixels_without_resizing(self):
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        out = draw_frame(frame.copy(), _make_record())
        self.assertEqual(out.shape, frame.shape)
        self.assertEqual(out.dtype, np.uint8)
        self.assertGreater(int(out.sum()), 0)

    def test_rejected_frame_still_draws_observation_point(self):
        """被门控拒绝的帧在旧版里仍会画黄点（画点在门控判断之前）。"""
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        record = _make_record(rejected=True, smooth=None, event="rejected")
        out = draw_frame(frame.copy(), record)
        self.assertGreater(int(out.sum()), 0)


if __name__ == "__main__":
    unittest.main()
