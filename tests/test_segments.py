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

from src.analysis import AnalysisResult, FrameRecord  # noqa: E402
from src.segments import collect_segments, replay_decisions, resolve_tracks  # noqa: E402

CFG = {
    "tracker": {
        "smooth_window_len": 5,
        "max_miss": 5,
        "static_frame_count": 3,
        "static_penalty": 120.0,
        "fast_speed_thresh": 100.0,
        "lambda_slow": 0.85,
        "lambda_fast": 0.65,
        "grid_size": 10,
        "dynamic_gain": 2.0,
        "dynamic_min_thresh": 60.0,
        "dynamic_max_thresh": 320.0,
    },
    "ransac": {"min_samples": 5, "max_iter": 80, "residual_thresh": 12},
    "detector": {
        "conf_min": 0.12,
        "max_box_area_ratio": 0.004,
        "max_aspect_ratio": 4.0,
        "roi_padding_ratio": 0.08,
    },
}
TMP_ROOT = PROJECT_ROOT / ".tmp-tests"


def cand(x, y, conf=0.8, is_static=False):
    """构造采集阶段的候选行：[x, y, is_static, conf, area_ratio, aspect_ratio, x1, y1, x2, y2]"""
    return [
        int(x),
        int(y),
        bool(is_static),
        float(conf),
        0.00036,
        1.01,
        int(x) - 9,
        int(y) - 9,
        int(x) + 9,
        int(y) + 9,
    ]


def rec(frame, event, smooth=None, candidates=None):
    return FrameRecord(
        frame=frame,
        event=event,
        smooth=smooth,
        candidates=candidates if candidates is not None else [cand(0, 0)],
    )


def obs(frame, x, y):
    return rec(frame, "obs", [float(x), float(y)])


class ReplayDecisionTest(unittest.TestCase):
    def test_accepts_consistent_observations(self):
        frames = [rec(f, "miss", candidates=[cand(100 + f, 100)]) for f in range(1, 4)]
        stats = replay_decisions(frames, CFG, verbose=False)
        self.assertEqual(stats["accepted"], 3)
        self.assertEqual(stats["rejected"], 0)
        self.assertTrue(all(r.event == "obs" for r in frames))
        self.assertTrue(all(r.smooth is not None for r in frames))

    def test_gate_rejects_large_jump(self):
        frames = [rec(f, "miss", candidates=[cand(100 + f, 100)]) for f in range(1, 3)]
        frames.append(rec(3, "miss", candidates=[cand(900, 600)]))  # 远处跳变
        stats = replay_decisions(frames, CFG, verbose=False)
        self.assertEqual(stats["accepted"], 2)
        self.assertEqual(stats["rejected"], 1)
        self.assertEqual(frames[2].event, "rejected")
        self.assertIsNone(frames[2].smooth)

    def test_no_candidates_means_miss(self):
        frames = [rec(1, "miss", candidates=[]), rec(2, "miss", candidates=[])]
        stats = replay_decisions(frames, CFG, verbose=False)
        self.assertEqual(stats["miss"], 2)
        self.assertTrue(all(r.event == "miss" for r in frames))

    def test_selected_candidate_recorded(self):
        frames = [rec(1, "miss", candidates=[cand(100, 100), cand(400, 400)])]
        replay_decisions(frames, CFG, verbose=False)
        self.assertIsNotNone(frames[0].chosen)
        self.assertEqual(len(frames[0].chosen), 2)


class SegmentRuleTest(unittest.TestCase):
    def test_single_segment_flushed_at_video_end(self):
        frames = [obs(1, 100, 100), obs(2, 110, 105), obs(3, 120, 112)]
        segments, clean_map = collect_segments(frames, CFG)
        self.assertEqual(len(segments), 1)
        self.assertEqual(segments[0]["start_frame"], 1)
        self.assertEqual(segments[0]["end_frame"], 3)
        self.assertEqual(segments[0]["n_raw"], 3)
        self.assertIn(0, clean_map)

    def test_five_missing_frames_do_not_close_segment(self):
        """max_miss=5：连续 5 帧丢失不切段，之后来的观测仍属于同一段。"""
        frames = [obs(1, 100, 100), obs(2, 110, 105)]
        frames += [rec(f, "miss") for f in range(3, 8)]
        frames += [obs(8, 200, 200), obs(9, 210, 205)]
        segments, _ = collect_segments(frames, CFG)
        self.assertEqual(len(segments), 1)
        self.assertEqual(segments[0]["n_raw"], 4)
        self.assertEqual(segments[0]["end_frame"], 9)

    def test_sixth_missing_frame_closes_segment(self):
        """第 6 帧连续丢失（missing > max_miss）才切段。"""
        frames = [obs(1, 100, 100), obs(2, 110, 105)]
        frames += [rec(f, "miss") for f in range(3, 9)]
        frames += [obs(9, 300, 300), obs(10, 310, 305)]
        segments, _ = collect_segments(frames, CFG)
        self.assertEqual(len(segments), 2)
        self.assertEqual((segments[0]["start_frame"], segments[0]["end_frame"]), (1, 2))
        self.assertEqual((segments[1]["start_frame"], segments[1]["end_frame"]), (9, 10))

    def test_rejected_frames_count_toward_missing(self):
        frames = [obs(1, 0, 0), obs(2, 1, 1)]
        frames += [rec(f, "rejected") for f in range(3, 9)]
        frames += [obs(9, 10, 10), obs(10, 11, 11)]
        segments, _ = collect_segments(frames, CFG)
        self.assertEqual(len(segments), 2)

    def test_empty_trails_are_not_recorded(self):
        frames = [rec(f, "miss") for f in range(1, 40)]
        segments, clean_map = collect_segments(frames, CFG)
        self.assertEqual(segments, [])
        self.assertEqual(clean_map, {})

    def test_segment_ids_are_sequential(self):
        frames = [obs(1, 0, 0), obs(2, 1, 1)]
        frames += [rec(f, "miss") for f in range(3, 9)]
        frames += [obs(9, 10, 10), obs(10, 11, 11)]
        frames += [rec(f, "miss") for f in range(11, 17)]
        frames += [obs(17, 20, 20)]
        segments, _ = collect_segments(frames, CFG)
        self.assertEqual([s["segment_id"] for s in segments], [0, 1, 2])

    def test_frame_segment_ids_are_assigned(self):
        frames = [obs(1, 0, 0), obs(2, 1, 1)]
        frames += [rec(3, "miss"), rec(4, "miss")]
        self.assertEqual([r.segment_id for r in frames], [None, None, None, None])
        collect_segments(frames, CFG)
        self.assertEqual([r.segment_id for r in frames], [0, 0, 0, 0])

    def test_short_segment_is_not_cleaned(self):
        frames = [obs(f, 100 + f, 100 + 2 * f) for f in range(1, 5)]
        segments, clean_map = collect_segments(frames, CFG)
        self.assertEqual(len(segments), 1)
        self.assertEqual(segments[0]["n_raw"], 4)
        self.assertEqual(segments[0]["n_cleaned"], 4)
        self.assertEqual(len(clean_map[0][0]), 4)

    def test_segment_bounds_use_observation_frames(self):
        frames = [obs(10, 0, 0), obs(11, 1, 1)]
        frames += [rec(f, "miss") for f in range(12, 18)]
        frames += [obs(18, 5, 5), obs(19, 6, 6)]
        segments, _ = collect_segments(frames, CFG)
        self.assertEqual(segments[0]["start_frame"], 10)
        self.assertEqual(segments[0]["end_frame"], 11)
        self.assertEqual(segments[1]["start_frame"], 18)


class ResolveTracksCsvTest(unittest.TestCase):
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

    def _analysis(self, frames):
        return AnalysisResult(
            video_path="videos/x.mp4",
            fps=50.0,
            width=1280,
            height=720,
            total_frames=len(frames),
            processed=len(frames),
            accepted=0,
            rejected=0,
            miss=0,
            segments=[],
            frames=frames,
            csv={},
        )

    def _track(self):
        """一段连续观测 + 一段 6 帧丢失 + 新一段观测。"""
        frames = [rec(f, "miss", candidates=[cand(100 + 2 * f, 100 + f)]) for f in range(1, 13)]
        frames += [rec(f, "miss", candidates=[]) for f in range(13, 19)]
        frames += [rec(f, "miss", candidates=[cand(500 + 2 * f, 300)]) for f in range(19, 25)]
        return frames

    def test_resolve_writes_three_csv_and_updates_analysis(self):
        analysis = self._analysis(self._track())
        resolve_tracks(analysis, CFG, out_path=self.out_path, verbose=False)

        for key in ("traj_csv", "segments_csv", "clean_csv"):
            self.assertTrue(os.path.isfile(analysis.csv[key]), key)
        self.assertGreaterEqual(len(analysis.segments), 1)
        self.assertEqual(
            analysis.accepted + analysis.rejected + analysis.miss, len(analysis.frames)
        )

        traj_rows = self._read(analysis.csv["traj_csv"])
        seg_rows = self._read(analysis.csv["segments_csv"])
        clean_rows = self._read(analysis.csv["clean_csv"])
        self.assertEqual(len(traj_rows), len(analysis.frames))
        self.assertEqual(len(seg_rows), len(analysis.segments))
        self.assertEqual(
            len(clean_rows), sum(int(s["n_cleaned"]) for s in seg_rows)
        )

    def test_save_traj_false_writes_no_csv(self):
        analysis = self._analysis(self._track())
        resolve_tracks(analysis, CFG, out_path=self.out_path, save_traj=False, verbose=False)
        self.assertEqual(analysis.csv, {})
        self.assertFalse(os.path.exists(os.path.join(self.tmp_dir, "clip.traj.csv")))
        self.assertGreaterEqual(len(analysis.segments), 1)

    def test_no_out_path_computes_without_writing(self):
        analysis = self._analysis(self._track())
        resolve_tracks(analysis, CFG, out_path=None, verbose=False)
        self.assertEqual(analysis.csv, {})
        self.assertEqual(os.listdir(self.tmp_dir), [])


if __name__ == "__main__":
    unittest.main()
