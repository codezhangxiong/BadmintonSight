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

from src.export import (  # noqa: E402
    CLEAN_FIELDS,
    SEGMENT_FIELDS,
    TRAJ_FIELDS,
    TrajectoryRecorder,
)


# 临时目录放在项目根下的新目录里：部分受限环境（如 CI 沙箱）只允许写工作区根，
# 系统 %TEMP% 下的新建子目录可能不可写。
TMP_ROOT = PROJECT_ROOT / ".tmp-tests"


class TrajectoryRecorderTest(unittest.TestCase):
    def setUp(self):
        TMP_ROOT.mkdir(parents=True, exist_ok=True)
        # 不要用 tempfile.TemporaryDirectory/tempfile.mkdtemp：它们会把目录 chmod 成 0700，
        # 在 Windows 上等于重置 ACL 并丢掉父目录继承的写授权（受限环境下会 PermissionError）。
        self.tmp_dir = os.path.join(str(TMP_ROOT), f"case_{uuid.uuid4().hex[:8]}")
        os.makedirs(self.tmp_dir)
        self.out_path = os.path.join(self.tmp_dir, "nested", "clip.mp4")

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    @staticmethod
    def _read_rows(path):
        with open(path, newline="", encoding="utf-8") as fh:
            return list(csv.DictReader(fh))

    def test_creates_three_sidecar_files_with_headers(self):
        rec = TrajectoryRecorder(self.out_path, enabled=True)
        rec.close()

        expected = {
            rec.traj_path: TRAJ_FIELDS,
            rec.segments_path: SEGMENT_FIELDS,
            rec.clean_path: CLEAN_FIELDS,
        }
        for path, fields in expected.items():
            self.assertTrue(os.path.isfile(path), f"缺少 sidecar 文件: {path}")
            with open(path, newline="", encoding="utf-8") as fh:
                header = next(csv.reader(fh))
            self.assertEqual(header, fields)

    def test_paths_follow_output_stem_and_create_parent_dir(self):
        rec = TrajectoryRecorder(self.out_path, enabled=True)
        rec.close()
        nested = os.path.join(self.tmp_dir, "nested")
        self.assertTrue(os.path.isdir(nested))
        self.assertEqual(os.path.basename(rec.traj_path), "clip.traj.csv")
        self.assertEqual(os.path.dirname(rec.traj_path), nested)

    def test_add_frame_records_obs_rejected_miss(self):
        rec = TrajectoryRecorder(self.out_path, enabled=True)
        candidate = {
            "confidence": 0.6383,
            "bbox": (624, 331, 642, 350),
            "is_static": False,
        }
        rec.add_frame(1, 0, "miss")
        rec.add_frame(2, 0, "obs", obs=(633, 340), smooth=(633.4, 340.2),
                      candidate=candidate, dist_to_prev=5.25, threshold=60.0)
        rec.add_frame(3, 0, "rejected", obs=(700, 100), candidate=candidate,
                      dist_to_prev=91.5, threshold=60.0)
        rec.close()

        rows = self._read_rows(rec.traj_path)
        self.assertEqual(len(rows), 3)
        self.assertEqual([r["event"] for r in rows], ["miss", "obs", "rejected"])

        self.assertEqual(rows[0]["obs_x"], "")
        self.assertEqual(rows[0]["smooth_x"], "")

        # obs 是整数像素、smooth 是浮点，格式化后分别不带/带小数位
        self.assertEqual(rows[1]["obs_x"], "633")
        self.assertEqual(rows[1]["smooth_x"], "633.40")
        self.assertEqual(rows[1]["conf"], "0.6383")
        self.assertEqual(rows[1]["bbox_x1"], "624")
        self.assertEqual(rows[1]["bbox_y2"], "350")
        self.assertEqual(rows[1]["is_static"], "0")
        self.assertEqual(rows[1]["dist_to_prev"], "5.25")
        self.assertEqual(rows[1]["threshold"], "60.00")

        # 被拒观测也要留下原始坐标，便于复盘门控
        self.assertEqual(rows[2]["obs_x"], "700")
        self.assertEqual(rows[2]["smooth_x"], "")

    def test_add_segment_and_clean_points(self):
        rec = TrajectoryRecorder(self.out_path, enabled=True)
        rec.add_segment(
            {
                "segment_id": 3,
                "start_frame": 10,
                "end_frame": 42,
                "n_raw": 33,
                "n_cleaned": 30,
                "first_x": 100.0,
                "first_y": 200.0,
                "last_x": 300.0,
                "last_y": 250.0,
            }
        )
        rec.add_clean_points(3, [(10, 100.0, 200.0), (11, 110.5, 210.25)])
        rec.add_clean_flat(3, [(120.0, 220.0)])
        rec.close()

        seg_rows = self._read_rows(rec.segments_path)
        self.assertEqual(len(seg_rows), 1)
        self.assertEqual(seg_rows[0]["segment_id"], "3")
        self.assertEqual(seg_rows[0]["n_raw"], "33")
        self.assertEqual(seg_rows[0]["n_cleaned"], "30")
        self.assertEqual(seg_rows[0]["last_x"], "300.00")

        clean_rows = self._read_rows(rec.clean_path)
        self.assertEqual(len(clean_rows), 3)
        self.assertEqual(clean_rows[0]["frame"], "10")
        self.assertEqual(clean_rows[1]["x"], "110.50")
        self.assertEqual(clean_rows[2]["frame"], "")  # 无帧号版本

    def test_disabled_creates_no_files(self):
        rec = TrajectoryRecorder(self.out_path, enabled=False)
        rec.add_frame(1, 0, "miss")
        rec.add_segment({"segment_id": 0})
        rec.add_clean_points(0, [(1, 2.0, 3.0)])
        rec.close()

        self.assertEqual(rec.summary()["traj_csv"], "")
        self.assertFalse(os.path.exists(os.path.dirname(self.out_path)))

    def test_summary_exposes_paths_when_enabled(self):
        rec = TrajectoryRecorder(self.out_path, enabled=True)
        summary = rec.summary()
        rec.close()
        self.assertTrue(summary["traj_csv"].endswith("clip.traj.csv"))
        self.assertTrue(summary["segments_csv"].endswith("clip.segments.csv"))
        self.assertTrue(summary["clean_csv"].endswith("clip.clean.csv"))

    def test_close_is_idempotent(self):
        rec = TrajectoryRecorder(self.out_path, enabled=True)
        rec.close()
        rec.close()  # 不应抛异常
        self.assertTrue(os.path.isfile(rec.traj_path))


if __name__ == "__main__":
    unittest.main()
