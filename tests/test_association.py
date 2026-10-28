import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.tracker.association import compute_cost, select_best_candidate  # noqa: E402


def cfg_with(association=True, margin=80.0, cont_margin=30.0, max_dist=0.0):
    return {
        "tracker": {
            "fast_speed_thresh": 100.0,
            "lambda_slow": 0.85,
            "lambda_fast": 0.65,
            "static_penalty": 120.0,
            "assoc_margin": margin,
            "assoc_cont_margin": cont_margin,
            "assoc_max_dist": max_dist,
        },
        "gates": {"association": association},
    }


def cand(x, y, conf, area_ratio=0.0004, is_static=False):
    return {
        "point": (x, y),
        "confidence": conf,
        "area_ratio": area_ratio,
        "is_static": is_static,
    }


REF = (0.0, 0.0)
EMPTY_TRAJ = []


class SelectBestCandidateTest(unittest.TestCase):
    def test_empty_candidates_returns_none(self):
        self.assertIsNone(select_best_candidate([], REF, EMPTY_TRAJ, cfg_with()))

    def test_single_candidate_is_returned(self):
        c = cand(10, 0, 0.9)
        self.assertIs(select_best_candidate([c], REF, EMPTY_TRAJ, cfg_with()), c)

    def test_gate_off_picks_lowest_cost(self):
        """gates.association=false → 纯代价最小（旧行为）。"""
        near_low_conf = cand(10, 0, 0.5)
        far_high_conf = cand(100, 0, 1.0)
        best = select_best_candidate([near_low_conf, far_high_conf], REF, EMPTY_TRAJ,
                                     cfg_with(association=False), last_obs=(10, 0))
        self.assertIs(best, near_low_conf)

    def test_tie_break_prefers_continuity(self):
        """代价接近时，选离上一观测更近的候选（连续性优先，而不是 conf）。"""
        near = cand(10, 0, 0.5)      # cost ≈ 8.60，到 last_obs 距离 0
        far = cand(100, 0, 0.95)     # cost ≈ 85.04，到 last_obs 距离 90
        best = select_best_candidate([near, far], REF, EMPTY_TRAJ,
                                     cfg_with(margin=80.0), last_obs=(10, 0))
        self.assertIs(best, near)

    def test_tie_break_uses_conf_when_continuity_close(self):
        """两候选到上一观测距离接近时，用置信度决胜。"""
        a = cand(10, 0, 0.5)         # cost ≈ 8.60
        b = cand(20, 0, 0.95)        # cost ≈ 17.04，差 8.4 < margin 80
        best = select_best_candidate([a, b], REF, EMPTY_TRAJ,
                                     cfg_with(margin=80.0), last_obs=(10, 0))
        self.assertIs(best, b)

    def test_tie_break_without_last_obs_falls_back_to_conf(self):
        near_low_conf = cand(10, 0, 0.5)
        far_high_conf = cand(100, 0, 0.95)  # 差 76.4 < margin 80
        best = select_best_candidate([near_low_conf, far_high_conf], REF, EMPTY_TRAJ,
                                     cfg_with(margin=80.0), last_obs=None)
        self.assertIs(best, far_high_conf)

    def test_tie_break_keeps_lowest_cost_when_other_conf_lower_and_close(self):
        near_high_conf = cand(10, 0, 0.95)
        far_low_conf = cand(100, 0, 0.2)
        best = select_best_candidate([near_high_conf, far_low_conf], REF, EMPTY_TRAJ,
                                     cfg_with(margin=80.0), last_obs=(10, 0))
        self.assertIs(best, near_high_conf)

    def test_margin_zero_disables_tie_break(self):
        near_low_conf = cand(10, 0, 0.5)
        far_high_conf = cand(100, 0, 0.95)
        best = select_best_candidate([near_low_conf, far_high_conf], REF, EMPTY_TRAJ,
                                     cfg_with(margin=0.0), last_obs=(10, 0))
        self.assertIs(best, near_low_conf)

    def test_small_margin_does_not_trigger(self):
        """代价差远大于 margin 时不做裁决。"""
        near_low_conf = cand(10, 0, 0.5)
        far_high_conf = cand(100, 0, 0.95)
        best = select_best_candidate([near_low_conf, far_high_conf], REF, EMPTY_TRAJ,
                                     cfg_with(margin=10.0), last_obs=(10, 0))
        self.assertIs(best, near_low_conf)

    def test_static_penalty_can_switch_choice(self):
        near_static = cand(10, 0, 0.9, is_static=True)
        far_normal = cand(100, 0, 0.9)
        best = select_best_candidate([near_static, far_normal], REF, EMPTY_TRAJ,
                                     cfg_with(margin=0.0))
        self.assertIs(best, far_normal)

    def test_max_dist_when_all_candidates_are_far_still_returns_best(self):
        """全部候选都超出 max_dist 时不返回 None，而是保留最优点（避免误丢快球）。"""
        a = cand(200, 0, 0.9)
        b = cand(300, 0, 0.8)
        best = select_best_candidate([a, b], REF, EMPTY_TRAJ, cfg_with(max_dist=100.0))
        self.assertIs(best, a)

    def test_max_dist_still_ranks_within_pool(self):
        """在范围内的候选里仍按代价 + tie 规则选。"""
        inside_high_conf = cand(50, 0, 0.99)   # 范围内
        outside = cand(500, 0, 0.99)           # 范围外
        best = select_best_candidate([outside, inside_high_conf], REF, EMPTY_TRAJ,
                                     cfg_with(max_dist=100.0))
        self.assertIs(best, inside_high_conf)


class CostSanityTest(unittest.TestCase):
    def test_confidence_has_small_weight_relative_to_distance(self):
        """记录现状：conf 权重只有 (1-λ)，所以 1px 距离 ≈ 4 个 conf 点，conf 容易被淹没。"""
        c1 = cand(0, 0, 0.0)
        c2 = cand(0, 0, 1.0)
        t1, _, _ = compute_cost(c1, REF, 0.85, static_penalty=120.0)
        t2, _, _ = compute_cost(c2, REF, 0.85, static_penalty=120.0)
        self.assertAlmostEqual(t1 - t2, 0.15, places=6)


if __name__ == "__main__":
    unittest.main()
