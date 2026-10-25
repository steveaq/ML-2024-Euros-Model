import unittest
from itertools import combinations
from hashlib import sha256
import json
import numpy as np
import pandas as pd
from euro_lab.engine import *


class EngineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.matches, cls.ranks = load_data()
        cls.features, cls.frame, cls.history = build_features(cls.matches, cls.ranks)
        cls.snapshot = team_snapshot(cls.history, cls.ranks)

    def test_bundled_files_match_manifest(self):
        manifest = json.loads((ROOT / "data/manifest.json").read_text())
        for name, source in manifest["sources"].items():
            self.assertEqual(
                sha256((ROOT / f"data/{name}.csv.gz").read_bytes()).hexdigest(),
                source["bundled_sha256"],
            )

    def test_forecast_never_sees_tournament_results(self):
        self.assertLess(self.features.date.max(), CUTOFF)
        self.assertLess(self.snapshot["Latest result"].max(), CUTOFF)
        self.assertEqual(set(self.snapshot.index), set(TEAMS))

    def test_ranking_release_never_in_future(self):
        for side in ["home", "away"]:
            f = self.frame
            self.assertTrue((f[f"rank_date_{side}"] <= f.date).all())
            self.assertTrue((f[f"{side}_history_end"] < f.date).all())

    def test_current_score_does_not_leak(self):
        date = pd.Timestamp("2023-09-12")
        altered = self.matches.copy()
        altered.loc[altered.date >= date, "home_score"] = 99
        f, _, _ = build_features(altered, self.ranks)
        old = self.features[self.features.date <= date].reset_index(drop=True)
        new = f[f.date <= date].reset_index(drop=True)
        pd.testing.assert_frame_equal(
            old[HOME_FEATURES + AWAY_FEATURES], new[HOME_FEATURES + AWAY_FEATURES]
        )

    def test_neutral_rates_are_order_invariant(self):
        model = fit_pair(self.frame, "Mean baseline")
        h, a = match_rates(model, self.snapshot, "France", "Germany")
        ra, rh = match_rates(model, self.snapshot, "Germany", "France")
        np.testing.assert_allclose([h, a], [rh, ra])

    def test_outcomes_partition_probability(self):
        for h, a in [(0.05, 6), (1.3, 1.3), (3, 0.2)]:
            p = outcome_probs(h, a)
            self.assertAlmostEqual(p.sum(), 1, places=10)
            self.assertTrue((p >= 0).all())
            np.testing.assert_allclose(p, outcome_probs(a, h)[::-1])
            self.assertLessEqual(goal_grid(h, a).sum(), 1 + 1e-12)

    def test_equal_rates_have_equal_win_chances(self):
        p = outcome_probs(1.4, 1.4)
        self.assertAlmostEqual(p[0], p[2])

    def test_all_third_place_allocations_are_valid(self):
        orders = {g: teams for g, teams in GROUPS.items()}
        team_group = {t: g for g, ts in GROUPS.items() for t in ts}
        for combo in combinations("ABCDEF", 4):
            ties = round_of_16(orders, combo)
            self.assertEqual(len(set(sum([list(t) for t in ties], []))), 16)
            self.assertTrue(all(team_group[h] != team_group[a] for h, a in ties))
            for h, a in ties:
                if a == orders[team_group[a]][2]:
                    self.assertIn(team_group[a], combo)

    def test_head_to_head_precedes_overall_goal_difference(self):
        scores = [
            ("A", "B", 1, 0),
            ("A", "C", 1, 0),
            ("A", "D", 0, 1),
            ("B", "C", 10, 0),
            ("B", "D", 10, 0),
            ("C", "D", 1, 0),
        ]
        order, _ = rank_group(list("ABCD"), scores, np.random.default_rng(1))
        self.assertEqual(order[:2], ["A", "B"])

    def test_stage_probabilities_and_reproducibility(self):
        matrix = {(h, a): np.array([1.3, 1.3]) for h in TEAMS for a in TEAMS if h != a}
        first, points, bracket = simulate(matrix, n=80, seed=11)
        second, _, _ = simulate(matrix, n=80, seed=11)
        pd.testing.assert_frame_equal(first, second)
        np.testing.assert_allclose(first.iloc[:, :5].sum().to_numpy(), [16, 8, 4, 2, 1])
        self.assertTrue((np.diff(first.iloc[:, :5].to_numpy(), axis=1) <= 0).all())
        np.testing.assert_allclose(points.sum(axis=1), 1)
        self.assertEqual(len(bracket), 15)
        self.assertEqual(
            bracket.stage.value_counts().to_dict(),
            {"Round of 16": 8, "Quarter-final": 4, "Semi-final": 2, "Final": 1},
        )

    def test_holdout_window_is_separate(self):
        _, test, metrics = evaluate(self.frame, "Mean baseline")
        self.assertGreaterEqual(test.date.min(), pd.Timestamp("2024-01-01"))
        self.assertLess(test.date.max(), CUTOFF)
        self.assertTrue(np.isfinite(list(metrics.values())).all())
        self.assertTrue((test[["pred_home", "pred_away"]] > 0).all().all())

    def test_group_review_has_36_untrained_matches(self):
        group = self.matches[
            (self.matches.tournament == "UEFA Euro")
            & (self.matches.date >= CUTOFF)
            & (self.matches.date <= "2024-06-26")
        ]
        self.assertEqual(len(group), 36)
        self.assertTrue(set(group.home_team) | set(group.away_team) == set(TEAMS))


if __name__ == "__main__":
    unittest.main()
