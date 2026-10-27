"""Weighted selection on hand-made model values (no GPU, no model).

    python -m unittest discover -s tests -t . -v
"""
import json
import shutil
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.config import ConfigError
from src.select import check_weights, minmax, score, sensitivity, tie_rule, write_ranking

BASE = {"macro_f1": 0.50, "inference_time": 0.20, "model_size": 0.15, "gradcam": 0.15}


def table():
    return pd.DataFrame([
        {"model": "Accurate", "arch": "a", "macro_f1": 0.90, "onnx_cpu_ms": 100.0, "model_size_mb": 300.0, "gradcam": 1.0},
        {"model": "Fast", "arch": "b", "macro_f1": 0.80, "onnx_cpu_ms": 10.0, "model_size_mb": 20.0, "gradcam": 1.0},
        {"model": "Middle", "arch": "c", "macro_f1": 0.85, "onnx_cpu_ms": 55.0, "model_size_mb": 160.0, "gradcam": 0.0},
    ])


class SelectionTests(unittest.TestCase):
    def test_minmax_direction_and_constant(self):
        s = pd.Series([1.0, 3.0, 2.0])
        self.assertEqual(list(minmax(s, True)), [0.0, 1.0, 0.5])
        self.assertEqual(list(minmax(s, False)), [1.0, 0.0, 0.5])
        self.assertEqual(list(minmax(pd.Series([2.0, 2.0]), True)), [1.0, 1.0])

    def test_score_is_weighted_sum_of_normalised_criteria(self):
        r = score(table(), BASE).set_index("arch")
        # a: f1 1, time 0, size 0, cam 1 -> 0.5 + 0.15
        self.assertAlmostEqual(r.loc["a", "score"], 0.65)
        # b: f1 0, time 1, size 1, cam 1 -> 0.2 + 0.15 + 0.15
        self.assertAlmostEqual(r.loc["b", "score"], 0.50)
        # c: f1 .5, time .5, size .5, cam 0 -> 0.25 + 0.1 + 0.075
        self.assertAlmostEqual(r.loc["c", "score"], 0.425)
        self.assertEqual(list(score(table(), BASE)["arch"]), ["a", "b", "c"])

    def test_sensitivity_reports_winner_changes(self):
        speed_heavy = {"macro_f1": 0.20, "inference_time": 0.40, "model_size": 0.25, "gradcam": 0.15}
        f1_heavy = {"macro_f1": 0.60, "inference_time": 0.15, "model_size": 0.10, "gradcam": 0.15}
        res = sensitivity(table(), "a", [f1_heavy, speed_heavy])
        self.assertEqual([(s["winner"], s["changed"]) for s in res], [("a", False), ("b", True)])

    def test_weights_are_validated(self):
        check_weights(BASE)
        with self.assertRaises(ConfigError):
            check_weights({**BASE, "macro_f1": 0.6})  # sums to 1.1
        with self.assertRaises(ConfigError):
            check_weights({"macro_f1": 1.0})


def tie_table():
    """A best by F1; B within one std and not significantly worse (tied, fastest); C within one std but
    significantly worse; D more than one std behind; E within, but no McNemar result."""
    t = pd.DataFrame([
        {"model": "A", "arch": "a", "macro_f1": 0.90, "macro_f1_std": 0.03, "onnx_cpu_ms": 100.0, "model_size_mb": 300.0, "gradcam": 1.0},
        {"model": "B", "arch": "b", "macro_f1": 0.88, "macro_f1_std": 0.02, "onnx_cpu_ms": 10.0, "model_size_mb": 20.0, "gradcam": 1.0},
        {"model": "C", "arch": "c", "macro_f1": 0.88, "macro_f1_std": 0.02, "onnx_cpu_ms": 5.0, "model_size_mb": 10.0, "gradcam": 1.0},
        {"model": "D", "arch": "d", "macro_f1": 0.80, "macro_f1_std": 0.02, "onnx_cpu_ms": 8.0, "model_size_mb": 15.0, "gradcam": 1.0},
        {"model": "E", "arch": "e", "macro_f1": 0.89, "macro_f1_std": 0.02, "onnx_cpu_ms": 9.0, "model_size_mb": 15.0, "gradcam": 1.0},
    ])
    pair = lambda a, b, ab, ba, p: {"model_a": a, "model_b": b, "a_right_b_wrong": ab, "a_wrong_b_right": ba,  # noqa: E731
                                    "p_holm": p}
    stats = {"attributes": {"genus": {"mcnemar": {"pairs": [
        pair("a", "b", 12, 9, 0.60), pair("a", "c", 30, 5, 0.001), pair("d", "a", 8, 14, 0.20)]}}}}
    return t, stats


class TieRuleTests(unittest.TestCase):
    def test_tied_models_are_ranked_by_speed_size_gradcam(self):
        table, stats = tie_table()
        out = tie_rule(score(table, BASE), stats, ["genus"], BASE).set_index("arch")
        self.assertEqual(out["tied"].to_dict(), {"a": True, "b": True, "c": False, "d": False, "e": False})
        self.assertIn("significantly worse", out.loc["c", "tie_note"])
        self.assertIn("gap", out.loc["d", "tie_note"])
        self.assertIn("no McNemar result", out.loc["e", "tie_note"])
        self.assertEqual(out.loc["b", "tie_rank"], 1)  # tied with A and faster, smaller
        self.assertEqual(out.loc["a", "tie_rank"], 2)
        # the rest follow in plain-score order
        rest = out[~out["tied"]].sort_values("tie_rank")
        self.assertEqual(list(rest["rank"]), sorted(rest["rank"]))

    def test_significance_needs_the_best_to_win_the_discordant_pairs(self):
        table, stats = tie_table()
        stats["attributes"]["genus"]["mcnemar"]["pairs"][1].update(a_right_b_wrong=5, a_wrong_b_right=30)
        out = tie_rule(score(table, BASE), stats, ["genus"], BASE).set_index("arch")
        self.assertTrue(out.loc["c", "tied"])  # significant, but C is the one with more right answers

    def test_per_seed_mcnemar_needs_significance_in_every_seed(self):
        table, stats = tie_table()
        pair = lambda a, b, ab, ba, p: {"model_a": a, "model_b": b, "a_right_b_wrong": ab,  # noqa: E731
                                        "a_wrong_b_right": ba, "p_holm": p}
        # C (pooled: significantly worse) is significant in seeds 0 and 1 only; B in no seed; D in all
        stats["attributes"]["genus"]["mcnemar_per_seed"] = {"seeds": {
            "0": [pair("a", "b", 4, 3, 0.9), pair("a", "c", 10, 1, 0.01), pair("d", "a", 1, 9, 0.02)],
            "1": [pair("a", "b", 4, 3, 0.9), pair("a", "c", 10, 2, 0.03), pair("d", "a", 1, 9, 0.02)],
            "2": [pair("a", "b", 4, 3, 0.9), pair("a", "c", 10, 5, 0.40), pair("d", "a", 1, 9, 0.02)]}}
        out = tie_rule(score(table, BASE), stats, ["genus"], BASE, mcnemar_mode="per_seed").set_index("arch")
        self.assertTrue(out.loc["c", "tied"])
        self.assertIn("2 of 3 seeds", out.loc["c", "tie_note"])
        self.assertEqual(out.loc["c", "mcnemar_p_holm"], "0.0100,0.0300,0.4000")
        self.assertTrue(out.loc["b", "tied"])
        self.assertFalse(out.loc["d", "tied"])  # significant in all seeds, and the gap is too large anyway
        self.assertFalse(out.loc["e", "tied"])  # no result
        pooled = tie_rule(score(table, BASE), stats, ["genus"], BASE).set_index("arch")
        self.assertFalse(pooled.loc["c", "tied"])
        with self.assertRaises(ValueError):
            tie_rule(score(table, BASE), stats, ["genus"], BASE, mcnemar_mode="other")

    def test_write_ranking_scores_genus_only_and_reports_both(self):
        tmp = Path(tempfile.mkdtemp())
        try:
            table, stats = tie_table()
            rows = []
            for attr, f1 in (("genus", None), ("species", {"a": 0.1, "b": 0.1, "c": 0.99, "d": 0.99, "e": 0.99})):
                for _, r in table.iterrows():
                    rows.append({"attribute": attr, "model": r["model"], "arch": r["arch"],
                                 "macro_f1_mean": r["macro_f1"] if f1 is None else f1[r["arch"]],
                                 "macro_f1_std": r["macro_f1_std"], "model_size_mb_mean": r["model_size_mb"],
                                 "latency_gpu_ms_mean": 1.0})
            pd.DataFrame(rows).to_csv(tmp / "summary.csv", index=False)
            profile = table.assign(onnx_size_mb=1.0, gflops=1.0, gradcam_pass=True)
            (tmp / "stats.json").write_text(json.dumps(stats), encoding="utf-8")
            cfg = {"run_name": "t", "models": {a: {} for a in "abcde"}, "data": {"main_attribute": "genus"},
                   "selection": {"weights": BASE, "attributes": ["genus"], "tie_rule": {"enabled": True, "alpha": 0.05},
                                 "sensitivity": []}}
            ranked, _ = write_ranking(cfg, tmp, profile)
            self.assertEqual(ranked.loc[0, "arch"], "b")  # tie-rule winner is the recommendation
            self.assertTrue(ranked.loc[0, "recommended"])
            by = ranked.set_index("arch")
            self.assertAlmostEqual(by.loc["c", "macro_f1"], 0.88)  # species (0.99) ignored
            md = (tmp / "ranking.md").read_text(encoding="utf-8")
            self.assertIn("## Plain weighted score", md)
            self.assertIn("## Tie rule (the recommendation)", md)
            self.assertIn("Scored attributes: genus", md)
            self.assertNotIn("McNemar test (reported, not used)", md)  # no per-seed results in stats.json
            seeds = {str(k): stats["attributes"]["genus"]["mcnemar"]["pairs"] for k in range(3)}
            stats["attributes"]["genus"]["mcnemar_per_seed"] = {"seeds": seeds}
            (tmp / "stats.json").write_text(json.dumps(stats), encoding="utf-8")
            cfg["selection"]["tie_rule"]["mcnemar"] = "per_seed"
            ranked, _ = write_ranking(cfg, tmp, profile)
            md = (tmp / "ranking.md").read_text(encoding="utf-8")
            self.assertIn("McNemar version used (selection.tie_rule.mcnemar): **per_seed**", md)
            self.assertIn("Tie rule with the pooled McNemar test (reported, not used)", md)
            self.assertIn("the tie set **does not change**", md)
            self.assertEqual(ranked.set_index("arch")["tied_pooled"].to_dict(), ranked.set_index("arch")["tied"].to_dict())
            cfg["selection"]["tie_rule"]["enabled"] = False
            plain, _ = write_ranking(cfg, tmp, profile)
            self.assertEqual(plain.loc[0, "rank"], 1)  # plain score decides when the rule is off
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
