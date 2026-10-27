"""Friedman and McNemar on hand-made results (no GPU, no model).

    python -m unittest discover -s tests -t . -v
"""
import json
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest, friedmanchisquare
from statsmodels.stats.multitest import multipletests

from src.stats import HOLDOUT_CAUTION, friedman, mcnemar_pairs, mcnemar_per_seed, run_stats

MODELS = ["m1", "m2", "m3"]


class StatsTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def per_seed(self, strategy, f1, correct=None):
        """f1[(arch, fold, seed)] -> macro_f1; correct[(arch, fold, seed)] -> list of 0/1 per test image."""
        rows = []
        for (arch, fold, seed), v in f1.items():
            pred_path = self.tmp / f"{arch}_{fold}_{seed}.csv"
            c = correct[(arch, fold, seed)] if correct else [1, 0]
            pd.DataFrame({"image_path": [f"f{fold}_img{i}.jpg" for i in range(len(c))], "crab_id": "x",
                          "true_label": "a", "pred_label": ["a" if k else "b" for k in c]}).to_csv(pred_path, index=False)
            rows.append({"attribute": "species", "arch": arch, "fold": fold, "seed": seed, "macro_f1": v,
                         "split_strategy": strategy, "predictions": str(pred_path)})
        return pd.DataFrame(rows)

    def test_friedman_kfold_uses_folds_averaged_over_seeds(self):
        rng = np.random.default_rng(0)
        f1 = {(m, k, s): float(rng.uniform()) for m in MODELS for k in range(5) for s in (0, 1)}
        res = friedman(self.per_seed("kfold", f1), "species", MODELS)
        self.assertEqual((res["block"], res["n_blocks"]), ("fold", 5))
        expected = friedmanchisquare(*[[np.mean([f1[(m, k, 0)], f1[(m, k, 1)]]) for k in range(5)] for m in MODELS])
        self.assertAlmostEqual(res["statistic"], expected.statistic)
        self.assertAlmostEqual(res["p_value"], expected.pvalue)
        self.assertNotIn("caution", res["note"])

    def test_friedman_holdout_uses_seeds_with_caution(self):
        f1 = {(m, 0, s): 0.5 + 0.1 * i + 0.01 * s for i, m in enumerate(MODELS) for s in range(3)}
        res = friedman(self.per_seed("holdout", f1), "species", MODELS)
        self.assertEqual((res["block"], res["n_blocks"]), ("seed", 3))
        self.assertEqual(res["note"], HOLDOUT_CAUTION)
        self.assertEqual(res["mean_rank"], {"m1": 3.0, "m2": 2.0, "m3": 1.0})

    def test_friedman_all_tied_does_not_crash(self):
        f1 = {(m, k, 0): 0.5 for m in MODELS for k in range(3)}
        res = friedman(self.per_seed("kfold", f1), "species", MODELS)
        self.assertIsNone(res["p_value"])
        self.assertIn("tied", res["note"])

    def test_mcnemar_counts_exact_p_and_holm(self):
        correct = {("m1", 0, 0): [1, 1, 1, 1, 1, 1, 0, 0],
                   ("m2", 0, 0): [0, 0, 0, 0, 0, 1, 0, 0],
                   ("m3", 0, 0): [1, 1, 1, 1, 1, 1, 0, 0]}
        f1 = {k: 0.5 for k in correct}
        pairs = {(r["model_a"], r["model_b"]): r for r in mcnemar_pairs(self.per_seed("holdout", f1, correct),
                                                                         "species", MODELS)}
        p12 = pairs[("m1", "m2")]
        self.assertEqual((p12["a_right_b_wrong"], p12["a_wrong_b_right"], p12["n_pairs"]), (5, 0, 8))
        self.assertAlmostEqual(p12["p_raw"], binomtest(0, 5, 0.5).pvalue)  # exact for few discordant pairs
        self.assertEqual(pairs[("m1", "m3")]["p_raw"], 1.0)  # identical predictions
        raw = [pairs[k]["p_raw"] for k in [("m1", "m2"), ("m1", "m3"), ("m2", "m3")]]
        holm = multipletests(raw, method="holm")[1]
        self.assertAlmostEqual(p12["p_holm"], holm[0])
        self.assertGreaterEqual(p12["p_holm"], p12["p_raw"])

    def test_mcnemar_pairs_by_fold_seed_and_image(self):
        # fold 1 exists only for m1 and m2: m3 pairs use fold 0 only
        correct = {("m1", 0, 0): [1, 0], ("m2", 0, 0): [0, 0], ("m3", 0, 0): [1, 1],
                   ("m1", 1, 0): [1, 1], ("m2", 1, 0): [0, 1]}
        pairs = {(r["model_a"], r["model_b"]): r
                 for r in mcnemar_pairs(self.per_seed("kfold", {k: 0.5 for k in correct}, correct), "species", MODELS)}
        self.assertEqual(pairs[("m1", "m2")]["n_pairs"], 4)
        self.assertEqual(pairs[("m1", "m2")]["a_right_b_wrong"], 2)
        self.assertEqual(pairs[("m1", "m3")]["n_pairs"], 2)

    def test_mcnemar_per_seed_tests_each_seed_alone(self):
        # m1 beats m2 on 30 images in seed 0 and on 30 in seed 1 but only on 3 in seed 2
        right, wrong = [1] * 30 + [0] * 4, [0] * 34
        correct = {("m1", 0, 0): right, ("m2", 0, 0): wrong, ("m3", 0, 0): right,
                   ("m1", 0, 1): right, ("m2", 0, 1): wrong, ("m3", 0, 1): right,
                   ("m1", 0, 2): [1] * 3 + [0] * 31, ("m2", 0, 2): wrong, ("m3", 0, 2): [1] * 3 + [0] * 31}
        df = self.per_seed("kfold", {k: 0.5 for k in correct}, correct)
        res = mcnemar_per_seed(df, "species", MODELS, alpha=0.05)
        self.assertEqual(sorted(res["seeds"]), [0, 1, 2])
        for seed, pairs in res["seeds"].items():
            self.assertEqual({r["n_pairs"] for r in pairs}, {34})  # every image once per seed
            pooled_alone = mcnemar_pairs(df[df["seed"] == seed], "species", MODELS)
            self.assertEqual([r["p_holm"] for r in pairs], [r["p_holm"] for r in pooled_alone])  # own Holm
        s12 = next(r for r in res["summary"] if (r["model_a"], r["model_b"]) == ("m1", "m2"))
        self.assertEqual((s12["n_seeds_significant"], s12["n_seeds"]), (2, 3))
        self.assertFalse(s12["significant_all_seeds_a_better"])
        pooled = {(r["model_a"], r["model_b"]): r for r in mcnemar_pairs(df, "species", MODELS)}
        self.assertEqual(pooled[("m1", "m2")]["n_pairs"], 102)  # pooled counts each image three times

    def test_run_stats_writes_valid_json(self):
        rng = np.random.default_rng(1)
        keys = [(m, k, s) for m in MODELS for k in range(3) for s in (0, 1)]
        correct = {key: list(rng.integers(0, 2, 6)) for key in keys}
        df = self.per_seed("kfold", {key: float(rng.uniform()) for key in keys}, correct)
        df.to_csv(self.tmp / "per_seed.csv", index=False)
        run_stats(self.tmp, MODELS)
        data = json.loads((self.tmp / "stats.json").read_text(encoding="utf-8"))  # strict JSON, no NaN
        sp = data["attributes"]["species"]
        self.assertEqual(len(sp["mcnemar"]["pairs"]), 3)
        self.assertEqual(sp["friedman"]["block"], "fold")
        self.assertTrue((self.tmp / "mcnemar.csv").exists())
        self.assertEqual(sorted(sp["mcnemar_per_seed"]["seeds"]), ["0", "1"])
        self.assertEqual(len(pd.read_csv(self.tmp / "mcnemar_per_seed.csv")), 2 * 3)


if __name__ == "__main__":
    unittest.main()
