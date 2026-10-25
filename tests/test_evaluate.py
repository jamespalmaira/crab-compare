"""Metric and results-table tests on hand-made predictions (no GPU, no model).

    python -m unittest discover -s tests -t . -v
"""
import math
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from src.evaluate import compute_metrics, save_confusion_matrix
from src.report import merge_rows, summarise

CLASSES = ["a", "b", "c"]


def one_hot_probs(pred, n=3, conf=0.8):
    p = np.full((len(pred), n), (1 - conf) / (n - 1))
    p[np.arange(len(pred)), pred] = conf
    return p


class MetricTests(unittest.TestCase):
    def test_perfect_predictions(self):
        y = np.array([0, 0, 1, 1, 2, 2])
        s, per_class, cm = compute_metrics(y, one_hot_probs(y), CLASSES)
        for k in ("accuracy", "macro_precision", "macro_recall", "macro_specificity", "macro_f1", "roc_auc_ovr"):
            self.assertAlmostEqual(s[k], 1.0, msg=k)
        self.assertTrue((cm == np.diag([2, 2, 2])).all())

    def test_known_confusion_and_specificity(self):
        y = np.array([0, 0, 0, 1, 1, 2])
        pred = np.array([0, 0, 1, 1, 2, 2])
        s, per_class, cm = compute_metrics(y, one_hot_probs(pred), CLASSES)
        self.assertAlmostEqual(s["accuracy"], 4 / 6)
        # class a: TN = 3 (rows b, c not predicted a), FP = 0 -> 1.0
        # class b: negatives = 4 (y != b), FP = 1 (an a predicted b) -> 3/4
        # class c: negatives = 5, FP = 1 (a b predicted c) -> 4/5
        np.testing.assert_allclose(per_class["specificity"], [1.0, 0.75, 0.8])
        self.assertAlmostEqual(s["macro_specificity"], np.mean([1.0, 0.75, 0.8]))
        np.testing.assert_allclose(per_class["recall"], [2 / 3, 1 / 2, 1.0])

    def test_class_missing_from_test_set_does_not_crash(self):
        y = np.array([0, 0, 1, 1])  # no class c in the test set
        pred = np.array([0, 2, 1, 1])  # but the model predicts c once
        with self.assertLogs("src.evaluate", "WARNING"):
            s, per_class, cm = compute_metrics(y, one_hot_probs(pred), CLASSES)
        self.assertEqual(s["n_test_classes"], 2)
        self.assertEqual(s["n_auc_classes"], 2)
        self.assertTrue(math.isnan(per_class.loc[2, "roc_auc"]))
        self.assertFalse(math.isnan(s["roc_auc_ovr"]))
        self.assertAlmostEqual(s["macro_recall"], np.mean([0.5, 1.0]))  # c left out of the average
        self.assertEqual(cm.shape, (3, 3))

    def test_single_class_test_set_gives_nan_auc(self):
        y = np.array([1, 1, 1])
        with self.assertLogs("src.evaluate", "WARNING"):
            s, _, _ = compute_metrics(y, one_hot_probs(np.array([1, 0, 1])), CLASSES)
        self.assertTrue(math.isnan(s["roc_auc_ovr"]))
        self.assertEqual(s["n_auc_classes"], 0)
        self.assertAlmostEqual(s["accuracy"], 2 / 3)

    def test_binary_auc_matches_sklearn(self):
        rng = np.random.default_rng(0)
        y = rng.integers(0, 2, 40)
        p1 = np.clip(y * 0.3 + rng.uniform(0, 0.7, 40), 0, 1)
        probs = np.stack([1 - p1, p1], 1)
        s, _, _ = compute_metrics(y, probs, ["neg", "pos"])
        self.assertAlmostEqual(s["roc_auc_ovr"], roc_auc_score(y, p1))

    def test_confusion_matrix_files(self):
        tmp = Path(tempfile.mkdtemp())
        try:
            save_confusion_matrix(np.array([[2, 1], [0, 3]]), ["x", "y"], tmp / "cm" / "run", "title")
            self.assertTrue((tmp / "cm" / "run.png").stat().st_size > 0)
            df = pd.read_csv(tmp / "cm" / "run.csv", index_col=0)
            self.assertEqual(df.loc["true_x", "pred_y"], 1)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


class TableTests(unittest.TestCase):
    def row(self, arch, seed, f1, fold=0, attribute="species"):
        base = {"attribute": attribute, "model": arch.upper(), "arch": arch, "fold": fold, "seed": seed}
        return {**base, **{m: f1 for m in ("accuracy", "macro_precision", "macro_recall", "macro_specificity",
                                           "macro_f1", "roc_auc_ovr", "best_val_macro_f1", "latency_gpu_ms",
                                           "latency_cpu_ms", "params_m", "model_size_mb")}}

    def test_merge_replaces_same_run_and_keeps_others(self):
        tmp = Path(tempfile.mkdtemp())
        try:
            path = tmp / "per_seed.csv"
            merge_rows(path, [self.row("m1", 0, 0.5), self.row("m1", 1, 0.7)])
            merged = merge_rows(path, [self.row("m1", 1, 0.9), self.row("m2", 0, 0.4)])
            self.assertEqual(len(merged), 3)
            self.assertEqual(merged.set_index(["arch", "seed"]).loc[("m1", 1), "macro_f1"], 0.9)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_summary_mean_and_std_across_seeds(self):
        per_seed = pd.DataFrame([self.row("m2", 0, 0.4), self.row("m1", 0, 0.5), self.row("m1", 1, 0.7)])
        s = summarise(per_seed, model_order=["m1", "m2"]).set_index("arch")
        self.assertEqual(list(s.index), ["m1", "m2"])
        self.assertAlmostEqual(s.loc["m1", "macro_f1_mean"], 0.6)
        self.assertAlmostEqual(s.loc["m1", "macro_f1_std"], np.std([0.5, 0.7], ddof=1))
        self.assertEqual(s.loc["m1", "n_runs"], 2)
        self.assertTrue(math.isnan(s.loc["m2", "macro_f1_std"]))  # one run: std undefined


if __name__ == "__main__":
    unittest.main()
