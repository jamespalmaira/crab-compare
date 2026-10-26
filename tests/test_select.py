"""Weighted selection on hand-made model values (no GPU, no model).

    python -m unittest discover -s tests -t . -v
"""
import unittest

import pandas as pd

from src.config import ConfigError
from src.select import check_weights, minmax, score, sensitivity

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


if __name__ == "__main__":
    unittest.main()
