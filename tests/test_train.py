"""Training-loop helpers that run without a GPU.

    python -m unittest discover -s tests -t . -v
"""
import unittest

from src.config import ConfigError, accumulation_steps, validate
from src.train import EarlyStopping


class EarlyStoppingTests(unittest.TestCase):
    def test_stops_after_patience_without_improvement(self):
        es = EarlyStopping(patience=2)
        improved = [es.step(s, e) for e, s in enumerate([0.5, 0.6, 0.6, 0.55], start=1)]
        self.assertEqual(improved, [True, True, False, False])
        self.assertEqual((es.best, es.best_epoch), (0.6, 2))  # a tie keeps the earlier epoch
        self.assertTrue(es.should_stop)

    def test_improvement_resets_patience(self):
        es = EarlyStopping(patience=2)
        for e, s in enumerate([0.5, 0.4, 0.7, 0.6], start=1):
            es.step(s, e)
        self.assertEqual(es.best_epoch, 3)
        self.assertFalse(es.should_stop)


class AccumulationTests(unittest.TestCase):
    def cfg(self, sizes, eff=32):
        return {"split": {"strategy": "holdout", "ratios": [0.7, 0.15, 0.15], "n_folds": 5},
                "training": {"effective_batch_size": eff},
                "models": {f"m{i}": {"batch_size": b} for i, b in enumerate(sizes)}}

    def test_every_model_reaches_the_same_effective_batch(self):
        cfg = self.cfg([8, 16, 16, 32, 8])
        validate(cfg)
        for arch, m in cfg["models"].items():
            self.assertEqual(m["batch_size"] * accumulation_steps(cfg, arch), 32)

    def test_batch_size_must_divide_effective_batch(self):
        with self.assertRaises(ConfigError):
            validate(self.cfg([12]))


if __name__ == "__main__":
    unittest.main()
