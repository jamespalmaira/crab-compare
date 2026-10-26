"""Resumable runs: completed (model, attribute, fold, seed) runs are skipped, never duplicated.

Training and evaluation are replaced by fast fakes; the loop, markers and tables are real.

    python -m unittest discover -s tests -t . -v
"""
import logging
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import pandas as pd
import yaml

import run
from scripts.make_dummy_data import make_dummy_dataset
from src.report import RUN_KEYS

ROOT = Path(__file__).resolve().parents[1]
MODELS = ["resnet50", "mobilenetv3_large_100"]


class FakeResult:
    def __init__(self, arch, fold, seed):
        self.info = {"arch": arch, "fold": fold, "seed": seed}
        self.model = None


def fake_evaluate(res, cfg, out_dir, device):
    keys = {"attribute": "species", "arch": res.info["arch"], "fold": res.info["fold"], "seed": res.info["seed"]}
    row = {**keys, "model": res.info["arch"], "best_val_macro_f1": 0.5, "best_epoch": 1, "epochs_trained": 1,
           "accuracy": 0.5, "macro_precision": 0.5, "macro_recall": 0.5, "macro_specificity": 0.5,
           "macro_f1": 0.5, "roc_auc_ovr": 0.5, "latency_gpu_ms": 1.0, "latency_cpu_ms": 1.0,
           "params_m": 1.0, "model_size_mb": 1.0}
    per_class = pd.DataFrame([{**keys, "class": c, "f1": 0.5} for c in ("species_A", "species_B")])
    return row, per_class, pd.DataFrame()


class ResumeTests(unittest.TestCase):
    def setUp(self):
        logging.disable(logging.WARNING)
        self.tmp = Path(tempfile.mkdtemp())
        csv = make_dummy_dataset(self.tmp / "d", n_species=2, crabs_per_species=10, photos_per_crab=2, size=32, seed=0)
        cfg = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
        cfg["results_dir"] = str(self.tmp / "results")
        cfg["data"].update(labels_csv=str(csv), external_test=None, min_groups_per_class=2)
        cfg["split"]["n_folds"] = 2
        self.config = self.tmp / "config.yaml"
        self.config.write_text(yaml.safe_dump(cfg), encoding="utf-8")
        self.out = self.tmp / "results" / "resume"
        self.calls = []

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)
        logging.disable(logging.NOTSET)

    def main(self, interrupt_at=None, extra=()):
        def fake_train(cfg, arch, rows, split, fold, seed, out_dir, device):
            if interrupt_at is not None and len(self.calls) == interrupt_at:
                raise KeyboardInterrupt  # the run is stopped part-way through this training
            self.calls.append((arch, fold, seed))
            return FakeResult(arch, fold, seed)

        with mock.patch.object(run, "train_one", fake_train), mock.patch.object(run, "evaluate_run", fake_evaluate), \
                mock.patch.object(run, "run_analysis"), mock.patch.object(run, "resolve_device"), \
                mock.patch.object(run, "limit_gpu_memory"):
            return run.main(["--config", str(self.config), "--attribute", "species", "--run-name", "resume",
                             "--models", *MODELS, "--seeds", "2", *extra])

    def test_interrupted_run_resumes_without_duplicates(self):
        with self.assertRaises(KeyboardInterrupt):
            self.main(interrupt_at=3)  # 2 models x 2 folds x 2 seeds = 8 runs; stopped during the 4th
        first = list(self.calls)
        self.assertEqual(len(first), 3)
        self.assertEqual(len(list((self.out / "completed").rglob("*.json"))), 3)
        self.assertEqual(len(pd.read_csv(self.out / "per_seed.csv")), 3)

        self.calls.clear()
        self.assertEqual(self.main(), 0)
        self.assertEqual(len(self.calls), 5)  # only the 5 not yet completed
        self.assertFalse(set(first) & set(self.calls))
        per_seed = pd.read_csv(self.out / "per_seed.csv")
        self.assertEqual(len(per_seed), 8)
        self.assertFalse(per_seed.duplicated(RUN_KEYS).any())
        per_class = pd.read_csv(self.out / "per_class.csv")
        self.assertEqual(len(per_class), 16)
        self.assertEqual(pd.read_csv(self.out / "summary.csv")["n_runs"].tolist(), [4, 4])

        self.calls.clear()
        self.assertEqual(self.main(), 0)  # everything done: nothing retrained, tables unchanged
        self.assertEqual(self.calls, [])
        self.assertEqual(len(pd.read_csv(self.out / "per_seed.csv")), 8)

    def test_markers_with_other_settings_are_not_resumed(self):
        with self.assertRaises(KeyboardInterrupt):
            self.main(interrupt_at=1)
        self.calls.clear()
        self.assertEqual(self.main(extra=["--epochs", "7"]), 2)  # refused with a message, nothing trained
        self.assertEqual(self.calls, [])


if __name__ == "__main__":
    unittest.main()
