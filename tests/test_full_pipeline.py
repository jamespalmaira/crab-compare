"""End to end on tiny dummy data: train genus and species, then the final analysis.

The analysis (statistics, Grad-CAM, ONNX, ranking, report) must cover genus only
(selection.attributes) and must not fail when species, the exploratory attribute,
is in the same results folder. Runs are also checked to go seed, fold, model.
Random weights, 1 epoch, 2 models; takes about a minute.

    python -m unittest discover -s tests -t . -v
"""
import json
import logging
import shutil
import tempfile
import unittest
from pathlib import Path

import pandas as pd
import yaml

import run
from scripts.make_dummy_data import make_dummy_dataset

ROOT = Path(__file__).resolve().parents[1]
MODELS = ["mobilenetv3_large_100", "resnet50"]
GENUS = {"species_A": "Scylla", "species_B": "Scylla", "species_C": "Portunus"}


class RunOrderTests(unittest.TestCase):
    def test_seed_then_fold_then_model(self):
        order = run.run_order("genus", ["m1", "m2"], [0, 1], [7, 8])
        self.assertEqual(order, [("genus", "m1", 0, 7), ("genus", "m2", 0, 7), ("genus", "m1", 1, 7),
                                 ("genus", "m2", 1, 7), ("genus", "m1", 0, 8), ("genus", "m2", 0, 8),
                                 ("genus", "m1", 1, 8), ("genus", "m2", 1, 8)])


class FullPipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        logging.disable(logging.WARNING)
        cls.tmp = Path(tempfile.mkdtemp())
        csv = make_dummy_dataset(cls.tmp / "d", n_species=3, crabs_per_species=15, photos_per_crab=2, size=48, seed=3)
        labels = pd.read_csv(csv)
        labels["genus"] = labels["species"].map(GENUS)
        labels.to_csv(csv, index=False)
        cfg = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
        cfg["results_dir"] = str(cls.tmp / "results")
        cfg["data"].update(labels_csv=str(csv), external_test=None, min_groups_per_class=2, num_workers=0)
        cfg["evaluation"].update(latency_warmup=1, latency_runs=2)
        cfg["analysis"]["gradcam_images"] = 2
        config = cls.tmp / "config.yaml"
        config.write_text(yaml.safe_dump(cfg), encoding="utf-8")

        cls.calls = []
        real_train = run.train_one

        def recording_train(cfg, arch, rows, split, fold, seed, out_dir, device):
            cls.calls.append((split["attribute"], arch, fold, seed))
            return real_train(cfg, arch, rows, split, fold, seed, out_dir, device)

        run.train_one = recording_train
        try:
            cls.rc = run.main(["--config", str(config), "--attribute", "all", "--run-name", "pipeline",
                               "--models", *MODELS, "--seeds", "2", "--folds", "0", "1", "--epochs", "1",
                               "--no-pretrained"])
        finally:
            run.train_one = real_train
        cls.out = cls.tmp / "results" / "pipeline"

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)
        logging.disable(logging.NOTSET)

    def test_runs_and_analysis_succeed(self):
        self.assertEqual(self.rc, 0)
        per_seed = pd.read_csv(self.out / "per_seed.csv")
        self.assertEqual(sorted(per_seed["attribute"].unique()), ["genus", "species"])  # species kept in the CSVs
        self.assertEqual(len(per_seed), 2 * 2 * 2 * 2)

    def test_training_order_is_seed_then_fold_then_model(self):
        expected = [(a, m, f, s) for a in ("genus", "species") for s in (0, 1) for f in (0, 1) for m in MODELS]
        self.assertEqual(self.calls, expected)

    def test_analysis_covers_genus_only(self):
        stats = json.loads((self.out / "stats.json").read_text(encoding="utf-8"))
        self.assertEqual(list(stats["attributes"]), ["genus"])
        self.assertEqual(set(pd.read_csv(self.out / "mcnemar.csv")["attribute"]), {"genus"})
        profile = pd.read_csv(self.out / "model_profile.csv")
        self.assertTrue(profile["profiled_run"].str.startswith("genus").all())
        self.assertEqual([p.name for p in (self.out / "gradcam").iterdir()], ["genus"])
        self.assertEqual(sorted(p.stem for p in (self.out / "onnx").glob("*.onnx")), sorted(MODELS))
        ranking = pd.read_csv(self.out / "ranking.csv")
        self.assertEqual(sorted(ranking["arch"]), sorted(MODELS))
        html = (self.out / "report.html").read_text(encoding="utf-8")
        self.assertIn("genus: Friedman", html)
        self.assertNotIn("species: Friedman", html)
        self.assertNotIn("<h3>species</h3>", html)
        self.assertIn("(in this folder: species)", html)


if __name__ == "__main__":
    unittest.main()
