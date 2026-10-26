"""Saved test-set predictions line up with the test images.

Uses the dummy dataset, the real test loader and predict(), with a tiny CPU
model so no GPU or timm weights are needed.

    python -m unittest discover -s tests -t . -v
"""
import copy
import logging
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from scripts.make_dummy_data import make_dummy_dataset
from src.config import load_config
from src.data import attribute_rows, build_datasets, build_loaders, get_split, load_labels
from src.evaluate import predictions_path, save_predictions
from src.train import predict

ROOT = Path(__file__).resolve().parents[1]
MODEL_CFG = {"input_size": (3, 32, 32), "mean": (0.5, 0.5, 0.5), "std": (0.5, 0.5, 0.5),
             "crop_pct": 1.0, "interpolation": "bilinear"}


class TinyModel(torch.nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        self.fc = torch.nn.Linear(3 * 32 * 32, n_classes)

    def forward(self, x):
        return self.fc(x.flatten(1))


class PredictionFileTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        logging.disable(logging.WARNING)
        cls.tmp = Path(tempfile.mkdtemp())
        cls.csv = make_dummy_dataset(cls.tmp / "dummy", n_species=3, crabs_per_species=6,
                                     photos_per_crab=3, size=40, seed=1)
        cfg = load_config(ROOT / "config.yaml")
        cfg["data"]["labels_csv"] = str(cls.csv)
        cls.cfg = copy.deepcopy(cfg)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)
        logging.disable(logging.NOTSET)

    def test_predictions_csv_matches_test_set(self):
        df = load_labels(self.csv)
        rows = attribute_rows(df, "species")
        split = get_split(self.cfg, rows, "species")
        classes = split["classes"]
        datasets = build_datasets(self.cfg, rows, split, 0, MODEL_CFG, seed=0)
        loaders = build_loaders(datasets, batch_size=4, num_workers=0, seed=0)
        torch.manual_seed(0)
        model = TinyModel(len(classes))
        probs, y_true = predict(model, loaders["test"], torch.device("cpu"))

        path = predictions_path(self.tmp / "results", "tiny", "species", seed=0, fold=0)
        self.assertEqual(path.name, "tiny_species_0_0.csv")
        save_predictions(path, datasets["test"], y_true, probs, classes)
        out = pd.read_csv(path)

        self.assertEqual(list(out.columns),
                         ["image_path", "crab_id", "true_label", "pred_label"] + [f"prob_{c}" for c in classes])
        test_rows = rows[rows.crab_id.isin(split["folds"][0]["test"])]
        self.assertEqual(len(out), len(test_rows))
        self.assertEqual(set(out.crab_id), set(split["folds"][0]["test"]))
        # each row's true label and crab_id are the ones labels.csv gives for that image
        truth = df.set_index("image_path")
        for _, r in out.iterrows():
            self.assertEqual(truth.loc[r.image_path, "species"], r.true_label)
            self.assertEqual(truth.loc[r.image_path, "crab_id"], r.crab_id)
        prob_cols = out[[f"prob_{c}" for c in classes]].to_numpy()
        np.testing.assert_allclose(prob_cols.sum(1), 1.0, atol=1e-5)
        self.assertEqual(list(out.pred_label), [classes[i] for i in prob_cols.argmax(1)])

    def test_misaligned_predictions_are_rejected(self):
        rows = attribute_rows(load_labels(self.csv), "species")
        split = get_split(self.cfg, rows, "species")
        datasets = build_datasets(self.cfg, rows, split, 0, MODEL_CFG, seed=0)
        n = len(datasets["test"])
        y = np.array(datasets["test"].targets)
        probs = np.full((n, 3), 1 / 3)
        with self.assertRaises(ValueError):
            save_predictions(self.tmp / "bad.csv", datasets["test"], y[:-1], probs[:-1], split["classes"])
        with self.assertRaises(ValueError):
            save_predictions(self.tmp / "bad.csv", datasets["test"], y[::-1], probs, split["classes"])


if __name__ == "__main__":
    unittest.main()
