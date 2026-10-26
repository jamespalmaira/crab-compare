"""Results by source and the source-dominance check (no GPU, no model).

    python -m unittest discover -s tests -t . -v
"""
import json
import logging
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.make_dummy_data import make_dummy_dataset
from src.config import load_config
from src.data import attribute_rows, build_datasets, get_split, load_labels, source_balance
from src.evaluate import per_source_metrics, save_predictions
from src.report import data_section, source_sections, summarise_sources

ROOT = Path(__file__).resolve().parents[1]
MODEL_CFG = {"input_size": (3, 32, 32), "mean": (0.5,) * 3, "std": (0.5,) * 3, "crop_pct": 1.0,
             "interpolation": "bilinear"}


class SourceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        logging.disable(logging.WARNING)
        cls.tmp = Path(tempfile.mkdtemp())
        # first species: 9 in 10 crabs from inaturalist; others spread over three sources
        cls.csv = make_dummy_dataset(cls.tmp / "public", n_species=3, crabs_per_species=10, photos_per_crab=2,
                                     size=32, seed=0, sources=["inaturalist", "gbif", "manual"], skew_source=True)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)
        logging.disable(logging.NOTSET)

    def test_blank_source_becomes_unknown(self):
        csv = self.tmp / "blank_source.csv"
        df = pd.read_csv(self.csv, dtype=str, keep_default_na=False)
        df.loc[0, "source"] = ""
        df.to_csv(csv, index=False)
        self.assertEqual(load_labels(csv).loc[0, "source"], "unknown")

    def test_dominant_source_is_flagged(self):
        rows = attribute_rows(load_labels(self.csv), "species")
        bal = source_balance(rows, "species", 0.8).set_index("class")
        self.assertTrue(bal.loc["species_A", "flagged"])  # 90% inaturalist
        self.assertEqual(bal.loc["species_A", "top_source"], "inaturalist")
        self.assertAlmostEqual(bal.loc["species_A", "top_share"], 0.9)
        self.assertFalse(bal.loc["species_B", "flagged"])
        self.assertEqual(json.loads(bal.loc["species_A", "images_by_source"]), {"gbif": 2, "inaturalist": 18})

    def test_threshold_is_strictly_greater_than(self):
        rows = pd.DataFrame({"species": ["a"] * 5, "source": ["x"] * 4 + ["y"]})
        self.assertFalse(source_balance(rows, "species", 0.8).loc[0, "flagged"])  # exactly 80%
        self.assertTrue(source_balance(rows, "species", 0.79).loc[0, "flagged"])

    def test_no_source_column_gives_empty_results(self):
        rows = pd.DataFrame({"species": ["a", "b"], "crab_id": ["1", "2"]})
        self.assertTrue(source_balance(rows, "species", 0.8).empty)
        self.assertTrue(per_source_metrics(pd.DataFrame({"true_label": ["a"], "pred_label": ["a"]}), ["a"]).empty)

    def test_per_source_metrics(self):
        pred = pd.DataFrame({
            "crab_id": ["1", "1", "2", "3", "4", "5"],
            "source": ["s1", "s1", "s1", "s2", "s2", "s2"],
            "true_label": ["a", "a", "b", "a", "b", "b"],
            "pred_label": ["a", "b", "b", "b", "b", "b"],
        })
        m = per_source_metrics(pred, ["a", "b", "c"]).set_index("source")
        self.assertEqual(m.loc["s1", "n_images"], 3)
        self.assertEqual(m.loc["s1", "n_groups"], 2)
        self.assertAlmostEqual(m.loc["s1", "accuracy"], 2 / 3)
        # s1: F1(a) = 2*1*0.5/1.5 = 2/3, F1(b) = 2*0.5*1/1.5 = 2/3; class c absent so not averaged
        self.assertAlmostEqual(m.loc["s1", "macro_f1"], 2 / 3)
        self.assertEqual(m.loc["s2", "n_classes_present"], 2)
        self.assertAlmostEqual(m.loc["s2", "accuracy"], 2 / 3)

    def test_predictions_carry_source(self):
        cfg = load_config(ROOT / "config.yaml")
        cfg["data"]["labels_csv"] = str(self.csv)
        rows = attribute_rows(load_labels(self.csv), "species")
        split = get_split(cfg, rows, "species")
        ds = build_datasets(cfg, rows, split, 0, MODEL_CFG, seed=0)["test"]
        n = len(ds)
        out = save_predictions(self.tmp / "pred.csv", ds, np.array(ds.targets), np.full((n, 3), 1 / 3),
                               split["classes"])
        self.assertEqual(list(out.columns[:5]), ["image_path", "crab_id", "source", "true_label", "pred_label"])
        truth = load_labels(self.csv).set_index("image_path")
        self.assertTrue((truth.loc[out["image_path"], "source"].to_numpy() == out["source"].to_numpy()).all())

    def test_source_summary_mean_and_std(self):
        per_source = pd.DataFrame([
            {"attribute": "species", "arch": "m", "model": "M", "fold": 0, "seed": s, "source": "s1",
             "n_images": 10, "accuracy": a, "macro_f1": a} for s, a in ((0, 0.5), (1, 0.7))])
        out = summarise_sources(per_source).iloc[0]
        self.assertEqual(out["n_runs"], 2)
        self.assertAlmostEqual(out["accuracy_mean"], 0.6)
        self.assertAlmostEqual(out["macro_f1_std"], np.std([0.5, 0.7], ddof=1))

    def test_report_sections(self):
        out = self.tmp / "report_bits"
        out.mkdir()
        self.assertIn("no <code>source</code> column", "".join(source_sections({"data": {}}, out)))
        rows = attribute_rows(load_labels(self.csv), "species")
        source_balance(rows, "species", 0.8).to_csv(out / "source_balance.csv", index=False)
        html = "".join(source_sections({"data": {}}, out))
        self.assertIn("1 of 3 class(es) flagged", html)
        self.assertIn("⚠ flagged", html)
        (out / "attributes.json").write_text(json.dumps({
            "species": {"n_images": 60, "n_groups": 30, "n_classes": 3, "groups_per_class": {"a": 10},
                        "min_groups_per_class": 30, "skip_reason": None},
            "disease": {"n_images": 0, "n_groups": 0, "n_classes": 0, "groups_per_class": {},
                        "min_groups_per_class": 30, "skip_reason": "0 labeled class(es)"}}), encoding="utf-8")
        html = "".join(data_section(out))
        self.assertIn("✓ used", html)
        self.assertIn("⚠ skipped", html)


if __name__ == "__main__":
    unittest.main()
