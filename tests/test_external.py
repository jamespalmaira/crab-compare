"""External test set (data.external_test): held out of every split, no observer on both sides.

    python -m unittest discover -s tests -t . -v
"""
import copy
import json
import logging
import shutil
import tempfile
import unittest
from pathlib import Path

import pandas as pd
import torch

from run import hold_out_external, select_attributes
from scripts.make_dummy_data import make_dummy_dataset
from src.config import load_config
from src.data import SPLIT_SETS, load_labels, split_external
from src.evaluate import evaluate_external
from src.report import RUN_KEYS, external_section, merge_rows

ROOT = Path(__file__).resolve().parents[1]
EXT = {"column": "country", "value": "PH", "name": "Philippines", "group_columns": ["crab_id", "observer_id"],
       "report_by": "genus"}
MODEL_CFG = {"input_size": (3, 32, 32), "mean": (0.5,) * 3, "std": (0.5,) * 3, "crop_pct": 1.0,
             "interpolation": "bilinear"}


def make_rows():
    """Genus/species rows over 4 countries. Observers o0-o4 shoot in PH and elsewhere; obs-level crab_id for
    gbif-like rows means crab_id differs but observer_id matches."""
    rows = []
    genera = {"Scylla": ["Scylla serrata", "Scylla olivacea"], "Portunus": ["Portunus pelagicus"],
              "Charybdis": ["Charybdis feriata"]}
    i = 0
    for genus, species in genera.items():
        for sp in species:
            for k in range(40):
                obs = f"o{k}"
                country = "PH" if k < 3 else ("TW", "SG", "TH")[k % 3]
                for p in range(2):
                    rows.append({"image_path": f"img{i}.jpg", "crab_id": f"observer:{obs}", "observer_id": obs,
                                 "genus": genus, "species": sp, "country": country})
                    i += 1
    # observer o3 shoots only outside PH under an observation-level crab_id, but also once in PH
    rows.append({"image_path": "x1.jpg", "crab_id": "observation:9001", "observer_id": "o3", "genus": "Scylla",
                 "species": "Scylla serrata", "country": "PH"})
    df = pd.DataFrame(rows)
    df["sex"] = df["maturity"] = df["disease"] = ""
    return df


class ExternalSplitTests(unittest.TestCase):
    def setUp(self):
        self.df = make_rows()

    def test_no_observer_on_both_sides(self):
        internal, external, info = split_external(self.df, EXT)
        self.assertTrue((external["country"] == "PH").all())
        self.assertFalse((internal["country"] == "PH").any())
        for col in ("crab_id", "observer_id"):
            self.assertFalse(set(internal[col]) & set(external[col]), col)
        # o0-o2 (PH) and o3 (one PH photo under another crab_id) are gone from the training side
        self.assertFalse({"o0", "o1", "o2", "o3"} & set(internal["observer_id"]))
        self.assertEqual(len(internal) + len(external) + info["removed_overlap"]["images"], len(self.df))

    def test_counts_are_reported(self):
        _, external, info = split_external(self.df, EXT)
        self.assertEqual(info["external"]["images"], len(external))
        self.assertEqual(info["external"]["groups"], external["crab_id"].nunique())
        removed = info["removed_overlap"]
        self.assertEqual(removed["groups"], 1)  # observer:o3 (its PH photo has another crab_id)
        self.assertEqual(removed["images"], 4 * 2)  # o3's non-PH photos: 4 species x 2 photos
        self.assertEqual(set(info["external"]["by_class"]), {"Scylla", "Portunus", "Charybdis"})
        self.assertEqual(info["removed_overlap"]["by_class"]["Scylla"]["images"], 4)

    def test_value_list_and_missing_column(self):
        internal, external, _ = split_external(self.df, {**EXT, "value": ["PH", "TW"]})
        self.assertEqual(set(external["country"]), {"PH", "TW"})
        with self.assertRaises(ValueError):
            split_external(self.df.drop(columns="country"), EXT)

    def test_cv_splits_never_contain_external_rows_or_observers(self):
        tmp = Path(tempfile.mkdtemp())
        try:
            logging.disable(logging.WARNING)
            cfg = load_config(ROOT / "config.yaml")
            cfg["data"]["labels_csv"] = str(tmp / "labels.csv")
            cfg["data"]["external_test"] = EXT
            cfg["data"]["min_groups_per_class"] = 5
            internal, external = hold_out_external(cfg, self.df, tmp)
            self.assertTrue((tmp / "external_split.json").exists())
            splits = select_attributes(cfg, internal, ["genus", "species"], tmp)
            ext_groups = set(external["crab_id"])
            ext_observers = set(external["observer_id"])
            crab_observer = dict(zip(self.df["crab_id"], self.df["observer_id"]))
            for attr, (rows, split) in splits.items():
                for fold in split["folds"]:
                    for name in SPLIT_SETS:
                        ids = set(fold[name])
                        self.assertFalse(ids & ext_groups, f"{attr} {name}")
                        self.assertFalse({crab_observer[c] for c in ids} & ext_observers, f"{attr} {name}")
        finally:
            logging.disable(logging.NOTSET)
            shutil.rmtree(tmp, ignore_errors=True)


class TinyModel(torch.nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        self.fc = torch.nn.Linear(3 * 32 * 32, n_classes)

    def forward(self, x):
        return self.fc(x.flatten(1))


class ExternalEvaluationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        logging.disable(logging.WARNING)
        cls.tmp = Path(tempfile.mkdtemp())
        csv = make_dummy_dataset(cls.tmp / "d", n_species=3, crabs_per_species=4, photos_per_crab=2, size=40, seed=2)
        cls.cfg = load_config(ROOT / "config.yaml")
        cls.cfg["data"]["labels_csv"] = str(csv)
        cls.rows = load_labels(csv)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)
        logging.disable(logging.NOTSET)

    def test_external_rows_predictions_and_report(self):
        torch.manual_seed(0)
        classes = ["species_A", "species_B"]  # species_C was "not in training"
        info = {"attribute": "species", "classes": classes, "arch": "tiny", "model": "Tiny", "fold": 2, "seed": 1}
        out_dir = self.tmp / "results"
        rows = evaluate_external(TinyModel(2), MODEL_CFG, info, self.rows, self.cfg, out_dir, torch.device("cpu"), 4)
        self.assertEqual([r["class"] for r in rows], ["ALL"] + classes)
        self.assertEqual(rows[0]["n_images"], 16)  # 2 species x 4 crabs x 2 photos; species_C left out
        self.assertEqual(rows[0]["n_groups"], 8)
        self.assertEqual([r["n_groups"] for r in rows[1:]], [4, 4])
        self.assertTrue(0 <= rows[0]["macro_f1"] <= 1)
        pred = pd.read_csv(out_dir / "predictions_external" / "tiny_species_1_2.csv")
        self.assertEqual(len(pred), 16)
        self.assertEqual(set(pred["true_label"]), set(classes))

        merged = merge_rows(out_dir / "external_metrics.csv", rows, keys=RUN_KEYS + ["class"])
        merged = merge_rows(out_dir / "external_metrics.csv", rows, keys=RUN_KEYS + ["class"])  # rerun replaces
        self.assertEqual(len(merged), 3)
        (out_dir / "external_split.json").write_text(json.dumps({
            "name": "Philippines", "column": "country", "values": ["PH"], "group_columns": ["crab_id"],
            "external": {"images": 16, "groups": 8, "by_class": {"species_A": {"images": 8, "groups": 4}}},
            "removed_overlap": {"images": 2, "groups": 1, "by_class": {}},
            "internal": {"images": 0, "groups": 0, "by_class": {}}}), encoding="utf-8")
        html = "".join(external_section(self.cfg, out_dir))
        self.assertIn("External test: Philippines", html)
        self.assertIn("Groups are small for some genera", html)
        self.assertIn("4 ⚠", html)  # 4 groups < 10

    def test_no_external_rows_for_the_attribute(self):
        info = {"attribute": "species", "classes": ["zzz"], "arch": "tiny", "model": "Tiny", "fold": 0, "seed": 0}
        self.assertEqual(evaluate_external(TinyModel(1), MODEL_CFG, info, self.rows, self.cfg, self.tmp / "r2",
                                           torch.device("cpu")), [])


if __name__ == "__main__":
    unittest.main()
