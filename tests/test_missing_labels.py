"""Attributes with missing labels: filtering, usage counts, skipping, group-safe splits.

The labels.csv here looks like public data: species on every row, sex and
maturity on a few crabs only, disease empty.

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

from run import select_attributes
from scripts.make_dummy_data import make_dummy_dataset
from src.config import ConfigError, load_config
from src.data import SPLIT_SETS, attribute_usage, load_labels, make_split, prepare_attribute

ROOT = Path(__file__).resolve().parents[1]


class MissingLabelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        logging.disable(logging.WARNING)
        cls.tmp = Path(tempfile.mkdtemp())
        # 3 species x 40 crabs x 2 photos; sex on 20% of crabs, maturity on 10%, disease blank everywhere
        cls.csv = make_dummy_dataset(cls.tmp / "sparse", n_species=3, crabs_per_species=40, photos_per_crab=2,
                                     size=32, seed=3, coverage={"sex": 0.2, "maturity": 0.1, "disease": 0.0})
        cls.df = load_labels(cls.csv)
        cfg = load_config(ROOT / "config.yaml")
        cfg["data"]["labels_csv"] = str(cls.csv)
        cls.cfg = cfg

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)
        logging.disable(logging.NOTSET)

    def test_dataset_is_sparse_as_intended(self):
        crabs = self.df.groupby("crab_id").first()
        self.assertEqual(len(crabs), 120)
        self.assertEqual((crabs["sex"] != "").sum(), 24)
        self.assertEqual((crabs["maturity"] != "").sum(), 12)
        self.assertTrue((self.df["disease"] == "").all())

    def test_usage_counts_only_labeled_rows(self):
        rows, usage = prepare_attribute(self.df, "sex", 30)
        self.assertTrue((rows["sex"] != "").all())
        self.assertEqual(usage["n_images"], 48)
        self.assertEqual(usage["n_groups"], 24)
        self.assertEqual(usage["n_classes"], 2)
        self.assertEqual(sum(usage["images_per_class"].values()), 48)
        self.assertEqual(set(usage["groups_per_class"]), {"female", "male"})

    def test_skip_rules(self):
        _, species = prepare_attribute(self.df, "species", 30)
        _, sex = prepare_attribute(self.df, "sex", 30)
        _, disease = prepare_attribute(self.df, "disease", 30)
        self.assertIsNone(species["skip_reason"])  # 40 groups per species
        self.assertIn("min_groups_per_class=30", sex["skip_reason"])  # about 12 per class
        self.assertIn("0 labeled class", disease["skip_reason"])
        _, sex_low = prepare_attribute(self.df, "sex", 5)
        self.assertIsNone(sex_low["skip_reason"])  # configurable threshold

    def test_single_class_is_skipped(self):
        one = self.df.assign(sex=self.df["sex"].replace("male", "female"))
        usage = attribute_usage(one[one["sex"] != ""], "sex", 1)
        self.assertIn("1 labeled class", usage["skip_reason"])

    def test_missing_attribute_column_is_treated_as_blank(self):
        csv = self.tmp / "no_disease.csv"
        pd.read_csv(self.csv, dtype=str, keep_default_na=False).drop(columns=["disease", "maturity"]).to_csv(csv, index=False)
        df = load_labels(csv)
        self.assertTrue((df["disease"] == "").all() and (df["maturity"] == "").all())
        _, usage = prepare_attribute(df, "disease", 30)
        self.assertIsNotNone(usage["skip_reason"])

    def test_split_on_filtered_rows_is_group_safe(self):
        # some photos of a labeled crab have no label: only its labeled photos are used
        df = self.df.copy()
        sex_crabs = sorted(df.loc[df["sex"] != "", "crab_id"].unique())
        half = df.index[(df["crab_id"] == sex_crabs[0]) & df["image_path"].str.endswith("_1.jpg")]
        df.loc[half, "sex"] = ""
        rows, _ = prepare_attribute(df, "sex", 5)
        self.assertEqual(set(rows["crab_id"]), set(sex_crabs))
        for strategy in ("holdout", "kfold"):
            split = make_split(rows, "sex", strategy, [0.7, 0.15, 0.15], 5, 42)
            for fold in split["folds"]:
                owner = {}
                for name in SPLIT_SETS:
                    for cid in fold[name]:
                        self.assertNotIn(cid, owner, f"{cid} in {owner.get(cid)} and {name}")
                        owner[cid] = name
                self.assertEqual(set(owner), set(sex_crabs))  # unlabeled crabs never enter the split

    def test_run_selects_usable_attributes_and_records_skips(self):
        cfg = copy.deepcopy(self.cfg)
        cfg["split"]["dir"] = "splits_select"
        out = self.tmp / "results"
        splits = select_attributes(cfg, self.df, ["species", "sex", "disease", "maturity"], out)
        self.assertEqual(list(splits), ["species"])
        saved = json.loads((out / "attributes.json").read_text(encoding="utf-8"))
        self.assertEqual(set(saved), {"species", "sex", "disease", "maturity"})
        self.assertIsNone(saved["species"]["skip_reason"])
        self.assertIsNotNone(saved["maturity"]["skip_reason"])

    def test_no_usable_attribute_is_a_clear_error(self):
        cfg = copy.deepcopy(self.cfg)
        cfg["split"]["dir"] = "splits_none"
        with self.assertRaises(ConfigError):
            select_attributes(cfg, self.df, ["sex", "disease", "maturity"], self.tmp / "results_none")


if __name__ == "__main__":
    unittest.main()
