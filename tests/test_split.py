"""Crab-level split tests on a generated dummy dataset.

    python -m unittest discover -s tests -t . -v
"""
import copy
import json
import logging
import shutil
import tempfile
import unittest
from pathlib import Path

import torch

from scripts.make_dummy_data import make_dummy_dataset
from src.config import load_config
from src.data import (ATTRIBUTES, SPLIT_SETS, SplitError, attribute_rows, build_datasets, check_split,
                      get_split, load_labels, make_split, split_path)

ROOT = Path(__file__).resolve().parents[1]
MODEL_CFG = {"input_size": (3, 64, 64), "mean": (0.5, 0.5, 0.5), "std": (0.5, 0.5, 0.5),
             "crop_pct": 0.875, "interpolation": "bicubic"}


class SplitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        logging.disable(logging.WARNING)
        cls.tmp = Path(tempfile.mkdtemp())
        cls.csv = make_dummy_dataset(cls.tmp / "dummy", n_species=3, crabs_per_species=10,
                                     photos_per_crab=4, size=72, seed=0)
        cls.df = load_labels(cls.csv)
        cls.base_cfg = load_config(ROOT / "config.yaml")
        cls.base_cfg["data"]["labels_csv"] = str(cls.csv)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)
        logging.disable(logging.NOTSET)

    def cfg(self, strategy):
        cfg = copy.deepcopy(self.base_cfg)
        cfg["split"]["strategy"] = strategy
        cfg["split"]["dir"] = f"splits_{self._testMethodName}"
        return cfg

    def all_splits(self):
        s = self.base_cfg["split"]
        for strategy in ("holdout", "kfold"):
            for attr in ATTRIBUTES:
                rows = attribute_rows(self.df, attr)
                yield strategy, attr, rows, make_split(rows, attr, strategy, s["ratios"], s["n_folds"], s["seed"])

    def test_no_crab_in_more_than_one_set(self):
        for strategy, attr, rows, split in self.all_splits():
            with self.subTest(strategy=strategy, attribute=attr):
                expected_folds = self.base_cfg["split"]["n_folds"] if strategy == "kfold" else 1
                self.assertEqual(len(split["folds"]), expected_folds)
                for fold in split["folds"]:
                    sets = [set(fold[name]) for name in SPLIT_SETS]
                    self.assertFalse(sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2])
                    self.assertEqual(sets[0] | sets[1] | sets[2], set(rows["crab_id"]))
                    # image level: every photo of a crab sits in that crab's set only
                    owner = {cid: name for name in SPLIT_SETS for cid in fold[name]}
                    image_sets = rows.groupby("crab_id")["image_path"].apply(set)
                    for cid, paths in image_sets.items():
                        others = [n for n in SPLIT_SETS if n != owner[cid]]
                        for n in others:
                            self.assertFalse(paths & set(rows[rows.crab_id.isin(fold[n])].image_path))

    def test_kfold_test_sets_partition_crabs(self):
        for strategy, attr, rows, split in self.all_splits():
            if strategy != "kfold":
                continue
            with self.subTest(attribute=attr):
                tests = [c for fold in split["folds"] for c in fold["test"]]
                self.assertEqual(sorted(tests), sorted(set(rows["crab_id"])))

    def test_holdout_ratios_and_stratification(self):
        rows = attribute_rows(self.df, "species")
        split = make_split(rows, "species", "holdout", [0.7, 0.15, 0.15], 5, 42)
        fold = split["folds"][0]
        self.assertTrue(split["stratified"])
        # 30 crabs: test = ceil(30*0.15) = 5, val = ceil(25*0.15/0.85) = 5 (sklearn rounds held-out up)
        self.assertEqual([len(fold[n]) for n in SPLIT_SETS], [20, 5, 5])
        species = rows.groupby("crab_id")["species"].first()
        for name in SPLIT_SETS:  # each set has every species
            self.assertEqual(set(species[fold[name]]), set(split["classes"]), name)

    def test_same_seed_same_split_and_different_seed_differs(self):
        rows = attribute_rows(self.df, "sex")
        a = make_split(rows, "sex", "holdout", [0.7, 0.15, 0.15], 5, 42)
        b = make_split(rows.sample(frac=1, random_state=1), "sex", "holdout", [0.7, 0.15, 0.15], 5, 42)
        c = make_split(rows, "sex", "holdout", [0.7, 0.15, 0.15], 5, 7)
        self.assertEqual(a, b)  # row order does not matter
        self.assertNotEqual(a["folds"], c["folds"])

    def test_split_saved_and_reused(self):
        for strategy in ("holdout", "kfold"):
            with self.subTest(strategy=strategy):
                cfg = self.cfg(strategy)
                rows = attribute_rows(self.df, "species")
                first = get_split(cfg, rows, "species")
                path = split_path(cfg, "species")
                self.assertTrue(path.exists())
                # tamper-proof reuse: the file on disk is what gets loaded
                saved = json.loads(path.read_text(encoding="utf-8"))
                self.assertEqual(saved, first)
                self.assertEqual(get_split(cfg, rows, "species"), first)

    def test_saved_split_mismatch_is_an_error(self):
        cfg = self.cfg("holdout")
        rows = attribute_rows(self.df, "species")
        get_split(cfg, rows, "species")
        with self.assertRaises(SplitError):  # a crab removed from labels.csv
            get_split(cfg, rows[rows.crab_id != "crab_000"], "species")
        cfg["split"]["ratios"] = [0.6, 0.2, 0.2]
        with self.assertRaises(SplitError):  # config changed under a saved split
            get_split(cfg, rows, "species")

    def test_check_split_catches_leak(self):
        rows = attribute_rows(self.df, "species")
        split = make_split(rows, "species", "holdout", [0.7, 0.15, 0.15], 5, 42)
        split["folds"][0]["test"].append(split["folds"][0]["train"][0])
        with self.assertRaises(SplitError):
            check_split(split)

    def test_blank_labels_skipped(self):
        df = self.df.copy()
        df.loc[df.crab_id == "crab_001", "maturity"] = ""
        rows = attribute_rows(df, "maturity")
        self.assertNotIn("crab_001", set(rows.crab_id))
        split = make_split(rows, "maturity", "kfold", [0.7, 0.15, 0.15], 5, 42)
        self.assertNotIn("crab_001", {c for f in split["folds"] for n in SPLIT_SETS for c in f[n]})

    def test_datasets_follow_split(self):
        cfg = self.cfg("holdout")
        rows = attribute_rows(self.df, "species")
        split = get_split(cfg, rows, "species")
        sets = build_datasets(cfg, rows, split, 0, MODEL_CFG, seed=0)
        for name in SPLIT_SETS:
            self.assertEqual(set(sets[name].crab_ids), set(split["folds"][0][name]))
        x, y = sets["train"][0]
        self.assertEqual(tuple(x.shape), (3, 64, 64))
        self.assertEqual(x.dtype, torch.float32)
        self.assertIn(y, range(len(split["classes"])))
        x_val, _ = sets["val"][0]
        self.assertTrue(torch.equal(x_val, sets["val"][0][0]))  # eval transform is deterministic


if __name__ == "__main__":
    unittest.main()
