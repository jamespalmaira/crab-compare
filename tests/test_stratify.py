"""Genus label, species stratification: every fold keeps each genus's species mix.

    python -m unittest discover -s tests -t . -v
"""
import copy
import hashlib
import json
import logging
import shutil
import tempfile
import unittest
from collections import Counter
from pathlib import Path

import pandas as pd

from run import groups_per_photo_class, summarise_split
from src.config import load_config
from src.data import (GENUS_ONLY, SPLIT_SETS, SplitError, attribute_rows, crab_finer, crab_strata, get_split, load_labels,
                      make_split, split_external)

ROOT = Path(__file__).resolve().parents[1]
REAL_LABELS = ROOT / "data" / "labels.csv"
SAVED_SPLIT = ROOT / "data" / "splits" / "genus_kfold_seed42.json"
MIN = 10  # min_stratum_groups
GROUPS = {  # (genus, species or "" for genus only) -> crab_id groups
    ("Scylla", "Scylla serrata"): 40, ("Scylla", "Scylla olivacea"): 30, ("Scylla", "Scylla tranquebarica"): 12,
    ("Scylla", "Scylla paramamosain"): 4, ("Scylla", ""): 6,
    ("Portunus", "Portunus pelagicus"): 35, ("Portunus", "Portunus sanguinolentus"): 15, ("Portunus", ""): 10,
    ("Charybdis", "Charybdis feriata"): 31,
}


def make_rows():
    rows = []
    for (genus, species), n in GROUPS.items():
        for i in range(n):
            cid = f"observer:{genus}:{species or 'g'}:{i}"
            for p in range(2):  # two photos per group
                rows.append({"image_path": f"{cid}_{p}.jpg", "crab_id": cid, "genus": genus, "species": species})
    return pd.DataFrame(rows)


def enough(min_groups=MIN):
    """Species (and genus-only buckets) with at least min_groups groups."""
    return {(g, s or GENUS_ONLY) for (g, s), n in GROUPS.items() if n >= min_groups}


class StratifyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows = make_rows()
        cls.fine = crab_finer(cls.rows, "species")
        cls.genus = cls.rows.groupby("crab_id")["genus"].first()

    def composition(self, ids):
        return {(self.genus[c], self.fine[c]) for c in ids}

    def test_strata_fold_small_species_into_their_genus(self):
        labels, fine, strata = crab_strata(self.rows, "genus", "species", MIN)
        counts = strata.value_counts()
        self.assertEqual(counts["Scylla / Scylla serrata"], 40 + 4 + 6)  # paramamosain and genus-only folded in
        self.assertEqual(counts["Scylla / Scylla tranquebarica"], 12)
        self.assertEqual(counts[f"Portunus / {GENUS_ONLY}"], 10)
        self.assertTrue((counts >= MIN).all())
        for cid, s in strata.items():  # a stratum never mixes genera
            self.assertTrue(s.startswith(labels[cid] + " / "))

    def test_every_fold_contains_every_species_with_enough_groups(self):
        for strategy in ("holdout", "kfold"):
            for seed in range(5):
                with self.subTest(strategy=strategy, seed=seed):
                    split = make_split(self.rows, "genus", strategy, [0.7, 0.15, 0.15], 5, seed,
                                       stratify_by="species", min_stratum_groups=MIN)
                    self.assertEqual(split["classes"], ["Charybdis", "Portunus", "Scylla"])  # the label is genus
                    self.assertEqual(split["stratified_on"], ["genus/species"])
                    self.assertEqual(len(split["folds"]), 5 if strategy == "kfold" else 1)
                    for k, fold in enumerate(split["folds"]):
                        for name in SPLIT_SETS:
                            missing = enough() - self.composition(fold[name])
                            self.assertFalse(missing, f"fold {k} {name} lacks {missing}")

    def test_genus_label_with_species_strata_stays_group_safe(self):
        split = make_split(self.rows, "genus", "kfold", [0.7, 0.15, 0.15], 5, 0, "species", MIN)
        tests = [c for f in split["folds"] for c in f["test"]]
        self.assertEqual(sorted(tests), sorted(self.genus.index))  # each group tested exactly once
        for fold in split["folds"]:
            sets = [set(fold[n]) for n in SPLIT_SETS]
            self.assertFalse(sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2])

    def test_species_mix_is_kept_close_to_the_whole(self):
        split = make_split(self.rows, "genus", "holdout", [0.7, 0.15, 0.15], 5, 0, "species", MIN)
        train = Counter(self.fine[c] for c in split["folds"][0]["train"] if self.genus[c] == "Scylla")
        total = Counter(self.fine[c] for c in self.genus.index if self.genus[c] == "Scylla")
        for species in ("Scylla serrata", "Scylla olivacea", "Scylla tranquebarica"):
            self.assertAlmostEqual(train[species] / sum(train.values()), total[species] / sum(total.values()), delta=0.05)

    def test_without_stratify_by_the_split_is_on_the_label(self):
        split = make_split(self.rows, "genus", "holdout", [0.7, 0.15, 0.15], 5, 0)
        self.assertEqual(split["stratified_on"], ["genus"])
        self.assertIsNone(split["stratify_by"])
        self.assertIsNone(split["strata"])

    def test_missing_finer_column_is_an_error(self):
        with self.assertRaises(SplitError):
            make_split(self.rows.drop(columns="species"), "genus", "holdout", [0.7, 0.15, 0.15], 5, 0, "species")

    def test_saved_split_records_and_checks_stratification(self):
        tmp = Path(tempfile.mkdtemp())
        try:
            cfg = {"data": {"labels_csv": str(tmp / "labels.csv")},
                   "split": {"strategy": "kfold", "ratios": [0.7, 0.15, 0.15], "n_folds": 5, "seed": 1,
                             "dir": "splits", "stratify_by": {"genus": "species"}, "min_stratum_groups": MIN}}
            split = get_split(cfg, self.rows, "genus")
            saved = json.loads(next((tmp / "splits").glob("genus_*.json")).read_text(encoding="utf-8"))
            self.assertEqual((saved["stratify_by"], saved["min_stratum_groups"]), ("species", MIN))
            self.assertEqual(saved["strata"]["Scylla / Scylla tranquebarica"], 12)
            self.assertEqual(get_split(cfg, self.rows, "genus"), split)  # reused
            cfg["split"]["min_stratum_groups"] = 5
            with self.assertRaises(SplitError):  # a saved split is never re-stratified silently
                get_split(cfg, self.rows, "genus")
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_composition_is_printed_for_every_fold(self):
        split = make_split(self.rows, "genus", "kfold", [0.7, 0.15, 0.15], 5, 0, "species", MIN)
        with self.assertLogs("run", logging.INFO) as logs:
            summarise_split(split, self.rows)
        lines = [m for m in logs.output if "by species" in m]
        self.assertEqual(len(lines), 5 * 3)  # every fold x every genus
        scylla = [m for m in lines if "Scylla by species" in m]
        self.assertTrue(all("Scylla tranquebarica" in m and GENUS_ONLY in m for m in scylla))
        self.assertTrue(all(m.count("|") == 2 for m in lines))  # train | val | test


def mixed_rows():
    """Groups with two photos of one genus; every fifth also has one photo of a second genus."""
    rows = []
    for genus, n in (("Scylla", 30), ("Portunus", 40), ("Charybdis", 25)):
        for i in range(n):
            cid = f"observer:{genus}:{i}"
            for p in range(2):
                rows.append({"image_path": f"{cid}_{p}.jpg", "crab_id": cid, "genus": genus,
                             "species": f"{genus} {'a' if i % 2 else 'b'}"})
            if i % 5 == 0:
                other = "Charybdis" if genus != "Charybdis" else "Portunus"
                rows.append({"image_path": f"{cid}_x.jpg", "crab_id": cid, "genus": other, "species": f"{other} a"})
    return pd.DataFrame(rows)


class FoldSummaryTests(unittest.TestCase):
    """The summary counts groups by each photo's own genus. It is reporting only: splits do not change."""
    # made by make_split on mixed_rows() before the summary was changed; the folds must never differ from this
    FOLDS_BEFORE_CHANGE = "1adbc3143f65c72c1a36ced55844b6d143e3ac7aa7fa06f76767d825537a5814"

    def setUp(self):
        self.rows = mixed_rows()
        self.split = make_split(self.rows, "genus", "kfold", [0.7, 0.15, 0.15], 5, 42, "species", MIN)

    def test_a_group_counts_under_each_genus_it_photographed(self):
        rows = pd.DataFrame({"crab_id": ["a", "a", "a", "b"], "genus": ["P", "P", "C", "P"]})
        self.assertEqual(groups_per_photo_class(rows, ["a", "b"], "genus", ["C", "P", "S"]), {"C": 1, "P": 2, "S": 0})

    def test_logged_counts_match_each_photos_own_genus(self):
        with self.assertLogs("run", logging.INFO) as logs:
            summarise_split(self.split, self.rows)
        lines = [m for m in logs.output if "  fold " in m]
        for fold, line in zip(self.split["folds"], lines):
            for name in SPLIT_SETS:
                part = self.rows[self.rows["crab_id"].isin(fold[name])]
                expect = {k: int(v) for k, v in part.groupby("genus")["crab_id"].nunique().sort_index().items()}
                self.assertIn(f"{name} {len(fold[name])} groups/{len(part)} imgs {expect}", line)
        # the fixture's mixed groups must make this differ from counting each group once
        ids = self.split["folds"][0]["test"]
        once = self.rows.groupby("crab_id")["genus"].first()[ids].value_counts().to_dict()
        self.assertNotEqual(groups_per_photo_class(self.rows, ids, "genus", sorted(once)), dict(sorted(once.items())))

    def test_species_lines_count_by_each_photos_own_genus(self):
        with self.assertLogs("run", logging.INFO) as logs:
            summarise_split(self.split, self.rows)
        line = next(m for m in logs.output if "    Charybdis by species" in m)
        test = self.rows[self.rows["crab_id"].isin(self.split["folds"][0]["test"]) & (self.rows["genus"] == "Charybdis")]
        n = test[test["species"] == "Charybdis a"]["crab_id"].nunique()
        self.assertIn(f"test Charybdis a {n}", line)  # includes groups whose most common genus is another

    def test_splits_are_identical_before_and_after(self):
        digest = hashlib.sha256(json.dumps(self.split["folds"], sort_keys=True).encode()).hexdigest()
        self.assertEqual(digest, self.FOLDS_BEFORE_CHANGE)

    def test_summary_changes_neither_the_split_nor_the_rows(self):
        split, rows = copy.deepcopy(self.split), self.rows.copy()
        summarise_split(self.split, self.rows)
        self.assertEqual(self.split, split)
        pd.testing.assert_frame_equal(self.rows, rows)

    @unittest.skipUnless(SAVED_SPLIT.exists() and REAL_LABELS.exists(), "real data is outside git")
    def test_real_split_is_reproduced_and_the_saved_file_stays(self):
        cfg = load_config(ROOT / "config.yaml")
        internal, _, _ = split_external(load_labels(REAL_LABELS), cfg["data"]["external_test"])
        s = cfg["split"]
        before = SAVED_SPLIT.read_bytes()
        fresh = make_split(attribute_rows(internal, "genus"), "genus", s["strategy"], s["ratios"], s["n_folds"], s["seed"],
                           s["stratify_by"]["genus"], s["min_stratum_groups"])
        self.assertEqual(fresh["folds"], json.loads(before)["folds"])
        summarise_split(fresh, attribute_rows(internal, "genus"))
        self.assertEqual(SAVED_SPLIT.read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
