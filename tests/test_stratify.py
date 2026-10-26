"""Genus label, species stratification: every fold keeps each genus's species mix.

    python -m unittest discover -s tests -t . -v
"""
import json
import logging
import shutil
import tempfile
import unittest
from collections import Counter
from pathlib import Path

import pandas as pd

from run import summarise_split
from src.data import GENUS_ONLY, SPLIT_SETS, SplitError, crab_finer, crab_strata, get_split, make_split

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


if __name__ == "__main__":
    unittest.main()
