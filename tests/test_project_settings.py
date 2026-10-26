"""The project decisions recorded in config.yaml (see handoff.md). Changing one should be deliberate.

    python -m unittest discover -s tests -t . -v
"""
import unittest
from pathlib import Path

from src.config import load_config

ROOT = Path(__file__).resolve().parents[1]
MODELS = ["efficientnet_b3", "resnet50", "densenet121", "mobilenetv3_large_100", "vit_base_patch16_224"]


class ProjectSettingsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg = load_config(ROOT / "config.yaml")

    def test_genus_is_the_main_task_and_species_exploratory(self):
        d = self.cfg["data"]
        self.assertEqual(d["main_attribute"], "genus")
        self.assertEqual(d["attributes"], ["genus", "species"])  # sex, maturity, disease out of scope
        self.assertEqual(d["min_groups_per_class"], 30)

    def test_five_fold_by_group_stratified_on_species_within_genus(self):
        s = self.cfg["split"]
        self.assertEqual((s["strategy"], s["n_folds"]), ("kfold", 5))
        self.assertEqual(s["stratify_by"], {"genus": "species"})

    def test_training_settings_unchanged(self):
        self.assertFalse(self.cfg["training"]["amp"])
        self.assertEqual(list(self.cfg["models"]), MODELS)
