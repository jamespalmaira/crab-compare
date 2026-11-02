"""The final-model step on tiny dummy data: fixed epochs, external test once, export, confidence table.

    python -m unittest discover -s tests -t . -v
"""
import json
import logging
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np
import onnxruntime as ort
import pandas as pd
import yaml

from scripts.final_model import main
from scripts.make_dummy_data import make_dummy_dataset
from src.final import (build_meta, check_held_out, confidence_table, cutoff_table, external_table, median_best_epoch,
                       select_golden_image, sha256_file)

ROOT = Path(__file__).resolve().parents[1]
ARCH = "mobilenetv3_large_100"


class FinalHelperTests(unittest.TestCase):
    def test_median_best_epoch_uses_only_this_model_and_attribute(self):
        df = pd.DataFrame({"arch": ["a"] * 3 + ["b"] + ["a"], "attribute": ["genus"] * 4 + ["species"],
                           "best_epoch": [4, 19, 37, 1, 2]})
        self.assertEqual(median_best_epoch(df, "a", "genus"), (19, [4, 19, 37]))
        with self.assertRaises(ValueError):
            median_best_epoch(df, "c", "genus")

    def test_check_held_out(self):
        rows = pd.DataFrame({"image_path": ["a"], "crab_id": ["x"], "observer_id": ["o"]})
        check_held_out(rows, pd.DataFrame({"image_path": ["b"], "crab_id": ["y"], "observer_id": ["p"]}))
        with self.assertRaises(ValueError):
            check_held_out(rows, pd.DataFrame({"image_path": ["b"], "crab_id": ["y"], "observer_id": ["o"]}))

    def test_external_table_marks_fewer_than_10_groups(self):
        t = external_table([{"class": "ALL", "n_images": 3, "n_groups": 11},
                            {"class": "A", "n_images": 2, "n_groups": 10, "recall": 0.5},
                            {"class": "B", "n_images": 1, "n_groups": 9, "recall": 1.0}]).set_index("genus")
        self.assertTrue(t.loc["A", "measurable"])
        self.assertFalse(t.loc["B", "measurable"])
        self.assertIn("not measurable", t.loc["B", "note"])

    def test_confidence_table_bins_and_scopes(self):
        p = pd.DataFrame({"image_path": ["i1", "i1", "i2", "i3"], "true_label": ["A", "A", "B", "B"],
                          "pred_label": ["A", "B", "B", "B"], "prob_A": [0.95, 0.45, 0.0, 0.3],
                          "prob_B": [0.05, 0.55, 1.0, 0.7]})
        t = confidence_table(p, ["A", "B"], edges=(0.25, 0.5, 0.9, 1.0))
        o = t[t["scope"] == "overall"].reset_index(drop=True)
        self.assertEqual(list(o["n_images"]), [0, 2, 2])  # 0.55, 0.7 | 0.95, 1.0 (1.0 is in the last bin)
        self.assertEqual(list(o["n_correct"]), [0, 1, 2])
        self.assertEqual(o.loc[1, "accuracy"], 0.5)
        self.assertTrue(np.isnan(o.loc[0, "accuracy"]))
        self.assertEqual(list(o["n_at_or_above"]), [4, 4, 2])
        self.assertEqual(o.loc[2, "share_at_or_above"], 0.5)
        self.assertEqual(o.loc[1, "accuracy_at_or_above"], 0.75)
        true_a = t[(t["scope"] == "true_genus") & (t["genus"] == "A")]
        pred_a = t[(t["scope"] == "predicted_genus") & (t["genus"] == "A")]
        self.assertEqual(true_a["n_images"].sum(), 2)
        self.assertEqual(pred_a["n_images"].sum(), 1)
        self.assertEqual(t["n_images"].sum(), 4 + 4 + 4)  # overall, by true genus, by predicted genus

    def test_cutoff_table_rates_include_the_cutoff(self):
        p = pd.DataFrame({"true_label": ["A", "A", "A", "B", "B"], "prob_A": [0.05, 0.3, 0.04, 0.05, 0.0]})
        t = cutoff_table(p, "A", cutoffs=(0.05, 0.5)).set_index("cutoff")
        self.assertEqual((t.loc[0.05, "n_A_at_or_above"], t.loc[0.05, "n_A"]), (2, 3))
        self.assertAlmostEqual(t.loc[0.05, "detection_rate"], 2 / 3)
        self.assertEqual((t.loc[0.05, "n_non_A_at_or_above"], t.loc[0.05, "n_non_A"]), (1, 2))
        self.assertEqual(t.loc[0.05, "false_alarm_rate"], 0.5)
        self.assertEqual((t.loc[0.5, "detection_rate"], t.loc[0.5, "false_alarm_rate"]), (0.0, 0.0))

    def test_golden_image_prefers_the_most_open_licence_then_path(self):
        ext = pd.DataFrame({"image_path": ["a", "c", "b", "d"], "license": ["CC-BY-NC", "CC0", "CC-BY", "CC0"]})
        self.assertEqual(select_golden_image(ext)["image_path"], "c")
        with self.assertRaises(ValueError):
            select_golden_image(ext[ext["license"] == "CC-BY-NC"])

    def test_meta_writes_the_resize_note_and_keeps_every_other_field(self):
        info = {"model": "M", "arch": "a", "attribute": "genus", "classes": ["A", "B"], "n_train": 1, "n_groups": 1,
                "epochs": 1, "seed": 0}
        data_cfg = {"input_size": (3, 224, 224), "mean": (0, 0, 0), "std": (1, 1, 1), "crop_pct": 0.875,
                    "interpolation": "bicubic"}
        meta = build_meta(info, data_cfg, "sha", "2026-11-01", "run", {"expected": "x"})
        self.assertEqual(meta["resize_note"], "Training and evaluation used cv2.INTER_CUBIC (bicubic, no antialiasing), "
                         "not PIL. The golden tensor was made the same way. crab-demo deliberately defaults to "
                         "PIL-style antialiasing for large phone photos.")
        self.assertEqual(meta["format_version"], 1)
        self.assertEqual(list(meta), ["format_version", "model_name", "timm_arch", "run_name", "date", "file", "sha256",
                                      "input", "heads", "note", "resize_note", "label_notes", "training", "golden"])

    @unittest.skipUnless((ROOT / "exports/densenet121_genus_v1/model_meta.json").exists(), "export is outside git")
    def test_resize_note_matches_the_export_and_is_the_only_addition(self):
        saved = json.loads((ROOT / "exports/densenet121_genus_v1/model_meta.json").read_text(encoding="utf-8"))
        info = {"model": saved["model_name"], "arch": saved["timm_arch"], "attribute": "genus",
                "classes": saved["heads"][0]["labels"], "n_train": saved["training"]["n_images"],
                "n_groups": saved["training"]["n_groups"], "epochs": saved["training"]["epochs"],
                "seed": saved["training"]["seed"]}
        data_cfg = {"input_size": (3, 224, 224), "mean": saved["input"]["mean"], "std": saved["input"]["std"],
                    "crop_pct": 0.875, "interpolation": "bicubic"}
        meta = build_meta(info, data_cfg, saved["sha256"], saved["date"], saved["run_name"], saved["golden"])
        self.assertEqual(json.loads(json.dumps(meta)), saved)


class FinalPipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        logging.disable(logging.WARNING)
        cls.tmp = Path(tempfile.mkdtemp())
        csv = make_dummy_dataset(cls.tmp / "d", n_species=2, crabs_per_species=20, photos_per_crab=2, size=48, seed=1)
        labels = pd.read_csv(csv)
        labels["genus"] = labels["species"]
        labels["source"] = "inaturalist"
        labels["observer_id"] = labels["crab_id"]
        crabs = sorted(labels["crab_id"].unique())
        labels["country"] = ["PH" if c in crabs[:2] or c in crabs[-2:] else "SG" for c in labels["crab_id"]]
        labels["license"] = "CC-BY"
        labels.to_csv(csv, index=False)
        cfg = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
        cfg["results_dir"] = str(cls.tmp / "results")
        cfg["data"].update(labels_csv=str(csv), min_groups_per_class=2, num_workers=0)
        cfg["training"]["pretrained"] = False
        cls.config = cls.tmp / "config.yaml"
        cls.config.write_text(yaml.safe_dump(cfg), encoding="utf-8")

        bench = cls.tmp / "results" / "bench"
        (bench / "predictions").mkdir(parents=True)
        pd.DataFrame({"arch": [ARCH] * 3, "attribute": ["genus"] * 3, "best_epoch": [1, 2, 9]}).to_csv(
            bench / "per_seed.csv", index=False)
        classes = sorted(labels["genus"].unique())
        pd.DataFrame({"image_path": ["i1", "i2"], "true_label": classes, "pred_label": classes[::-1],
                      f"prob_{classes[0]}": [0.2, 0.9], f"prob_{classes[1]}": [0.8, 0.1]}).to_csv(
            bench / "predictions" / f"{ARCH}_genus_0_0.csv", index=False)
        cls.bench = bench
        cls.run_dir = cls.tmp / "results" / "final"
        cls.export = cls.tmp / "export"
        cls.args = ["--config", str(cls.config), "--run-name", "final", "--benchmark", "bench", "--arch", ARCH,
                    "--export-dir", str(cls.export)]
        for cmd in ("train", "export", "confidence"):
            main([cmd] + cls.args)
        cls.labels = labels

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)
        logging.disable(logging.NOTSET)

    def test_trains_on_the_whole_training_side_for_the_median_epoch(self):
        run = json.loads((self.run_dir / "final_run.json").read_text(encoding="utf-8"))
        self.assertEqual(run["epochs"], 2)
        self.assertEqual(run["n_train"], int((self.labels["country"] != "PH").sum()))
        self.assertEqual(run["n_groups"], self.labels.loc[self.labels["country"] != "PH", "crab_id"].nunique())
        self.assertEqual(run["schedule_epochs"], 50)  # the benchmark's schedule, cut off at the fixed epoch
        self.assertEqual(run["optimizer_steps"], 2 * run["steps_per_epoch"])

    def test_external_test_is_run_once(self):
        with self.assertRaises(SystemExit):
            main(["train"] + self.args)
        ext = pd.read_csv(self.run_dir / "external_metrics.csv")
        self.assertEqual(ext.loc[ext["class"] == "ALL", "n_images"].iloc[0], int((self.labels["country"] == "PH").sum()))
        self.assertTrue((self.run_dir / "external_per_genus.csv").exists())

    def test_meta_has_one_head_and_the_preprocessing(self):
        meta = json.loads((self.export / "model_meta.json").read_text(encoding="utf-8"))
        self.assertEqual([h["name"] for h in meta["heads"]], ["genus"])
        self.assertEqual(meta["heads"][0]["labels"], sorted(self.labels["genus"].unique()))
        self.assertEqual(meta["heads"][0]["output_type"], "logits")
        self.assertEqual(meta["input"]["input_size"], [3, 224, 224])
        self.assertEqual(meta["input"]["preprocessing"]["resize_shorter_side"], 256)
        self.assertEqual(len(meta["input"]["mean"]), 3)
        self.assertIn("not expert-verified", meta["note"])
        self.assertEqual(meta["sha256"], sha256_file(self.export / "model.onnx"))

    def test_golden_vectors_reproduce_with_onnx_runtime(self):
        exp = json.loads((self.export / "golden" / "expected.json").read_text(encoding="utf-8"))
        x = np.load(self.export / "golden" / "input_tensor.npy")
        raw = np.fromfile(self.export / "golden" / "input_tensor_f32le.bin", dtype="<f4").reshape(x.shape)
        np.testing.assert_array_equal(x, raw)
        session = ort.InferenceSession(str(self.export / "model.onnx"), providers=["CPUExecutionProvider"])
        out = session.run(None, {"input": x})[0][0]
        np.testing.assert_allclose(out, exp["logits"], rtol=exp["tolerance"]["rtol"], atol=exp["tolerance"]["atol"])
        self.assertTrue((self.export / exp["image"]).exists())
        self.assertAlmostEqual(sum(exp["probabilities"].values()), 1.0, places=5)

    def test_every_file_has_a_correct_sha256(self):
        lines = (self.export / "SHA256SUMS.txt").read_text(encoding="utf-8").splitlines()
        listed = {line.split("  ", 1)[1]: line.split("  ", 1)[0] for line in lines}
        files = {p.relative_to(self.export).as_posix() for p in self.export.rglob("*") if p.is_file()} - {"SHA256SUMS.txt"}
        self.assertEqual(set(listed), files)
        self.assertIn("confidence_table.csv", listed)
        for name, digest in listed.items():
            self.assertEqual(sha256_file(self.export / name), digest)

    def test_the_benchmark_is_only_read(self):
        self.assertEqual({p.name for p in self.bench.iterdir()}, {"per_seed.csv", "predictions"})


if __name__ == "__main__":
    unittest.main()
