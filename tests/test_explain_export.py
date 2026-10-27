"""Grad-CAM, ONNX export and FLOPs on CPU with random-weight models.

Uses one CNN (MobileNetV3) and ViT, the two target-layer cases. No weights are
downloaded. Takes about half a minute.

    python -m unittest discover -s tests -t . -v
"""
import logging
import os
import shutil
import tempfile
import unittest
import warnings
from pathlib import Path

import pandas as pd
import torch

from scripts.make_dummy_data import make_dummy_dataset
from src.config import load_config
from src.export import check_onnx, export_onnx, flops, onnx_input_size, onnx_latency, time_onnx
from src.gradcam import TARGET_LAYERS, run_gradcam, safe_filename
from src.models import create_model, model_data_config

ROOT = Path(__file__).resolve().parents[1]
CPU = torch.device("cpu")


class ExplainExportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        warnings.filterwarnings("ignore")
        logging.disable(logging.WARNING)
        cls.tmp = Path(tempfile.mkdtemp())
        csv = make_dummy_dataset(cls.tmp / "dummy", n_species=2, crabs_per_species=2, photos_per_crab=1,
                                 size=64, seed=0)
        df = pd.read_csv(csv)
        cls.sample = pd.DataFrame({"image_path": df["image_path"][:2], "crab_id": df["crab_id"][:2],
                                   "true_label": df["species"][:2], "pred_label": df["species"][:2]})
        cls.image_root = csv.parent
        cls.aug = load_config(ROOT / "config.yaml")["augmentation"]

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)
        logging.disable(logging.NOTSET)

    def test_every_configured_model_has_a_target_layer(self):
        self.assertEqual(set(TARGET_LAYERS), set(load_config(ROOT / "config.yaml")["models"]))

    def test_gradcam_cnn_and_vit(self):
        for arch in ("mobilenetv3_large_100", "vit_base_patch16_224"):
            with self.subTest(arch=arch):
                torch.manual_seed(0)
                model = create_model(arch, 2, pretrained=False)
                out = self.tmp / "cam" / arch
                res = run_gradcam(model, arch, self.sample, self.image_root, model_data_config(model), self.aug,
                                  out, CPU, arch)
                self.assertTrue(res["gradcam_pass"], res["gradcam_error"])
                self.assertEqual(len(list(out.glob("0*.png"))), 2)
                self.assertTrue((out / "grid.png").exists())

    def test_safe_filename(self):
        self.assertEqual(safe_filename('00_observer:inaturalist:6834136_true-A_pred-B'),
                         "00_observer_inaturalist_6834136_true-A_pred-B")
        self.assertEqual(safe_filename('a<b>c"d/e\\f|g?h*i.'), "a_b_c_d_e_f_g_h_i")

    def test_gradcam_crab_id_with_colons(self):
        # real crab_ids are observer:<source>:<id>; colons are not allowed in Windows filenames
        model = create_model("mobilenetv3_large_100", 2, pretrained=False)
        sample = self.sample.assign(crab_id=["observer:inaturalist:6834136", "observer:gbif:1"])
        out = self.tmp / "cam_colons"
        res = run_gradcam(model, "mobilenetv3_large_100", sample, self.image_root, model_data_config(model),
                          self.aug, out, CPU, "colons")
        self.assertTrue(res["gradcam_pass"], res["gradcam_error"])
        names = sorted(p.name for p in out.glob("0*.png"))
        self.assertEqual(len(names), 2)
        self.assertTrue(names[0].startswith("00_observer_inaturalist_6834136_true-"), names)
        self.assertFalse(any(":" in n for n in names))

    def test_gradcam_failure_is_recorded_not_raised(self):
        model = create_model("mobilenetv3_large_100", 2, pretrained=False)
        bad = self.sample.assign(image_path="missing.jpg")
        res = run_gradcam(model, "mobilenetv3_large_100", bad, self.image_root, model_data_config(model),
                          self.aug, self.tmp / "bad", CPU, "bad")
        self.assertFalse(res["gradcam_pass"])
        self.assertIn("missing.jpg", res["gradcam_error"])

    def test_onnx_export_matches_pytorch(self):
        torch.manual_seed(0)
        model = create_model("mobilenetv3_large_100", 3, pretrained=False).eval()
        size = tuple(model_data_config(model)["input_size"])
        path = export_onnx(model, size, self.tmp / "onnx" / "m.onnx")
        res = check_onnx(model, path, size, rtol=1e-3, atol=1e-4)
        self.assertTrue(res["onnx_outputs_match"], res["onnx_max_abs_diff"])
        self.assertGreater(res["onnx_size_mb"], 1)
        self.assertEqual(onnx_input_size(path), size)

        t = time_onnx(path, size, threads=2, warmup=1, runs=5)
        self.assertLessEqual(t["q1_ms"], t["median_ms"])
        self.assertLessEqual(t["median_ms"], t["q3_ms"])
        self.assertAlmostEqual(t["iqr_ms"], t["q3_ms"] - t["q1_ms"])
        self.assertEqual((t["threads"], t["warmup"], t["runs"]), (2, 1, 5))

        # the protocol: one new process per model and repeat, median of the per-process medians
        rows, summary = onnx_latency({"m": (path, size)}, {"threads": 2, "warmup": 1, "runs": 3, "repeats": 3})
        self.assertEqual(list(rows["repeat"]), [1, 2, 3])
        self.assertEqual(rows["pid"].nunique(), 3)
        self.assertNotIn(os.getpid(), set(rows["pid"]))
        self.assertAlmostEqual(summary["m"]["onnx_cpu_ms"], float(rows["median_ms"].median()))
        self.assertEqual((summary["m"]["onnx_threads"], summary["m"]["onnx_repeats"]), (2, 3))

    def test_flops_do_not_modify_the_model(self):
        model = create_model("mobilenetv3_large_100", 3, pretrained=False)
        keys = set(model.state_dict())
        f = flops(model, (3, 224, 224))
        self.assertAlmostEqual(f["gflops"], 2 * f["gmacs"])
        self.assertGreater(f["gmacs"], 0.1)
        self.assertEqual(set(model.state_dict()), keys)  # thop counters stay on the copy


if __name__ == "__main__":
    unittest.main()
