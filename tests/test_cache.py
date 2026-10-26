"""The optional pre-resized image cache (data.cache_longer_side).

    python -m unittest discover -s tests -t . -v
"""
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from src.data import CrabDataset, build_image_cache, build_transforms, image_root

MODEL_CFG = {"input_size": (3, 64, 64), "mean": (0.5, 0.5, 0.5), "std": (0.5, 0.5, 0.5),
             "crop_pct": 0.875, "interpolation": "bicubic"}
AUG = {"hflip_p": 0.5, "rotate_limit": 15, "rotate_p": 0.5,
       "color_jitter": {"brightness": 0.2, "contrast": 0.2, "saturation": 0.2, "hue": 0.02, "p": 0.8},
       "resized_crop_scale": [0.7, 1.0]}
SIZES = {"a/big.jpg": (1200, 800), "a/small.jpg": (300, 200), "b/tall.png": (400, 900), "b/edge.jpg": (512, 384)}


class ImageCacheTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        rng = np.random.default_rng(0)
        for rel, (w, h) in SIZES.items():
            p = self.tmp / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(rng.integers(0, 255, (h, w, 3), dtype=np.uint8)).save(p)
        self.rows = pd.DataFrame({"image_path": list(SIZES), "crab_id": ["c1", "c1", "c2", "c3"],
                                  "genus": ["Scylla", "Portunus", "Scylla", "Portunus"]})
        self.rows.to_csv(self.tmp / "labels.csv", index=False)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def cfg(self, side):
        return {"data": {"labels_csv": str(self.tmp / "labels.csv"), "cache_longer_side": side}}

    def test_off_by_default_reads_the_originals(self):
        self.assertEqual(image_root(self.cfg(None)), self.tmp)
        self.assertFalse((self.tmp / "cache").exists())

    def test_cache_resizes_only_large_images_and_is_built_once(self):
        root = image_root(self.cfg(512))
        self.assertEqual(root, self.tmp / "cache" / "longer_512")
        with Image.open(root / "a/big.jpg") as im:
            self.assertEqual(im.size, (512, 341))
        with Image.open(root / "b/tall.png") as im:
            self.assertEqual(max(im.size), 512)
        for rel in ("a/small.jpg", "b/edge.jpg"):  # at or under the limit: copied unchanged
            self.assertEqual((root / rel).read_bytes(), (self.tmp / rel).read_bytes())
        self.assertEqual(build_image_cache(self.tmp, list(SIZES), root, 512), 0)  # already built

    def test_tensors_have_the_same_shapes_with_and_without_the_cache(self):
        for train in (True, False):
            tf = build_transforms(MODEL_CFG, AUG, train=train, seed=0)
            plain = CrabDataset(self.rows, image_root(self.cfg(None)), "genus", ["Portunus", "Scylla"], tf)
            cached = CrabDataset(self.rows, image_root(self.cfg(512)), "genus", ["Portunus", "Scylla"], tf)
            for i in range(len(self.rows)):
                (x0, y0), (x1, y1) = plain[i], cached[i]
                self.assertEqual(tuple(x0.shape), (3, 64, 64))
                self.assertEqual(x0.shape, x1.shape)
                self.assertEqual(x0.dtype, x1.dtype)
                self.assertEqual(y0, y1)


if __name__ == "__main__":
    unittest.main()
