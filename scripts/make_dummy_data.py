"""Generate a small fake crab dataset for testing the pipeline.

Each fake crab gets one value per attribute and several photos. Images are
random noise with a coloured disc whose colour depends on species, so a model
can learn something, but they are not crab photos.

    python scripts/make_dummy_data.py --out data/dummy
"""
import argparse
import csv
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

SPECIES_COLOURS = [(200, 60, 40), (40, 160, 60), (50, 80, 200), (200, 180, 40), (150, 50, 170)]
OTHER_CLASSES = {
    "sex": ["female", "male"],
    "disease": ["diseased", "healthy"],
    "maturity": ["immature", "mature"],
}


def make_dummy_dataset(out_dir, n_species=3, crabs_per_species=10, photos_per_crab=4, size=128, seed=0):
    rng = np.random.default_rng(seed)
    out_dir = Path(out_dir)
    (out_dir / "images").mkdir(parents=True, exist_ok=True)

    species = [f"species_{chr(ord('A') + i)}" for i in range(n_species)]
    n_crabs = n_species * crabs_per_species
    # Balanced, shuffled values per attribute, so every class has several crabs
    others = {attr: rng.permutation(np.resize(values, n_crabs)) for attr, values in OTHER_CLASSES.items()}

    rows = []
    for c in range(n_crabs):
        sp = c // crabs_per_species
        crab_id = f"crab_{c:03d}"
        for p in range(photos_per_crab):
            img = rng.integers(0, 256, (size, size, 3), dtype=np.uint8)
            im = Image.fromarray(img)
            r = int(size * rng.uniform(0.2, 0.35))
            cx, cy = rng.integers(r, size - r, 2)
            ImageDraw.Draw(im).ellipse([cx - r, cy - r, cx + r, cy + r], fill=SPECIES_COLOURS[sp % len(SPECIES_COLOURS)])
            rel = f"images/{crab_id}_{p}.jpg"
            im.save(out_dir / rel, quality=90)
            rows.append({"image_path": rel, "crab_id": crab_id, "species": species[sp],
                         **{attr: others[attr][c] for attr in OTHER_CLASSES}})

    csv_path = out_dir / "labels.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["image_path", "crab_id", "species", "sex", "disease", "maturity"])
        w.writeheader()
        w.writerows(rows)
    return csv_path


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="data/dummy")
    ap.add_argument("--species", type=int, default=3)
    ap.add_argument("--crabs-per-species", type=int, default=10)
    ap.add_argument("--photos-per-crab", type=int, default=4)
    ap.add_argument("--size", type=int, default=128)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    path = make_dummy_dataset(a.out, a.species, a.crabs_per_species, a.photos_per_crab, a.size, a.seed)
    n = a.species * a.crabs_per_species
    print(f"Wrote {n} crabs x {a.photos_per_crab} photos = {n * a.photos_per_crab} images; labels: {path}")


if __name__ == "__main__":
    main()
