"""Generate a small fake crab dataset for testing the pipeline.

Each fake crab gets one value per attribute and several photos. Images are
random noise with a coloured disc whose colour depends on species, so a model
can learn something, but they are not crab photos.

Options mimic public data: --coverage blanks an attribute for most crabs
(e.g. sex=0.2 keeps sex labels on 20% of crabs), --drop removes an attribute
column, and --sources adds a source column (--skew-source gives the first
species almost only the first source).

    python scripts/make_dummy_data.py --out data/dummy
    python scripts/make_dummy_data.py --out data/dummy_public --crabs-per-species 40 --photos-per-crab 2 \
        --coverage sex=0.2 maturity=0.1 --drop disease --sources inaturalist gbif manual --skew-source
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


def make_dummy_dataset(out_dir, n_species=3, crabs_per_species=10, photos_per_crab=4, size=128, seed=0,
                       coverage=None, drop=(), sources=None, skew_source=False):
    rng = np.random.default_rng(seed)
    out_dir = Path(out_dir)
    (out_dir / "images").mkdir(parents=True, exist_ok=True)
    coverage = coverage or {}

    species = [f"species_{chr(ord('A') + i)}" for i in range(n_species)]
    n_crabs = n_species * crabs_per_species
    # Balanced, shuffled values per attribute, so every class has several crabs
    others = {attr: rng.permutation(np.resize(values, n_crabs)) for attr, values in OTHER_CLASSES.items()}
    # Crabs that keep each attribute's label (the rest are left blank)
    # (only draws random numbers when asked, so default datasets stay identical)
    labeled = {attr: (set(rng.permutation(n_crabs)[:int(round(coverage[attr] * n_crabs))])
                      if coverage.get(attr, 1.0) < 1.0 else set(range(n_crabs)))
               for attr in OTHER_CLASSES}

    rows = []
    for c in range(n_crabs):
        sp = c // crabs_per_species
        crab_id = f"crab_{c:03d}"
        source = None
        if sources and skew_source and sp == 0:
            # 9 in 10 crabs of the first species come from the first source
            source = sources[0] if c % 10 else sources[1 % len(sources)]
        elif sources:
            source = sources[c % len(sources)]
        for p in range(photos_per_crab):
            img = rng.integers(0, 256, (size, size, 3), dtype=np.uint8)
            im = Image.fromarray(img)
            r = int(size * rng.uniform(0.2, 0.35))
            cx, cy = rng.integers(r, size - r, 2)
            ImageDraw.Draw(im).ellipse([cx - r, cy - r, cx + r, cy + r], fill=SPECIES_COLOURS[sp % len(SPECIES_COLOURS)])
            rel = f"images/{crab_id}_{p}.jpg"
            im.save(out_dir / rel, quality=90)
            row = {"image_path": rel, "crab_id": crab_id, "species": species[sp],
                   **{attr: (others[attr][c] if c in labeled[attr] else "") for attr in OTHER_CLASSES}}
            if source:
                row["source"] = source
            rows.append(row)

    fields = ["image_path", "crab_id", "species", "sex", "disease", "maturity"] + (["source"] if sources else [])
    fields = [f for f in fields if f not in drop]
    csv_path = out_dir / "labels.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
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
    ap.add_argument("--coverage", nargs="*", default=[], metavar="ATTR=FRACTION",
                    help="fraction of crabs that keep a label, e.g. sex=0.2")
    ap.add_argument("--drop", nargs="*", default=[], help="attribute columns to leave out, e.g. disease")
    ap.add_argument("--sources", nargs="*", default=None, help="add a source column with these values")
    ap.add_argument("--skew-source", action="store_true", help="first species comes almost only from the first source")
    a = ap.parse_args()
    coverage = {k: float(v) for k, v in (item.split("=") for item in a.coverage)}
    path = make_dummy_dataset(a.out, a.species, a.crabs_per_species, a.photos_per_crab, a.size, a.seed,
                              coverage, a.drop, a.sources, a.skew_source)
    n = a.species * a.crabs_per_species
    print(f"Wrote {n} crabs x {a.photos_per_crab} photos = {n * a.photos_per_crab} images; labels: {path}")


if __name__ == "__main__":
    main()
