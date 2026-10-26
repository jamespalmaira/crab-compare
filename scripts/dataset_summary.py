"""Final dataset totals per genus: every build stage, then the external split, plus disk space.

Reads the builder's stage_counts.csv for a run and data/labels.csv, applies the bench's own
external split (config.yaml data.external_test, the same code run.py uses), and prints photos
and observer groups per genus at each step.

    python scripts/dataset_summary.py --run download
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import load_config  # noqa: E402
from src.data import load_labels, split_external  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def counts(df, by="genus"):
    """{genus: 'photos/groups'} plus ALL."""
    out = {g: f"{len(x)}/{x['crab_id'].nunique()}" for g, x in df.groupby(by)}
    out["ALL"] = f"{len(df)}/{df['crab_id'].nunique()}"
    return out


def disk_usage(paths):
    files = [p for root in paths if root.exists() for p in root.rglob("*") if p.is_file()]
    return len(files), sum(p.stat().st_size for p in files)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, help="builder run name under data/external/runs/")
    ap.add_argument("--config", default=str(ROOT / "config.yaml"))
    args = ap.parse_args()

    cfg = load_config(args.config)
    data_dir = ROOT / "data"
    stages = pd.read_csv(data_dir / "external" / "runs" / args.run / "stage_counts.csv")
    stages["cell"] = stages["photos"].astype(str) + "/" + stages["groups_observer"].astype(str)
    order = list(dict.fromkeys(stages["stage"]))
    table = stages.pivot(index="genus", columns="stage", values="cell").reindex(columns=order)

    labels = load_labels(Path(cfg["data"]["labels_csv"]))
    ext = cfg["data"].get("external_test")
    if ext:
        internal, external, info = split_external(labels, {**ext, "report_by": "genus"})
        kept = set(internal["image_path"]) | set(external["image_path"])
        removed = labels[~labels["image_path"].isin(kept)]
        name = ext.get("name", ext["value"])
        for col, frame in ((f"external test ({name})", external), ("removed: observer also in external", removed),
                           ("training side (5-fold CV)", internal)):
            table[col] = pd.Series(counts(frame))
    table = table.fillna("0/0")
    table = table.reindex([g for g in table.index if g != "ALL"] + ["ALL"])

    print("Photos/observer groups per genus (groups: distinct crab_id = observer)\n")
    with pd.option_context("display.width", 250, "display.max_columns", 20):
        print(table.T.to_string())
    n_files, size = disk_usage([data_dir / "external" / s for s in ("inaturalist", "gbif")])
    n_review, size_review = disk_usage([data_dir / "external" / "review"])
    print(f"\nlabels.csv: {len(labels)} photos, {labels['crab_id'].nunique()} observer groups, "
          f"{labels['species'].replace('', pd.NA).nunique()} species + genus-only")
    print(f"images on disk: {n_files} files, {size / 1e6:.1f} MB (data/external/inaturalist, gbif); "
          f"contact sheets: {n_review} files, {size_review / 1e6:.1f} MB (data/external/review)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
