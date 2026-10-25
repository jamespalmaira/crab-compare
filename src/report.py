"""Results tables: per_seed.csv, per_class.csv and summary.csv.

per_seed.csv keeps one row per (attribute, model, fold, seed) run. Re-running
a run replaces its row; other rows already in the file are kept, so
attributes or models can be run separately into the same results folder.
summary.csv is rebuilt from per_seed.csv: mean and standard deviation (n-1)
across all runs of a model and attribute (seeds, and folds for k-fold CV).
"""
from pathlib import Path

import pandas as pd

RUN_KEYS = ["attribute", "arch", "fold", "seed"]
SUMMARY_METRICS = [
    "accuracy", "macro_precision", "macro_recall", "macro_specificity", "macro_f1", "roc_auc_ovr",
    "best_val_macro_f1", "latency_gpu_ms", "latency_cpu_ms", "params_m", "model_size_mb",
]


def merge_rows(path, new_rows, keys=RUN_KEYS):
    """Write new rows into a CSV, replacing rows with the same keys."""
    path = Path(path)
    new = pd.DataFrame(new_rows) if not isinstance(new_rows, pd.DataFrame) else new_rows
    if path.exists():
        old = pd.read_csv(path)
        replaced = old.set_index(keys).index.isin(new.set_index(keys).index)
        new = pd.concat([old[~replaced], new], ignore_index=True)
    new = new.sort_values(keys, kind="stable").reset_index(drop=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    new.to_csv(path, index=False)
    return new


def summarise(per_seed, model_order=None):
    """Mean and std per (attribute, model) across seeds and folds."""
    groups = per_seed.groupby(["attribute", "arch", "model"], sort=False)
    rows = []
    for (attribute, arch, model), g in groups:
        row = {"attribute": attribute, "model": model, "arch": arch, "n_runs": len(g),
               "folds": g["fold"].nunique(), "seeds": g["seed"].nunique()}
        for m in SUMMARY_METRICS:
            row[f"{m}_mean"] = g[m].mean()
            row[f"{m}_std"] = g[m].std(ddof=1)  # NaN for a single run
        rows.append(row)
    summary = pd.DataFrame(rows)
    if model_order:  # list models in config order
        summary["_order"] = summary["arch"].map({a: i for i, a in enumerate(model_order)})
        summary = summary.sort_values(["attribute", "_order"], kind="stable").drop(columns="_order")
    return summary.reset_index(drop=True)


def write_tables(out_dir, run_rows, class_rows, model_order=None):
    out_dir = Path(out_dir)
    per_seed = merge_rows(out_dir / "per_seed.csv", run_rows)
    merge_rows(out_dir / "per_class.csv", pd.concat(class_rows, ignore_index=True), keys=RUN_KEYS + ["class"])
    summary = summarise(per_seed, model_order)
    summary.to_csv(out_dir / "summary.csv", index=False)
    return per_seed, summary
