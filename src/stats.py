"""Statistical comparison of the models: Friedman and pairwise McNemar.

Friedman (per attribute) compares the five models on test macro-F1:
  - k-fold CV: blocks are folds; each fold's value is macro-F1 averaged over seeds.
  - single holdout: blocks are seeds. All seeds share one test set, so the
    blocks are not independent samples; the result is marked accordingly.

McNemar (per attribute, every pair of models) uses the saved test-set
predictions, pooled over all runs (folds and seeds) and paired by
(fold, seed, image). p-values are Holm-corrected across the pairs.
"""
import itertools
import json
import logging
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, rankdata
from statsmodels.stats.contingency_tables import mcnemar
from statsmodels.stats.multitest import multipletests

log = logging.getLogger(__name__)

HOLDOUT_CAUTION = "seeds share one test set; interpret with caution"
POOLED_CAUTION = ("pooled over folds and seeds; with several seeds the same test images appear more than once, "
                  "so pairs are not fully independent")


def _clean(x):
    """JSON-safe values: NaN/inf -> None, numpy scalars -> Python, applied recursively."""
    if isinstance(x, dict):
        return {str(k): _clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_clean(v) for v in x]
    if isinstance(x, np.generic):
        x = x.item()
    if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
        return None
    return x


def friedman(per_seed, attribute, model_order):
    df = per_seed[per_seed["attribute"] == attribute]
    strategy = df["split_strategy"].iloc[0]
    if strategy == "kfold":
        block, note = "fold", "blocks are folds (macro-F1 averaged over seeds within each fold)"
        table = df.groupby(["fold", "arch"])["macro_f1"].mean().unstack("arch")
    else:
        block, note = "seed", HOLDOUT_CAUTION
        table = df.pivot_table(index="seed", columns="arch", values="macro_f1")
    models = [m for m in model_order if m in table.columns]
    table = table[models].dropna()
    out = {"test": "friedman", "metric": "macro_f1", "split_strategy": strategy, "block": block,
           "n_blocks": int(len(table)), "n_models": len(models), "note": note,
           "statistic": None, "p_value": None,
           "mean_rank": {}, "block_values": table.round(6).to_dict(orient="index")}
    if len(models) < 3 or len(table) < 2:
        out["note"] += f"; not run: needs >= 3 models and >= 2 {block}s (have {len(models)}, {len(table)})"
        return out
    # rank 1 = best macro-F1 within each block
    ranks = np.vstack([rankdata(-row) for row in table.to_numpy()])
    out["mean_rank"] = {m: float(r) for m, r in zip(models, ranks.mean(0))}
    if np.allclose(table.to_numpy(), table.to_numpy()[:, :1]):
        out["note"] += "; all models tied in every block, test undefined"
        return out
    stat, p = friedmanchisquare(*[table[m].to_numpy() for m in models])
    out["statistic"], out["p_value"] = _clean(float(stat)), _clean(float(p))
    return out


def load_predictions(per_seed, attribute):
    """Long table of test predictions for one attribute: arch, fold, seed, image_path, correct."""
    frames = []
    for _, r in per_seed[per_seed["attribute"] == attribute].iterrows():
        p = pd.read_csv(r["predictions"])
        frames.append(pd.DataFrame({"arch": r["arch"], "fold": r["fold"], "seed": r["seed"],
                                    "image_path": p["image_path"], "correct": p["true_label"] == p["pred_label"]}))
    return pd.concat(frames, ignore_index=True)


def mcnemar_pairs(per_seed, attribute, model_order):
    preds = load_predictions(per_seed, attribute)
    wide = preds.pivot_table(index=["fold", "seed", "image_path"], columns="arch", values="correct", aggfunc="first")
    models = [m for m in model_order if m in wide.columns]
    rows = []
    for a, b in itertools.combinations(models, 2):
        pair = wide[[a, b]].dropna().astype(bool)
        n01 = int((pair[a] & ~pair[b]).sum())  # a right, b wrong
        n10 = int((~pair[a] & pair[b]).sum())  # a wrong, b right
        both = int((pair[a] & pair[b]).sum())
        neither = int((~pair[a] & ~pair[b]).sum())
        discordant = n01 + n10
        if discordant == 0:
            p, stat, method = 1.0, 0.0, "no discordant pairs"
        else:
            exact = discordant < 25
            res = mcnemar([[both, n01], [n10, neither]], exact=exact, correction=not exact)
            p, stat = float(res.pvalue), float(res.statistic)
            method = "exact binomial" if exact else "chi-square with continuity correction"
        rows.append({"model_a": a, "model_b": b, "n_pairs": int(len(pair)), "a_right_b_wrong": n01,
                     "a_wrong_b_right": n10, "both_right": both, "both_wrong": neither,
                     "statistic": stat, "p_raw": p, "method": method})
    if rows:
        corrected = multipletests([r["p_raw"] for r in rows], method="holm")[1]
        for r, pc in zip(rows, corrected):
            r["p_holm"] = float(pc)
    return rows


def run_stats(out_dir, model_order, alpha=0.05):
    out_dir = Path(out_dir)
    per_seed = pd.read_csv(out_dir / "per_seed.csv")
    results = {"alpha": alpha, "correction": "holm", "attributes": {}}
    for attribute in per_seed["attribute"].unique():
        pairs = mcnemar_pairs(per_seed, attribute, model_order)
        for r in pairs:
            r["significant_holm"] = bool(r["p_holm"] < alpha)
        results["attributes"][attribute] = {
            "friedman": friedman(per_seed, attribute, model_order),
            "mcnemar": {"note": POOLED_CAUTION, "pairs": pairs},
        }
    with open(out_dir / "stats.json", "w", encoding="utf-8") as f:
        json.dump(_clean(results), f, indent=1, allow_nan=False)
    pairs = [{"attribute": a, **r} for a, v in results["attributes"].items() for r in v["mcnemar"]["pairs"]]
    pd.DataFrame(pairs).to_csv(out_dir / "mcnemar.csv", index=False)
    return results
