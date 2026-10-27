"""Weighted backbone selection (build step 6).

One pre-written rule picks a single backbone. Each model is scored on four
criteria, each min-max normalised across the models to [0, 1] (1 = best), then
combined with the weights in config.yaml:

  macro_f1        mean test macro-F1 over selection.attributes   higher is better
  inference_time  ONNX Runtime CPU latency, batch size 1         lower is better
  model_size      fp32 weight file size (MB)                      lower is better
  gradcam         Grad-CAM compatibility check (1 / 0)           higher is better

Only selection.attributes (genus) are scored; exploratory attributes are
reported but never affect the choice. If every model has the same value on a
criterion, it cannot separate them and all get 1.0 on it. Ties in the plain
score are broken by macro-F1, then inference time. GPU latency is reported as
a secondary column only.

The tie rule (selection.tie_rule, on by default) is the recommendation: models
that are not significantly worse than the best by McNemar (Holm) and within one
standard deviation of its macro-F1 are ranked by speed, size and Grad-CAM only.
Both rankings are reported.
"""
import json
from pathlib import Path

import pandas as pd

from src.config import ConfigError

CRITERIA = {  # name -> (column, higher_is_better)
    "macro_f1": ("macro_f1", True),
    "inference_time": ("onnx_cpu_ms", False),
    "model_size": ("model_size_mb", False),
    "gradcam": ("gradcam", True),
}


def check_weights(weights, label="selection.weights"):
    if set(weights) != set(CRITERIA):
        raise ConfigError(f"{label} must have exactly these keys: {sorted(CRITERIA)}; got {sorted(weights)}")
    if any(w < 0 for w in weights.values()) or abs(sum(weights.values()) - 1.0) > 1e-6:
        raise ConfigError(f"{label} must be non-negative and sum to 1; got {weights}")


def model_table(summary, profile, model_order, attributes=None):
    """One row per model with the raw value of every criterion. Macro-F1 (and its std across runs)
    is taken over `attributes` only (selection.attributes); other attributes are not scored."""
    if attributes:
        missing_attrs = sorted(set(attributes) - set(summary["attribute"]))
        if missing_attrs:
            raise ValueError(f"selection.attributes {missing_attrs} have no results in summary.csv")
        summary = summary[summary["attribute"].isin(attributes)]
    per_attr = summary.pivot_table(index="arch", columns="attribute", values="macro_f1_mean")
    per_attr_std = summary.pivot_table(index="arch", columns="attribute", values="macro_f1_std", dropna=False)
    missing = per_attr.isna().any(axis=1)
    if missing.any():
        raise ValueError(f"Models without results for every attribute: {list(per_attr.index[missing])}")
    prof = profile.set_index("arch")
    rows = []
    for arch in model_order:
        if arch not in per_attr.index or arch not in prof.index:
            continue
        s = summary[summary["arch"] == arch]
        rows.append({
            "model": s["model"].iloc[0], "arch": arch,
            "macro_f1": float(per_attr.loc[arch].mean()), "n_attributes": int(per_attr.shape[1]),
            "macro_f1_std": float(per_attr_std.loc[arch].mean()) if arch in per_attr_std.index else float("nan"),
            "attributes": ",".join(per_attr.columns),
            "onnx_cpu_ms": float(prof.loc[arch, "onnx_cpu_ms"]),
            "model_size_mb": float(s["model_size_mb_mean"].mean()),
            "gradcam": 1.0 if bool(prof.loc[arch, "gradcam_pass"]) else 0.0,
            "gpu_latency_ms": float(s["latency_gpu_ms_mean"].mean()),
            "onnx_size_mb": float(prof.loc[arch, "onnx_size_mb"]),
            "gflops": float(prof.loc[arch, "gflops"]),
        })
    return pd.DataFrame(rows)


def minmax(values, higher_is_better):
    lo, hi = values.min(), values.max()
    if hi - lo < 1e-12:
        return pd.Series(1.0, index=values.index)
    norm = (values - lo) / (hi - lo)
    return norm if higher_is_better else 1.0 - norm


def score(table, weights):
    t = table.copy()
    t["score"] = 0.0
    for name, (col, higher) in CRITERIA.items():
        t[f"{name}_norm"] = minmax(t[col], higher)
        t["score"] += weights[name] * t[f"{name}_norm"]
    t = t.sort_values(["score", "macro_f1", "onnx_cpu_ms"], ascending=[False, False, True], kind="stable")
    t.insert(0, "rank", range(1, len(t) + 1))
    return t.reset_index(drop=True)


TIE_CRITERIA = ("inference_time", "model_size", "gradcam")
MCNEMAR_TEXT = {
    "pooled": "Pooled: one test over all folds and seeds; every test image counts once per seed, so the pairs are "
              "not independent and the p-values are too small.",
    "per_seed": "Per seed (added 2026-10-27, after the first analysis): one test per seed over all folds, so every "
                "test image counts once, Holm-corrected within each seed; a model is significantly worse only when "
                "it is significant in every seed. The p-values are listed per seed.",
}


MCNEMAR_MODES = ("pooled", "per_seed")


def _mcnemar_pairs(stats, attributes, mode="pooled"):
    """{(attribute, model_a, model_b): [pair result, ...]}: one pooled result, or one per seed."""
    out = {}
    for attr in attributes:
        st = stats.get("attributes", {}).get(attr, {})
        if mode == "pooled":
            groups = [(st.get("mcnemar") or {}).get("pairs", [])]
        else:
            seeds = (st.get("mcnemar_per_seed") or {}).get("seeds", {})
            groups = [seeds[k] for k in sorted(seeds, key=int)]
        for pairs in groups:
            for p in pairs:
                out.setdefault((attr, p["model_a"], p["model_b"]), []).append(p)
    return out


def _worse(p, best_arch, alpha):
    """True when the best model is significantly better than the other model of the pair p."""
    best_right = p["a_right_b_wrong"] if p["model_a"] == best_arch else p["a_wrong_b_right"]
    model_right = p["a_wrong_b_right"] if p["model_a"] == best_arch else p["a_right_b_wrong"]
    return p["p_holm"] < alpha and best_right > model_right


def tie_rule(ranked, stats, attributes, weights, alpha=0.05, mcnemar_mode="pooled"):
    """Re-rank with the tie rule. Adds tied, f1_gap, best_f1_std, mcnemar_p_holm, tie_note, tie_score, tie_rank.

    The best model has the highest mean macro-F1 (on the selection attributes). Another model is
    tied with it when it is not significantly worse by McNemar (Holm-corrected, on every
    selection attribute) and its mean macro-F1 is within one standard deviation (of the best
    model, across runs) of the best. With mcnemar_mode "pooled" one test over all runs decides;
    with "per_seed" a model is significantly worse only when the test is significant (best
    model better) in every seed. Tied models are ranked by speed, size and Grad-CAM only, with
    their usual weights; the other models follow in plain-score order. A model with no McNemar
    result against the best is not tied.
    """
    if mcnemar_mode not in MCNEMAR_MODES:
        raise ValueError(f"selection.tie_rule.mcnemar must be one of {MCNEMAR_MODES}, not {mcnemar_mode!r}")
    t = ranked.copy()
    best = t.sort_values(["macro_f1", "rank"], ascending=[False, True]).iloc[0]
    best_std = 0.0 if pd.isna(best["macro_f1_std"]) else float(best["macro_f1_std"])
    pairs = _mcnemar_pairs(stats, attributes, mcnemar_mode)
    tied, gaps, pvals, notes = [], [], [], []
    for _, r in t.iterrows():
        gap = float(best["macro_f1"] - r["macro_f1"])
        gaps.append(gap)
        if r["arch"] == best["arch"]:
            tied.append(True), pvals.append(""), notes.append("best macro-F1")
            continue
        worse, ps, note = False, [], ""
        for attr in attributes:
            found = pairs.get((attr, best["arch"], r["arch"])) or pairs.get((attr, r["arch"], best["arch"]))
            if not found:
                worse, note = True, f"no McNemar result for {attr}"
                break
            ps.append(",".join(f"{p['p_holm']:.4f}" for p in found))
            hits = [_worse(p, best["arch"], alpha) for p in found]
            if all(hits):
                worse = True
                note = f"significantly worse on {attr}" + (f" in all {len(hits)} seeds" if mcnemar_mode == "per_seed" else "")
            elif any(hits):
                note = f"significantly worse in {sum(hits)} of {len(hits)} seeds on {attr}, not all"
        within = gap <= best_std + 1e-12
        if not worse and not within:
            note = f"macro-F1 gap {gap:.3f} > best std {best_std:.3f}"
        tied.append(not worse and within)
        pvals.append(";".join(ps))
        notes.append(note or "tied")
    t["tied"], t["f1_gap"], t["best_f1_std"], t["mcnemar_p_holm"], t["tie_note"] = tied, gaps, best_std, pvals, notes
    t["tie_score"] = sum(weights[k] * t[f"{k}_norm"] for k in TIE_CRITERIA)
    t["_order"] = [(0, -ts, -f1) if tie else (1, rk, 0) for tie, ts, f1, rk in
                   zip(t["tied"], t["tie_score"], t["macro_f1"], t["rank"])]
    t = t.sort_values("_order", kind="stable").drop(columns="_order").reset_index(drop=True)
    t["tie_rank"] = range(1, len(t) + 1)
    return t


def sensitivity(table, base_winner, weight_sets):
    out = []
    for i, w in enumerate(weight_sets, start=1):
        check_weights(w, f"selection.sensitivity[{i}]")
        ranked = score(table, w)
        out.append({"weights": w, "winner": ranked.loc[0, "arch"], "winner_model": ranked.loc[0, "model"],
                    "changed": ranked.loc[0, "arch"] != base_winner,
                    "order": " > ".join(ranked["model"])})
    return out


def fmt_weights(w):
    return " / ".join(f"{round(100 * w[k])}" for k in CRITERIA)


def tie_mcnemar_mode(cfg):
    """selection.tie_rule.mcnemar: which McNemar version decides the tie rule (default pooled, the original)."""
    return str((cfg["selection"].get("tie_rule") or {}).get("mcnemar", "pooled"))


def selection_settings(cfg):
    """(selection attributes, tie rule on?, alpha). Attributes default to the main attribute."""
    sel = cfg["selection"]
    attrs = list(sel.get("attributes") or [cfg["data"].get("main_attribute", "genus")])
    tr = sel.get("tie_rule") or {}
    return attrs, bool(tr.get("enabled", True)), float(tr.get("alpha", 0.05))


def write_ranking(cfg, out_dir, profile=None):
    """ranking.csv and ranking.md. Returns (ranked, sensitivity): ranked has the plain weighted rank
    (`rank`), the tie-rule rank (`tie_rank`, `tied`, ...) and `recommended`, in recommendation order."""
    out_dir = Path(out_dir)
    sel = cfg["selection"]
    check_weights(sel["weights"])
    attrs, tie_on, alpha = selection_settings(cfg)
    summary = pd.read_csv(out_dir / "summary.csv")
    profile = profile if profile is not None else pd.read_csv(out_dir / "model_profile.csv")
    stats_path = out_dir / "stats.json"
    stats = json.loads(stats_path.read_text(encoding="utf-8")) if stats_path.exists() else {}
    table = model_table(summary, profile, list(cfg["models"]), attrs)
    plain = score(table, sel["weights"])
    mode = tie_mcnemar_mode(cfg)
    tied = tie_rule(plain, stats, attrs, sel["weights"], alpha, mode)
    other_mode = "pooled" if mode == "per_seed" else "per_seed"
    has_other = other_mode == "pooled" or all("mcnemar_per_seed" in stats.get("attributes", {}).get(a, {})
                                               for a in attrs)
    other = tie_rule(plain, stats, attrs, sel["weights"], alpha, other_mode) if has_other else None
    if other is not None:
        o = other.set_index("arch")
        tied[f"tied_{other_mode}"] = tied["arch"].map(o["tied"])
        tied[f"tie_rank_{other_mode}"] = tied["arch"].map(o["tie_rank"])
        tied[f"mcnemar_p_holm_{other_mode}"] = tied["arch"].map(o["mcnemar_p_holm"])
    final = tied if tie_on else tied.sort_values("rank").reset_index(drop=True)
    final["recommended"] = [i == 0 for i in range(len(final))]
    winner = final.loc[0]
    sens = sensitivity(table, plain.loc[0, "arch"], sel.get("sensitivity", []))
    final.to_csv(out_dir / "ranking.csv", index=False)

    w = sel["weights"]
    rule = "tie rule" if tie_on else "plain weighted score"
    plain_winner = plain.loc[0]
    lines = [
        f"# Backbone ranking: {cfg['run_name']}", "",
        f"**Recommended backbone: {winner['model']} (`{winner['arch']}`)**, by the {rule}"
        + (f" (selection.tie_rule.enabled: true). The plain weighted score ranks {plain_winner['model']} first."
           if tie_on else "."), "",
        f"Scored attributes: {', '.join(attrs)} (selection.attributes). Any other attribute (exploratory results) "
        "is reported in summary.csv and report.html but does not affect the ranking.", "",
        "## Plain weighted score", "",
        "Score = " + " + ".join(f"{w[k]:.2f} x {k}" for k in CRITERIA) + ", each criterion min-max "
        "normalised across the models (1 = best). Macro-F1 is the mean test macro-F1 over the cross-validation runs "
        f"of {', '.join(attrs)}; inference time is ONNX Runtime CPU latency at batch size 1; "
        "model size is the fp32 weight file; Grad-CAM is pass (1) or fail (0). "
        "A criterion on which all models are equal gives every model 1.0.", "",
        "| Rank | Model | Score | Macro-F1 | norm | ONNX CPU ms | norm | Size MB | norm | Grad-CAM | GPU ms (secondary) |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for _, r in plain.iterrows():
        lines.append(f"| {r['rank']} | {r['model']} | {r['score']:.3f} | {r['macro_f1']:.3f} | {r['macro_f1_norm']:.2f} "
                     f"| {r['onnx_cpu_ms']:.1f} | {r['inference_time_norm']:.2f} | {r['model_size_mb']:.1f} "
                     f"| {r['model_size_norm']:.2f} | {'pass' if r['gradcam'] else 'fail'} | {r['gpu_latency_ms']:.1f} |")
    lines += ["", "## Tie rule" + (" (the recommendation)" if tie_on else " (reported, not used)"), "",
              f"A model is tied with the best macro-F1 model ({tied.loc[tied['f1_gap'].idxmin(), 'model']}) when it is "
              f"not significantly worse by McNemar (Holm-corrected, alpha {alpha}) and its mean macro-F1 is within one "
              f"standard deviation of the best ({tied['best_f1_std'].iloc[0]:.3f}). Tied models are ranked by inference "
              "time, model size and Grad-CAM only (same weights); the rest follow in plain-score order.", "",
              f"McNemar version used (selection.tie_rule.mcnemar): **{mode}**. "
              + MCNEMAR_TEXT[mode], "",
              "| Rank | Model | Tied | Macro-F1 | Gap to best | McNemar p (Holm) vs best | Speed/size/Grad-CAM score | Note |",
              "|---|---|---|---|---|---|---|---|"]
    for _, r in tied.iterrows():
        lines.append(f"| {r['tie_rank']} | {r['model']} | {'yes' if r['tied'] else 'no'} | {r['macro_f1']:.3f} "
                     f"| {r['f1_gap']:.3f} | {r['mcnemar_p_holm'] or '-'} | {r['tie_score']:.3f} | {r['tie_note']} |")
    if other is not None:
        lines += ["", f"### Tie rule with the {other_mode} McNemar test (reported, not used)", "",
                  MCNEMAR_TEXT[other_mode], "",
                  "| Rank | Model | Tied | McNemar p (Holm) vs best | Note |", "|---|---|---|---|---|"]
        for _, r in other.iterrows():
            lines.append(f"| {r['tie_rank']} | {r['model']} | {'yes' if r['tied'] else 'no'} "
                         f"| {r['mcnemar_p_holm'] or '-'} | {r['tie_note']} |")
        same_set = set(tied.loc[tied["tied"], "arch"]) == set(other.loc[other["tied"], "arch"])
        same_winner = tied.loc[0, "arch"] == other.loc[0, "arch"]
        lines += ["", f"Compared with the {other_mode} version, the tie set "
                  f"**{'does not change' if same_set else 'changes'}** ({mode}: "
                  f"{', '.join(tied.loc[tied['tied'], 'model'])}; {other_mode}: {', '.join(other.loc[other['tied'], 'model'])}) "
                  f"and the tie-rule winner **{'does not change' if same_winner else 'changes'}** ({mode}: "
                  f"{tied.loc[0, 'model']}; {other_mode}: {other.loc[0, 'model']})."]
    lines += ["", "## Sensitivity check (plain weighted score)", "",
              f"Weights are macro-F1 / inference time / model size / Grad-CAM (%). Base: {fmt_weights(w)}.", "",
              "| Weights | Winner | Winner changed | Full order |", "|---|---|---|---|"]
    for s in sens:
        lines.append(f"| {fmt_weights(s['weights'])} | {s['winner_model']} | {'yes' if s['changed'] else 'no'} "
                     f"| {s['order']} |")
    stable = not any(s["changed"] for s in sens)
    lines += ["", f"The plain-score winner {'does not change' if stable else 'changes'} under the alternative weight sets."]
    (out_dir / "ranking.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return final, sens
