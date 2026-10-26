"""Weighted backbone selection (build step 6).

One pre-written rule picks a single backbone for all attributes. Each model
is scored on four criteria, each min-max normalised across the models to
[0, 1] (1 = best), then combined with the weights in config.yaml:

  macro_f1        mean test macro-F1 across attributes      higher is better
  inference_time  ONNX Runtime CPU latency, batch size 1   lower is better
  model_size      fp32 weight file size (MB)                lower is better
  gradcam         Grad-CAM compatibility check (1 / 0)     higher is better

If every model has the same value on a criterion, it cannot separate them and
all get 1.0 on it. Ties in the final score are broken by macro-F1, then
inference time. GPU latency is reported as a secondary column only.
"""
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


def model_table(summary, profile, model_order):
    """One row per model with the raw value of every criterion."""
    per_attr = summary.pivot_table(index="arch", columns="attribute", values="macro_f1_mean")
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


def write_ranking(cfg, out_dir, profile=None):
    out_dir = Path(out_dir)
    sel = cfg["selection"]
    check_weights(sel["weights"])
    summary = pd.read_csv(out_dir / "summary.csv")
    profile = profile if profile is not None else pd.read_csv(out_dir / "model_profile.csv")
    table = model_table(summary, profile, list(cfg["models"]))
    ranked = score(table, sel["weights"])
    winner = ranked.loc[0]
    sens = sensitivity(table, winner["arch"], sel.get("sensitivity", []))
    ranked.to_csv(out_dir / "ranking.csv", index=False)

    w = sel["weights"]
    lines = [
        f"# Backbone ranking: {cfg['run_name']}", "",
        f"**Recommended backbone: {winner['model']} (`{winner['arch']}`)**, score {winner['score']:.3f}.", "",
        "Score = " + " + ".join(f"{w[k]:.2f} x {k}" for k in CRITERIA) + ", each criterion min-max "
        "normalised across the models (1 = best). Macro-F1 is the mean over "
        f"{int(winner['n_attributes'])} attribute(s); inference time is ONNX Runtime CPU latency at batch size 1; "
        "model size is the fp32 weight file; Grad-CAM is pass (1) or fail (0). "
        "A criterion on which all models are equal gives every model 1.0.", "",
        "| Rank | Model | Score | Macro-F1 | norm | ONNX CPU ms | norm | Size MB | norm | Grad-CAM | GPU ms (secondary) |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for _, r in ranked.iterrows():
        lines.append(f"| {r['rank']} | {r['model']} | {r['score']:.3f} | {r['macro_f1']:.3f} | {r['macro_f1_norm']:.2f} "
                     f"| {r['onnx_cpu_ms']:.1f} | {r['inference_time_norm']:.2f} | {r['model_size_mb']:.1f} "
                     f"| {r['model_size_norm']:.2f} | {'pass' if r['gradcam'] else 'fail'} | {r['gpu_latency_ms']:.1f} |")
    lines += ["", "## Sensitivity check", "",
              f"Weights are macro-F1 / inference time / model size / Grad-CAM (%). Base: {fmt_weights(w)}.", "",
              "| Weights | Winner | Winner changed | Full order |", "|---|---|---|---|"]
    for s in sens:
        lines.append(f"| {fmt_weights(s['weights'])} | {s['winner_model']} | {'yes' if s['changed'] else 'no'} "
                     f"| {s['order']} |")
    stable = not any(s["changed"] for s in sens)
    lines += ["", f"The winner {'does not change' if stable else 'changes'} under the alternative weight sets."]
    (out_dir / "ranking.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return ranked, sens
