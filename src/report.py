"""Results tables (per_seed.csv, per_class.csv, summary.csv) and report.html.

per_seed.csv keeps one row per (attribute, model, fold, seed) run. Re-running
a run replaces its row; other rows already in the file are kept, so
attributes or models can be run separately into the same results folder.
summary.csv is rebuilt from per_seed.csv: mean and standard deviation (n-1)
across all runs of a model and attribute (seeds, and folds for k-fold CV).

report.html is one self-contained page (inline SVG charts, no scripts) that
links to the confusion matrices and Grad-CAM overlays in the same folder.
"""
import html
import json
import math
import os
from datetime import datetime
from pathlib import Path

import pandas as pd

from src.select import selection_settings

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


def summarise_sources(per_source, model_order=None):
    """Mean and std of per-source test accuracy and macro-F1 across runs."""
    g = per_source.groupby(["attribute", "arch", "model", "source"], sort=False)
    summary = g.agg(n_runs=("seed", "size"), n_images_mean=("n_images", "mean"),
                    accuracy_mean=("accuracy", "mean"), accuracy_std=("accuracy", "std"),
                    macro_f1_mean=("macro_f1", "mean"), macro_f1_std=("macro_f1", "std")).reset_index()
    if model_order:
        summary["_order"] = summary["arch"].map({a: i for i, a in enumerate(model_order)})
        summary = summary.sort_values(["attribute", "_order", "source"], kind="stable").drop(columns="_order")
    return summary.reset_index(drop=True)


def write_tables(out_dir, run_rows, class_rows, model_order=None, source_rows=None, external_rows=None):
    out_dir = Path(out_dir)
    per_seed = merge_rows(out_dir / "per_seed.csv", run_rows)
    if external_rows:
        merge_rows(out_dir / "external_metrics.csv", external_rows, keys=RUN_KEYS + ["class"])
    merge_rows(out_dir / "per_class.csv", pd.concat(class_rows, ignore_index=True), keys=RUN_KEYS + ["class"])
    source_rows = [s for s in (source_rows or []) if not s.empty]
    if source_rows:
        per_source = merge_rows(out_dir / "per_source.csv", pd.concat(source_rows, ignore_index=True),
                                keys=RUN_KEYS + ["source"])
        summarise_sources(per_source, model_order).to_csv(out_dir / "source_summary.csv", index=False)
    summary = summarise(per_seed, model_order)
    summary.to_csv(out_dir / "summary.csv", index=False)
    return per_seed, summary


# ---------------------------------------------------------------- report.html

CSS = """
:root {
  color-scheme: light;
  --page: #f9f9f7; --surface: #fcfcfb; --ink: #0b0b0b; --ink-2: #52514e; --muted: #898781;
  --grid: #e1e0d9; --axis: #c3c2b7; --border: rgba(11,11,11,0.10); --series-1: #2a78d6;
  --good: #006300; --critical: #d03b3b; --banner: #fff4dc;
}
@media (prefers-color-scheme: dark) {
  :root:where(:not([data-theme="light"])) {
    color-scheme: dark;
    --page: #0d0d0d; --surface: #1a1a19; --ink: #f0efec; --ink-2: #c3c2b7; --muted: #898781;
    --grid: #2c2c2a; --axis: #383835; --border: rgba(255,255,255,0.10); --series-1: #3987e5;
    --good: #0ca30c; --critical: #e66767; --banner: #3a2f12;
  }
}
:root[data-theme="dark"] {
  color-scheme: dark;
  --page: #0d0d0d; --surface: #1a1a19; --ink: #f0efec; --ink-2: #c3c2b7; --muted: #898781;
  --grid: #2c2c2a; --axis: #383835; --border: rgba(255,255,255,0.10); --series-1: #3987e5;
  --good: #0ca30c; --critical: #e66767; --banner: #3a2f12;
}
* { box-sizing: border-box; }
body { margin: 0; background: var(--page); color: var(--ink);
       font: 15px/1.5 system-ui, -apple-system, "Segoe UI", sans-serif; }
main { max-width: 1080px; margin: 0 auto; padding: 24px 16px 64px; }
h1 { font-size: 24px; margin: 0 0 4px; } h2 { font-size: 18px; margin: 36px 0 8px; }
h3 { font-size: 15px; margin: 20px 0 6px; }
p, li { color: var(--ink-2); } .meta { color: var(--muted); font-size: 13px; }
.card { background: var(--surface); border: 1px solid var(--border); border-radius: 8px; padding: 16px; }
.banner { background: var(--banner); border-radius: 8px; padding: 10px 14px; margin: 16px 0; color: var(--ink); }
.winner { font-size: 20px; font-weight: 600; color: var(--ink); }
.scroll { overflow-x: auto; }
table { border-collapse: collapse; width: 100%; font-size: 13px; background: var(--surface); }
th, td { padding: 6px 8px; border-bottom: 1px solid var(--grid); text-align: right; white-space: nowrap; }
th { color: var(--ink-2); font-weight: 600; } th:first-child, td:first-child { text-align: left; }
td { font-variant-numeric: tabular-nums; }
.pass { color: var(--good); } .fail { color: var(--critical); }
.charts { display: grid; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); gap: 16px; }
svg text { font: 12px system-ui, -apple-system, "Segoe UI", sans-serif; font-variant-numeric: tabular-nums; }
.gallery { display: grid; grid-template-columns: repeat(auto-fill, minmax(240px, 1fr)); gap: 12px; }
.gallery figure { margin: 0; } .gallery img { width: 100%; border-radius: 6px; border: 1px solid var(--border); }
.gallery figcaption { font-size: 13px; color: var(--ink-2); }
a { color: var(--series-1); }
"""


def _e(x):
    return html.escape(str(x))


def _num(x, digits=3):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "–"
    return f"{x:.{digits}f}"


def _pm(mean, std, digits=3):
    if std is None or (isinstance(std, float) and math.isnan(std)):
        return _num(mean, digits)
    return f"{_num(mean, digits)} ± {_num(std, digits)}"


def _rel(path, out_dir):
    return os.path.relpath(Path(path), Path(out_dir)).replace(os.sep, "/")


def _nice_max(v):
    if v <= 0:
        return 1.0
    exp = 10 ** math.floor(math.log10(v))
    for m in (1, 2, 2.5, 5, 10):
        if v <= m * exp:
            return m * exp
    return 10 * exp


def bar_chart(title, labels, values, errors=None, unit="", digits=3, x_max=None):
    """Horizontal single-series bar chart as inline SVG, with hover tooltips."""
    width, left, right, row_h, top = 520, 132, 64, 30, 28
    plot_w = width - left - right
    height = top + row_h * len(labels) + 26
    x_max = x_max or _nice_max(max((v + (e or 0)) for v, e in zip(values, errors or [0] * len(values))))
    sx = lambda v: left + plot_w * max(0.0, min(v, x_max)) / x_max  # noqa: E731
    parts = [f'<svg viewBox="0 0 {width} {height}" width="100%" role="img" aria-label="{_e(title)}">',
             f'<text x="0" y="16" fill="var(--ink)" font-weight="600">{_e(title)}</text>']
    for i in range(5):  # gridlines and ticks
        v = x_max * i / 4
        x = sx(v)
        parts.append(f'<line x1="{x:.1f}" x2="{x:.1f}" y1="{top}" y2="{height - 22}" stroke="var(--grid)" stroke-width="1"/>')
        parts.append(f'<text x="{x:.1f}" y="{height - 6}" fill="var(--muted)" text-anchor="middle">'
                     f'{v:g}{_e(unit)}</text>')
    parts.append(f'<line x1="{left}" x2="{left}" y1="{top}" y2="{height - 22}" stroke="var(--axis)" stroke-width="1"/>')
    for i, (label, v) in enumerate(zip(labels, values)):
        y = top + i * row_h + 7
        h = row_h - 14
        x1 = sx(v)
        w = max(0.0, x1 - left)
        r = min(4.0, w / 2, h / 2)
        err = errors[i] if errors else None
        tip = f"{label}: {_num(v, digits)}{unit}" + (f" ± {_num(err, digits)}" if err and not math.isnan(err) else "")
        parts.append(f'<g><title>{_e(tip)}</title>'
                     f'<rect x="0" y="{top + i * row_h}" width="{width}" height="{row_h}" fill="transparent"/>'
                     f'<text x="{left - 8}" y="{y + h / 2 + 4:.1f}" fill="var(--ink-2)" text-anchor="end">{_e(label)}</text>')
        if w > 0:  # bar anchored at the baseline, rounded at the data end
            parts.append(f'<path d="M{left},{y} H{x1 - r:.1f} Q{x1:.1f},{y} {x1:.1f},{y + r:.1f} '
                         f'V{y + h - r:.1f} Q{x1:.1f},{y + h} {x1 - r:.1f},{y + h} H{left} Z" fill="var(--series-1)"/>')
        end = x1
        if err and not math.isnan(err):
            lo, hi = sx(v - err), sx(v + err)
            cy = y + h / 2
            parts.append(f'<path d="M{lo:.1f},{cy} H{hi:.1f} M{lo:.1f},{cy - 4} V{cy + 4} M{hi:.1f},{cy - 4} V{cy + 4}" '
                         f'stroke="var(--ink-2)" stroke-width="1.5" fill="none"/>')
            end = max(end, hi)
        parts.append(f'<text x="{end + 6:.1f}" y="{y + h / 2 + 4:.1f}" fill="var(--ink-2)">{_num(v, digits)}</text></g>')
    parts.append("</svg>")
    return "".join(parts)


def _table(header, rows):
    head = "".join(f"<th>{_e(h)}</th>" for h in header)
    body = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows)
    return f'<div class="scroll"><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>'


def _p(p):
    if p is None or (isinstance(p, float) and math.isnan(p)):
        return "–"
    return "&lt; 0.0001" if p < 1e-4 else f"{p:.4f}"


def data_section(out_dir):
    """Which attributes were trained and which were skipped, from attributes.json."""
    path = Path(out_dir) / "attributes.json"
    if not path.exists():
        return []
    usage = json.loads(path.read_text(encoding="utf-8"))
    rows = []
    for attr, u in usage.items():
        status = ("<span class='fail'>⚠ skipped</span>" if u["skip_reason"] else "<span class='pass'>✓ used</span>")
        groups = ", ".join(f"{_e(k)} {v}" for k, v in u["groups_per_class"].items()) or "–"
        rows.append([_e(attr), status, u["n_images"], u["n_groups"], u["n_classes"], groups,
                     _e(u["skip_reason"] or "")])
    min_groups = next(iter(usage.values()))["min_groups_per_class"] if usage else "?"
    return ["<h2>Data used</h2><p>Each attribute uses only the rows labeled for it. Groups are distinct "
            "<code>crab_id</code> values (an individual, observer, observation or filename ID, not a verified "
            f"individual). An attribute is skipped with fewer than 2 classes or any class below {min_groups} groups.</p>",
            _table(["Attribute", "Status", "Images", "Groups", "Classes", "Groups per class", "Reason"], rows)]


def _only(df, attrs):
    return df[df["attribute"].isin(attrs)] if attrs else df


def source_sections(cfg, out_dir, attrs=None):
    """Per-source test results and the source-dominance check, when labels.csv has a source column."""
    out_dir = Path(out_dir)
    summary_path, balance_path = out_dir / "source_summary.csv", out_dir / "source_balance.csv"
    if not summary_path.exists() and not balance_path.exists():
        return ["<h2>Results by source</h2><p class='meta'>labels.csv has no <code>source</code> column, "
                "so there are no per-source results.</p>"]
    s = []
    if summary_path.exists():
        src = _only(pd.read_csv(summary_path), attrs)
        s.append("<h2>Results by source</h2><p>Test accuracy and macro-F1 per source, mean ± std over runs. "
                 "Macro-F1 averages the classes present in that source's test images; a source with few test "
                 "images gives noisy numbers.</p>")
        for attribute, g in src.groupby("attribute", sort=False):
            s.append(f"<h3>{_e(attribute)}</h3>")
            s.append(_table(["Model", "Source", "Runs", "Test images (mean)", "Accuracy", "Macro-F1"],
                            [[_e(r["model"]), _e(r["source"]), int(r["n_runs"]), _num(r["n_images_mean"], 1),
                              _pm(r["accuracy_mean"], r["accuracy_std"]), _pm(r["macro_f1_mean"], r["macro_f1_std"])]
                             for _, r in g.iterrows()]))
    if balance_path.exists():
        bal = pd.read_csv(balance_path)
        flagged = bal[bal["flagged"]]
        thr = float(bal["threshold"].iloc[0]) if len(bal) else cfg["data"].get("source_dominance_threshold", 0.8)
        s.append(f"<h3>Source balance</h3><p>Classes where one source supplies more than {thr:.0%} of the images "
                 f"are flagged: a model can learn that source's camera, background or framing instead of the crab. "
                 f"<b>{len(flagged)} of {len(bal)} class(es) flagged.</b></p>")
        rows = []
        for _, r in bal.iterrows():
            flag = "<span class='fail'>⚠ flagged</span>" if r["flagged"] else "ok"
            by_source = ", ".join(f"{_e(k)} {v}" for k, v in json.loads(r["images_by_source"]).items())
            rows.append([_e(r["attribute"]), _e(r["class"]), r["n_images"], _e(r["top_source"]),
                         f"{100 * r['top_share']:.0f}%", flag, by_source])
        s.append(_table(["Attribute", "Class", "Images", "Largest source", "Its share", "Flag", "Images by source"],
                        rows))
    return s


SMALL_GROUPS = 10  # external classes with fewer groups are marked small


def external_section(cfg, out_dir, attrs=None):
    """The external test set (data.external_test): what was held out, and every model's results on it."""
    out_dir = Path(out_dir)
    split_path, metrics_path = out_dir / "external_split.json", out_dir / "external_metrics.csv"
    if not split_path.exists():
        return []
    info = json.loads(split_path.read_text(encoding="utf-8"))
    name = info.get("name", "external")
    ext, rem = info["external"], info["removed_overlap"]
    s = [f"<h2>External test: {_e(name)}</h2>",
         f"<p>Images with <code>{_e(info['column'])}</code> = {_e(', '.join(info['values']))} are held out of every "
         f"train, validation and cross-validation test split, and each trained run is tested on them as a local check. "
         f"{ext['images']} images in {ext['groups']} groups are held out. Every observer who also appears in them "
         f"(matched on {_e(', '.join(info['group_columns']))}) was removed from the training side: "
         f"{rem['images']} images in {rem['groups']} groups.</p>"]
    classes = sorted(set(ext["by_class"]) | set(rem["by_class"]))
    if classes:
        s.append(_table(["Class", "Held-out images", "Held-out groups", "Removed from training: images", "groups"],
                        [[_e(c), ext["by_class"].get(c, {}).get("images", 0), ext["by_class"].get(c, {}).get("groups", 0),
                          rem["by_class"].get(c, {}).get("images", 0), rem["by_class"].get(c, {}).get("groups", 0)]
                         for c in classes]))
    if not metrics_path.exists():
        s.append("<p class='meta'>No external results yet.</p>")
        return s
    m = _only(pd.read_csv(metrics_path), attrs)
    s.append(f"<p><b>Groups are small for some genera</b> (⚠ marks fewer than {SMALL_GROUPS} groups). Results on this "
             "set describe how the models did on these photos; they are not precise estimates, and per-class numbers "
             "for small classes should not be over-interpreted.</p>")
    for attribute, g in m.groupby("attribute", sort=False):
        overall = g[g["class"] == "ALL"].groupby(["arch", "model"], sort=False)
        s.append(f"<h3>{_e(attribute)}: overall</h3>")
        s.append(_table(["Model", "Runs", "Images", "Groups", "Accuracy", "Macro-F1"],
                        [[_e(model), len(r), int(r["n_images"].iloc[0]), int(r["n_groups"].iloc[0]),
                          _pm(r["accuracy"].mean(), r["accuracy"].std()), _pm(r["macro_f1"].mean(), r["macro_f1"].std())]
                         for (arch, model), r in overall]))
        per = g[g["class"] != "ALL"].groupby(["arch", "model", "class"], sort=False)
        rows = []
        for (arch, model, cls), r in per:
            n_groups = int(r["n_groups"].iloc[0])
            small = " ⚠" if n_groups < SMALL_GROUPS else ""
            rows.append([_e(model), _e(cls), int(r["n_images"].iloc[0]), f"{n_groups}{small}",
                         _pm(r["recall"].mean(), r["recall"].std()), _pm(r["f1"].mean(), r["f1"].std())])
        s.append(f"<h3>{_e(attribute)}: per class</h3>")
        s.append(_table(["Model", "Class", "Images", "Groups", "Recall", "F1"], rows))
    s.append("<p class='meta'>Per-run rows: external_metrics.csv; predictions: predictions_external/; confusion "
             "matrices: confusion_matrices_external/.</p>")
    return s


def write_report(cfg, out_dir, ranked, sensitivity):
    out_dir = Path(out_dir)
    attrs = selection_settings(cfg)[0]  # the report covers the analysed attributes (genus) only
    summary = pd.read_csv(out_dir / "summary.csv")
    others = sorted(set(summary["attribute"]) - set(attrs))
    summary = _only(summary, attrs)
    per_seed = _only(pd.read_csv(out_dir / "per_seed.csv"), attrs)
    profile = pd.read_csv(out_dir / "model_profile.csv").set_index("arch")
    stats = json.loads((out_dir / "stats.json").read_text(encoding="utf-8"))
    order = list(cfg["models"])
    first = per_seed.iloc[0]
    winner = ranked.iloc[0]
    test_run = (not bool(first["pretrained"])) or "dummy" in str(cfg["data"]["labels_csv"]).lower()
    s = []

    s.append(f"<h1>Crab backbone comparison</h1><p class='meta'>Run <b>{_e(cfg['run_name'])}</b> · generated "
             f"{datetime.now():%Y-%m-%d %H:%M} · labels {_e(cfg['data']['labels_csv'])}</p>")
    if test_run:
        s.append("<div class='banner'><b>Test run.</b> Dummy data and/or random weights (no pretraining); "
                 "these numbers check the pipeline and say nothing about the models on real crab photos.</div>")
    seeds = sorted(per_seed["seed"].unique().tolist())
    folds = sorted(per_seed["fold"].unique().tolist())
    rule = "tie rule" if selection_settings(cfg)[1] else "plain weighted score"
    s.append(f"<div class='card'><div class='meta'>Recommended backbone ({rule}, scored on "
             f"{_e(', '.join(selection_settings(cfg)[0]))} only; config.yaml)</div>"
             f"<div class='winner'>{_e(winner['model'])} <span class='meta'>({_e(winner['arch'])}), score "
             f"{winner['score']:.3f}</span></div><p class='meta'>Split {_e(first['split_strategy'])} (split seed "
             f"{_e(first['split_seed'])}, folds {folds}) · training seeds {seeds} · max epochs "
             f"{_e(cfg['training']['epochs'])} · effective batch {_e(first['effective_batch_size'])} · AMP "
             f"{_e(first['amp'])} · pretrained {_e(first['pretrained'])}<br>GPU {_e(first['gpu'])} · torch "
             f"{_e(first['torch'])} · CUDA {_e(first['cuda'])} · cuDNN {_e(first['cudnn'])} · timm "
             f"{_e(first['timm'])}</p></div>")

    s.extend(data_section(out_dir))

    # comparison table and charts, per attribute
    s.append("<h2>Comparison</h2><p>Test-set metrics, mean ± std over "
             f"{int(summary['n_runs'].max())} run(s) per model (seeds{', folds' if len(folds) > 1 else ''}). "
             "Macro averages are over the classes present in each test set. GPU latency is PyTorch fp32, batch 1; "
             "ONNX CPU latency is ONNX Runtime, batch 1.</p>")
    for attribute, g in summary.groupby("attribute", sort=False):
        rows = []
        for _, r in g.iterrows():
            p = profile.loc[r["arch"]] if r["arch"] in profile.index else None
            cam = ("<span class='pass'>✓ pass</span>" if p is not None and bool(p["gradcam_pass"])
                   else "<span class='fail'>✗ fail</span>")
            rows.append([_e(r["model"]), _pm(r["macro_f1_mean"], r["macro_f1_std"]),
                         _pm(r["accuracy_mean"], r["accuracy_std"]), _pm(r["macro_precision_mean"], r["macro_precision_std"]),
                         _pm(r["macro_recall_mean"], r["macro_recall_std"]),
                         _pm(r["macro_specificity_mean"], r["macro_specificity_std"]),
                         _pm(r["roc_auc_ovr_mean"], r["roc_auc_ovr_std"]), _num(r["latency_gpu_ms_mean"], 1),
                         _num(p["onnx_cpu_ms"], 1) if p is not None else "–", _num(r["params_m_mean"], 1),
                         _num(p["gflops"], 2) if p is not None else "–", _num(r["model_size_mb_mean"], 1), cam])
        s.append(f"<h3>{_e(attribute)}</h3>")
        s.append(_table(["Model", "Macro-F1", "Accuracy", "Macro precision", "Macro recall", "Macro specificity",
                         "ROC-AUC (OvR)", "GPU ms", "ONNX CPU ms", "Params (M)", "GFLOPs", "Size (MB)", "Grad-CAM"], rows))
        charts = bar_chart(f"Macro-F1, {attribute} (mean ± std)", list(g["model"]), list(g["macro_f1_mean"]),
                           list(g["macro_f1_std"]), x_max=1.0)
        s.append(f"<div class='charts'><div class='card'>{charts}</div>")
        lat = profile.reindex(order).dropna(subset=["onnx_cpu_ms"])
        s.append(f"<div class='card'>{bar_chart('ONNX Runtime CPU latency, batch 1 (ms)', list(lat['model']), list(lat['onnx_cpu_ms']), digits=1)}"
                 "</div></div>")
    s.append("<p class='meta'>GFLOPs from thop (2 × MACs); thop does not count ViT's attention matrix products "
             "(about 1.4 GFLOPs for ViT-B/16), so ViT is slightly undercounted.</p>")

    s.extend(source_sections(cfg, out_dir, attrs))
    s.extend(external_section(cfg, out_dir, attrs))

    # statistics
    s.append("<h2>Statistical tests</h2>")
    for attribute, st in stats["attributes"].items():
        fr = st["friedman"]
        s.append(f"<h3>{_e(attribute)}: Friedman (macro-F1)</h3>")
        caution = "caution" in fr["note"]
        s.append(f"<p>{'<b>' if caution else ''}{_e(fr['note'])}{'</b>' if caution else ''}. "
                 f"Blocks: {fr['n_blocks']} {_e(fr['block'])}(s), {fr['n_models']} models. "
                 f"χ² = {_num(fr['statistic'], 3)}, p = {_p(fr['p_value'])}.</p>")
        if fr["mean_rank"]:
            s.append(_table(["Model", "Mean rank (1 = best)"],
                            [[_e(cfg["models"][a]["name"]), _num(v, 2)] for a, v in fr["mean_rank"].items()]))
        mc = st["mcnemar"]
        s.append(f"<h3>{_e(attribute)}: McNemar, every pair</h3><p>{_e(mc['note'])}. "
                 f"Holm correction across {len(mc['pairs'])} pairs, α = {stats['alpha']}.</p>")
        s.append(_table(["Model A", "Model B", "Paired images", "A right, B wrong", "A wrong, B right", "p (raw)",
                         "p (Holm)", "Significant", "Method"],
                        [[_e(cfg["models"][r["model_a"]]["name"]), _e(cfg["models"][r["model_b"]]["name"]),
                          r["n_pairs"], r["a_right_b_wrong"], r["a_wrong_b_right"], _p(r["p_raw"]), _p(r["p_holm"]),
                          "yes" if r["significant_holm"] else "no", _e(r["method"])] for r in mc["pairs"]]))

    # ranking
    w = cfg["selection"]["weights"]
    sel_attrs, tie_on, alpha = selection_settings(cfg)
    plain = ranked.sort_values("rank")
    s.append("<h2>Ranking</h2><p>Only " + ", ".join(f"<b>{_e(a)}</b>" for a in sel_attrs) + " is scored "
             "(selection.attributes) and this report covers only that; exploratory results such as species "
             + (f"(in this folder: {_e(', '.join(others))}) " if others else "")
             + "stay in summary.csv and per_seed.csv and do not affect the "
             f"choice. The recommendation uses the <b>{'tie rule' if tie_on else 'plain weighted score'}</b>; both "
             "rankings are shown. Full method in ranking.md.</p>")
    s.append("<h3>Plain weighted score</h3><p>Score = " + " + ".join(f"{v:.2f} × {_e(k)}" for k, v in w.items()) +
             ", each criterion min-max normalised across the models (1 = best). Inference time is ONNX Runtime CPU "
             "latency at batch 1; GPU latency is shown for reference only.</p>")
    s.append(_table(["Rank", "Model", "Score", "Macro-F1", "norm", "ONNX CPU ms", "norm", "Size MB", "norm",
                     "Grad-CAM", "GPU ms (secondary)"],
                    [[r["rank"], f"<b>{_e(r['model'])}</b>" if r["rank"] == 1 else _e(r["model"]), _num(r["score"]),
                      _num(r["macro_f1"]), _num(r["macro_f1_norm"], 2), _num(r["onnx_cpu_ms"], 1),
                      _num(r["inference_time_norm"], 2), _num(r["model_size_mb"], 1), _num(r["model_size_norm"], 2),
                      "pass" if r["gradcam"] else "fail", _num(r["gpu_latency_ms"], 1)] for _, r in plain.iterrows()]))
    tie = ranked.sort_values("tie_rank")
    s.append(f"<h3>Tie rule{' (the recommendation)' if tie_on else ' (reported, not used)'}</h3><p>A model is tied "
             f"with the best macro-F1 model when it is not significantly worse by McNemar (Holm-corrected, alpha {alpha}) "
             f"and its mean macro-F1 is within one standard deviation of the best ({_num(tie['best_f1_std'].iloc[0])}). "
             "Tied models are ranked by inference time, model size and Grad-CAM only (same weights); the others "
             "follow in plain-score order.</p>")
    s.append(_table(["Rank", "Model", "Tied", "Macro-F1", "Gap to best", "McNemar p (Holm) vs best",
                     "Speed/size/Grad-CAM score", "Note"],
                    [[r["tie_rank"], f"<b>{_e(r['model'])}</b>" if r["tie_rank"] == 1 else _e(r["model"]),
                      "yes" if r["tied"] else "no", _num(r["macro_f1"]), _num(r["f1_gap"]),
                      _e(r["mcnemar_p_holm"]) if isinstance(r["mcnemar_p_holm"], str) and r["mcnemar_p_holm"] else "–",
                      _num(r["tie_score"]), _e(r["tie_note"])] for _, r in tie.iterrows()]))
    changed = any(x["changed"] for x in sensitivity)
    s.append("<h3>Sensitivity check (plain weighted score)</h3><p>Weights as macro-F1 / inference time / model size / "
             f"Grad-CAM (%). The winner <b>{'changes' if changed else 'does not change'}</b> under the alternative "
             "weights.</p>")
    s.append(_table(["Weights", "Winner", "Winner changed", "Order"],
                    [[" / ".join(str(round(100 * x["weights"][k])) for k in w), _e(x["winner_model"]),
                      "yes" if x["changed"] else "no", _e(x["order"])] for x in sensitivity]))

    # explainability and links
    s.append("<h2>Grad-CAM overlays</h2><p>One representative trained checkpoint per model, fp32. Pass means "
             "Grad-CAM ran and returned finite maps of the input size. Click a grid for full size; single overlays are "
             "in each model's folder.</p><div class='gallery'>")
    for arch in order:
        if arch not in profile.index:
            continue
        p = profile.loc[arch]
        status = "pass" if bool(p["gradcam_pass"]) else "fail"
        cap = (f"{_e(p['model'])}: <span class='{status}'>{status}</span> · layer {_e(p['gradcam_target_layer'])} · "
               f"{_e(p['gradcam_degenerate'] if isinstance(p['gradcam_degenerate'], str) else '')} · "
               f"{_e(p['profiled_run'])}")
        if isinstance(p["gradcam_grid"], str) and p["gradcam_grid"]:
            src = _rel(p["gradcam_grid"], out_dir)
            s.append(f"<figure><a href='{_e(src)}'><img src='{_e(src)}' alt='Grad-CAM {_e(p['model'])}'></a>"
                     f"<figcaption>{cap} · <a href='{_e(_rel(Path(p['gradcam_grid']).parent, out_dir))}/'>folder</a>"
                     f"</figcaption></figure>")
        else:
            s.append(f"<figure><figcaption>{cap}<br>{_e(p['gradcam_error'])}</figcaption></figure>")
    s.append("</div>")

    s.append("<h2>Confusion matrices</h2>")
    rows = []
    for _, r in per_seed.sort_values(["attribute", "arch", "fold", "seed"]).iterrows():
        cm = _rel(r["confusion_matrix"], out_dir)
        rows.append([_e(r["attribute"]), _e(r["model"]), r["fold"], r["seed"], _num(r["macro_f1"]),
                     f"<a href='{_e(cm)}'>PNG</a> · <a href='{_e(cm[:-4])}.csv'>CSV</a> · "
                     f"<a href='{_e(_rel(r['predictions'], out_dir))}'>predictions</a>"])
    s.append(_table(["Attribute", "Model", "Fold", "Seed", "Test macro-F1", "Files"], rows))
    s.append("<p class='meta'>Also in this folder: per_seed.csv, per_class.csv, summary.csv, attributes.json, "
             "per_source.csv, source_summary.csv and source_balance.csv (when there is a source column), stats.json, "
             "mcnemar.csv, model_profile.csv, ranking.csv, ranking.md, onnx/, checkpoints/, logs/.</p>")

    page = ("<!doctype html><html lang='en'><head><meta charset='utf-8'>"
            "<meta name='viewport' content='width=device-width, initial-scale=1'>"
            f"<title>Crab backbone comparison</title><style>{CSS}</style></head><body><main>"
            + "".join(s) + "</main></body></html>")
    path = out_dir / "report.html"
    path.write_text(page, encoding="utf-8")
    return path
