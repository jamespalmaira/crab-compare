"""Post-training analysis of a results folder (build steps 5-7).

Only the analysed attributes (selection.attributes: genus) are analysed;
other attributes in the folder (species, exploratory) stay in the CSVs but
get no statistics, Grad-CAM, ranking or report section. Statistics use every
run of the analysed attributes. Grad-CAM, ONNX and FLOPs are properties of the
architecture, so each model is checked once, on a representative trained
checkpoint: the first analysed attribute, lowest fold, lowest seed. Their
columns are added to per_seed.csv and summary.csv by model.
"""
import json
import logging
from pathlib import Path

import pandas as pd
import torch

from src.data import image_root
from src.export import check_onnx, export_onnx, flops, onnx_input_size, onnx_latency
from src.gradcam import run_gradcam, sample_predictions
from src.models import create_model
from src.select import selection_settings
from src.stats import run_stats

log = logging.getLogger(__name__)

PROFILE_COLUMNS = ["gmacs", "gflops", "onnx_cpu_ms", "onnx_size_mb", "onnx_outputs_match", "gradcam_pass"]


def representative_runs(per_seed, model_order, attribute_order):
    rank = {a: i for i, a in enumerate(attribute_order)}
    df = per_seed.assign(_attr=per_seed["attribute"].map(rank).fillna(len(rank)))
    reps = {}
    for arch in model_order:
        runs = df[df["arch"] == arch].sort_values(["_attr", "fold", "seed"])
        if len(runs):
            reps[arch] = runs.iloc[0]
    return reps


def load_trained(row, device):
    ckpt = torch.load(row["checkpoint"], map_location="cpu", weights_only=False)
    model = create_model(ckpt["arch"], len(ckpt["classes"]), pretrained=False)
    model.load_state_dict(ckpt["state_dict"])
    return model.to(device).eval(), ckpt


ANALYSIS_STEPS = ("stats", "gradcam", "onnx", "latency")  # ranking and report always run after these
GRADCAM_COLUMNS = ["gradcam_target_layer", "gradcam_pass", "gradcam_degenerate", "gradcam_n_images",
                   "gradcam_error", "gradcam_grid"]
ONNX_COLUMNS = ["gmacs", "gflops", "params_m_thop", "onnx_max_abs_diff", "onnx_outputs_match", "onnx_size_mb",
                "onnx_path"]
LATENCY_COLUMNS = ["onnx_cpu_ms", "onnx_cpu_iqr_ms", "onnx_cpu_medians_ms", "onnx_threads", "onnx_runs",
                   "onnx_warmup", "onnx_repeats", "ort_version"]
STEP_COLUMNS = {"gradcam": GRADCAM_COLUMNS, "onnx": ONNX_COLUMNS, "latency": LATENCY_COLUMNS}


def profile_models(cfg, out_dir, device, model_order, steps=ANALYSIS_STEPS):
    """Grad-CAM, ONNX export and parity check with FLOPs, and the ONNX CPU latency protocol, per model.

    Only the given steps are run; with an existing model_profile.csv, the columns of the other
    steps are kept from it, so e.g. Grad-CAM can be rerun without re-timing latency.
    """
    out_dir = Path(out_dir)
    attrs = selection_settings(cfg)[0]
    per_seed = pd.read_csv(out_dir / "per_seed.csv")
    per_seed = per_seed[per_seed["attribute"].isin(attrs)]
    if per_seed.empty:
        raise ValueError(f"No runs of the analysed attribute(s) {attrs} (selection.attributes) in {out_dir}")
    reps = representative_runs(per_seed, model_order, attrs)
    a = cfg.get("analysis", {})
    path = out_dir / "model_profile.csv"
    old = pd.read_csv(path).set_index("arch") if path.exists() else None
    # a step that is not run keeps its earlier columns (those present: an older analysis may lack some)
    missing = [s for s in STEP_COLUMNS if s not in steps and (old is None or STEP_COLUMNS[s][0] not in old.columns)]
    if missing:
        raise ValueError(f"Step(s) {missing} are not run and have no earlier results in {path}; run them too")
    images = image_root(cfg)  # the cache when data.cache_longer_side is set, as in training
    rows, to_time = [], {}
    for arch, rep in reps.items():
        name = cfg["models"][arch]["name"]
        row = {"arch": arch, "model": name, "profiled_run": f"{rep['attribute']} fold {rep['fold']} seed {rep['seed']}",
               "checkpoint": rep["checkpoint"]}
        if old is not None and arch in old.index:
            for s in STEP_COLUMNS:
                if s not in steps:
                    row.update({c: old.loc[arch, c] for c in STEP_COLUMNS[s] if c in old.columns})
        if "gradcam" in steps or "onnx" in steps:
            model, ckpt = load_trained(rep, device)
            input_size = tuple(ckpt["data_cfg"]["input_size"])
            log.info("Profiling %s using %s", arch, rep["checkpoint"])
            if "gradcam" in steps:
                sample = sample_predictions(rep["predictions"], a.get("gradcam_images", 8), seed=a.get("gradcam_seed", 0))
                cam = run_gradcam(model, arch, sample, images, ckpt["data_cfg"], cfg["augmentation"],
                                  out_dir / "gradcam" / rep["attribute"] / arch, device,
                                  f"{name} - {rep['attribute']} - fold {rep['fold']} seed {rep['seed']}")
                row.update({k: v for k, v in cam.items() if k != "arch"})
                log.info("%s: Grad-CAM %s (%s)", arch, "pass" if cam["gradcam_pass"] else "FAIL",
                         cam["gradcam_degenerate"])
            if "onnx" in steps:
                onnx_path = export_onnx(model, input_size, out_dir / "onnx" / f"{arch}.onnx", a.get("onnx_opset", 17))
                res = check_onnx(model, onnx_path, input_size, a.get("onnx_rtol", 0.001), a.get("onnx_atol", 0.0001))
                model.to(device)
                row.update(**flops(model, input_size), **res)
                log.info("%s: ONNX match %s max|diff| %.2e, %.1f MB | %.2f GFLOPs", arch, res["onnx_outputs_match"],
                         res["onnx_max_abs_diff"], res["onnx_size_mb"], row["gflops"])
            del model
            torch.cuda.empty_cache()
        if "latency" in steps:
            to_time[arch] = (row["onnx_path"], onnx_input_size(row["onnx_path"]))
        rows.append(row)
    if to_time:
        protocol = a["onnx_latency"]
        log.info("ONNX latency protocol: %s", protocol)
        per_repeat, summary = onnx_latency(to_time, protocol)
        per_repeat.to_csv(out_dir / "onnx_latency.csv", index=False)
        for row in rows:
            row.update(summary[row["arch"]])
    profile = pd.DataFrame(rows)
    profile.to_csv(path, index=False)
    return profile


def add_profile_columns(out_dir, profile):
    """Add per-model FLOPs, ONNX and Grad-CAM columns to per_seed.csv and summary.csv."""
    cols = profile.set_index("arch")[PROFILE_COLUMNS]
    for name in ("per_seed.csv", "summary.csv"):
        path = Path(out_dir) / name
        df = pd.read_csv(path).drop(columns=PROFILE_COLUMNS, errors="ignore")
        df.join(cols, on="arch").to_csv(path, index=False)


def analyze(cfg, out_dir, device, steps=ANALYSIS_STEPS):
    model_order = list(cfg["models"])
    if "stats" in steps:
        stats = run_stats(out_dir, model_order, cfg.get("analysis", {}).get("stats_alpha", 0.05),
                          attributes=selection_settings(cfg)[0])
    else:
        stats = json.loads((Path(out_dir) / "stats.json").read_text(encoding="utf-8"))
    profile = profile_models(cfg, out_dir, device, model_order, steps)
    add_profile_columns(out_dir, profile)
    return stats, profile
