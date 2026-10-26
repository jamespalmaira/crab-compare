"""Post-training analysis of a results folder (build steps 5-7).

Statistics use every run. Grad-CAM, ONNX and FLOPs are properties of the
architecture, so each model is checked once, on a representative trained
checkpoint: the first attribute (config order), lowest fold, lowest seed.
Their columns are added to per_seed.csv and summary.csv by model.
"""
import logging
from pathlib import Path

import pandas as pd
import torch

from src.export import check_onnx, export_onnx, flops
from src.gradcam import run_gradcam, sample_predictions
from src.models import create_model
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


def profile_models(cfg, out_dir, device, model_order):
    out_dir = Path(out_dir)
    per_seed = pd.read_csv(out_dir / "per_seed.csv")
    reps = representative_runs(per_seed, model_order, cfg["data"]["attributes"])
    a, ev = cfg.get("analysis", {}), cfg.get("evaluation", {})
    image_root = Path(cfg["data"]["labels_csv"]).parent
    rows = []
    for arch, rep in reps.items():
        model, ckpt = load_trained(rep, device)
        input_size = tuple(ckpt["data_cfg"]["input_size"])
        name = cfg["models"][arch]["name"]
        log.info("Profiling %s using %s", arch, rep["checkpoint"])

        sample = sample_predictions(rep["predictions"], a.get("gradcam_images", 8), seed=a.get("gradcam_seed", 0))
        cam = run_gradcam(model, arch, sample, image_root, ckpt["data_cfg"], cfg["augmentation"],
                          out_dir / "gradcam" / rep["attribute"] / arch, device,
                          f"{name} - {rep['attribute']} - fold {rep['fold']} seed {rep['seed']}")

        onnx_path = export_onnx(model, input_size, out_dir / "onnx" / f"{arch}.onnx", a.get("onnx_opset", 17))
        onnx_res = check_onnx(model, onnx_path, input_size, a.get("onnx_rtol", 0.001), a.get("onnx_atol", 0.0001),
                              ev.get("latency_warmup", 10), ev.get("latency_runs", 50))
        model.to(device)
        row = {"arch": arch, "model": name, "profiled_run": f"{rep['attribute']} fold {rep['fold']} seed {rep['seed']}",
               "checkpoint": rep["checkpoint"], **flops(model, input_size), **onnx_res, **cam}
        log.info("%s: Grad-CAM %s (%s) | ONNX match %s max|diff| %.2e, CPU %.2f ms, %.1f MB | %.2f GFLOPs",
                 arch, "pass" if cam["gradcam_pass"] else "FAIL", cam["gradcam_degenerate"],
                 onnx_res["onnx_outputs_match"], onnx_res["onnx_max_abs_diff"], onnx_res["onnx_cpu_ms"],
                 onnx_res["onnx_size_mb"], row["gflops"])
        rows.append(row)
        del model
        torch.cuda.empty_cache()
    profile = pd.DataFrame(rows)
    profile.to_csv(out_dir / "model_profile.csv", index=False)
    return profile


def add_profile_columns(out_dir, profile):
    """Add per-model FLOPs, ONNX and Grad-CAM columns to per_seed.csv and summary.csv."""
    cols = profile.set_index("arch")[PROFILE_COLUMNS]
    for name in ("per_seed.csv", "summary.csv"):
        path = Path(out_dir) / name
        df = pd.read_csv(path).drop(columns=PROFILE_COLUMNS, errors="ignore")
        df.join(cols, on="arch").to_csv(path, index=False)


def analyze(cfg, out_dir, device):
    model_order = list(cfg["models"])
    stats = run_stats(out_dir, model_order, cfg.get("analysis", {}).get("stats_alpha", 0.05))
    profile = profile_models(cfg, out_dir, device, model_order)
    add_profile_columns(out_dir, profile)
    return stats, profile
