"""Timing check on the real data: speed, GPU memory and data loading only. Not a result.

Checks that the real data loads for one attribute (images, groups, classes, class weights, fold
sizes, composition per fold, the external set held out with no observer on both sides), then
trains each model for 1 epoch on one fold with one seed, with the configured micro-batches,
effective batch and amp setting, and reports training and validation seconds, the test and
external evaluation seconds, peak GPU memory, and any out-of-memory or non-finite error.
From those it estimates the time of the full run. Metrics from a timing run must never be
reported or analysed: the run name must start with timing_.

    python scripts/timing_check.py --run-name timing_20261026
"""
import argparse
import gc
import json
import logging
import sys
import time
import warnings
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from run import apply_overrides, hold_out_external, select_attributes  # noqa: E402
from src.config import ConfigError, load_config  # noqa: E402
from src.data import class_weights, crab_finer, load_labels  # noqa: E402
from src.evaluate import evaluate_external, evaluate_run  # noqa: E402
from src.train import GPUOutOfMemory, NonFiniteLoss, limit_gpu_memory, resolve_device, train_one  # noqa: E402

log = logging.getLogger("timing")
MiB = 2**20


def check_data(cfg, df_all, rows, external, split):
    """Print what the bench will train on and assert the external set is kept out."""
    attr, classes = split["attribute"], split["classes"]
    log.info("%s: %d images, %d groups, classes %s", attr, len(rows), rows["crab_id"].nunique(), classes)
    log.info("groups per class (an observer with photos of two genera counts in both): %s",
             rows.groupby(attr)["crab_id"].nunique().to_dict())
    mixed = (rows.groupby("crab_id")[attr].nunique() > 1).sum()
    log.info("observers with photos of more than one %s: %d (each stays whole in one set)", attr, mixed)

    finer = (cfg["split"].get("stratify_by") or {}).get(attr)  # e.g. species within genus
    ext_ids = set(external["crab_id"]) | set(external.get("observer_id", pd.Series(dtype=str)))
    ext_paths = set(external["image_path"])
    for k, fold in enumerate(split["folds"]):
        sets = {name: set(ids) for name, ids in fold.items() if name in ("train", "val", "test")}
        names = list(sets)
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                assert not sets[a] & sets[b], f"fold {k}: {a} and {b} share groups"
        assert set().union(*sets.values()) == set(rows["crab_id"]), f"fold {k}: sets do not cover every group"
        parts = []
        for name, ids in sets.items():
            part = rows[rows["crab_id"].isin(ids)]
            assert not set(part["image_path"]) & ext_paths, f"fold {k} {name}: external image inside"
            assert not (set(part["crab_id"]) | set(part.get("observer_id", pd.Series(dtype=str)))) & ext_ids, \
                f"fold {k} {name}: an external observer is inside"
            groups = part.groupby(attr)["crab_id"].nunique().reindex(classes, fill_value=0).to_dict()
            parts.append(f"{name} {len(ids)} groups/{len(part)} imgs, groups by photo {attr} {groups}")
        log.info("fold %d: %s", k, " | ".join(parts))
        if finer:
            for cls in classes:
                comp = []
                for name, ids in sets.items():
                    part = rows[rows["crab_id"].isin(ids) & (rows[attr] == cls)]
                    sp = Counter(part[finer].where(part[finer] != "", "(genus only)"))
                    comp.append(f"{name} " + ", ".join(f"{s} {n}" for s, n in sorted(sp.items())))
                log.info("    %s photos by %s: %s", cls, finer, " | ".join(comp))
    log.info("external (%s): %d images, %d groups; no external image or observer in any fold: OK",
             cfg["data"]["external_test"].get("name", "external"), len(external), external["crab_id"].nunique())

    train0 = rows[rows["crab_id"].isin(split["folds"][0]["train"])]
    targets = train0[attr].map({c: i for i, c in enumerate(classes)}).to_numpy()
    w = class_weights(targets, len(classes))
    log.info("class weights, fold 0 train (inverse frequency): %s",
             {c: round(float(x), 3) for c, x in zip(classes, w)})


def time_model(cfg, arch, rows, split, external, out_dir, device, no_eval=False):
    out = {"arch": arch, "model": cfg["models"][arch]["name"], "batch_size": cfg["models"][arch]["batch_size"],
           "status": "ok"}
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    try:
        res = train_one(cfg, arch, rows, split, 0, cfg["training"]["seeds"][0], out_dir, device)
        hist = res.info["history"]
        for h in hist:
            out[f"train_s_e{h['epoch']}"], out[f"val_s_e{h['epoch']}"] = h["train_s"], h["val_s"]
        steady = hist[1:] or hist  # epoch 1 carries the one-time data-loader start-up
        out.update(train_s=float(np.mean([h["train_s"] for h in steady])),
                   val_s=float(np.mean([h["val_s"] for h in steady])),
                   startup_s=(hist[0]["train_s"] + hist[0]["val_s"])
                   - float(np.mean([h["train_s"] + h["val_s"] for h in steady])) if len(hist) > 1 else 0.0,
                   n_train=res.info["n_train"], n_val=res.info["n_val"],
                   steps_per_epoch=res.info["steps_per_epoch"], accumulation_steps=res.info["accumulation_steps"],
                   num_workers=cfg["data"]["num_workers"], cache=cfg["data"].get("cache_longer_side") or 0)
        if device.type == "cuda":
            out.update(peak_alloc_mib=torch.cuda.max_memory_allocated() / MiB,
                       peak_reserved_mib=torch.cuda.max_memory_reserved() / MiB)
        out["eval_s"] = float("nan")
        if not no_eval:
            start = time.perf_counter()
            evaluate_run(res, cfg, out_dir, device)  # test set, latency, model size: timed, not reported
            evaluate_external(res.model, res.data_cfg, res.info, external, cfg, out_dir, device,
                              cfg["models"][arch]["batch_size"])
            out["eval_s"] = time.perf_counter() - start
        del res
    except GPUOutOfMemory as e:
        out["status"] = f"out of memory: {e}"
    except NonFiniteLoss as e:
        out["status"] = f"non-finite: {e}"
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return out


def estimate(timings, n_folds, n_seeds, max_epochs):
    """Upper bound (every run to max_epochs) and a likely figure (early stop at about half).
    Per run: the one-time start-up, steady-state epochs, then test and external evaluation."""
    rows = []
    for t in timings:
        if t["status"] != "ok":
            continue
        epoch = t["train_s"] + t["val_s"]
        runs = n_folds * n_seeds
        fixed = t.get("startup_s", 0.0) + t["eval_s"]
        rows.append({"model": t["model"], "epoch_s": epoch, "runs": runs,
                     "upper_h": runs * (epoch * max_epochs + fixed) / 3600,
                     "likely_h": runs * (epoch * max_epochs / 2 + fixed) / 3600})
    return pd.DataFrame(rows)


def main(argv=None):
    warnings.filterwarnings("ignore", category=UserWarning)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(ROOT / "config.yaml"))
    ap.add_argument("--run-name", required=True, help="must start with timing_")
    ap.add_argument("--attribute", default=None, help="default: data.main_attribute")
    ap.add_argument("--models", nargs="+", default=["all"])
    ap.add_argument("--epochs", type=int, default=1, help="epochs per model; from 2, epoch 1 is reported apart as "
                    "start-up and the estimate uses the mean of the later epochs")
    ap.add_argument("--num-workers", type=int, default=None, help="override data.num_workers")
    ap.add_argument("--cache-longer-side", type=int, default=None, help="override data.cache_longer_side (0 = off)")
    ap.add_argument("--no-eval", action="store_true", help="skip the timed test and external evaluation")
    ap.add_argument("--data-only", action="store_true", help="only the data check, no training")
    args = ap.parse_args(argv)
    if not args.run_name.startswith("timing_"):
        raise SystemExit("--run-name must start with timing_ (timing runs are excluded from any analysis)")

    cfg = load_config(args.config)
    full = {"seeds": list(cfg["training"]["seeds"]), "epochs": cfg["training"]["epochs"],
            "n_folds": cfg["split"]["n_folds"]}
    apply_overrides(cfg, argparse.Namespace(labels=None, run_name=args.run_name, epochs=args.epochs,
                                            no_pretrained=False, num_workers=args.num_workers,
                                            min_groups_per_class=None, seeds=1))
    if args.cache_longer_side is not None:
        cfg["data"]["cache_longer_side"] = args.cache_longer_side or None
    attr = args.attribute or cfg["data"].get("main_attribute", "genus")
    out_dir = Path(cfg.get("results_dir", "results")) / cfg["run_name"]
    df = load_labels(cfg["data"]["labels_csv"])
    internal, external = hold_out_external(cfg, df, out_dir)
    rows, split = select_attributes(cfg, internal, [attr], out_dir)[attr]
    check_data(cfg, df, rows, external, split)
    if args.data_only:
        return 0

    device = resolve_device(cfg)
    limit_gpu_memory(cfg, device)
    models = list(cfg["models"]) if args.models == ["all"] else args.models
    log.info("timing: %s | fold 0 | seed %s | 1 epoch | pretrained %s | amp %s | effective batch %d",
             models, cfg["training"]["seeds"], cfg["training"]["pretrained"], cfg["training"]["amp"],
             cfg["training"]["effective_batch_size"])
    timings = []
    for arch in models:
        timings.append(time_model(cfg, arch, rows, split, external, out_dir, device, args.no_eval))
        log.info("%s", json.dumps(timings[-1]))
        pd.DataFrame(timings).to_csv(out_dir / "timing.csv", index=False)

    est = estimate(timings, full["n_folds"], len(full["seeds"]), full["epochs"])
    est.to_csv(out_dir / "timing_estimate.csv", index=False)
    print(f"\nTiming per model ({args.epochs} epoch(s), fold 0, seed 0, num_workers {cfg['data']['num_workers']}, "
          f"cache {cfg['data'].get('cache_longer_side') or 'off'}). train_s/val_s: mean of epochs 2+ "
          "(epoch 1 if only one); startup_s: epoch 1 minus that")
    cols = [c for c in pd.DataFrame(timings).columns if c not in ("arch", "n_train", "n_val", "steps_per_epoch")]
    print(pd.DataFrame(timings)[cols].round(1).to_string(index=False))
    print(f"\nFull run: {len(models)} models x {full['n_folds']} folds x {len(full['seeds'])} seeds "
          f"{full['seeds']} x up to {full['epochs']} epochs")
    print(est.round(2).to_string(index=False))
    print(f"TOTAL upper bound {est['upper_h'].sum():.1f} h, likely (early stop at ~{full['epochs'] // 2} epochs) "
          f"{est['likely_h'].sum():.1f} h")
    return 0 if all(t["status"] == "ok" for t in timings) else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except ConfigError as e:
        log.error("%s", e)
        sys.exit(2)
