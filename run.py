"""Crab model comparison: command-line entry point.

    python run.py --config config.yaml --attribute species --models all --seeds 3
    python run.py --config config.yaml --attribute all --models resnet50 efficientnet_b3

Trains every selected model on the same crab-level split (every fold of it
for k-fold CV) for each training seed, keeping the best-val-macro-F1
checkpoint per run under results/<run_name>/.
"""
import argparse
import gc
import logging
import sys
import warnings
from collections import Counter
from pathlib import Path

import torch

from src.config import ConfigError, load_config
from src.data import ATTRIBUTES, SPLIT_SETS, SplitError, attribute_rows, get_split, load_labels
from src.evaluate import evaluate_run
from src.report import write_tables
from src.train import GPUOutOfMemory, NonFiniteLoss, limit_gpu_memory, resolve_device, train_one

log = logging.getLogger("run")


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="Train and compare five backbones on the crab dataset.")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--attribute", default="species", choices=list(ATTRIBUTES) + ["all"])
    ap.add_argument("--models", nargs="+", default=["all"], help="'all' or timm model names from config.yaml")
    ap.add_argument("--seeds", type=int, default=None, help="use the first N seeds in training.seeds (default: all)")
    ap.add_argument("--labels", default=None, help="override data.labels_csv")
    ap.add_argument("--run-name", default=None, help="override run_name (output folder under results/)")
    ap.add_argument("--epochs", type=int, default=None, help="override training.epochs (e.g. 1 for a smoke test)")
    ap.add_argument("--no-pretrained", action="store_true", help="random init instead of downloading weights")
    ap.add_argument("--num-workers", type=int, default=None, help="override data.num_workers")
    ap.add_argument("--folds", type=int, nargs="+", default=None, help="only these fold indices (k-fold)")
    ap.add_argument("--splits-only", action="store_true", help="make/check the splits and stop")
    return ap.parse_args(argv)


def apply_overrides(cfg, args):
    if args.labels:
        cfg["data"]["labels_csv"] = args.labels
    if args.run_name:
        cfg["run_name"] = args.run_name
    if args.epochs is not None:
        cfg["training"]["epochs"] = args.epochs
    if args.no_pretrained:
        cfg["training"]["pretrained"] = False
    if args.num_workers is not None:
        cfg["data"]["num_workers"] = args.num_workers
    seeds = cfg["training"]["seeds"]
    if args.seeds is not None:
        if not 1 <= args.seeds <= len(seeds):
            raise ConfigError(f"--seeds {args.seeds}: config.yaml lists {len(seeds)} seeds {seeds}")
        cfg["training"]["seeds"] = seeds[:args.seeds]


def summarise_split(split, rows):
    attr = split["attribute"]
    crab_class = rows.groupby("crab_id")[attr].first()
    log.info("%s: %d crabs, %d images, classes %s, strategy=%s, stratified=%s",
             attr, split["n_crabs"], len(rows), split["classes"], split["strategy"], split["stratified"])
    for k, fold in enumerate(split["folds"]):
        parts = []
        for name in SPLIT_SETS:
            ids = fold[name]
            n_img = rows["crab_id"].isin(ids).sum()
            per_class = dict(sorted(Counter(crab_class[ids]).items()))
            parts.append(f"{name} {len(ids)} crabs/{n_img} imgs {per_class}")
        log.info("  fold %d: %s", k, " | ".join(parts))


def main(argv=None):
    warnings.filterwarnings("ignore", category=UserWarning)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = parse_args(argv)
    try:
        cfg = load_config(args.config)
        if cfg["training"].get("amp"):
            log.warning("training.amp is true. The project decision is amp: false for all five models "
                        "(see handoff.md): on the GTX 1650, AMP was slower, and a cuDNN fp16 bug gives "
                        "batch-size-dependent NaNs in ResNet-50.")
        apply_overrides(cfg, args)
        models = list(cfg["models"]) if args.models == ["all"] else args.models
        unknown = [m for m in models if m not in cfg["models"]]
        if unknown:
            raise ConfigError(f"Models not in config.yaml: {unknown}")
        attributes = cfg["data"]["attributes"] if args.attribute == "all" else [args.attribute]

        df = load_labels(cfg["data"]["labels_csv"])
        splits = {}
        for attr in attributes:
            rows = attribute_rows(df, attr)
            splits[attr] = (rows, get_split(cfg, rows, attr))
            summarise_split(splits[attr][1], rows)
        if args.splits_only:
            return 0

        device = resolve_device(cfg)
        limit_gpu_memory(cfg, device)
        out_dir = Path(cfg.get("results_dir", "results")) / cfg["run_name"]
        seeds = cfg["training"]["seeds"]
        log.info("Run %s -> %s | models %s | seeds %s | epochs %d | pretrained %s | amp %s",
                 cfg["run_name"], out_dir, models, seeds, cfg["training"]["epochs"],
                 cfg["training"]["pretrained"], cfg["training"]["amp"])

        run_rows, class_rows = [], []
        for attr in attributes:
            rows, split = splits[attr]
            folds = args.folds if args.folds is not None else range(len(split["folds"]))
            for arch in models:
                for fold in folds:
                    if not 0 <= fold < len(split["folds"]):
                        raise ConfigError(f"--folds {fold}: split has {len(split['folds'])} fold(s)")
                    for seed in seeds:
                        res = train_one(cfg, arch, rows, split, fold, seed, out_dir, device)
                        row, per_class = evaluate_run(res, cfg, out_dir, device)
                        run_rows.append(row)
                        class_rows.append(per_class)
                        # write after every run so finished results survive a later failure
                        write_tables(out_dir, run_rows, class_rows, model_order=list(cfg["models"]))
                        del res
                        gc.collect()
                        torch.cuda.empty_cache()
    except (ConfigError, SplitError, FileNotFoundError, ValueError, GPUOutOfMemory, NonFiniteLoss) as e:
        log.error("%s", e)
        return 2

    for r in run_rows:
        log.info("%-10s %-24s fold %d seed %d | val F1 %.4f (epoch %d/%d) | test acc %.4f macro-F1 %.4f",
                 r["attribute"], r["arch"], r["fold"], r["seed"], r["best_val_macro_f1"], r["best_epoch"],
                 r["epochs_trained"], r["accuracy"], r["macro_f1"])
    log.info("Wrote %s, %s and %s", out_dir / "per_seed.csv", out_dir / "per_class.csv", out_dir / "summary.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
