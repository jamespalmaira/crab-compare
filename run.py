"""Crab model comparison: command-line entry point.

    python run.py --config config.yaml --models all --seeds 3           # main attribute (genus)
    python run.py --config config.yaml --attribute species --models all  # secondary, exploratory
    python run.py --config config.yaml --attribute all --models resnet50 efficientnet_b3

Trains every selected model on the same crab-level split (every fold of it
for k-fold CV) for each training seed, keeping the best-val-macro-F1
checkpoint per run under results/<run_name>/.
"""
import argparse
import gc
import json
import logging
import sys
import time
import warnings
from pathlib import Path

import pandas as pd
import torch

from src.config import ConfigError, load_config
from src.analyze import ANALYSIS_STEPS, analyze
from src.data import (ATTRIBUTES, GENUS_ONLY, SPLIT_SETS, SplitError, data_settings, get_split, load_labels,
                      prepare_attribute, source_balance, split_external)
from src.select import write_ranking
from src.evaluate import evaluate_external, evaluate_run
from src.report import merge_rows, write_report, write_tables
from src.train import GPUOutOfMemory, NonFiniteLoss, limit_gpu_memory, resolve_device, train_one

log = logging.getLogger("run")


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="Train and compare five backbones on the crab dataset.")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--attribute", default=None, choices=list(ATTRIBUTES) + ["all"],
                    help="default: data.main_attribute in config.yaml (genus)")
    ap.add_argument("--models", nargs="+", default=["all"], help="'all' or timm model names from config.yaml")
    ap.add_argument("--seeds", type=int, default=None, help="use the first N seeds in training.seeds (default: all)")
    ap.add_argument("--labels", default=None, help="override data.labels_csv")
    ap.add_argument("--run-name", default=None, help="override run_name (output folder under results/)")
    ap.add_argument("--epochs", type=int, default=None, help="override training.epochs (e.g. 1 for a smoke test)")
    ap.add_argument("--no-pretrained", action="store_true", help="random init instead of downloading weights")
    ap.add_argument("--num-workers", type=int, default=None, help="override data.num_workers")
    ap.add_argument("--folds", type=int, nargs="+", default=None, help="only these fold indices (k-fold)")
    ap.add_argument("--min-groups-per-class", type=int, default=None, help="override data.min_groups_per_class")
    ap.add_argument("--splits-only", action="store_true", help="make/check the splits and stop")
    ap.add_argument("--analyze-only", action="store_true",
                    help="skip training; run statistics, Grad-CAM, ONNX and FLOPs on results/<run_name>/")
    ap.add_argument("--analyze-steps", nargs="+", default=list(ANALYSIS_STEPS), choices=list(ANALYSIS_STEPS),
                    help="with --analyze-only: run only these analysis steps (default: all); the others are kept "
                         "from the earlier analysis. Ranking and report.html are always rebuilt")
    return ap.parse_args(argv)


def run_analysis(cfg, out_dir, device, steps=ANALYSIS_STEPS):
    log.info("Analysis steps: %s", ", ".join(steps))
    stats, profile = analyze(cfg, out_dir, device, steps)
    for attribute, st in stats["attributes"].items():
        fr = st["friedman"]
        log.info("%s Friedman over %d %s(s): chi2 %s p %s (%s)", attribute, fr["n_blocks"], fr["block"],
                 fr["statistic"], fr["p_value"], fr["note"])
        for r in st["mcnemar"]["pairs"]:
            log.info("%s McNemar %s vs %s: %d/%d discordant, p raw %.4f, Holm %.4f", attribute, r["model_a"],
                     r["model_b"], r["a_right_b_wrong"], r["a_wrong_b_right"], r["p_raw"], r["p_holm"])
    log.info("Wrote stats.json, mcnemar.csv, model_profile.csv, gradcam/ and onnx/ in %s", out_dir)

    ranked, sens = write_ranking(cfg, out_dir, profile)
    for _, r in ranked.iterrows():
        log.info("tie-rule rank %d  %-16s tied %-5s | plain rank %d score %.3f | macro-F1 %.4f (%s)", r["tie_rank"],
                 r["model"], r["tied"], r["rank"], r["score"], r["macro_f1"], r["tie_note"])
    log.info("Recommended backbone: %s (%s)", ranked.loc[0, "model"], ranked.loc[0, "arch"])
    for s in sens:
        log.info("sensitivity %s -> winner %s%s", s["weights"], s["winner_model"], " (CHANGED)" if s["changed"] else "")
    log.info("Wrote ranking.csv and ranking.md in %s", out_dir)

    log.info("Wrote %s", write_report(cfg, out_dir, ranked, sens))
    return ranked, sens


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
    if args.min_groups_per_class is not None:
        cfg["data"]["min_groups_per_class"] = args.min_groups_per_class
    seeds = cfg["training"]["seeds"]
    if args.seeds is not None:
        if not 1 <= args.seeds <= len(seeds):
            raise ConfigError(f"--seeds {args.seeds}: config.yaml lists {len(seeds)} seeds {seeds}")
        cfg["training"]["seeds"] = seeds[:args.seeds]


def run_order(attr, models, folds, seeds):
    """Seed first, then fold, then model: each finished seed is a full comparison of every model on
    every fold, so a run stopped part-way still leaves complete seeds to analyse."""
    return [(attr, arch, fold, seed) for seed in seeds for fold in folds for arch in models]


def run_settings(cfg, arch):
    """What a finished run depends on besides its (attribute, arch, fold, seed) key. A completion
    marker with different settings is not resumed: the results would not be comparable."""
    return {"training": {k: v for k, v in cfg["training"].items() if k != "seeds"},
            "augmentation": cfg["augmentation"], "split": cfg["split"], "model": cfg["models"][arch],
            "labels_csv": str(cfg["data"]["labels_csv"]), "cache_longer_side": cfg["data"].get("cache_longer_side"),
            "external_test": cfg["data"].get("external_test")}


def marker_path(out_dir, attr, arch, fold, seed):
    return Path(out_dir) / "completed" / attr / f"{arch}_fold{fold}_seed{seed}.json"


def completed(cfg, out_dir, attr, arch, fold, seed):
    """True if this combination already finished in out_dir with the same settings."""
    path = marker_path(out_dir, attr, arch, fold, seed)
    if not path.exists():
        return False
    saved = json.loads(path.read_text(encoding="utf-8"))["settings"]
    if saved != json.loads(json.dumps(run_settings(cfg, arch))):
        raise ConfigError(f"{path} was made with different settings; use a new --run-name, or delete the "
                          "folder's completed/ markers to retrain everything")
    return True


def mark_completed(cfg, out_dir, attr, arch, fold, seed, seconds, rows):
    """data_settings is a record only, kept apart from settings: completed() compares settings, and adding
    to them would make every marker written before this field existed count as different."""
    path = marker_path(out_dir, attr, arch, fold, seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps({"attribute": attr, "arch": arch, "fold": fold, "seed": seed,
                               "seconds": round(seconds, 1), "settings": run_settings(cfg, arch),
                               "data_settings": data_settings(cfg, rows, attr)}, indent=1),
                   encoding="utf-8")
    tmp.replace(path)  # written only after the run's rows are in the tables


def save_attribute_usage(out_dir, usages):
    """Record which attributes were used or skipped (merged with earlier runs into this folder)."""
    path = Path(out_dir) / "attributes.json"
    saved = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    saved.update(usages)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(saved, indent=1), encoding="utf-8")


def hold_out_external(cfg, df, out_dir):
    """(internal rows, external rows or None). With data.external_test, the external rows and every
    observer who appears in them are taken out before any split is made; external_split.json
    records the counts."""
    ext = cfg["data"].get("external_test")
    if not ext:
        return df, None
    internal, external, info = split_external(df, ext)
    info["name"] = ext.get("name", "external")
    path = Path(out_dir) / "external_split.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(info, indent=1), encoding="utf-8")
    return internal, external


def select_attributes(cfg, df, attributes, out_dir):
    """Labeled rows and the saved split for every usable attribute; skipped ones are logged."""
    min_groups = cfg["data"].get("min_groups_per_class", 30)
    threshold = cfg["data"].get("source_dominance_threshold", 0.8)
    splits, usages, balances = {}, {}, []
    for attr in attributes:
        rows, usages[attr] = prepare_attribute(df, attr, min_groups)
        if usages[attr]["skip_reason"]:
            continue
        balances.append(source_balance(rows, attr, threshold))
        splits[attr] = (rows, get_split(cfg, rows, attr))
        summarise_split(splits[attr][1], rows)
    save_attribute_usage(out_dir, usages)
    balances = [b for b in balances if not b.empty]
    if balances:
        merge_rows(Path(out_dir) / "source_balance.csv", pd.concat(balances, ignore_index=True),
                   keys=["attribute", "class"])
    if not splits:
        raise ConfigError(f"No attribute has enough labels to train ({attributes}); see the skip messages above")
    return splits


def groups_per_photo_class(rows, ids, attr, classes):
    """Groups per class, by each photo's own label: a group with photos of two classes counts in both."""
    part = rows[rows["crab_id"].isin(ids)]
    return {k: int(v) for k, v in part.groupby(attr)["crab_id"].nunique().reindex(classes, fill_value=0).items()}


def summarise_split(split, rows):
    """Log what each fold holds. Reporting only: the split itself stratifies a group on its most common label."""
    attr, classes = split["attribute"], split["classes"]
    finer = split.get("stratify_by")
    log.info("%s: %d groups, %d images, classes %s, strategy=%s, stratified=%s%s; groups per class are counted by "
             "each photo's own %s, so a group with photos of two classes counts in both",
             attr, split["n_crabs"], len(rows), classes, split["strategy"], split["stratified"],
             f" on {split['stratified_on']}" if finer else "", attr)
    for k, fold in enumerate(split["folds"]):
        parts = []
        for name in SPLIT_SETS:
            ids = fold[name]
            n_img = rows["crab_id"].isin(ids).sum()
            parts.append(f"{name} {len(ids)} groups/{n_img} imgs {groups_per_photo_class(rows, ids, attr, classes)}")
        log.info("  fold %d: %s", k, " | ".join(parts))
        if finer:  # each class's composition by the finer key (groups per species), per set
            for cls in classes:
                comp = []
                for name in SPLIT_SETS:
                    part = rows[rows["crab_id"].isin(fold[name]) & (rows[attr] == cls)]
                    fine = part[finer].where(part[finer] != "", GENUS_ONLY)
                    groups = part.groupby(fine)["crab_id"].nunique()
                    comp.append(f"{name} " + ", ".join(f"{v} {n}" for v, n in groups.items()))
                log.info("    %s by %s: %s", cls, finer, " | ".join(comp))


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
        attribute = args.attribute or cfg["data"].get("main_attribute", "genus")
        attributes = cfg["data"]["attributes"] if attribute == "all" else [attribute]
        out_dir = Path(cfg.get("results_dir", "results")) / cfg["run_name"]

        if args.analyze_only:
            if cfg["run_name"].startswith("timing_"):
                raise ConfigError(f"{cfg['run_name']} is a timing run (speed and memory only); "
                                  "timing runs are excluded from any analysis")
            if not (out_dir / "per_seed.csv").exists():
                raise FileNotFoundError(f"No per_seed.csv in {out_dir}; train first or check --run-name")
            device = resolve_device(cfg)
            limit_gpu_memory(cfg, device)
            run_analysis(cfg, out_dir, device, tuple(args.analyze_steps))
            return 0

        df = load_labels(cfg["data"]["labels_csv"])
        df, external = hold_out_external(cfg, df, out_dir)
        splits = select_attributes(cfg, df, attributes, out_dir)
        if args.splits_only:
            return 0

        device = resolve_device(cfg)
        limit_gpu_memory(cfg, device)
        seeds = cfg["training"]["seeds"]
        log.info("Run %s -> %s | models %s | seeds %s | epochs %d | pretrained %s | amp %s",
                 cfg["run_name"], out_dir, models, seeds, cfg["training"]["epochs"],
                 cfg["training"]["pretrained"], cfg["training"]["amp"])

        run_rows, class_rows, source_rows, external_rows = [], [], [], []
        plan = []
        for attr, (rows, split) in splits.items():
            folds = args.folds if args.folds is not None else range(len(split["folds"]))
            for fold in folds:
                if not 0 <= fold < len(split["folds"]):
                    raise ConfigError(f"--folds {fold}: split has {len(split['folds'])} fold(s)")
            plan += run_order(attr, models, folds, seeds)
        todo = [p for p in plan if not completed(cfg, out_dir, *p)]
        if len(todo) < len(plan):
            log.info("Resuming %s: %d of %d run(s) already completed, skipped", out_dir, len(plan) - len(todo),
                     len(plan))
        durations = []
        for attr, arch, fold, seed in todo:
            rows, split = splits[attr]
            started = time.perf_counter()
            res = train_one(cfg, arch, rows, split, fold, seed, out_dir, device)
            row, per_class, per_source = evaluate_run(res, cfg, out_dir, device)
            run_rows.append(row)
            class_rows.append(per_class)
            source_rows.append(per_source)
            if external is not None:
                external_rows += evaluate_external(res.model, res.data_cfg, res.info, external, cfg,
                                                   out_dir, device, cfg["models"][arch]["batch_size"])
            # write after every run so finished results survive a later failure
            write_tables(out_dir, run_rows, class_rows, list(cfg["models"]), source_rows, external_rows)
            durations.append(time.perf_counter() - started)
            mark_completed(cfg, out_dir, attr, arch, fold, seed, durations[-1], rows)
            left = len(todo) - len(durations)
            eta = int(sum(durations) / len(durations) * left / 60)
            log.info("Progress: %d/%d runs done (%d this session), last %.1f min, ETA %dh%02dm for %d left",
                     len(plan) - left, len(plan), len(durations), durations[-1] / 60, eta // 60, eta % 60, left)
            if not [p for p in todo[len(durations):] if p[0] == attr and p[3] == seed]:
                log.info("Seed %s of %s complete: every fold and model done, a full comparison", seed, attr)
            del res
            gc.collect()
            torch.cuda.empty_cache()

        for r in run_rows:
            log.info("%-10s %-24s fold %d seed %d | val F1 %.4f (epoch %d/%d) | test acc %.4f macro-F1 %.4f",
                     r["attribute"], r["arch"], r["fold"], r["seed"], r["best_val_macro_f1"], r["best_epoch"],
                     r["epochs_trained"], r["accuracy"], r["macro_f1"])
        log.info("Wrote %s, %s and %s", out_dir / "per_seed.csv", out_dir / "per_class.csv", out_dir / "summary.csv")
        run_analysis(cfg, out_dir, device)
    except (ConfigError, SplitError, FileNotFoundError, ValueError, GPUOutOfMemory, NonFiniteLoss) as e:
        log.error("%s", e)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
