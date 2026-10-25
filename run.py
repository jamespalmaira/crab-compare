"""Crab model comparison: command-line entry point.

    python run.py --config config.yaml --attribute species --models all --seeds 3
    python run.py --config config.yaml --attribute all --models resnet50 efficientnet_b3

Build step 1 only prepares and checks the crab-level splits. Training is
added in build step 3.
"""
import argparse
import logging
import sys
from collections import Counter

from src.config import ConfigError, load_config
from src.data import ATTRIBUTES, SPLIT_SETS, SplitError, attribute_rows, get_split, load_labels

log = logging.getLogger("run")


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="Train and compare five backbones on the crab dataset.")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--attribute", default="species", choices=list(ATTRIBUTES) + ["all"])
    ap.add_argument("--models", nargs="+", default=["all"], help="'all' or timm model names from config.yaml")
    ap.add_argument("--seeds", type=int, default=None, help="number of training seeds (default: all in config)")
    ap.add_argument("--labels", default=None, help="override data.labels_csv")
    return ap.parse_args(argv)


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
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = parse_args(argv)
    try:
        cfg = load_config(args.config)
        if cfg["training"].get("amp"):
            log.warning("training.amp is true. The project decision is amp: false for all five models "
                        "(see handoff.md): on the GTX 1650, AMP was slower, and a cuDNN fp16 bug gives "
                        "batch-size-dependent NaNs in ResNet-50.")
        if args.labels:
            cfg["data"]["labels_csv"] = args.labels
        models = list(cfg["models"]) if args.models == ["all"] else args.models
        unknown = [m for m in models if m not in cfg["models"]]
        if unknown:
            raise ConfigError(f"Models not in config.yaml: {unknown}")
        attributes = cfg["data"]["attributes"] if args.attribute == "all" else [args.attribute]

        df = load_labels(cfg["data"]["labels_csv"])
        for attr in attributes:
            rows = attribute_rows(df, attr)
            summarise_split(get_split(cfg, rows, attr), rows)
    except (ConfigError, SplitError, FileNotFoundError, ValueError) as e:
        log.error("%s", e)
        return 2

    log.info("Splits ready for models %s. Training is not built yet (build step 3).", models)
    return 0


if __name__ == "__main__":
    sys.exit(main())
