"""Load and validate config.yaml."""
from pathlib import Path

import yaml

STRATEGIES = ("holdout", "kfold")


class ConfigError(ValueError):
    pass


def load_config(path):
    path = Path(path)
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    validate(cfg)
    return cfg


def validate(cfg):
    split = cfg["split"]
    if split["strategy"] not in STRATEGIES:
        raise ConfigError(f"split.strategy must be one of {STRATEGIES}, got {split['strategy']!r}")
    ratios = split["ratios"]
    if len(ratios) != 3 or abs(sum(ratios) - 1.0) > 1e-6 or min(ratios) <= 0:
        raise ConfigError(f"split.ratios must be three positive numbers summing to 1, got {ratios}")
    if split["strategy"] == "kfold" and split["n_folds"] < 2:
        raise ConfigError("split.n_folds must be at least 2")

    eff = cfg["training"]["effective_batch_size"]
    for arch, m in cfg["models"].items():
        bs = m["batch_size"]
        if bs < 1 or eff % bs != 0:
            raise ConfigError(
                f"models.{arch}.batch_size={bs} must divide training.effective_batch_size={eff} "
                "so every model trains at the same effective batch size"
            )


def accumulation_steps(cfg, arch):
    """Micro-batches per optimizer step for this model."""
    return cfg["training"]["effective_batch_size"] // cfg["models"][arch]["batch_size"]
