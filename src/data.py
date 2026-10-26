"""Labels, crab-level splits, augmentation and data loaders.

Splits are made over crab_id, never over images, so every photo of one crab
lands in exactly one of train / val / test. A split is saved to disk the first
time it is made and reused by every model afterwards. Each attribute uses only
the rows labeled for it, and its split is made on those rows.
"""
import json
import logging
import random
from pathlib import Path

import albumentations as A
import cv2
import numpy as np
import pandas as pd
import torch
from albumentations.pytorch import ToTensorV2
from PIL import Image
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from torch.utils.data import DataLoader, Dataset

log = logging.getLogger(__name__)

ATTRIBUTES = ("species", "sex", "disease", "maturity")
REQUIRED_COLUMNS = ("image_path", "crab_id")  # attribute columns may be missing or blank
SPLIT_SETS = ("train", "val", "test")

_INTERPOLATION = {"bilinear": cv2.INTER_LINEAR, "bicubic": cv2.INTER_CUBIC, "nearest": cv2.INTER_NEAREST}


class SplitError(RuntimeError):
    pass


# ---------------------------------------------------------------- labels

def load_labels(csv_path):
    """labels.csv with image_path and crab_id; attribute and source columns are optional.

    A missing attribute column is treated as all blank. crab_id is a grouping
    key (an individual, observer, observation or filename ID): all rows with
    the same crab_id always stay in the same split set.
    """
    df = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{csv_path} is missing required columns: {missing}")
    for attr in ATTRIBUTES:
        if attr not in df.columns:
            log.info("%s has no %s column; treating it as unlabeled", csv_path, attr)
            df[attr] = ""
    columns = list(REQUIRED_COLUMNS) + list(ATTRIBUTES) + (["source"] if "source" in df.columns else [])
    df = df[columns].apply(lambda col: col.str.strip())
    if (df["crab_id"] == "").any() or (df["image_path"] == "").any():
        raise ValueError(f"{csv_path} has rows with an empty crab_id or image_path")
    if "source" in df.columns:
        df["source"] = df["source"].replace("", "unknown")
    return df


def attribute_rows(df, attribute):
    """Rows that have a label for this attribute (blank labels are skipped)."""
    if attribute not in ATTRIBUTES:
        raise ValueError(f"Unknown attribute {attribute!r}; expected one of {ATTRIBUTES}")
    rows = df[df[attribute] != ""]
    skipped = len(df) - len(rows)
    if skipped:
        log.info("%s: leaving out %d of %d image(s) with no label", attribute, skipped, len(df))
    return rows.reset_index(drop=True)


def attribute_usage(rows, attribute, min_groups_per_class):
    """What one attribute would train on, and why it must be skipped (None if usable)."""
    groups = rows.groupby(attribute)["crab_id"].nunique().sort_index()
    images = rows[attribute].value_counts().sort_index()
    usage = {
        "attribute": attribute, "n_images": int(len(rows)), "n_groups": int(rows["crab_id"].nunique()),
        "n_classes": int(len(groups)), "groups_per_class": {k: int(v) for k, v in groups.items()},
        "images_per_class": {k: int(v) for k, v in images.items()},
        "min_groups_per_class": int(min_groups_per_class), "skip_reason": None,
    }
    if len(groups) < 2:
        usage["skip_reason"] = f"{len(groups)} labeled class(es); at least 2 are needed"
    else:
        low = {k: int(v) for k, v in groups.items() if v < min_groups_per_class}
        if low:
            usage["skip_reason"] = (f"class(es) below data.min_groups_per_class={min_groups_per_class} "
                                    f"groups (crab_id): {low}")
    return usage


def source_balance(rows, attribute, threshold):
    """Per class: image share of its largest source; flagged when above the threshold.

    Returns an empty frame when labels.csv has no source column. A flagged
    class can be learned from source-specific cues (camera, background,
    framing) instead of the crab itself.
    """
    if "source" not in rows.columns or rows.empty:
        return pd.DataFrame()
    counts = rows.groupby([attribute, "source"]).size().unstack(fill_value=0)
    out = []
    for cls, per_source in counts.iterrows():
        total = int(per_source.sum())
        top = per_source.idxmax()
        share = float(per_source.max() / total)
        out.append({"attribute": attribute, "class": cls, "n_images": total,
                    "n_sources": int((per_source > 0).sum()), "top_source": top, "top_share": share,
                    "flagged": share > threshold, "threshold": threshold,
                    "images_by_source": json.dumps({k: int(v) for k, v in per_source.items() if v > 0})})
    balance = pd.DataFrame(out)
    for _, r in balance[balance["flagged"]].iterrows():
        log.warning("%s class %s: %.0f%% of its %d images come from source %r (above %.0f%%)", attribute,
                    r["class"], 100 * r["top_share"], r["n_images"], r["top_source"], 100 * threshold)
    return balance


def prepare_attribute(df, attribute, min_groups_per_class):
    """Filter to labeled rows, log what is used, and return (rows, usage)."""
    rows = attribute_rows(df, attribute)
    usage = attribute_usage(rows, attribute, min_groups_per_class)
    log.info("%s: %d images, %d groups (crab_id), %d classes; groups per class %s", attribute,
             usage["n_images"], usage["n_groups"], usage["n_classes"], usage["groups_per_class"])
    if usage["skip_reason"]:
        log.warning("Skipping attribute %s: %s", attribute, usage["skip_reason"])
    return rows, usage


def crab_labels(rows, attribute):
    """One label per crab (its most common image label), used for stratifying."""
    per_crab = rows.groupby("crab_id")[attribute]
    mixed = sorted(cid for cid, labels in per_crab if labels.nunique() > 1)
    if mixed:
        log.warning("%s: %d crab(s) have mixed labels across photos; stratifying by the most "
                    "common one: %s", attribute, len(mixed), mixed[:10])
    return per_crab.agg(lambda s: s.value_counts().sort_index().idxmax()).sort_index()


# ---------------------------------------------------------------- splitting

def _two_way(crab_ids, labels, frac, seed):
    """Split crabs into (rest, held_out), stratified when every class allows it."""
    try:
        rest, held, _, _ = train_test_split(crab_ids, labels, test_size=frac,
                                            random_state=seed, stratify=labels)
        return sorted(rest), sorted(held), True
    except ValueError as e:
        log.warning("Stratified split not possible (%s); falling back to a random crab split", e)
        rest, held = train_test_split(crab_ids, test_size=frac, random_state=seed)
        return sorted(rest), sorted(held), False


def make_split(rows, attribute, strategy, ratios, n_folds, seed):
    labels = crab_labels(rows, attribute)
    crab_ids, y = list(labels.index), list(labels.values)
    train_r, val_r, test_r = ratios
    val_frac = val_r / (train_r + val_r)
    folds, stratified = [], True

    if strategy == "holdout":
        trval, test, s1 = _two_way(crab_ids, y, test_r, seed)
        train, val, s2 = _two_way(trval, [labels[c] for c in trval], val_frac, seed)
        folds.append({"train": train, "val": val, "test": test})
        stratified = s1 and s2
    elif strategy == "kfold":
        if len(crab_ids) < n_folds:
            raise SplitError(f"{attribute}: {len(crab_ids)} crabs is fewer than n_folds={n_folds}")
        if labels.value_counts().min() >= n_folds:
            splitter = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
        else:
            log.warning("%s: some class has fewer than %d crabs; using unstratified KFold",
                        attribute, n_folds)
            splitter = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
            stratified = False
        for k, (trval_idx, test_idx) in enumerate(splitter.split(crab_ids, y)):
            trval = [crab_ids[i] for i in trval_idx]
            train, val, s = _two_way(trval, [labels[c] for c in trval], val_frac, seed + k)
            folds.append({"train": train, "val": val, "test": sorted(crab_ids[i] for i in test_idx)})
            stratified = stratified and s
    else:
        raise ValueError(f"Unknown split strategy {strategy!r}")

    split = {
        "attribute": attribute,
        "strategy": strategy,
        "seed": seed,
        "ratios": list(ratios),
        "n_folds": n_folds if strategy == "kfold" else None,
        "classes": sorted(labels.unique()),
        "stratified": stratified,
        "n_crabs": len(crab_ids),
        "folds": folds,
    }
    check_split(split)
    return split


def check_split(split):
    """Raise if any crab appears in more than one set of a fold."""
    for k, fold in enumerate(split["folds"]):
        seen = {}
        for name in SPLIT_SETS:
            for cid in fold[name]:
                if cid in seen:
                    raise SplitError(f"Fold {k}: crab {cid!r} is in both {seen[cid]} and {name}")
                seen[cid] = name
        if not fold["train"] or not fold["val"] or not fold["test"]:
            raise SplitError(f"Fold {k}: a set is empty "
                             f"(train={len(fold['train'])}, val={len(fold['val'])}, test={len(fold['test'])})")
    if split["strategy"] == "kfold":
        tests = [c for fold in split["folds"] for c in fold["test"]]
        if len(tests) != len(set(tests)):
            raise SplitError("k-fold test sets overlap")


def split_path(cfg, attribute):
    s = cfg["split"]
    root = Path(cfg["data"]["labels_csv"]).parent / s["dir"]
    return root / f"{attribute}_{s['strategy']}_seed{s['seed']}.json"


def get_split(cfg, rows, attribute):
    """Load the saved split for this attribute, or make and save it once."""
    s = cfg["split"]
    path = split_path(cfg, attribute)
    if path.exists():
        with open(path, encoding="utf-8") as f:
            split = json.load(f)
        _check_matches(split, rows, attribute, s, path)
        check_split(split)
        log.info("Reusing saved split %s", path)
        return split

    split = make_split(rows, attribute, s["strategy"], s["ratios"], s["n_folds"], s["seed"])
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(split, f, indent=1)
    log.info("Saved new split %s", path)
    return split


def _check_matches(split, rows, attribute, s, path):
    """A saved split must match the current config and labels; never regenerate silently."""
    expected = {"attribute": attribute, "strategy": s["strategy"], "seed": s["seed"],
                "ratios": list(s["ratios"]),
                "n_folds": s["n_folds"] if s["strategy"] == "kfold" else None}
    diffs = {k: (split.get(k), v) for k, v in expected.items() if split.get(k) != v}
    saved_crabs = {c for fold in split["folds"][:1] for name in SPLIT_SETS for c in fold[name]}
    current_crabs = set(rows["crab_id"])
    if saved_crabs != current_crabs:
        diffs["crab_ids"] = (f"{len(saved_crabs)} saved", f"{len(current_crabs)} in labels.csv")
    current_classes = sorted(rows[attribute].unique())
    if split["classes"] != current_classes:
        diffs["classes"] = (split["classes"], current_classes)
    if diffs:
        raise SplitError(f"Saved split {path} does not match the current config/labels: {diffs}. "
                         "Delete the file to make a new split (all models must then be re-run).")


# ---------------------------------------------------------------- transforms

def build_transforms(model_cfg, aug, train, seed=None):
    """model_cfg is timm's resolved data config (input_size, mean, std, crop_pct, ...)."""
    size = model_cfg["input_size"][1]
    interp = _INTERPOLATION.get(model_cfg.get("interpolation"), cv2.INTER_LINEAR)
    normalize = [A.Normalize(mean=model_cfg["mean"], std=model_cfg["std"]), ToTensorV2()]
    if train:
        cj = aug["color_jitter"]
        steps = [
            A.RandomResizedCrop(size=(size, size), scale=tuple(aug["resized_crop_scale"]),
                                interpolation=interp),
            A.HorizontalFlip(p=aug["hflip_p"]),
            A.Rotate(limit=aug["rotate_limit"], p=aug["rotate_p"], interpolation=interp,
                     border_mode=cv2.BORDER_REFLECT_101),
            A.ColorJitter(brightness=cj["brightness"], contrast=cj["contrast"],
                          saturation=cj["saturation"], hue=cj["hue"], p=cj["p"]),
        ]
    else:
        scale_size = int(np.floor(size / model_cfg.get("crop_pct", 1.0)))
        steps = [A.SmallestMaxSize(max_size=scale_size, interpolation=interp),
                 A.CenterCrop(height=size, width=size)]
    return A.Compose(steps + normalize, seed=seed)


# ---------------------------------------------------------------- dataset and loaders

class CrabDataset(Dataset):
    def __init__(self, rows, image_root, attribute, classes, transform):
        self.image_paths = list(rows["image_path"])  # as written in labels.csv
        self.paths = [Path(image_root) / p for p in self.image_paths]
        self.crab_ids = list(rows["crab_id"])
        self.sources = list(rows["source"]) if "source" in rows.columns else None
        class_to_idx = {c: i for i, c in enumerate(classes)}
        self.targets = [class_to_idx[v] for v in rows[attribute]]
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        with Image.open(self.paths[i]) as im:
            image = np.asarray(im.convert("RGB"))
        return self.transform(image=image)["image"], self.targets[i]


def class_weights(targets, n_classes):
    """Inverse-frequency weights for the loss, computed on the training set."""
    counts = np.bincount(targets, minlength=n_classes).astype(float)
    if (counts == 0).any():
        log.warning("Classes with no training images get weight 0: %s", np.flatnonzero(counts == 0).tolist())
    weights = np.where(counts > 0, counts.sum() / (n_classes * np.maximum(counts, 1)), 0.0)
    return torch.tensor(weights, dtype=torch.float32)


def _seed_worker(worker_id):
    s = torch.initial_seed() % 2**32
    np.random.seed(s)
    random.seed(s)
    info = torch.utils.data.get_worker_info()
    info.dataset.transform.set_random_seed(s)


def build_datasets(cfg, rows, split, fold, model_cfg, seed):
    """Datasets for one fold; every model gets the same crabs in each set."""
    image_root = Path(cfg["data"]["labels_csv"]).parent
    attribute, classes = split["attribute"], split["classes"]
    sets = {}
    for name in SPLIT_SETS:
        ids = set(split["folds"][fold][name])
        part = rows[rows["crab_id"].isin(ids)].reset_index(drop=True)
        tf = build_transforms(model_cfg, cfg["augmentation"], train=(name == "train"), seed=seed)
        sets[name] = CrabDataset(part, image_root, attribute, classes, tf)
    return sets


def build_loaders(datasets, batch_size, num_workers, seed):
    g = torch.Generator().manual_seed(seed)
    if len(datasets["train"]) < batch_size:
        raise SplitError(f"Training set has {len(datasets['train'])} images, fewer than batch_size={batch_size}")
    common = dict(batch_size=batch_size, num_workers=num_workers, pin_memory=torch.cuda.is_available(),
                  worker_init_fn=_seed_worker if num_workers > 0 else None,
                  # keep workers between epochs; on Windows each restart re-imports torch
                  persistent_workers=num_workers > 0)
    return {
        # drop_last avoids a size-1 batch, which BatchNorm cannot train on
        "train": DataLoader(datasets["train"], shuffle=True, drop_last=True, generator=g, **common),
        "val": DataLoader(datasets["val"], shuffle=False, **common),
        "test": DataLoader(datasets["test"], shuffle=False, **common),
    }
