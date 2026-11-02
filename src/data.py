"""Labels, crab-level splits, augmentation and data loaders.

Splits are made over crab_id, never over images, so every photo of one crab
lands in exactly one of train / val / test. A split is saved to disk the first
time it is made and reused by every model afterwards. Each attribute uses only
the rows labeled for it, and its split is made on those rows.
"""
import json
import logging
import os
import random
import shutil
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

ATTRIBUTES = ("genus", "species", "sex", "disease", "maturity")
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
    df = df.apply(lambda col: col.str.strip())  # other columns (source, country, observer_id, ...) are kept
    if (df["crab_id"] == "").any() or (df["image_path"] == "").any():
        raise ValueError(f"{csv_path} has rows with an empty crab_id or image_path")
    if "source" in df.columns:
        df["source"] = df["source"].replace("", "unknown")
    return df


GROUP_COLUMNS = ("crab_id", "observer_id")  # columns that identify who took a photo


def split_external(df, ext):
    """Hold out an external test set: rows where ext['column'] equals ext['value'] (or one of a list).

    Returns (internal, external, info). Every observer that appears in the external set
    (by crab_id, and by observer_id when that column exists) is removed from the internal
    side, so no observer's photos are on both sides. info counts what was held out and
    what the observer overlap removed, overall and per main-attribute class.
    """
    column = ext["column"]
    if column not in df.columns:
        raise ValueError(f"data.external_test.column {column!r} is not in labels.csv")
    values = ext["value"] if isinstance(ext["value"], (list, tuple)) else [ext["value"]]
    values = [str(v) for v in values]
    is_ext = df[column].isin(values)
    external, rest = df[is_ext], df[~is_ext]
    group_cols = [c for c in ext.get("group_columns", GROUP_COLUMNS) if c in df.columns]
    overlap = pd.Series(False, index=rest.index)
    for c in group_cols:
        ext_ids = set(external[c]) - {""}
        overlap |= rest[c].isin(ext_ids)
    removed, internal = rest[overlap], rest[~overlap]
    by = ext.get("report_by")

    def per(frame):
        if not by or by not in frame.columns:
            return {}
        return {k: {"images": int(len(g)), "groups": int(g["crab_id"].nunique())}
                for k, g in frame[frame[by] != ""].groupby(by)}

    info = {
        "column": column, "values": values, "group_columns": group_cols,
        "external": {"images": int(len(external)), "groups": int(external["crab_id"].nunique()), "by_class": per(external)},
        "removed_overlap": {"images": int(len(removed)), "groups": int(removed["crab_id"].nunique()),
                            "by_class": per(removed)},
        "internal": {"images": int(len(internal)), "groups": int(internal["crab_id"].nunique()), "by_class": per(internal)},
    }
    log.info("External test (%s in %s): %d images, %d groups held out; observer overlap removed %d images, "
             "%d groups from the training side; %d images, %d groups left for cross-validation",
             column, values, len(external), info["external"]["groups"], len(removed), info["removed_overlap"]["groups"],
             len(internal), info["internal"]["groups"])
    return internal.reset_index(drop=True), external.reset_index(drop=True), info


def data_settings(cfg, rows, attribute):
    """Data-side settings to record with a run: loader workers, image cache and the photo cap.

    The per-observer cap is applied by the dataset builder, so it comes from the run whose labels.csv
    is this labels.csv (all None if none is). The largest group and the photos per class show what
    the rows actually hold; run.py itself caps nothing.
    """
    labels = Path(cfg["data"]["labels_csv"])
    cap = {"builder_run": None, "max_photos_per_group": None, "group_cap_seed": None}
    content = labels.read_bytes() if labels.exists() else None
    for path in sorted((labels.parent / "external" / "runs").glob("*/labels.csv")):
        run_json = path.parent / "run.json"
        if path.read_bytes() == content and run_json.exists():
            built = json.loads(run_json.read_text(encoding="utf-8"))
            cap = {"builder_run": path.parent.name, "max_photos_per_group": built.get("max_photos_per_group"),
                   "group_cap_seed": built.get("group_cap_seed")}
            break
    cap["largest_group_photos"] = int(rows.groupby(["crab_id", attribute]).size().max())
    cap["photos_per_class"] = {k: int(v) for k, v in rows[attribute].value_counts().sort_index().items()}
    return {"num_workers": cfg["data"]["num_workers"], "cache_longer_side": cfg["data"].get("cache_longer_side"),
            "photo_cap": cap}


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

GENUS_ONLY = "(genus only)"
OTHER = "(other)"


def _mode(values):
    return values.value_counts().sort_index().idxmax()


def crab_finer(rows, finer):
    """One finer value per crab (e.g. its species); blank -> '(genus only)'."""
    values = rows[finer].where(rows[finer] != "", GENUS_ONLY)
    return rows.assign(_finer=values).groupby("crab_id")["_finer"].agg(_mode).sort_index()


def crab_strata(rows, attribute, finer, min_groups):
    """Per crab: its label (e.g. genus) and a stratum '<label> / <finer>' (e.g. '<genus> / <species>').

    A stratum with fewer than `min_groups` crabs is folded into the largest stratum of the
    same label that has enough, or into '<label> / (other)' if none has. So every stratum
    can be stratified on, and the label is never mixed across strata.
    """
    labels = crab_labels(rows, attribute)
    fine = crab_finer(rows, finer).reindex(labels.index)
    pairs = {c: (labels[c], fine[c]) for c in labels.index}
    counts = pd.Series(list(pairs.values())).value_counts()
    target = {}
    for label in sorted(set(labels)):
        keys = sorted((k for k in counts.index if k[0] == label), key=lambda k: (-counts[k], k[1]))
        big = [k for k in keys if counts[k] >= min_groups]
        for k in keys:
            target[k] = k if counts[k] >= min_groups else (big[0] if big else (label, OTHER))
    strata = pd.Series({c: " / ".join(target[p]) for c, p in pairs.items()}).sort_index()
    return labels, fine, strata


def _two_way(crab_ids, options, frac, seed):
    """Split crabs into (rest, held_out, level). `options` are (level, per-crab key) pairs tried
    in order (e.g. species strata, then genus); level is 'none' for a random split."""
    for level, keys in options:
        y = [keys[c] for c in crab_ids]
        try:
            rest, held = train_test_split(crab_ids, test_size=frac, random_state=seed, stratify=y)
            return sorted(rest), sorted(held), level
        except ValueError as e:
            log.warning("Stratifying on %s not possible (%s); trying the next option", level, e)
    log.warning("Falling back to a random crab split")
    rest, held = train_test_split(crab_ids, test_size=frac, random_state=seed)
    return sorted(rest), sorted(held), "none"


def make_split(rows, attribute, strategy, ratios, n_folds, seed, stratify_by=None, min_stratum_groups=10):
    """Group-safe split over crab_id. With `stratify_by` (a finer column, e.g. species for the
    genus label), each step stratifies on '<label> / <finer>' strata, falling back to the label
    and then to a random split."""
    if rows.empty:
        raise SplitError(f"{attribute}: no labeled rows to split")
    if stratify_by and stratify_by not in rows.columns:
        raise SplitError(f"{attribute}: stratify_by column {stratify_by!r} is not in labels.csv")
    labels = crab_labels(rows, attribute)
    options, strata = [(attribute, labels)], None
    if stratify_by:
        _, _, strata = crab_strata(rows, attribute, stratify_by, min_stratum_groups)
        options.insert(0, (f"{attribute}/{stratify_by}", strata))
    crab_ids = list(labels.index)
    train_r, val_r, test_r = ratios
    val_frac = val_r / (train_r + val_r)
    folds, levels = [], []

    if strategy == "holdout":
        trval, test, l1 = _two_way(crab_ids, options, test_r, seed)
        train, val, l2 = _two_way(trval, options, val_frac, seed)
        folds.append({"train": train, "val": val, "test": test})
        levels += [l1, l2]
    elif strategy == "kfold":
        if len(crab_ids) < n_folds:
            raise SplitError(f"{attribute}: {len(crab_ids)} crabs is fewer than n_folds={n_folds}")
        usable = [(level, keys) for level, keys in options if keys.value_counts().min() >= n_folds]
        if usable:
            level, keys = usable[0]
            splitter, y = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed), list(keys.values)
        else:
            log.warning("%s: some class has fewer than %d crabs; using unstratified KFold", attribute, n_folds)
            level, splitter, y = "none", KFold(n_splits=n_folds, shuffle=True, random_state=seed), None
        levels.append(level)
        for k, (trval_idx, test_idx) in enumerate(splitter.split(crab_ids, y)):
            trval = [crab_ids[i] for i in trval_idx]
            train, val, lv = _two_way(trval, options, val_frac, seed + k)
            folds.append({"train": train, "val": val, "test": sorted(crab_ids[i] for i in test_idx)})
            levels.append(lv)
    else:
        raise ValueError(f"Unknown split strategy {strategy!r}")

    split = {
        "attribute": attribute,
        "strategy": strategy,
        "seed": seed,
        "ratios": list(ratios),
        "n_folds": n_folds if strategy == "kfold" else None,
        "classes": sorted(labels.unique()),
        "stratified": "none" not in levels,
        "stratify_by": stratify_by,
        "min_stratum_groups": min_stratum_groups if stratify_by else None,
        "stratified_on": sorted(set(levels)),
        "strata": {k: int(v) for k, v in strata.value_counts().sort_index().items()} if strata is not None else None,
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

    stratify_by, min_stratum = stratify_settings(s, attribute)
    split = make_split(rows, attribute, s["strategy"], s["ratios"], s["n_folds"], s["seed"], stratify_by, min_stratum)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(split, f, indent=1)
    log.info("Saved new split %s", path)
    return split


def stratify_settings(s, attribute):
    """(finer stratification column or None, min_stratum_groups) for one attribute from config split."""
    stratify_by = (s.get("stratify_by") or {}).get(attribute)
    return stratify_by, (int(s.get("min_stratum_groups", 10)) if stratify_by else None)


def _check_matches(split, rows, attribute, s, path):
    """A saved split must match the current config and labels; never regenerate silently."""
    stratify_by, min_stratum = stratify_settings(s, attribute)
    expected = {"attribute": attribute, "strategy": s["strategy"], "seed": s["seed"],
                "ratios": list(s["ratios"]),
                "n_folds": s["n_folds"] if s["strategy"] == "kfold" else None,
                "stratify_by": stratify_by, "min_stratum_groups": min_stratum}
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


# ---------------------------------------------------------------- image cache

def build_image_cache(src_root, image_paths, cache_root, longer_side):
    """Pre-resized copies of the images, under the same relative paths. Returns files written.

    An image whose longer side is above longer_side is resized to it (Lanczos, aspect kept) and
    saved as JPEG quality 95; any other image is copied byte for byte. A cached file newer than
    its source is kept, so the cache is built once. It depends on nothing but the images and
    longer_side, so every model reads the same cached images.
    """
    src_root, cache_root = Path(src_root), Path(cache_root)
    written = 0
    for rel in image_paths:
        src, dst = src_root / rel, cache_root / rel
        if dst.exists() and dst.stat().st_mtime >= src.stat().st_mtime:
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        tmp = dst.with_name(dst.name + ".tmp")
        with Image.open(src) as im:
            if max(im.size) > longer_side:
                scale = longer_side / max(im.size)
                small = im.convert("RGB").resize((max(1, round(im.width * scale)), max(1, round(im.height * scale))),
                                                 Image.LANCZOS)
                small.save(tmp, format="JPEG", quality=95)
            else:
                shutil.copyfile(src, tmp)
        os.replace(tmp, dst)
        written += 1
    if written:
        log.info("Image cache %s: %d file(s) written (longer side %d)", cache_root, written, longer_side)
    return written


def image_root(cfg):
    """The folder image_path is relative to: labels.csv's folder, or with data.cache_longer_side set,
    the pre-resized cache next to it (data/cache/longer_<N>/, built on first use for every row)."""
    root = Path(cfg["data"]["labels_csv"]).parent
    side = cfg["data"].get("cache_longer_side")
    if not side:
        return root
    cache = root / "cache" / f"longer_{int(side)}"
    paths = pd.read_csv(cfg["data"]["labels_csv"], usecols=["image_path"], dtype=str)["image_path"]
    build_image_cache(root, paths, cache, int(side))
    return cache


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
    root = image_root(cfg)
    attribute, classes = split["attribute"], split["classes"]
    sets = {}
    for name in SPLIT_SETS:
        ids = set(split["folds"][fold][name])
        part = rows[rows["crab_id"].isin(ids)].reset_index(drop=True)
        tf = build_transforms(model_cfg, cfg["augmentation"], train=(name == "train"), seed=seed)
        sets[name] = CrabDataset(part, root, attribute, classes, tf)
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
