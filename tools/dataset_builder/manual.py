"""Importers for datasets downloaded by hand (Roboflow, Kaggle, Zenodo, GitHub). Nothing is scraped.

Each import is described by a small YAML file (see imports/example_*.yaml.example) with
its source name, license and URL. Images are read from disk; nothing is downloaded.
"""
import csv
import json
import logging
import re
from pathlib import Path

import yaml
from PIL import Image

from . import labels
from .records import new_record, skip

log = logging.getLogger(__name__)

TYPES = ("folder_csv", "roboflow_yolo", "roboflow_coco", "filename_id")
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
RESERVED = {"inaturalist", "gbif"}
# Roboflow names augmented copies <stem>_jpg.rf.<hash>.jpg; the stem identifies the source image.
RF_COPY = re.compile(r"^(?P<stem>.+?)_(?:jpe?g|png|bmp|webp)\.rf\.[0-9a-f]+$", re.I)


class ImportSpecError(ValueError):
    pass


def load_spec(path, root):
    """The import YAML, checked. `path` in it is relative to the project root."""
    with open(path, encoding="utf-8") as f:
        spec = yaml.safe_load(f) or {}
    for field in ("name", "type", "path", "license", "url"):
        if not spec.get(field):
            raise ImportSpecError(f"{path}: '{field}' is required")
    if spec["type"] not in TYPES:
        raise ImportSpecError(f"{path}: type must be one of {TYPES}")
    if not re.fullmatch(r"[a-z0-9_-]+", spec["name"]) or spec["name"] in RESERVED:
        raise ImportSpecError(f"{path}: name must be lowercase letters, digits, _ or -, and not {sorted(RESERVED)}")
    if spec["type"] == "filename_id":
        if not spec.get("regex"):
            raise ImportSpecError(f"{path}: filename_id imports need a 'regex'")
        re.compile(spec["regex"])
    spec["path"] = Path(root) / spec["path"]
    spec.setdefault("class_map", {})
    spec.setdefault("min_box_size", 32)
    spec.setdefault("box_padding", 0.0)
    spec["spec_file"] = str(path)
    return spec


def _class_entry(spec, name):
    """{'species', 'sex', 'maturity'} for a dataset class or folder name, or None if unmapped."""
    entry = spec["class_map"].get(name)
    if entry is None:
        return None
    return {"species": entry} if isinstance(entry, str) else dict(entry)


def _record(spec, cfg, species, rel, local_path, sex_raw="", mat_raw="", label_from="", **extra):
    maps = cfg["labels"]
    sex = labels.map_value(sex_raw, maps["sex"])
    mat = labels.map_value(mat_raw, maps["maturity"])
    name = spec["name"]
    defaults = dict(
        source=name, species=species, photo_id=f"{name}:{rel}", local_path=str(local_path),
        observation_url=spec["url"], license_raw=spec["license"], dataset=spec["spec_file"],
        attribution=spec.get("attribution", ""), country=spec.get("country", ""),
        sex=sex, sex_source=f"manual:{name}:{label_from}" if sex else "", sex_raw=sex_raw or "",
        maturity=mat, maturity_source=f"manual:{name}:{label_from}" if mat else "", maturity_raw=mat_raw or "",
    )
    defaults.update(extra)
    return new_record(**defaults)


def _images(folder):
    return sorted(p for p in Path(folder).rglob("*") if p.suffix.lower() in IMAGE_EXTS)


def folder_csv(spec, cfg):
    """A folder of images plus a CSV with a 'file' column and optional species, sex, maturity,
    observer_id, observation_id, id (file-level crab ID) and license columns."""
    base = spec["path"]
    name = spec["name"]
    out = []
    with open(base / spec.get("csv", "labels.csv"), newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            row = {k.strip(): (v or "").strip() for k, v in row.items() if k}
            rel = row["file"]
            extra = {}
            for col, field in (("observer_id", "observer_id"), ("observation_id", "observation_id"), ("id", "file_id")):
                if row.get(col):
                    extra[field] = f"{name}:{row[col]}"
            if row.get("license"):
                extra["license_raw"] = row["license"]
            out.append(_record(spec, cfg, row.get("species") or spec.get("species", ""), rel, base / rel,
                               row.get("sex", ""), row.get("maturity", ""), "csv", **extra))
    return out


def parse_file_id(rx, filename):
    """The crab ID in a filename: the named group 'id', else group 1, else the whole match; '' if none."""
    m = rx.search(filename)
    if not m:
        return ""
    if "id" in m.groupdict():
        return m.group("id") or ""
    return m.group(1) if m.groups() else m.group(0)


def filename_id(spec, cfg):
    """ReID-style folders: the crab ID comes from each filename via `regex` (named group 'id' or group 1).

    The species comes from `species`, or from the parent folder name through `class_map`.
    Files whose name does not match are skipped (an unparsed ID would weaken the group split).
    """
    base, name = spec["path"], spec["name"]
    rx = re.compile(spec["regex"])
    out = []
    for p in _images(base):
        rel = p.relative_to(base).as_posix()
        entry = _class_entry(spec, p.parent.name) or {"species": spec.get("species", "")}
        file_id = parse_file_id(rx, p.name)
        rec = _record(spec, cfg, entry.get("species", ""), rel, p, entry.get("sex", ""), entry.get("maturity", ""),
                      "class_map", file_id=f"{name}:{file_id}" if file_id else "")
        if not file_id:
            skip(rec, "file_id_not_parsed")
        out.append(rec)
    return out


def _source_stem(path):
    m = RF_COPY.match(Path(path).stem)
    return m.group("stem") if m else Path(path).stem


def _crop_records(spec, cfg, img_path, rel, boxes, size):
    """One record per labeled box. `boxes` are (class_name, x0, y0, x1, y1) in pixels."""
    W, H = size
    name = spec["name"]
    out = []
    for k, (cls, x0, y0, x1, y1) in enumerate(boxes):
        pad_x, pad_y = (x1 - x0) * spec["box_padding"], (y1 - y0) * spec["box_padding"]
        box = (max(0, int(round(x0 - pad_x))), max(0, int(round(y0 - pad_y))), min(W, int(round(x1 + pad_x))), min(H, int(round(y1 + pad_y))))
        entry = _class_entry(spec, cls)
        rec = _record(spec, cfg, (entry or {}).get("species", ""), f"{rel}#{k}", img_path,
                      (entry or {}).get("sex", ""), (entry or {}).get("maturity", ""), "class_map",
                      crop_box=json.dumps(box), observation_id=f"{name}:{_source_stem(rel)}",
                      width=box[2] - box[0], height=box[3] - box[1], taxon_name=cls)
        if entry is None:
            skip(rec, f"class_not_mapped:{cls}")
        elif min(box[2] - box[0], box[3] - box[1]) < spec["min_box_size"]:
            skip(rec, "box_too_small")
        out.append(rec)
    return out


def _yolo_names(base):
    files = sorted(base.rglob("data.yaml"))
    if not files:
        raise ImportSpecError(f"{base}: no data.yaml found (expected a Roboflow YOLO export)")
    names = yaml.safe_load(files[0].read_text(encoding="utf-8")).get("names", [])
    return dict(names) if isinstance(names, dict) else dict(enumerate(names))


def roboflow_yolo(spec, cfg):
    """Roboflow YOLO export: <split>/images/*.jpg with <split>/labels/*.txt (boxes or polygons)."""
    base = spec["path"]
    names = _yolo_names(base)
    out = []
    for img in _images(base):
        if img.parent.name != "images":
            continue
        label = img.parent.parent / "labels" / f"{img.stem}.txt"
        if not label.exists():
            continue
        with Image.open(img) as im:
            W, H = im.size
        boxes = []
        for line in label.read_text(encoding="utf-8").splitlines():
            parts = line.split()
            if len(parts) < 5:
                continue
            cls, vals = names.get(int(parts[0]), parts[0]), [float(v) for v in parts[1:]]
            if len(vals) == 4:
                cx, cy, w, h = vals
                x0, y0, x1, y1 = cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2
            else:  # polygon: x1 y1 x2 y2 ...
                xs, ys = vals[0::2], vals[1::2]
                x0, y0, x1, y1 = min(xs), min(ys), max(xs), max(ys)
            boxes.append((cls, x0 * W, y0 * H, x1 * W, y1 * H))
        out += _crop_records(spec, cfg, img, img.relative_to(base).as_posix(), boxes, (W, H))
    return out


def roboflow_coco(spec, cfg):
    """Roboflow COCO export: <split>/_annotations.coco.json next to the images."""
    base = spec["path"]
    files = sorted(base.rglob("*.coco.json")) or sorted(base.rglob("*annotations*.json"))
    if not files:
        raise ImportSpecError(f"{base}: no COCO annotation JSON found")
    out = []
    for ann_file in files:
        coco = json.loads(ann_file.read_text(encoding="utf-8"))
        cats = {c["id"]: c["name"] for c in coco.get("categories", [])}
        by_image = {}
        for a in coco.get("annotations", []):
            x, y, w, h = a["bbox"]
            by_image.setdefault(a["image_id"], []).append((cats.get(a["category_id"], str(a["category_id"])), x, y, x + w, y + h))
        for im in coco.get("images", []):
            img = ann_file.parent / im["file_name"]
            if not img.exists() or im["id"] not in by_image:
                continue
            size = (im.get("width"), im.get("height"))
            if not all(size):
                with Image.open(img) as pil:
                    size = pil.size
            out += _crop_records(spec, cfg, img, img.relative_to(base).as_posix(), by_image[im["id"]], size)
    return out


IMPORTERS = {"folder_csv": folder_csv, "filename_id": filename_id,
             "roboflow_yolo": roboflow_yolo, "roboflow_coco": roboflow_coco}


def run_import(spec, cfg):
    records = IMPORTERS[spec["type"]](spec, cfg)
    log.info("import %s (%s): %d records", spec["name"], spec["type"], len(records))
    return records
