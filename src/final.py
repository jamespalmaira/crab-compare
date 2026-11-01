"""The final model for the app: one backbone trained on the whole training side, then exported.

This is a separate step after the benchmark and changes nothing in it. There is no validation
split, so no early stopping: the model trains for a fixed number of epochs, the median best epoch
of the same backbone's benchmark runs. Everything else matches the benchmark, including the cosine
schedule over training.epochs, which is cut off at that epoch so every epoch has the learning rate
it had in the benchmark. The app gets logits from the ONNX model and applies softmax itself.
"""
import hashlib
import json
import logging
import math
import shutil
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import pandas as pd
import torch
from PIL import Image

from src.config import accumulation_steps
from src.data import CrabDataset, build_loaders, build_transforms, class_weights, image_root
from src.export import check_onnx, export_onnx
from src.models import create_model, model_data_config
from src.train import (accumulated_step, amp_enabled, environment_info, make_scaler, oom_guard, run_log, set_seed)

log = logging.getLogger(__name__)

CONFIDENCE_EDGES = (0.25, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99, 1.00)
GOLDEN_LICENSES = ("CC0", "CC-BY")  # the golden image ships with the export, so only open licences
NOT_MEASURABLE_BELOW_GROUPS = 10
LABEL_NOTE = ("Labels are community identifications from public sources (iNaturalist, GBIF), not expert-verified. "
              "crab_id is an observer grouping key, not an individual crab.")
LABEL_NOTES = {"Atergatis": "Atergatis floridus: a reported toxic look-alike. It is not an edible crab."}


def median_best_epoch(per_seed, arch, attribute):
    """(fixed number of epochs, sorted best epochs): the median best epoch over the benchmark runs of arch."""
    rows = per_seed[(per_seed["arch"] == arch) & (per_seed["attribute"] == attribute)]
    if rows.empty:
        raise ValueError(f"No benchmark runs of {arch} for {attribute}")
    return int(math.ceil(rows["best_epoch"].median())), sorted(int(v) for v in rows["best_epoch"])


def check_held_out(rows, external):
    """Raise if any external image or observer is on the training side."""
    if set(rows["image_path"]) & set(external["image_path"]):
        raise ValueError("An external image is on the training side")
    for col in ("crab_id", "observer_id"):
        if col in rows.columns and col in external.columns:
            if (set(rows[col]) & set(external[col])) - {""}:
                raise ValueError(f"An external {col} is on the training side")


def train_final(cfg, arch, rows, attribute, epochs, seed, out_dir, device):
    """Train on every row for a fixed number of epochs. Returns (model, data_cfg, info)."""
    t, m = cfg["training"], cfg["models"][arch]
    classes = sorted(rows[attribute].unique())
    bs, accum = m["batch_size"], accumulation_steps(cfg, arch)
    amp = amp_enabled(cfg, device)
    name = f"{arch}_final_seed{seed}"
    ckpt_path = Path(out_dir) / "checkpoints" / f"{name}.pth"
    log_path = Path(out_dir) / "logs" / f"{name}.log"
    rows = rows.reset_index(drop=True)

    with run_log(log_path), oom_guard(m["name"], arch, bs):
        set_seed(seed)
        model = create_model(arch, len(classes), pretrained=t["pretrained"]).to(device)
        data_cfg = model_data_config(model)
        tf = build_transforms(data_cfg, cfg["augmentation"], train=True, seed=seed)
        dataset = CrabDataset(rows, image_root(cfg), attribute, classes, tf)
        # same loader settings as the benchmark; the val and test loaders are never iterated
        loader = build_loaders({"train": dataset, "val": dataset, "test": dataset}, bs, cfg["data"]["num_workers"],
                               seed)["train"]
        steps_per_epoch = len(loader) // accum
        info = {"run": name, "attribute": attribute, "arch": arch, "model": m["name"], "fold": "all", "seed": seed,
                "classes": classes, "n_train": len(dataset), "n_groups": int(rows["crab_id"].nunique()),
                "images_per_class": rows[attribute].value_counts().sort_index().to_dict(),
                "batch_size": bs, "accumulation_steps": accum, "effective_batch_size": t["effective_batch_size"],
                "amp": amp, "pretrained": t["pretrained"], "epochs": epochs, "schedule_epochs": t["epochs"],
                "lr": t["lr"], "weight_decay": t["weight_decay"], "steps_per_epoch": steps_per_epoch,
                "input_size": list(data_cfg["input_size"]), **environment_info(device)}
        log.info("run %s", json.dumps(info))

        criterion = torch.nn.CrossEntropyLoss(weight=class_weights(dataset.targets, len(classes)).to(device))
        optimizer = torch.optim.AdamW(model.parameters(), lr=t["lr"], weight_decay=t["weight_decay"])
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=t["epochs"] * steps_per_epoch)
        scaler = make_scaler(amp)
        step, history = 0, []
        for epoch in range(1, epochs + 1):
            start = time.perf_counter()
            model.train()
            losses, group = [], []
            for x, y in loader:
                group.append((x.to(device, non_blocking=True), y.to(device, non_blocking=True)))
                if len(group) < accum:
                    continue
                losses.append(accumulated_step(model, group, criterion, optimizer, scaler, amp, arch, step))
                scheduler.step()
                step += 1
                group = []
            history.append({"epoch": epoch, "train_loss": float(np.mean(losses))})
            log.info("%s epoch %d/%d train_loss %.4f lr %.2e %.1fs", name, epoch, epochs, float(np.mean(losses)),
                     scheduler.get_last_lr()[0], time.perf_counter() - start)

        ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        info.update(optimizer_steps=step, checkpoint=str(ckpt_path), log=str(log_path), history=history)
        torch.save({"state_dict": model.state_dict(), "arch": arch, "classes": classes, "attribute": attribute,
                    "seed": seed, "epoch": epochs, "data_cfg": data_cfg, "info": info}, ckpt_path)
        log.info("%s done: %d epochs, %d optimizer steps, checkpoint %s", name, epochs, step, ckpt_path)
    return model, data_cfg, info


def external_table(metric_rows):
    """Per-class external rows with images, groups and recall; below 10 groups recall is not measurable."""
    rows = []
    for r in metric_rows:
        if r["class"] == "ALL":
            continue
        measurable = r["n_groups"] >= NOT_MEASURABLE_BELOW_GROUPS
        rows.append({"genus": r["class"], "images": r["n_images"], "groups": r["n_groups"], "recall": r["recall"],
                     "measurable": measurable, "note": "" if measurable else "not measurable (fewer than 10 groups)"})
    return pd.DataFrame(rows)


def _confidence_rows(frame, scope, genus, edges):
    total, out = len(frame), []
    conf = frame["confidence"].to_numpy()
    bins = np.clip(np.searchsorted(edges, conf, side="right") - 1, 0, len(edges) - 2)
    for k in range(len(edges) - 1):
        part, above = frame[bins == k], frame[bins >= k]
        out.append({"scope": scope, "genus": genus, "bin_low": edges[k], "bin_high": edges[k + 1],
                    "n_images": len(part), "n_distinct_images": part["image_path"].nunique(),
                    "n_correct": int(part["correct"].sum()),
                    "accuracy": float(part["correct"].mean()) if len(part) else float("nan"),
                    "n_at_or_above": len(above), "share_at_or_above": len(above) / total if total else float("nan"),
                    "accuracy_at_or_above": float(above["correct"].mean()) if len(above) else float("nan")})
    return out


def confidence_table(predictions, classes, edges=CONFIDENCE_EDGES):
    """Accuracy and images by confidence bin (top softmax probability), overall and per genus.

    Bins are [low, high), the last one includes 1. Per genus comes two ways: by true genus, and by
    predicted genus, which is all the app can see. n_images counts predictions (an image once per
    seed), n_distinct_images counts each photo once. The *_at_or_above columns show what a threshold
    at bin_low would keep.
    """
    df = predictions.copy()
    df["confidence"] = df[[f"prob_{c}" for c in classes]].max(axis=1)
    df["correct"] = df["true_label"] == df["pred_label"]
    rows = _confidence_rows(df, "overall", "", edges)
    for scope, col in (("true_genus", "true_label"), ("predicted_genus", "pred_label")):
        for c in classes:
            rows += _confidence_rows(df[df[col] == c], scope, c, edges)
    return pd.DataFrame(rows)


def select_golden_image(external, licenses=GOLDEN_LICENSES):
    """The fixed golden image: first by image_path among the most open licence, never chosen by a prediction."""
    for lic in licenses:
        rows = external[external["license"] == lic].sort_values("image_path")
        if len(rows):
            return rows.iloc[0]
    raise ValueError(f"No external image has a licence in {licenses}")


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_sha256s(export_dir):
    """SHA256SUMS.txt (sha256sum format) for every file in the folder except itself."""
    export_dir = Path(export_dir)
    files = sorted(p for p in export_dir.rglob("*") if p.is_file() and p.name != "SHA256SUMS.txt")
    lines = [f"{sha256_file(p)}  {p.relative_to(export_dir).as_posix()}" for p in files]
    (export_dir / "SHA256SUMS.txt").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    return lines


def build_meta(info, data_cfg, onnx_sha256, created, run_name, golden):
    size = data_cfg["input_size"]
    return {
        "format_version": 1,
        "model_name": info["model"],
        "timm_arch": info["arch"],
        "run_name": run_name,
        "date": created,
        "file": "model.onnx",
        "sha256": onnx_sha256,
        "input": {
            "name": "input", "layout": "NCHW", "shape": [None, *size], "dtype": "float32", "color": "RGB",
            "input_size": list(size), "mean": list(data_cfg["mean"]), "std": list(data_cfg["std"]),
            "preprocessing": {
                "steps": ["decode to RGB", "resize shorter side (keep aspect)", "center crop", "scale to [0, 1]",
                          "subtract mean, divide by std", "HWC to CHW"],
                "resize_shorter_side": int(np.floor(size[1] / data_cfg.get("crop_pct", 1.0))),
                "interpolation": data_cfg.get("interpolation"), "crop_pct": data_cfg.get("crop_pct"),
                "center_crop": size[1], "exif_orientation_applied": False},
        },
        "heads": [{"name": info["attribute"], "output_name": "logits", "output_type": "logits",
                   "activation_in_app": "softmax", "num_classes": len(info["classes"]), "labels": info["classes"]}],
        "note": LABEL_NOTE,
        "label_notes": {c: n for c, n in LABEL_NOTES.items() if c in info["classes"]},
        "training": {"n_images": info["n_train"], "n_groups": info["n_groups"], "epochs": info["epochs"],
                     "seed": info["seed"], "excludes": "every Philippine photo and observer (external test set)"},
        "golden": golden,
    }


def export_final(ckpt_path, external, cfg, export_dir, run_name, created):
    """model.onnx, model_meta.json, golden vectors and SHA-256s. Returns a dict of what was checked."""
    export_dir = Path(export_dir)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    info, data_cfg = ckpt["info"], ckpt["data_cfg"]
    model = create_model(ckpt["arch"], len(ckpt["classes"]), pretrained=False)
    model.load_state_dict(ckpt["state_dict"])
    model = model.cpu().float().eval()
    size = tuple(data_cfg["input_size"])
    a = cfg.get("analysis", {})
    rtol, atol = a.get("onnx_rtol", 0.001), a.get("onnx_atol", 0.0001)

    export_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = export_onnx(model, size, export_dir / "model.onnx", a.get("onnx_opset", 17))
    parity = check_onnx(model, onnx_path, size, rtol, atol)

    # golden vectors, preprocessed exactly as in evaluation
    row = select_golden_image(external)
    src = Path(image_root(cfg)) / row["image_path"]
    golden_dir = export_dir / "golden"
    golden_dir.mkdir(exist_ok=True)
    image_file = golden_dir / f"image{src.suffix.lower()}"
    shutil.copyfile(src, image_file)
    with Image.open(src) as im:
        rgb = np.asarray(im.convert("RGB"))
    tensor = build_transforms(data_cfg, cfg["augmentation"], train=False)(image=rgb)["image"].float()
    array = tensor.numpy().astype(np.float32)[None]
    np.save(golden_dir / "input_tensor.npy", array)
    array.astype("<f4").tofile(golden_dir / "input_tensor_f32le.bin")
    with torch.no_grad():
        logits_pt = model(tensor[None]).numpy()
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    logits_ort = session.run(None, {"input": array})[0]
    golden_diff = float(np.abs(logits_pt - logits_ort).max())
    golden_match = bool(np.allclose(logits_ort, logits_pt, rtol=rtol, atol=atol))
    probs = torch.softmax(torch.from_numpy(logits_pt), dim=1)[0].numpy()
    classes = ckpt["classes"]
    golden = {"image": f"golden/{image_file.name}", "source_image_path": row["image_path"], "license": row["license"],
              "attribution": str(row.get("attribution", "")), "observation_id": str(row.get("observation_id", "")),
              "community_label": row[info["attribute"]], "input_tensor": "golden/input_tensor.npy",
              "input_tensor_raw": "golden/input_tensor_f32le.bin", "input_shape": list(array.shape),
              "logits": [float(v) for v in logits_pt[0]],
              "probabilities": {c: float(p) for c, p in zip(classes, probs)},
              "predicted_label": classes[int(probs.argmax())], "tolerance": {"rtol": rtol, "atol": atol},
              "logits_source": "PyTorch fp32, CPU", "onnx_max_abs_diff": golden_diff}
    (golden_dir / "expected.json").write_text(json.dumps(golden, indent=1), encoding="utf-8")

    meta = build_meta(info, data_cfg, sha256_file(onnx_path), created, run_name,
                      {"expected": "golden/expected.json"})
    (export_dir / "model_meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    return {"onnx_outputs_match": parity["onnx_outputs_match"], "onnx_max_abs_diff": parity["onnx_max_abs_diff"],
            "onnx_size_mb": parity["onnx_size_mb"], "golden_match": golden_match, "golden_max_abs_diff": golden_diff,
            "golden": golden, "meta": meta}
