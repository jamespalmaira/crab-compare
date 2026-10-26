"""Test-set metrics, confusion matrices, latency and model size.

Macro metrics average over the classes present in the test set. With a
crab-level stratified split every class is normally present; on a small test
set a missing class is left out of the macro average (and reported) instead
of counting as zero. ROC-AUC is one-vs-rest, computed only for classes that
have both positive and negative test images, so a missing class never
crashes the evaluation.
"""
import io
import logging
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402
import torch  # noqa: E402
from sklearn.metrics import confusion_matrix, f1_score, precision_recall_fscore_support, roc_auc_score  # noqa: E402

from src.models import param_count  # noqa: E402
from src.train import predict, run_name  # noqa: E402

log = logging.getLogger(__name__)


# ---------------------------------------------------------------- metrics

def roc_auc_ovr(y_true, probs, n_classes):
    """Macro one-vs-rest ROC-AUC over classes with both positives and negatives in y_true."""
    per_class = {}
    for c in range(n_classes):
        pos = y_true == c
        if 0 < pos.sum() < len(y_true):
            per_class[c] = float(roc_auc_score(pos, probs[:, c]))
        else:
            per_class[c] = float("nan")
    scored = [v for v in per_class.values() if not np.isnan(v)]
    return (float(np.mean(scored)) if scored else float("nan")), per_class


def compute_metrics(y_true, probs, classes):
    """Return (summary dict, per-class DataFrame, confusion matrix)."""
    n = len(classes)
    labels = list(range(n))
    y_pred = probs.argmax(1)
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0)

    tp = np.diag(cm)
    fp = cm.sum(0) - tp
    tn = cm.sum() - cm.sum(1) - fp
    negatives = tn + fp
    specificity = np.where(negatives > 0, tn / np.maximum(negatives, 1), np.nan)
    auc_macro, auc_per_class = roc_auc_ovr(y_true, probs, n)

    present = support > 0
    if not present.all():
        log.warning("Classes missing from the test set (left out of macro averages): %s",
                    [classes[i] for i in np.flatnonzero(~present)])

    def macro(values):
        v = np.asarray(values, dtype=float)[present]
        v = v[~np.isnan(v)]
        return float(v.mean()) if len(v) else float("nan")

    summary = {
        "accuracy": float((y_pred == y_true).mean()),
        "macro_precision": macro(precision),
        "macro_recall": macro(recall),
        "macro_specificity": macro(specificity),
        "macro_f1": macro(f1),
        "roc_auc_ovr": auc_macro,
        "n_test_images": int(len(y_true)),
        "n_test_classes": int(present.sum()),
        "n_auc_classes": int(sum(not np.isnan(v) for v in auc_per_class.values())),
    }
    per_class = pd.DataFrame({
        "class": classes, "support": support, "precision": precision, "recall": recall, "f1": f1,
        "specificity": specificity, "roc_auc": [auc_per_class[c] for c in labels],
    })
    return summary, per_class, cm


def save_confusion_matrix(cm, classes, stem, title):
    stem = Path(stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(cm, index=[f"true_{c}" for c in classes], columns=[f"pred_{c}" for c in classes]) \
        .to_csv(stem.with_suffix(".csv"))
    size = max(4, 0.9 * len(classes) + 2)
    fig, ax = plt.subplots(figsize=(size, size * 0.85))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, xticklabels=classes, yticklabels=classes, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title, fontsize=9)
    fig.tight_layout()
    fig.savefig(stem.with_suffix(".png"), dpi=120)
    plt.close(fig)


def predictions_path(out_dir, arch, attribute, seed, fold):
    return Path(out_dir) / "predictions" / f"{arch}_{attribute}_{seed}_{fold}.csv"


def save_predictions(path, dataset, y_true, probs, classes):
    """One row per test image, in dataset order (the test loader does not shuffle)."""
    if len(dataset) != len(y_true) or list(dataset.targets) != list(y_true):
        raise ValueError(f"Predictions do not line up with the test set ({len(y_true)} vs {len(dataset)} images)")
    df = pd.DataFrame({
        "image_path": dataset.image_paths,
        "crab_id": dataset.crab_ids,
        "true_label": [classes[i] for i in y_true],
        "pred_label": [classes[i] for i in probs.argmax(1)],
    })
    if dataset.sources is not None:
        df.insert(2, "source", dataset.sources)
    for i, c in enumerate(classes):
        df[f"prob_{c}"] = probs[:, i]
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return df


def per_source_metrics(pred_df, classes):
    """Test accuracy and macro-F1 per source (macro over the classes present in that source)."""
    if "source" not in pred_df.columns:
        return pd.DataFrame()
    rows = []
    for source, g in pred_df.groupby("source", sort=True):
        present = sorted(set(g["true_label"]), key=classes.index)
        rows.append({
            "source": source, "n_images": int(len(g)), "n_groups": int(g["crab_id"].nunique()),
            "n_classes_present": len(present),
            "accuracy": float((g["true_label"] == g["pred_label"]).mean()),
            "macro_f1": float(f1_score(g["true_label"], g["pred_label"], labels=present, average="macro",
                                       zero_division=0)),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- speed and size

@torch.no_grad()
def measure_latency(model, input_size, device, warmup, runs):
    """Mean fp32 inference time per image in ms (batch of 1, after warm-up)."""
    model.eval()
    x = torch.randn(1, *input_size, device=device)
    for _ in range(warmup):
        model(x)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    start = time.perf_counter()
    for _ in range(runs):
        model(x)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return (time.perf_counter() - start) * 1000 / runs


def model_size_mb(model):
    """Size of the serialized weights (state_dict) in MB (10^6 bytes)."""
    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    return buf.tell() / 1e6


# ---------------------------------------------------------------- one run

def evaluate_run(result, cfg, out_dir, device):
    """Test-set evaluation of a trained run; returns (per-seed row, per-class rows)."""
    info, model = result.info, result.model
    classes = info["classes"]
    probs, y_true = predict(model, result.loaders["test"], device, info["arch"])
    summary, per_class, cm = compute_metrics(y_true, probs, classes)
    pred_path = predictions_path(out_dir, info["arch"], info["attribute"], info["seed"], info["fold"])
    pred_df = save_predictions(pred_path, result.datasets["test"], y_true, probs, classes)
    per_source = per_source_metrics(pred_df, classes)

    name = run_name(info["arch"], info["fold"], info["seed"])
    stem = Path(out_dir) / "confusion_matrices" / info["attribute"] / name
    save_confusion_matrix(cm, classes, stem, f"{info['model']} - {info['attribute']} - fold {info['fold']} seed {info['seed']}")

    ev = cfg.get("evaluation", {})
    warmup, runs = ev.get("latency_warmup", 10), ev.get("latency_runs", 50)
    input_size = tuple(result.data_cfg["input_size"])
    latency_gpu = measure_latency(model, input_size, device, warmup, runs) if device.type == "cuda" else float("nan")
    latency_cpu = float("nan")
    if ev.get("latency_cpu", True):
        model.to("cpu")
        latency_cpu = measure_latency(model, input_size, torch.device("cpu"), warmup, runs)
        model.to(device)

    row = {
        "attribute": info["attribute"], "model": info["model"], "arch": info["arch"],
        "fold": info["fold"], "seed": info["seed"],
        **summary,
        "best_val_macro_f1": info["best_val_macro_f1"], "best_epoch": info["best_epoch"],
        "epochs_trained": info["epochs_trained"],
        "latency_gpu_ms": latency_gpu, "latency_cpu_ms": latency_cpu, "cpu_threads": torch.get_num_threads(),
        "params_m": param_count(model) / 1e6, "model_size_mb": model_size_mb(model),
        "batch_size": info["batch_size"], "accumulation_steps": info["accumulation_steps"],
        "effective_batch_size": info["effective_batch_size"], "amp": info["amp"],
        "pretrained": info["pretrained"], "split_strategy": info["split_strategy"], "split_seed": info["split_seed"],
        "gpu": info["gpu"], "torch": info["torch"], "cuda": info["cuda"], "cudnn": info["cudnn"],
        "timm": info["timm"], "checkpoint": info["checkpoint"], "log": info["log"],
        "confusion_matrix": str(stem.with_suffix(".png")), "predictions": str(pred_path),
    }
    keys = {"attribute": info["attribute"], "arch": info["arch"], "fold": info["fold"], "seed": info["seed"]}
    per_class = per_class.assign(**keys)[list(keys) + list(per_class.columns)]
    if not per_source.empty:
        per_source = per_source.assign(model=info["model"], **keys)[
            list(keys) + ["model"] + list(per_source.columns)]
    log.info("%s test: acc %.4f macro-F1 %.4f ROC-AUC %.4f | latency GPU %.2f ms CPU %.2f ms | %.1fM params %.1f MB",
             name, summary["accuracy"], summary["macro_f1"], summary["roc_auc_ovr"], latency_gpu, latency_cpu,
             row["params_m"], row["model_size_mb"])
    for _, s in per_source.iterrows():
        log.info("%s test, source %s: %d images, acc %.4f macro-F1 %.4f", name, s["source"], s["n_images"],
                 s["accuracy"], s["macro_f1"])
    return row, per_class, per_source
