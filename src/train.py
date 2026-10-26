"""Training loop and the GPU handling every run shares.

Every model is trained the same way: AdamW, a cosine schedule over all
optimizer steps, class-weighted cross-entropy, gradient accumulation to one
effective batch size, and early stopping on validation macro-F1. Only the
micro-batch size differs per model (to fit GPU memory).
"""
import json
import logging
import math
import platform
import random
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score

from src.config import accumulation_steps
from src.data import build_datasets, build_loaders, class_weights
from src.models import create_model, model_data_config

log = logging.getLogger(__name__)


class GPUOutOfMemory(RuntimeError):
    pass


class NonFiniteLoss(RuntimeError):
    pass


class NonFiniteOutput(NonFiniteLoss):
    pass


def resolve_device(cfg):
    want = cfg.get("device", "auto")
    if want == "cpu":
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda", torch.cuda.current_device())
    if want == "cuda":
        raise RuntimeError("device: cuda was requested but torch.cuda.is_available() is False")
    log.warning("No CUDA GPU available; running on CPU. Training all five models will be slow.")
    return torch.device("cpu")


def limit_gpu_memory(cfg, device):
    """Cap PyTorch's allocator below the card's dedicated VRAM.

    On Windows the NVIDIA driver can spill into system RAM instead of raising
    out-of-memory, which makes training silently very slow. With a cap, going
    over raises an OutOfMemoryError that oom_guard turns into a clear message.
    """
    if device.type != "cuda":
        return None
    frac = cfg.get("gpu", {}).get("memory_fraction", 0.85)
    torch.cuda.set_per_process_memory_fraction(frac, device)
    total = torch.cuda.get_device_properties(device).total_memory
    return frac * total


def amp_enabled(cfg, device):
    return bool(cfg["training"].get("amp", False)) and device.type == "cuda"


def make_scaler(enabled):
    return torch.amp.GradScaler("cuda", enabled=enabled)


@contextmanager
def oom_guard(name, arch, batch_size):
    """Turn a CUDA out-of-memory error into a short message naming the model and batch size."""
    try:
        yield
    except torch.OutOfMemoryError:
        raise GPUOutOfMemory(
            f"{name} ({arch}) ran out of GPU memory at batch_size={batch_size}. "
            f"Lower models.{arch}.batch_size in config.yaml (it must still divide "
            "training.effective_batch_size). Batch size and model list are not changed automatically."
        ) from None


def check_finite(loss, arch, step):
    if not torch.isfinite(loss):
        raise NonFiniteLoss(f"{arch}: loss is {loss.item()} at step {step}. "
                            "Try training.amp: false or a lower learning rate.")


def accumulated_step(model, batches, criterion, optimizer, scaler, amp, arch, step):
    """One optimizer step over len(batches) micro-batches (gradient accumulation).

    The loss of each micro-batch is divided by the number of micro-batches, so
    the update matches a single batch of the effective size.
    """
    accum = len(batches)
    optimizer.zero_grad(set_to_none=True)
    total = 0.0
    for x, y in batches:
        with torch.autocast(device_type=x.device.type, dtype=torch.float16, enabled=amp):
            loss = criterion(model(x), y)
        check_finite(loss.detach(), arch, step)
        scaler.scale(loss / accum).backward()
        total += loss.item()
    scaler.step(optimizer)
    scaler.update()
    return total / accum


# ---------------------------------------------------------------- run setup

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def environment_info(device):
    import albumentations
    import sklearn
    import timm
    info = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "timm": timm.__version__,
        "albumentations": albumentations.__version__,
        "sklearn": sklearn.__version__,
        "numpy": np.__version__,
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else "none",
    }
    return info


class EarlyStopping:
    """Stop after `patience` epochs without a strictly better score (ties keep the earlier epoch)."""

    def __init__(self, patience):
        self.patience = patience
        self.best = -math.inf
        self.best_epoch = None
        self.bad_epochs = 0

    def step(self, score, epoch):
        """Return True if this epoch is the new best."""
        if score > self.best:
            self.best, self.best_epoch, self.bad_epochs = score, epoch, 0
            return True
        self.bad_epochs += 1
        return False

    @property
    def should_stop(self):
        return self.bad_epochs >= self.patience


@torch.no_grad()
def predict(model, loader, device, arch=""):
    """Class probabilities and targets for a loader. Always eval mode and fp32.

    Evaluation never uses autocast: fp16 eval gives batch-size-dependent NaNs
    in ResNet-50 on the GTX 1650 (see handoff.md).
    """
    was_training = model.training
    model.eval()
    probs, targets = [], []
    for x, y in loader:
        logits = model(x.to(device, non_blocking=True)).float()
        if not torch.isfinite(logits).all():
            raise NonFiniteOutput(f"{arch}: non-finite model outputs in evaluation (batch of {len(x)})")
        probs.append(torch.softmax(logits, dim=1).cpu())
        targets.append(y)
    model.train(was_training)
    return torch.cat(probs).numpy(), torch.cat(targets).numpy()


def macro_f1(targets, probs, n_classes):
    return float(f1_score(targets, probs.argmax(1), labels=list(range(n_classes)),
                          average="macro", zero_division=0))


def run_name(arch, fold, seed):
    return f"{arch}_fold{fold}_seed{seed}"


@contextmanager
def run_log(path):
    """Copy log messages of one run into its own file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(path, mode="w", encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    root = logging.getLogger()
    root.addHandler(handler)
    try:
        yield
    finally:
        root.removeHandler(handler)
        handler.close()


# ---------------------------------------------------------------- training loop

class TrainResult:
    def __init__(self, info, model, datasets, loaders, data_cfg):
        self.info, self.model, self.datasets, self.loaders, self.data_cfg = info, model, datasets, loaders, data_cfg


def train_one(cfg, arch, rows, split, fold, seed, out_dir, device):
    """Train one model for one attribute, fold and seed; return the best-val-F1 model."""
    t, m = cfg["training"], cfg["models"][arch]
    attribute, classes = split["attribute"], split["classes"]
    n_classes = len(classes)
    bs, accum = m["batch_size"], accumulation_steps(cfg, arch)
    amp = amp_enabled(cfg, device)
    name = run_name(arch, fold, seed)
    ckpt_path = Path(out_dir) / "checkpoints" / attribute / f"{name}.pth"
    log_path = Path(out_dir) / "logs" / attribute / f"{name}.log"

    with run_log(log_path), oom_guard(m["name"], arch, bs):
        set_seed(seed)
        model = create_model(arch, n_classes, pretrained=t["pretrained"]).to(device)
        data_cfg = model_data_config(model)
        datasets = build_datasets(cfg, rows, split, fold, data_cfg, seed)
        loaders = build_loaders(datasets, bs, cfg["data"]["num_workers"], seed)
        steps_per_epoch = len(loaders["train"]) // accum
        if steps_per_epoch == 0:
            raise ValueError(f"{arch}: {len(datasets['train'])} training images is fewer than one "
                             f"effective batch of {t['effective_batch_size']}")

        info = {
            "run": name, "attribute": attribute, "arch": arch, "model": m["name"], "fold": fold, "seed": seed,
            "split_strategy": split["strategy"], "split_seed": split["seed"], "classes": classes,
            "n_train": len(datasets["train"]), "n_val": len(datasets["val"]), "n_test": len(datasets["test"]),
            "batch_size": bs, "accumulation_steps": accum, "effective_batch_size": t["effective_batch_size"],
            "amp": amp, "pretrained": t["pretrained"], "epochs_max": t["epochs"], "lr": t["lr"],
            "weight_decay": t["weight_decay"], "early_stopping_patience": t["early_stopping_patience"],
            "steps_per_epoch": steps_per_epoch, "input_size": list(data_cfg["input_size"]),
            **environment_info(device),
        }
        log.info("run %s", json.dumps(info))

        criterion = torch.nn.CrossEntropyLoss(weight=class_weights(datasets["train"].targets, n_classes).to(device))
        optimizer = torch.optim.AdamW(model.parameters(), lr=t["lr"], weight_decay=t["weight_decay"])
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=t["epochs"] * steps_per_epoch)
        scaler = make_scaler(amp)
        stopper = EarlyStopping(t["early_stopping_patience"])
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        step = 0
        history = []  # per epoch: loss, val macro-F1, and training and validation seconds kept apart

        for epoch in range(1, t["epochs"] + 1):
            start = time.perf_counter()
            model.train()
            losses, group = [], []
            for x, y in loaders["train"]:
                group.append((x.to(device, non_blocking=True), y.to(device, non_blocking=True)))
                if len(group) < accum:
                    continue
                losses.append(accumulated_step(model, group, criterion, optimizer, scaler, amp, arch, step))
                scheduler.step()
                step += 1
                group = []
            # a trailing partial group is dropped, so every step uses the full effective batch
            if device.type == "cuda":
                torch.cuda.synchronize()
            train_s = time.perf_counter() - start

            val_start = time.perf_counter()
            val_probs, val_targets = predict(model, loaders["val"], device, arch)
            val_f1 = macro_f1(val_targets, val_probs, n_classes)
            val_s = time.perf_counter() - val_start
            improved = stopper.step(val_f1, epoch)
            if improved:
                torch.save({"state_dict": model.state_dict(), "arch": arch, "classes": classes,
                            "attribute": attribute, "fold": fold, "seed": seed, "epoch": epoch,
                            "val_macro_f1": val_f1, "data_cfg": data_cfg, "info": info}, ckpt_path)
            history.append({"epoch": epoch, "train_loss": float(np.mean(losses)), "val_macro_f1": val_f1,
                            "train_s": train_s, "val_s": val_s})
            log.info("%s epoch %d/%d train_loss %.4f val_macro_f1 %.4f lr %.2e %.1fs (train %.1fs, val %.1fs)%s",
                     name, epoch, t["epochs"], float(np.mean(losses)), val_f1, scheduler.get_last_lr()[0],
                     time.perf_counter() - start, train_s, val_s, " *best" if improved else "")
            if stopper.should_stop:
                log.info("%s early stop: no val macro-F1 improvement for %d epochs", name, stopper.patience)
                break

        best = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(best["state_dict"])
        info.update(epochs_trained=epoch, best_epoch=stopper.best_epoch, best_val_macro_f1=stopper.best,
                    checkpoint=str(ckpt_path), log=str(log_path), history=history)
        log.info("%s done: best epoch %d, val macro-F1 %.4f, checkpoint %s", name, stopper.best_epoch,
                 stopper.best, ckpt_path)
    return TrainResult(info, model, datasets, loaders, data_cfg)
