"""Training building blocks shared by every model.

Build step 3 adds the full training loop here. For now this holds the GPU
handling every run needs: device choice, the memory cap, mixed precision,
gradient accumulation, and clear errors for out-of-memory and NaN losses.
"""
import logging
from contextlib import contextmanager

import torch

log = logging.getLogger(__name__)


class GPUOutOfMemory(RuntimeError):
    pass


class NonFiniteLoss(RuntimeError):
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
    return bool(cfg["training"].get("amp", True)) and device.type == "cuda"


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
