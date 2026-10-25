"""Forward/backward check for every model on the GPU, with peak memory.

For each model in config.yaml this runs full optimizer steps (AdamW, mixed
precision per config, gradient accumulation up to the effective batch size)
on random input at the model's own input size, and reports peak GPU memory
at its configured micro-batch size. No data or pretrained weights are used.

    python scripts/check_models.py                # configured batch sizes
    python scripts/check_models.py --probe 4 8 16 32 64   # also try other sizes
"""
import argparse
import gc
import logging
import sys
import time
import warnings
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import accumulation_steps, load_config  # noqa: E402
from src.models import create_model, dummy_batch, model_data_config, param_count  # noqa: E402
from src.train import (GPUOutOfMemory, NonFiniteLoss, accumulated_step, amp_enabled,  # noqa: E402
                       limit_gpu_memory, make_scaler, oom_guard, resolve_device)

NUM_CLASSES = 4
MiB = 2**20


def run_check(cfg, arch, batch_size, accum, device, amp, steps=2):
    """Run `steps` optimizer steps; return (peak allocated MiB, peak reserved MiB, ms per micro-batch)."""
    name = cfg["models"][arch]["name"]
    torch.manual_seed(0)
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    model = optimizer = None
    try:
        with oom_guard(name, arch, batch_size):
            model = create_model(arch, NUM_CLASSES, pretrained=False).to(device).train()
            data_cfg = model_data_config(model)
            t = cfg["training"]
            optimizer = torch.optim.AdamW(model.parameters(), lr=t["lr"], weight_decay=t["weight_decay"])
            criterion = torch.nn.CrossEntropyLoss()
            scaler = make_scaler(amp)
            batches = [dummy_batch(data_cfg, batch_size, NUM_CLASSES, device) for _ in range(accum)]
            # first step allocates AdamW state; time the later ones
            accumulated_step(model, batches, criterion, optimizer, scaler, amp, arch, 0)
            torch.cuda.synchronize(device)
            start = time.perf_counter()
            for s in range(1, steps):
                accumulated_step(model, batches, criterion, optimizer, scaler, amp, arch, s)
            torch.cuda.synchronize(device)
            ms = (time.perf_counter() - start) * 1000 / max(1, (steps - 1) * accum)
        return (torch.cuda.max_memory_allocated(device) / MiB,
                torch.cuda.max_memory_reserved(device) / MiB, ms, data_cfg["input_size"][1])
    finally:
        del model, optimizer
        gc.collect()
        torch.cuda.empty_cache()


def main():
    warnings.filterwarnings("ignore", category=UserWarning)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--probe", type=int, nargs="*", default=[], help="extra micro-batch sizes to try")
    ap.add_argument("--no-amp", action="store_true", help="override training.amp to false (fp32)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.no_amp:
        cfg["training"]["amp"] = False
    device = resolve_device(cfg)
    if device.type != "cuda":
        print("FAIL: no CUDA GPU available; this check needs the GPU.")
        return 1
    amp = amp_enabled(cfg, device)
    cap = limit_gpu_memory(cfg, device)
    props = torch.cuda.get_device_properties(device)
    eff = cfg["training"]["effective_batch_size"]
    print(f"GPU {props.name}, {props.total_memory / MiB:.0f} MiB total, allocator cap {cap / MiB:.0f} MiB "
          f"({cfg['gpu']['memory_fraction']:.0%}); torch {torch.__version__}, CUDA {torch.version.cuda}; "
          f"AMP {'on' if amp else 'off'}; effective batch {eff}\n")

    failures = []
    header = f"{'model':<16}{'arch':<24}{'input':>6}{'params':>9}{'batch':>7}{'accum':>7}" \
             f"{'peak alloc':>12}{'peak resv':>11}{'ms/batch':>10}  result"
    print(header)
    print("-" * len(header))
    for arch, m in cfg["models"].items():
        bs, accum = m["batch_size"], accumulation_steps(cfg, arch)
        params = param_count(create_model(arch, NUM_CLASSES, pretrained=False)) / 1e6
        try:
            alloc, resv, ms, size = run_check(cfg, arch, bs, accum, device, amp)
            print(f"{m['name']:<16}{arch:<24}{size:>6}{params:>8.1f}M{bs:>7}{accum:>7}"
                  f"{alloc:>9.0f} MiB{resv:>7.0f} MiB{ms:>10.0f}  OK")
        except (GPUOutOfMemory, NonFiniteLoss) as e:
            failures.append(arch)
            print(f"{m['name']:<16}{arch:<24}{'':>6}{params:>8.1f}M{bs:>7}{accum:>7}{'':>34}  FAIL")
            print(f"    {e}")

    if args.probe:
        print("\nProbe: one AdamW step per micro-batch size (peak allocated MiB; OOM = over the cap, "
              "NaN = non-finite loss)")
        print(f"{'model':<16}" + "".join(f"{b:>9}" for b in args.probe))
        for arch, m in cfg["models"].items():
            cells = []
            for b in args.probe:
                try:
                    alloc, *_ = run_check(cfg, arch, b, 1, device, amp, steps=1)
                    cells.append(f"{alloc:>9.0f}")
                except GPUOutOfMemory:
                    cells.append(f"{'OOM':>9}")
                except NonFiniteLoss:
                    cells.append(f"{'NaN':>9}")
            print(f"{m['name']:<16}" + "".join(cells))

    if failures:
        print(f"\nFAIL: {failures} did not pass at their configured batch size.")
        return 1
    print("\nAll models passed forward + backward at their configured batch size.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
