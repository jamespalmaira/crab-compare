"""Smoke test: short training runs on the dummy dataset, AMP on vs off.

For each model in config.yaml this trains for a few optimizer steps on the
dummy data (real loaders, augmentation, class-weighted loss, configured
micro-batch and gradient accumulation), once with AMP on and once with AMP
off. It reports peak GPU memory, seconds per optimizer step, whether the
training loss stayed finite, and whether eval-mode outputs on the val set
were finite. Weights are random (pretrained=False); nothing is saved.

Then it applies one global rule: AMP on for all models only if every model
stayed finite with AMP on, otherwise AMP off for all models.

    python scripts/make_dummy_data.py --out data/dummy
    python scripts/smoke_train.py --labels data/dummy/labels.csv
"""
import argparse
import copy
import gc
import logging
import sys
import time
import warnings
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import accumulation_steps, load_config  # noqa: E402
from src.data import (attribute_rows, build_datasets, build_loaders, class_weights,  # noqa: E402
                      get_split, load_labels)
from src.models import create_model, model_data_config  # noqa: E402
from src.train import (GPUOutOfMemory, NonFiniteLoss, accumulated_step, limit_gpu_memory,  # noqa: E402
                       make_scaler, oom_guard, resolve_device)

MiB = 2**20


def micro_batches(loader, device):
    """Endless stream of training micro-batches, re-shuffled each pass."""
    while True:
        for x, y in loader:
            yield x.to(device, non_blocking=True), y.to(device, non_blocking=True)


@torch.no_grad()
def eval_finite(model, loader, device, amp):
    model.eval()
    sizes, ok = [], True
    for x, _ in loader:
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp):
            out = model(x.to(device))
        sizes.append(len(x))
        ok = ok and bool(torch.isfinite(out).all())
    model.train()
    return ok, sizes


def smoke_run(cfg, rows, split, arch, amp, steps, device):
    name, bs = cfg["models"][arch]["name"], cfg["models"][arch]["batch_size"]
    accum = accumulation_steps(cfg, arch)
    t = cfg["training"]
    seed = t["seeds"][0]
    torch.manual_seed(seed)
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    result = {"peak": None, "s_step": None, "train": "", "eval": "", "eval_sizes": []}
    model = optimizer = None
    try:
        with oom_guard(name, arch, bs):
            n_classes = len(split["classes"])
            model = create_model(arch, n_classes, pretrained=False).to(device).train()
            datasets = build_datasets(cfg, rows, split, 0, model_data_config(model), seed)
            loaders = build_loaders(datasets, bs, cfg["data"]["num_workers"], seed)
            criterion = torch.nn.CrossEntropyLoss(weight=class_weights(datasets["train"].targets, n_classes).to(device))
            optimizer = torch.optim.AdamW(model.parameters(), lr=t["lr"], weight_decay=t["weight_decay"])
            scaler = make_scaler(amp)
            stream = micro_batches(loaders["train"], device)
            times = []
            try:
                for step in range(steps):
                    batches = [next(stream) for _ in range(accum)]
                    torch.cuda.synchronize(device)
                    start = time.perf_counter()
                    accumulated_step(model, batches, criterion, optimizer, scaler, amp, arch, step)
                    torch.cuda.synchronize(device)
                    times.append(time.perf_counter() - start)
                result["train"] = f"finite ({steps}/{steps})"
            except NonFiniteLoss:
                result["train"] = f"NaN at step {step + 1}/{steps}"
            ok, sizes = eval_finite(model, loaders["val"], device, amp)
            result["eval"], result["eval_sizes"] = ("finite" if ok else "NaN"), sizes
            timed = times[1:] or times  # first step includes cuDNN warm-up
            result["s_step"] = sum(timed) / len(timed) if timed else None
            result["peak"] = torch.cuda.max_memory_allocated(device) / MiB
    except GPUOutOfMemory as e:
        result["train"] = "OOM"
        result["error"] = str(e)
    finally:
        del model, optimizer
        gc.collect()
        torch.cuda.empty_cache()
    return result


def main():
    warnings.filterwarnings("ignore", category=UserWarning)
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--labels", default="data/dummy/labels.csv")
    ap.add_argument("--attribute", default="species")
    ap.add_argument("--steps", type=int, default=20, help="optimizer steps per run")
    args = ap.parse_args()

    cfg = load_config(args.config)
    cfg["data"]["labels_csv"] = args.labels
    device = resolve_device(cfg)
    if device.type != "cuda":
        print("FAIL: no CUDA GPU available; this smoke test needs the GPU.")
        return 1
    limit_gpu_memory(cfg, device)
    rows = attribute_rows(load_labels(args.labels), args.attribute)
    split = get_split(cfg, rows, args.attribute)
    props = torch.cuda.get_device_properties(device)
    print(f"GPU {props.name}; torch {torch.__version__}, CUDA {torch.version.cuda}, cuDNN {torch.backends.cudnn.version()}; "
          f"{args.labels} attribute={args.attribute}, train {len(split['folds'][0]['train'])} crabs; "
          f"{args.steps} optimizer steps per run, effective batch {cfg['training']['effective_batch_size']}\n")

    header = f"{'model':<16}{'batch x acc':>12}{'AMP':>5}{'peak MiB':>10}{'s/step':>8}  {'train loss':<20}{'eval (val) outputs':<20}"
    print(header)
    print("-" * len(header))
    all_finite_with_amp = True
    for arch, m in cfg["models"].items():
        for amp in (True, False):
            run_cfg = copy.deepcopy(cfg)
            run_cfg["training"]["amp"] = amp
            r = smoke_run(run_cfg, rows, split, arch, amp, args.steps, device)
            if amp and not (r["train"].startswith("finite") and r["eval"] == "finite"):
                all_finite_with_amp = False
            peak = f"{r['peak']:.0f}" if r["peak"] is not None else "-"
            sps = f"{r['s_step']:.2f}" if r["s_step"] is not None else "-"
            ev = f"{r['eval']} {r['eval_sizes']}" if r["eval"] else "-"
            print(f"{m['name']:<16}{m['batch_size']:>7} x {accumulation_steps(cfg, arch):<2}{'on' if amp else 'off':>5}"
                  f"{peak:>10}{sps:>8}  {r['train']:<20}{ev:<20}")
            if "error" in r:
                print(f"    {r['error']}")

    print("\nRule: AMP on for all five models only if every model stays finite with AMP on; otherwise off for all.")
    print(f"Recommendation: training.amp: {'true' if all_finite_with_amp else 'false'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
