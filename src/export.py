"""ONNX export, onnxruntime check and CPU latency; FLOPs with thop.

Uses the TorchScript-based exporter (dynamo=False), which needs no extra
packages. The exported model has a dynamic batch axis; outputs are compared
with PyTorch fp32 on CPU for a batch of 2.
"""
import copy
import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import pandas as pd
import thop
import torch

log = logging.getLogger(__name__)


def export_onnx(model, input_size, path, opset=17):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    model = model.cpu().float().eval()
    torch.onnx.export(model, (torch.randn(1, *input_size),), str(path), input_names=["input"],
                      output_names=["logits"], dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
                      opset_version=opset, dynamo=False)
    return path


def check_onnx(model, path, input_size, rtol, atol, seed=0):
    """Compare onnxruntime with PyTorch fp32 on CPU for a batch of 2. Latency is timed by onnx_latency."""
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(2, *input_size, generator=g)
    with torch.no_grad():
        ref = model.cpu().float().eval()(x).numpy()
    out = session.run(None, {"input": x.numpy()})[0]
    diff = float(np.abs(out - ref).max())
    match = bool(np.allclose(out, ref, rtol=rtol, atol=atol))
    return {"onnx_max_abs_diff": diff, "onnx_outputs_match": match,
            "onnx_size_mb": Path(path).stat().st_size / 1e6, "onnx_path": str(path)}


def onnx_input_size(path):
    """(C, H, W) of an exported model's input (the batch axis is dynamic)."""
    shape = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"]).get_inputs()[0].shape
    return tuple(int(v) for v in shape[1:])


def time_onnx(path, input_size, threads, warmup, runs, seed=0):
    """One ONNX Runtime CPU timing pass in this process, batch size 1, a fixed number of intra-op threads.

    Each of the `runs` timed calls is measured on its own; returns the median, quartiles and IQR in ms.
    """
    so = ort.SessionOptions()
    so.intra_op_num_threads = int(threads)
    so.inter_op_num_threads = 1
    so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    session = ort.InferenceSession(str(path), sess_options=so, providers=["CPUExecutionProvider"])
    x1 = np.random.default_rng(seed).standard_normal((1, *input_size)).astype(np.float32)
    for _ in range(warmup):
        session.run(None, {"input": x1})
    times = []
    for _ in range(runs):
        start = time.perf_counter()
        session.run(None, {"input": x1})
        times.append((time.perf_counter() - start) * 1000)
    q1, med, q3 = np.percentile(times, [25, 50, 75])
    return {"median_ms": float(med), "q1_ms": float(q1), "q3_ms": float(q3), "iqr_ms": float(q3 - q1),
            "mean_ms": float(np.mean(times)), "min_ms": float(np.min(times)), "max_ms": float(np.max(times)),
            "threads": int(threads), "warmup": int(warmup), "runs": int(runs), "ort_version": ort.__version__,
            "pid": os.getpid()}


def onnx_latency(models, protocol):
    """Latency protocol (analysis.onnx_latency): every repeat times each model in its own new process.

    models maps arch -> (onnx path, input size). Repeats are interleaved (repeat 1 of every model, then
    repeat 2, ...) so slow drift on the machine is spread over the models. Returns (per-repeat rows,
    per-model summary): onnx_cpu_ms is the median of the per-process medians.
    """
    threads, warmup = int(protocol["threads"]), int(protocol["warmup"])
    runs, repeats = int(protocol["runs"]), int(protocol["repeats"])
    rows = []
    for rep in range(1, repeats + 1):
        for arch, (path, input_size) in models.items():
            cmd = [sys.executable, "-m", "src.export", "time", str(path), ",".join(map(str, input_size)),
                   str(threads), str(warmup), str(runs)]
            out = subprocess.run(cmd, capture_output=True, text=True, check=True,
                                 cwd=Path(__file__).resolve().parents[1])
            res = json.loads(out.stdout.strip().splitlines()[-1])
            rows.append({"arch": arch, "repeat": rep, **res})
            log.info("%s latency repeat %d/%d (pid %d): median %.2f ms, IQR %.2f ms (%d threads, %d runs)",
                     arch, rep, repeats, res["pid"], res["median_ms"], res["iqr_ms"], threads, runs)
    df = pd.DataFrame(rows)
    summary = {arch: {"onnx_cpu_ms": float(g["median_ms"].median()), "onnx_cpu_iqr_ms": float(g["iqr_ms"].median()),
                      "onnx_cpu_medians_ms": ";".join(f"{v:.2f}" for v in g["median_ms"]),
                      "onnx_threads": threads, "onnx_runs": runs, "onnx_warmup": warmup, "onnx_repeats": repeats,
                      "ort_version": g["ort_version"].iloc[0]}
               for arch, g in df.groupby("arch", sort=False)}
    return df, summary


def flops(model, input_size):
    """thop count for one image: MACs and FLOPs (= 2 x MACs) in billions.

    thop counts nn.Module layers only, so ViT's attention matrix products
    (about 0.7 GMACs for ViT-B/16) are not included.
    """
    m = copy.deepcopy(model).cpu().float().eval()  # thop adds counters to modules; keep the original clean
    macs, params = thop.profile(m, inputs=(torch.randn(1, *input_size),), verbose=False)
    return {"gmacs": macs / 1e9, "gflops": 2 * macs / 1e9, "params_m_thop": params / 1e6}


if __name__ == "__main__":
    # One timing pass in a fresh process (used by onnx_latency):
    #   python -m src.export time <model.onnx> <C,H,W> <threads> <warmup> <runs>
    if len(sys.argv) != 7 or sys.argv[1] != "time":
        sys.exit("usage: python -m src.export time <model.onnx> <C,H,W> <threads> <warmup> <runs>")
    _, _, onnx_file, size, n_threads, n_warmup, n_runs = sys.argv
    print(json.dumps(time_onnx(onnx_file, tuple(int(v) for v in size.split(",")), int(n_threads), int(n_warmup),
                               int(n_runs))))
