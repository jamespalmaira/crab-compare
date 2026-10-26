"""ONNX export, onnxruntime check and CPU latency; FLOPs with thop.

Uses the TorchScript-based exporter (dynamo=False), which needs no extra
packages. The exported model has a dynamic batch axis; outputs are compared
with PyTorch fp32 on CPU for a batch of 2.
"""
import copy
import logging
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
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


def check_onnx(model, path, input_size, rtol, atol, warmup, runs, seed=0):
    """Compare onnxruntime with PyTorch and time ORT on CPU at batch size 1."""
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(2, *input_size, generator=g)
    with torch.no_grad():
        ref = model.cpu().float().eval()(x).numpy()
    out = session.run(None, {"input": x.numpy()})[0]
    diff = float(np.abs(out - ref).max())
    match = bool(np.allclose(out, ref, rtol=rtol, atol=atol))

    x1 = np.random.default_rng(seed).standard_normal((1, *input_size)).astype(np.float32)
    for _ in range(warmup):
        session.run(None, {"input": x1})
    start = time.perf_counter()
    for _ in range(runs):
        session.run(None, {"input": x1})
    latency = (time.perf_counter() - start) * 1000 / runs
    return {"onnx_max_abs_diff": diff, "onnx_outputs_match": match, "onnx_cpu_ms": latency,
            "onnx_size_mb": Path(path).stat().st_size / 1e6, "onnx_path": str(path)}


def flops(model, input_size):
    """thop count for one image: MACs and FLOPs (= 2 x MACs) in billions.

    thop counts nn.Module layers only, so ViT's attention matrix products
    (about 0.7 GMACs for ViT-B/16) are not included.
    """
    m = copy.deepcopy(model).cpu().float().eval()  # thop adds counters to modules; keep the original clean
    macs, params = thop.profile(m, inputs=(torch.randn(1, *input_size),), verbose=False)
    return {"gmacs": macs / 1e9, "gflops": 2 * macs / 1e9, "params_m_thop": params / 1e6}
