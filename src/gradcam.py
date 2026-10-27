"""Grad-CAM compatibility check and overlays (fp32).

CNNs use their last convolutional block as the target layer. ViT uses the
norm layer of its last transformer block with a reshape_transform that drops
the class token and folds the patch tokens back into a 14x14 grid.

A model passes when Grad-CAM runs, returns maps of the input's height and
width, and every value is finite. A map that is flat (no variation) is
reported as degenerate but still passes: that reflects an untrained model,
not an incompatible architecture.
"""
import logging
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402
from pytorch_grad_cam import GradCAM  # noqa: E402
from pytorch_grad_cam.utils.image import show_cam_on_image  # noqa: E402

from src.data import build_transforms  # noqa: E402

log = logging.getLogger(__name__)

TARGET_LAYERS = {
    "efficientnet_b3": ("blocks[-1]", lambda m: m.blocks[-1]),
    "resnet50": ("layer4[-1]", lambda m: m.layer4[-1]),
    "densenet121": ("features.denseblock4", lambda m: m.features.denseblock4),
    "mobilenetv3_large_100": ("blocks[-1]", lambda m: m.blocks[-1]),
    "vit_base_patch16_224": ("blocks[-1].norm1", lambda m: m.blocks[-1].norm1),
}


UNSAFE_FILENAME = re.compile(r'[<>:"/\\|?*\x00-\x1f]')  # characters Windows does not allow in a filename


def safe_filename(text):
    """text with every character Windows disallows replaced by '_' (crab_id is e.g. observer:inaturalist:123)."""
    return UNSAFE_FILENAME.sub("_", str(text)).rstrip(" .")


def vit_reshape_transform(model):
    prefix = getattr(model, "num_prefix_tokens", 1)
    h, w = model.patch_embed.grid_size

    def reshape(tokens):
        x = tokens[:, prefix:, :]
        return x.reshape(x.size(0), h, w, x.size(2)).permute(0, 3, 1, 2)
    return reshape


def load_images(pred_df, image_root, data_cfg, aug_cfg):
    """Eval-transformed tensors plus the matching un-normalized RGB images in [0, 1]."""
    tf = build_transforms(data_cfg, aug_cfg, train=False)
    mean = np.array(data_cfg["mean"]).reshape(1, 1, 3)
    std = np.array(data_cfg["std"]).reshape(1, 1, 3)
    tensors, rgbs = [], []
    for p in pred_df["image_path"]:
        with Image.open(Path(image_root) / p) as im:
            x = tf(image=np.asarray(im.convert("RGB")))["image"]
        tensors.append(x)
        rgbs.append(np.clip(x.permute(1, 2, 0).numpy() * std + mean, 0, 1).astype(np.float32))
    return torch.stack(tensors), rgbs


def run_gradcam(model, arch, pred_df, image_root, data_cfg, aug_cfg, out_dir, device, title):
    """Run Grad-CAM on the given test images; save overlays and a grid. Returns the check result."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    layer_name, get_layer = TARGET_LAYERS[arch]
    result = {"arch": arch, "gradcam_target_layer": layer_name, "gradcam_pass": False,
              "gradcam_degenerate": None, "gradcam_n_images": len(pred_df), "gradcam_error": "",
              "gradcam_grid": ""}
    try:
        model = model.to(device).float().eval()
        x, rgbs = load_images(pred_df, image_root, data_cfg, aug_cfg)
        reshape = vit_reshape_transform(model) if arch.startswith("vit") else None
        with GradCAM(model=model, target_layers=[get_layer(model)], reshape_transform=reshape) as cam:
            maps = cam(input_tensor=x.to(device), targets=None)  # target: each image's top class
        h, w = x.shape[-2:]
        if maps.shape != (len(x), h, w):
            raise ValueError(f"Grad-CAM maps have shape {maps.shape}, expected {(len(x), h, w)}")
        if not np.isfinite(maps).all():
            raise ValueError("Grad-CAM maps contain non-finite values")
        flat = [float(m.max() - m.min()) < 1e-6 for m in maps]
        result["gradcam_degenerate"] = f"{sum(flat)}/{len(flat)} flat maps"

        overlays = [show_cam_on_image(rgb, m, use_rgb=True) for rgb, m in zip(rgbs, maps)]
        for i, (ov, (_, r)) in enumerate(zip(overlays, pred_df.iterrows())):
            name = safe_filename(f"{i:02d}_{r['crab_id']}_true-{r['true_label']}_pred-{r['pred_label']}") + ".png"
            Image.fromarray(ov).save(out_dir / name)
        result["gradcam_grid"] = str(save_grid(overlays, pred_df, out_dir / "grid.png", title))
        result["gradcam_pass"] = True
    except Exception as e:  # any failure means the architecture failed the check
        result["gradcam_error"] = f"{type(e).__name__}: {e}"
        log.warning("%s Grad-CAM check failed: %s", arch, result["gradcam_error"])
    return result


def save_grid(overlays, pred_df, path, title):
    n = len(overlays)
    cols = min(4, n)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(2.6 * cols, 3.3 * rows), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    for ax, ov, (_, r) in zip(axes.flat, overlays, pred_df.iterrows()):
        ax.imshow(ov)
        ax.set_title(f"true {r['true_label']}\npred {r['pred_label']}", fontsize=8)
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)
    return path


def sample_predictions(pred_csv, n, seed=0):
    """A fixed sample of test images from a run's predictions file."""
    df = pd.read_csv(pred_csv)
    return df.sample(n=min(n, len(df)), random_state=seed).reset_index(drop=True)
