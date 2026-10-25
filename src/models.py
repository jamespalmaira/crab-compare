"""timm model factory for the five backbones.

The model list lives in config.yaml (models:). Input size, mean, std and
crop_pct are read from each model's own timm config, never set by hand.
"""
import timm
import torch


def create_model(arch, num_classes, pretrained=True):
    return timm.create_model(arch, pretrained=pretrained, num_classes=num_classes)


def model_data_config(model):
    """timm's preprocessing config for this model: input_size, mean, std, crop_pct, interpolation."""
    return timm.data.resolve_data_config({}, model=model)


def param_count(model):
    return sum(p.numel() for p in model.parameters())


def dummy_batch(data_cfg, batch_size, num_classes, device):
    c, h, w = data_cfg["input_size"]
    x = torch.randn(batch_size, c, h, w, device=device)
    y = torch.randint(0, num_classes, (batch_size,), device=device)
    return x, y
