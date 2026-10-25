import sys
import timm
import torch

models = {
    "EfficientNet-B3": "efficientnet_b3",
    "ResNet-50": "resnet50",
    "DenseNet-121": "densenet121",
    "MobileNetV3": "mobilenetv3_large_100",
    "ViT": "vit_base_patch16_224",
}

failures = []
for name, arch in models.items():
    try:
        m = timm.create_model(arch, pretrained=False, num_classes=2).eval()
        size = timm.data.resolve_data_config({}, model=m)["input_size"][1]
        with torch.no_grad():
            out = m(torch.randn(1, 3, size, size))
        assert tuple(out.shape) == (1, 2), out.shape
        print(f"  OK   {name} ({arch}) output {tuple(out.shape)}")
    except Exception as e:
        failures.append(name)
        print(f"  FAIL {name} ({arch}): {e}")

packages = ["pytorch_grad_cam", "onnx", "onnxruntime", "sklearn", "scipy",
            "statsmodels", "albumentations", "cv2", "pandas", "matplotlib"]
for pkg in packages:
    try:
        __import__(pkg)
        print(f"  OK   import {pkg}")
    except Exception as e:
        failures.append(pkg)
        print(f"  FAIL import {pkg}: {e}")

print(f"\nPython {sys.version.split()[0]}, torch {torch.__version__}, "
      f"CUDA {torch.cuda.is_available()}")
sys.exit(1 if failures else 0)
