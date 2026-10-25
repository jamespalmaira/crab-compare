# Handout for Claude Code: Crab Model Comparison Tool

## Goal
Build a command-line Python tool that trains five image classifiers on the same crab dataset under identical settings, evaluates them, and produces a ranked comparison report. The result tells us which backbone to use in Dissertation 3.

**Not in scope:** no mobile app, no paper, no web UI. The app comes later.

## Environment
- Python 3.10 to 3.12, fresh virtual environment.
- Install with `pip install -r requirements.txt` (file provided). For an NVIDIA GPU, install the CUDA build of torch/torchvision first.
- Use a GPU if available (`cuda`), otherwise fall back to CPU and warn the user.

## The five models (all available through `timm`)

| Name in report | timm model name | Input size |
|---|---|---|
| EfficientNet-B3 | `efficientnet_b3` | 300 (use 288 or 300; keep consistent with the pretrained config) |
| ResNet-50 | `resnet50` | 224 |
| DenseNet-121 | `densenet121` | 224 |
| MobileNetV3 | `mobilenetv3_large_100` | 224 |
| ViT | `vit_base_patch16_224` | 224 |

- Create each with `timm.create_model(name, pretrained=True, num_classes=N)`.
- Read mean, std and input size from `timm.data.resolve_data_config` so preprocessing matches each model.
- Model names go in a config dict so more can be added later.

## Data
- Folder layout: `data/<attribute>/<class_name>/*.jpg`, or a `labels.csv` with columns `image_path, crab_id, species, sex, disease, maturity`.
- The tool trains **one classifier per attribute** (species, sex, disease, maturity). The attribute is selected by a `--attribute` flag, or `all` to loop through them.
- **Split by `crab_id`, not by image**, so photos of the same crab never appear in both train and test. Use 70/15/15, stratified where possible, with a fixed seed. Save the split to disk so every model reuses it.
- Handle class imbalance with class-weighted loss, and report macro metrics.
- Augmentation (identical for all models): random flip, small rotation, color jitter, random resized crop. Apply to training only.

## Training protocol (identical across models)
- Optimizer AdamW, same learning rate, weight decay, batch size, and max epochs for all models, set in `config.yaml`.
- Cosine LR schedule, early stopping on validation macro-F1.
- Mixed precision on GPU.
- Run **3 to 5 seeds** per model (configurable). Save the best checkpoint per run.
- Everything configurable through `config.yaml` and CLI flags.

## Metrics (on the test set, per model, per seed)
- Accuracy, macro precision, macro recall, macro specificity, macro F1, ROC-AUC (one-vs-rest)
- Per-class precision, recall, F1
- Confusion matrix (saved as PNG and CSV)
- Inference time per image (ms): measured on CPU and GPU, with warm-up runs, averaged over many runs
- Parameter count, FLOPs, model file size in MB

## Statistical comparison
- **Friedman test** across the five models using per-seed (or per-fold) macro-F1.
- **McNemar test** between pairs of models on test-set predictions (use the same seed or the best run). Report p-values with a note on multiple comparisons.
- Output mean and standard deviation across seeds for every metric.

## Explainability check
- Run Grad-CAM (`pytorch-grad-cam`) on a sample of test images for every model and save the overlays.
- CNNs: use the last convolutional block as the target layer.
- ViT: use a `reshape_transform` and the last transformer block's norm layer as the target.
- Purpose: confirm each backbone is compatible with Grad-CAM before Dissertation 3. No quantitative scoring is needed.

## Mobile feasibility check
- Export each trained model to ONNX and run it with `onnxruntime` to confirm it exports cleanly and to record file size and latency.
- (Optional) TFLite conversion is a later step, not required here.

## Model selection output
Produce a weighted score per model. Weights are configurable, defaulting to:
- 50% macro-F1
- 20% inference time (lower is better)
- 15% model size (lower is better)
- 15% Grad-CAM compatibility (pass/fail from the check above)

Min-max normalise each criterion before weighting. Print the ranking and name the recommended backbone.

## Outputs
Write everything to `results/<run_name>/`:
- `summary.csv`: one row per model/attribute with mean and std for each metric
- `per_seed.csv`: raw results for every run
- `stats.json`: Friedman and McNemar results
- `ranking.csv` and `ranking.md`: weighted selection result
- `confusion_matrices/`, `gradcam/`, `checkpoints/`, `onnx/`
- `report.html`: a single page with the comparison table, bar charts of macro-F1 and latency, and the ranking

## Suggested project structure
```
crab-compare/
  requirements.txt
  config.yaml
  data/
  src/
    data.py        # dataset, crab_id-based split, augmentation
    models.py      # timm model factory for the five models
    train.py       # training loop, early stopping, seeds
    evaluate.py    # metrics, confusion matrix, latency
    stats.py       # Friedman, McNemar
    gradcam.py     # explainability check
    export.py      # ONNX export and benchmark
    select.py      # weighted ranking
    report.py      # HTML and CSV outputs
  run.py           # CLI entry point
```

## CLI example
```
python run.py --config config.yaml --attribute species --models all --seeds 3
python run.py --config config.yaml --attribute all --models resnet50 efficientnet_b3
```

## Build order
1. Project skeleton, `config.yaml`, data loader and `crab_id` split.
2. Model factory. Verify all five models create and run a forward pass.
3. Training loop on one model and one attribute, then extend to all five.
4. Metrics and the results table.
5. Statistics, Grad-CAM, ONNX export and latency.
6. Weighted ranking and report.
7. Test the whole pipeline on a small **synthetic or dummy dataset** (a few folders of random images) before real data arrives.

## Rules
- Do not use a different split, augmentation, or hyperparameters for any single model. Fairness is the point.
- Fix random seeds and log them.
- Keep code simple and readable. Add a short README with setup and usage.
- Ask before downloading large datasets or changing the model list.
