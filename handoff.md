# Handoff: Training and Test Bench (Dissertation 2)

## Purpose
A command-line training and comparison bench. It trains five image classifiers on the same crab dataset under identical settings, evaluates them, and produces a ranked report. The top-ranked backbone is carried into Dissertation 3.

Out of scope for this bench: the mobile app, the multi-task model, decision-support rules, and any paper.

## Current status (as of 2026-10-24)

| Item | Status |
|---|---|
| Requirements file (`requirements.txt`) | Written. Not yet installed or tested. |
| Build handout (`CLAUDE_CODE_HANDOUT.md`) | Written. |
| Setup guide (`CLAUDE_CODE_SETUP.md`) | Written, including git and GitHub steps. |
| Project folder and git repository | Not created yet. No git repo exists in `/home/claude`. |
| GitHub account and private repo | Not confirmed. User must create them personally. |
| Python environment | Not created. |
| Code (`src/`, `run.py`, `config.yaml`) | Not started. |
| Crab dataset | Not collected or labeled yet. |
| Labeling guide and expert | Not confirmed. |

Nothing has been installed or executed in this environment yet.

## Decisions already made
- **Models (five, via `timm`):** EfficientNet-B3, ResNet-50, DenseNet-121, MobileNetV3 (`mobilenetv3_large_100`), ViT (`vit_base_patch16_224`).
- **Attributes in D2:** species, sex, disease, maturity. Age and hormonal indicators are deferred. Hormones are not measurable from photos; the D3 proxy is "visible reproductive and molting indicators." Final list to be confirmed with the adviser before training.
- **Split by individual crab, not by image.** All photos of one crab stay in one set. Stratified where possible. Fixed seed, saved to disk, reused by every model.
- **One backbone for all attributes.** Selection uses one pre-written rule, not per-attribute picks by hand.
- **Selection score:** 50% macro-F1, 20% inference time, 15% model size, 15% Grad-CAM compatibility. Min-max normalised. Weights fixed before results are seen.
- **Fairness rules:** identical augmentation, optimizer, learning rate, batch size, epochs and early-stopping rule for every model.
- **Statistics:** Friedman test across models, McNemar test for pairs.
- **No containers.** A virtual environment is sufficient for now.

## Open questions (must be resolved before training)
1. **Sample size:** distinct crabs per species. Target is at least 50 per species; floor is about 30 with an honest caveat. Count crabs, not photos.
2. **Validation strategy:** the handout specifies a 70/15/15 crab-level split with 3 to 5 seeds. If the dataset is small, switch to 5-fold crab-level cross-validation. Decide before training.
3. **Attribute list:** confirm with adviser whether maturity and egg-bearing/molt are included in D2.
4. **Labeling:** who labels, labeling guide written, and inter-rater agreement (Cohen's kappa) planned.
5. **Git and GitHub:** user creates the account and a private `crab-compare` repository, then provides username and the git name/email. Claude Code must not invent these.
6. **Mobile app:** not in this bench. Current recommendation is a basic on-device speed test in D2 and the full app in the final dissertation. Confirm with adviser.

## Next steps, in order
1. User creates the GitHub account and private repo (see `CLAUDE_CODE_SETUP.md`, section 2a).
2. Run `CLAUDE_CODE_SETUP.md` in Claude Code: prerequisites, git init, virtual environment, PyTorch (GPU or CPU build), `requirements.txt`, verification script. Stop and report results before building.
3. Build the bench in the order given in `CLAUDE_CODE_HANDOUT.md`:
   1. Project skeleton, `config.yaml`, data loader, crab-level split
   2. Model factory; forward-pass check for all five models
   3. Training loop on one model and one attribute, then all five
   4. Metrics and results table
   5. Statistics, Grad-CAM, ONNX export and latency
   6. Weighted ranking and report
   7. Test on a dummy dataset before any real photos are used
4. Commit after each completed step.
5. Collect and label data only after the pipeline passes on dummy data.

## Expected layout
```
crab-compare/
  requirements.txt
  config.yaml
  data/              (not in git)
  src/
    data.py  models.py  train.py  evaluate.py  stats.py  gradcam.py  export.py  select.py  report.py
  run.py
  results/           (not in git)
```

## Commands (target interface)
```
python run.py --config config.yaml --attribute species --models all --seeds 3
python run.py --config config.yaml --attribute all --models resnet50 efficientnet_b3
```

## Rules for whoever continues this work
- Do not change the split, augmentation or hyperparameters for one model only.
- Do not choose the backbone by hand or per attribute.
- Do not promise exact age or hormone levels in any output.
- Do not force-push or commit data, checkpoints or results.
- Ask the user before downloading large datasets or changing the model list.
- Log random seeds, GPU model, CUDA version and package versions with every result.

## Reference files
- `requirements.txt`
- `CLAUDE_CODE_HANDOUT.md`
- `CLAUDE_CODE_SETUP.md`
