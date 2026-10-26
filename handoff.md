# Handoff: Training and Test Bench (Dissertation 2)

## Purpose
A command-line training and comparison bench. It trains five image classifiers on the same crab dataset under identical settings, evaluates them, and produces a ranked report. The top-ranked backbone is carried into Dissertation 3. The dataset is built from public sources only (see "Data" below).

Out of scope for this bench: the mobile app, the multi-task model, decision-support rules, and any paper.

## Current status (as of 2026-10-26)

| Item | Status |
|---|---|
| Requirements file (`requirements.txt`) | Written, installed and verified. |
| Build handout (`CLAUDE_CODE_HANDOUT.md`) | Written. |
| Setup guide (`CLAUDE_CODE_SETUP.md`) | Written, including git and GitHub steps. |
| Project folder and git repository | Done. Git repo on branch `main`. |
| GitHub account and private repo | Done. Remote `origin` is the private `crab-compare` repo; `main` is pushed. |
| Python environment | Done. `.venv` with Python 3.10.19, torch 2.14.1+cu126, CUDA available (True). GPU: NVIDIA GeForce GTX 1650, 4 GB VRAM. |
| Environment verification (`verify_env.py`) | Done. All five models and all package imports pass. |
| Code (`src/`, `run.py`, `config.yaml`) | Build steps 1–7 done and tested on dummy data only (holdout and 5-fold, 1 epoch, 1–2 seeds, random weights): group-safe split on `crab_id`, model factory, training loop, evaluation with saved predictions, Friedman/McNemar (Holm), Grad-CAM check, ONNX export and latency, thop FLOPs, weighted ranking with sensitivity check, and `report.html`. Ready for public data: each attribute uses only its labeled rows and is skipped (with a message) below 2 classes or 30 groups per class; with a `source` column it reports per-source accuracy/macro-F1 and flags classes where one source supplies > 80% of images. `python run.py --analyze-only` re-runs steps 5–7 on an existing results folder. Not yet run with pretrained weights or real data. |
| Crab dataset | Public datasets only (iNaturalist, GBIF, manually imported sets). Not assembled into `labels.csv` yet. The dataset builder (`tools/dataset_builder/`, see its README) is written and tested with mocked APIs; it resolves genera and species, also collects observations identified only to genus, fetches metadata, filters licenses, removes duplicates, writes `crab_id` by `--group-by`, and reports per-genus and per-species counts against the thresholds. No images downloaded yet. Genus dry run (2026-10-26, research grade, observer-level groups): Philippines only, *Portunus* 47 groups is the only trainable genus (*Scylla* 14, *Charybdis* 3); widened to 11 countries, *Scylla* 217, *Portunus* 451 and *Charybdis* 65 are all trainable, but every genus is > 80% iNaturalist and over half its photos come from one or two countries (*Scylla* 46% Taiwan, *Portunus* 57% Singapore). Optional *Atergatis*: 97 groups widened. Results in `data/external/runs/genus_*`. |
| Labels | Taken from the public sources; no in-house labeling. **Genus is the main label**; species is secondary and exploratory. Sex and maturity are weak and unverified. Disease is dropped from D2 (no public source provides it). |

## Decisions already made
- **Models (five, via `timm`):** EfficientNet-B3, ResNet-50, DenseNet-121, MobileNetV3 (`mobilenetv3_large_100`), ViT (`vit_base_patch16_224`).
- **Data sources: public datasets only** (decided 2026-10-26): iNaturalist, GBIF, and manually imported sets. `labels.csv` has a `source` column naming where each image came from.
- **Main task: genus-level classification** (decided 2026-10-26; `config.yaml` `data.main_attribute: genus`). Species-level classification is kept as a **secondary, exploratory** result. Reasons:
  - **Label reliability for *Scylla*:** the *Scylla* species are hard to tell apart from photos, so public species-level identifications are the least reliable labels in the set, and many observations are identified only to genus. Genus labels are far more dependable.
  - **Thin data for *S. tranquebarica*:** it has too few groups to train on as a species (21 observer-level groups in the earlier widened dry run, below the 30-group threshold), but its photos still count towards *Scylla*.
  - The dataset builder also collects observations identified only to genus (`id_rank = genus`), which a species-level task would have to drop.
- **Attributes in D2:** genus (main), species (secondary, exploratory), sex, maturity (`config.yaml` `data.attributes`). **Disease is dropped** from D2 because no public source provides it; the code still accepts a disease column but it is not configured. **Sex and maturity labels are weak and unverified**: they come from observers' notes or annotations in the public sources, not from an expert, and must be reported as such. Age and hormonal indicators are deferred. Hormones are not measurable from photos; the D3 proxy is "visible reproductive and molting indicators." Final list to be confirmed with the adviser before training.
- **Missing labels:** each attribute is trained and evaluated only on rows where its label is non-empty. An attribute is skipped, with a logged reason, if it has fewer than 2 classes or any class has fewer than `data.min_groups_per_class` (30) distinct `crab_id` groups. Used and skipped attributes are recorded in `results/<run>/attributes.json` and `report.html`.
- **Split by `crab_id` group, not by image.** `crab_id` is a grouping key, not a verified individual: depending on the source it is an observer, an observation, or an ID taken from the filename. All rows with the same `crab_id` stay in one set. Stratified where possible: the genus split is stratified on genus/species strata (`split.stratify_by`), with genus-only groups as their own stratum and strata under 10 groups folded into their genus, so every fold keeps each genus's species mix and every species with at least 10 groups appears in train, val and test of every fold. Fixed seed, saved to disk, reused by every model. The split cannot stop the same physical crab from appearing under two different groups (for example the same animal posted twice, or the same photo in two sources).
- **Source checks:** with a `source` column, every run reports test accuracy and macro-F1 per source, and any class where one source supplies more than 80% of the images (`data.source_dominance_threshold`) is flagged, since a model can learn that source's camera, background or framing instead of the crab.
- **Taxa (`tools/dataset_builder/species.yaml`):** genera *Scylla*, *Portunus*, *Charybdis*; species *S. serrata*, *S. olivacea*, *S. tranquebarica*, *S. paramamosain*, *P. pelagicus*, *P. sanguinolentus*, *C. feriata*. *Atergatis* / *Atergatis floridus* is an **optional candidate**: counted on its own line, left out of the totals and `labels.csv` unless `--include-optional`. The list still needs a taxonomist's check.
- **Quality grade:** research grade by default. Genus-rank iNaturalist `needs_id` observations are included only with `genus_needs_id: true` or `--include-genus-needs-id`, and are flagged `review_flag = genus_needs_id`.
- **One backbone for all attributes.** Selection uses one pre-written rule, not per-attribute picks by hand.
- **Selection score:** 50% macro-F1, 20% inference time, 15% model size, 15% Grad-CAM compatibility. Min-max normalised. Weights fixed before results are seen.
- **Fairness rules:** identical augmentation, optimizer, learning rate, batch size, epochs and early-stopping rule for every model. "Batch size" means the effective batch size (32): each model has its own micro-batch to fit the 4 GB GPU, and gradient accumulation brings every model to the same 32.
- **Mixed precision (AMP): off for all five models (`training.amp: false`), final (decided 2026-10-26).** One global setting, never per model. Reasons, measured on the GTX 1650 (4 GB), torch 2.14.1+cu126, cuDNN 9.10.2:
  - All ten configurations (five models, AMP on and off) fit in 4 GB at their configured batch sizes. The tightest is EfficientNet-B3 in fp32, at about 3.4 GB including the CUDA context.
  - AMP was slower on this GPU, which has no tensor cores. fp32 training steps were 1.3–3.8× faster depending on the model.
  - A cuDNN fp16 bug in ResNet-50's `layer4.0.downsample.0` (1×1 conv, 1024→2048, stride 2) produces NaNs that depend on batch size, not on the images. In eval it hits batch sizes 3–6, 11–13, 20 and larger, which leftover val/test batches can be. It is not an overflow: the fp32 output is below 4. The same layer is fine with cuDNN off or in fp32.
  - The AMP code path is kept. `run.py` prints a warning if `amp: true` is set.
- **Statistics:** Friedman test across models, McNemar test for pairs.
- **No containers.** A virtual environment is sufficient for now.

## Open questions (must be resolved before training)
1. **Sample size:** distinct `crab_id` groups per class. The bench skips any attribute with a class under 30 groups; the target is at least 50 per genus (and per species for the exploratory species task). Count groups, not photos.
2. **Validation strategy:** the handout specifies a 70/15/15 group-level split with 3 to 5 seeds. If the dataset is small, switch to 5-fold group-level cross-validation. Decide before training.
3. **Attribute list:** disease is dropped. Confirm with the adviser whether the weak sex and maturity labels are reported in D2 (and if so, as exploratory results with their label source stated), and whether egg-bearing/molt is included.
4. **Label quality:** which public labels to accept (for example iNaturalist research-grade only), whether to spot-check a sample of species, sex and maturity labels against a reference, and whether to report agreement (Cohen's kappa) on that sample.
5. **Grouping key per source:** which field becomes `crab_id` for each source (observation ID, observer, or filename ID), and how to find duplicates across sources (the same photo or animal in iNaturalist and GBIF) before splitting.
6. **Source confounding:** what to do when a class is flagged (one source > 80% of its images): rebalance, add sources, or report it as a limitation.
7. **Image licences:** record each image's licence and attribution from the source, and check that the licences allow use in the dissertation.
8. **Selection and the exploratory attributes:** the ranking averages test macro-F1 over every trained attribute, so the exploratory species, sex and maturity results currently count as much as genus. Decide (before results are seen) whether selection should use genus only, or keep the average.
9. **Atergatis floridus:** decide whether the optional candidate joins the task (as a fourth genus or a "not a target crab" class) or stays out.
10. **Genus-rank `needs_id`:** decide whether to include these flagged records, and whether to require `identifications=most_agree` for them.
11. **Mobile app:** not in this bench. Current recommendation is a basic on-device speed test in D2 and the full app in the final dissertation. Confirm with adviser.

Resolved: the GitHub account and private `crab-compare` repository exist, and git name/email are set by the user (Claude Code must still not invent or change them).

## Next steps, in order
Done: GitHub repo, environment setup and verification, and build steps 1–7 of `CLAUDE_CODE_HANDOUT.md`, tested on dummy data.
1. Resolve the open questions above with the adviser.
2. Review the dry-run counts in `data/external/runs/`, then (after approval) download with the dataset builder, which writes `data/labels.csv`: `image_path`, `crab_id` (the agreed grouping key), `genus`, `species` (blank when identified only to genus), `sex`, `maturity` (blank where unknown), `source`, `id_rank`, `quality_grade`, `review_flag`; duplicates across sources removed.
3. Run `python run.py --attribute all --splits-only` to see which attributes are usable, the genus/species composition of every fold, and the source-balance warnings before training.
4. First real run with pretrained weights (needs internet once, about 550 MB of weights).
5. Commit code changes after each step; never commit data or results.

## Expected layout
```
crab-compare/
  requirements.txt
  config.yaml
  data/              (not in git)
  src/
    config.py  data.py  models.py  train.py  evaluate.py  stats.py  gradcam.py  export.py  select.py
    analyze.py  report.py
  scripts/           make_dummy_data.py, check_models.py, smoke_train.py
  tests/             python -m unittest discover -s tests -t .
  run.py
  results/           (not in git)
```

## Commands (target interface)
```
python run.py --config config.yaml --models all --seeds 3                      # main task: genus
python run.py --config config.yaml --attribute species --models all --seeds 3  # secondary, exploratory
python run.py --config config.yaml --attribute all --models resnet50 efficientnet_b3
python run.py --config config.yaml --attribute all --splits-only      # check usable attributes and sources
python run.py --config config.yaml --run-name <run> --analyze-only    # re-run statistics, ranking, report
```

## Rules for whoever continues this work
- Do not change the split, augmentation or hyperparameters for one model only.
- Do not choose the backbone by hand or per attribute.
- Do not promise exact age or hormone levels in any output.
- Do not force-push or commit data, checkpoints or results.
- Ask the user before downloading large datasets or changing the model list.
- Log random seeds, GPU model, CUDA version and package versions with every result.
- Do not describe `crab_id` as a verified individual, or sex and maturity labels as verified, in any output.

## Reference files
- `requirements.txt`
- `CLAUDE_CODE_HANDOUT.md`
- `CLAUDE_CODE_SETUP.md`
