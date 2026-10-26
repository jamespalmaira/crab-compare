# Handoff: Training and Test Bench (Dissertation 2)

## Purpose
A command-line training and comparison bench. It trains five image classifiers on the same crab dataset under identical settings, evaluates them, and produces a ranked report. The top-ranked backbone is carried into Dissertation 3. The dataset is built from public sources only (see "Data" below). **The main task is genus-level classification** of four crab genera.

Out of scope for this bench: the mobile app, the multi-task model, decision-support rules, sex, maturity and disease, and any paper.

## Current status (as of 2026-10-26)

| Item | Status |
|---|---|
| Requirements file (`requirements.txt`) | Written, installed and verified. |
| Build handout (`CLAUDE_CODE_HANDOUT.md`) | Written. It predates the decisions below; where they differ, this file wins. |
| Setup guide (`CLAUDE_CODE_SETUP.md`) | Written, including git and GitHub steps. |
| Project folder and git repository | Done. Git repo on branch `main`. |
| GitHub account and private repo | Done. Remote `origin` is the private `crab-compare` repo; `main` is pushed. |
| Python environment | Done. `.venv` with Python 3.10.19, torch 2.14.1+cu126, CUDA available (True). GPU: NVIDIA GeForce GTX 1650, 4 GB VRAM. |
| Environment verification (`verify_env.py`) | Done. All five models and all package imports pass. |
| Code (`src/`, `run.py`, `config.yaml`) | Build steps 1–7 done and tested on dummy data only (random weights, 1 epoch): group-safe 5-fold split on `crab_id` stratified on species within genus, model factory, training loop, evaluation with saved predictions, the external Philippine test, Friedman/McNemar (Holm), Grad-CAM check, ONNX export and latency, thop FLOPs, genus-only ranking with the tie rule and a sensitivity check, and `report.html`. `python run.py --analyze-only` re-runs steps 5–7 on an existing results folder. Not yet run with pretrained weights or real data. **No training has been run.** |
| Crab dataset | Public datasets only (iNaturalist, GBIF). Built by `tools/dataset_builder/` (see its README) with the settings below; images downloaded 2026-10-26 (run `download`, 2,724 files, 560 MB), then rebuilt offline without GBIF preserved specimens (run `cleaned`). `data/labels.csv`: 1,524 photos, 594 observer groups (1,514 iNaturalist, 10 GBIF human observations; all identified to species). Training side after holding out the Philippines (photos / observer groups): *Atergatis* 131 / 59, *Charybdis* 73 / 46, *Portunus* 889 / 388, *Scylla* 285 / 143 (all ≥ 30 groups). Philippine external test: 119 / 64 (*Charybdis* 3 / 2). Contact sheets in `data/external/review/`, to be reviewed before training. |
| Labels | Taken from the public sources; no in-house labeling, and no label is verified. **Genus is the main label**; species is secondary and exploratory. |

## Decisions already made
- **Models (five, via `timm`):** EfficientNet-B3, ResNet-50, DenseNet-121, MobileNetV3 (`mobilenetv3_large_100`), ViT (`vit_base_patch16_224`).
- **Data sources: public datasets only** (decided 2026-10-26): iNaturalist and GBIF (manual imports are supported but none are used). `labels.csv` has `source`, `country`, `genus`, `species`, `id_rank` and `quality_grade` columns.
- **Main task: genus-level classification** (decided 2026-10-26; `config.yaml` `data.main_attribute: genus`). Species-level classification is kept as a **secondary, exploratory** result and is never scored. Reasons:
  - **Label reliability for *Scylla*:** the *Scylla* species are hard to tell apart from photos, so public species-level identifications are the least reliable labels in the set, and some observations are identified only to genus. Genus labels are far more dependable.
  - **Thin data for *S. tranquebarica*:** 21 observer groups in the widened dry run, below the 30-group threshold, so it cannot be trained as a species; its photos still count towards *Scylla*.
- **Genera (four):** *Scylla*, *Portunus*, *Charybdis* and *Atergatis*. ***Atergatis floridus* is included as a reported toxic look-alike of edible crabs**, for the project's safety angle; **it is not an edible crab.**
- **Species excluded: *Scylla paramamosain*** (`exclude_species` in the builder config). It has no Philippine photos, so its presence in the Philippines is uncertain, which conflicts with the "Philippine species" premise. It is fetched and counted, then left out of `labels.csv` (kept in `metadata.csv` with `skip_reason = excluded_species`). To confirm with a taxonomist.
- **Countries: the widened list** (PH, ID, MY, BN, SG, TH, VN, KH, TW, TL, PG), because **the Philippines alone supports only one trainable genus** (Philippine dry run: *Portunus* 47 observer groups; *Scylla* 14; *Charybdis* 3).
- **The data is about 95% non-Philippine, and genus is tied to country** (for example *Scylla* is 46% Taiwan and *Portunus* 57% Singapore). So **every Philippine photo is held out as an external test set** (`data.external_test`: `country = PH`), never used in any train, validation or cross-validation test split, and **every observer who appears in it is removed from the training side** (matched on `crab_id` and `observer_id`), so no person's photos are on both sides. Each trained run is tested on it, and `external_metrics.csv` and `report.html` give per-genus images and groups. It is the local check, not a precise estimate: some genera have very few Philippine groups (in the cleaned data *Charybdis* has 2 groups and 3 photos, *Scylla* 13 groups). **Tell the adviser that the dataset is mostly non-Philippine and that the Philippine external test is the local check**; it is a limitation, clearly stated.
- **GBIF preserved specimens are excluded** (decided 2026-10-26; builder `exclude_basis_of_record: [PRESERVED_SPECIMEN, MATERIAL_SAMPLE]`, a rule, not a list of images). The contact sheets showed specimens in jars and scanned ledger pages. Museum material is **not the app's target images** (field and market photos of live or fresh crabs), and its look (jars, trays, labels, ledger scans) is **genus-specific, so a model could use it as a shortcut** instead of the crab. The records stay in `metadata.csv` with `skip_reason = excluded_basis_of_record`; `LIVING_SPECIMEN` is kept. Effect (photos / groups before → after): *Atergatis* 372/97 → 335/75, *Charybdis* 105/65 → 89/49, *Portunus* 1847/451 → 1818/437, *Scylla* 428/172 → 414/159. All 31 genus-rank (genus-only) photos were museum records, so every remaining photo is identified to species.
- **Manual removals: `data/removals.csv`** (columns `id`: an `image_path` or `photo_id`; `reason`), applied by the builder when `labels.csv` is built, logged and counted (`skip_reason = removed_manually`). **Removals are made only from visual review of the contact sheets before training, never from model results.**
  - **Photo review (2026-10-26):** every one of the 1,524 photos then in `labels.csv` was looked at on contact sheets (`python scripts/review_sheets.py sheets`, 30 per sheet with `photo_id`), against criteria fixed beforehand: no crab visible, detached part or empty molt/shell only, cooked or served, crab too small to see, non-crab subject, corrupted or duplicate screenshot. Hard photos (blurry, dark, partial, in hand, basket, market, boat, water) were kept, and no photo was judged on its genus or species. Verdicts are in **`data/review_log.csv`** (`photo_id`, `genus`, `verdict` usable/unusable/unsure, `reason`, `confidence`): 1,386 usable, 60 unusable with high confidence, 78 unsure. The user audited a random sample of 40 (20 removals, 20 usable), approved the 60 removals, and decided to keep all 78 unsure photos. The 60 are in `data/removals.csv`: about 45 empty *Portunus* shells or molts on beaches, 13 detached claws or legs, 2 cooked crabs and 1 sea star (*Portunus* 53, *Charybdis* 7). Effect on labels.csv: 1,524 photos / 594 groups → 1,464 / 571; training side *Charybdis* 46 → 44 groups, *Portunus* 388 → 366, *Atergatis* 59 → 60 (an observer whose Philippine photos were all removed no longer drops out with the external set), *Scylla* 143 unchanged. Every genus stays above 30 training-side groups.
  - **Removals are applied after the 5-per-observer group cap, not before (changed 2026-10-26).** Before, a removal freed a slot and the cap filled it with another photo from the same observer, which nobody had reviewed. Now the cap's choice is made first (and does not depend on the removals file), and a removal just leaves a gap, so **every photo in `labels.csv` is one that was reviewed** (checked after the rebuild: all 1,464 have a verdict in `review_log.csv`). The cost is a few photos fewer for observers who lost one. The builder reports any removal that the cap had already left out of `labels.csv`. Note that the `groups_observer` column in `counts.csv` counts every kept photo before the cap, so it can be one or two groups above the stage table (e.g. *Charybdis* 47 vs 46); the stage table and `scripts/dataset_summary.py` are the figures to quote.
  - `data/removals.csv` and `data/review_log.csv` are the only files in `data/` that are versioned (`.gitignore` un-ignores them), so every removal and the review behind it are reproducible: commit them with each change. A new download or a different cap seed changes which photos are in `labels.csv`; review any photo without a verdict before training. Rebuild after editing it with `python -m tools.dataset_builder --rebuild-from download --run-name cleaned --overwrite-labels` (offline, no downloads).
- **Quality grade: iNaturalist research grade only.** Genus-rank `needs_id` observations are **off** (they would add label doubt, and *Scylla* is already trainable). The code path stays (`genus_needs_id`, flagged `review_flag = genus_needs_id`).
- **Grouping: `crab_id` = observer**, the strictest key. `crab_id` is a grouping key, not a verified individual crab. **At most 5 photos per observer within each genus** go into `labels.csv` (`max_photos_per_group: 5`, fixed seed), so a few prolific photographers do not dominate; the cap is within the genus so no genus loses a group. `metadata.csv` keeps every photo.
- **Validation: 5-fold cross-validation by observer group**, stratified on species within genus (`split.stratify_by`: genus-only groups form their own stratum, strata under 10 groups fold into their genus), so every fold keeps each genus's species mix. *Charybdis*, the smallest genus (65 groups), gives about 13 test groups per fold. Fixed seed, saved to disk, reused by every model. The split cannot stop the same physical crab from appearing under two groups (the same animal posted by two people, or the same photo in two sources).
- **Missing labels:** each attribute is trained and evaluated only on rows where its label is non-empty, and is skipped (with a logged reason) below 2 classes or below `data.min_groups_per_class` (30) groups in any class.
- **Class imbalance:** the class-weighted loss and macro-F1 are kept, because the genera are very uneven (*Portunus* 1847 photos before the cap, *Charybdis* 105).
- **Out of scope: sex, maturity and disease.** No public source gives disease, and the sex and maturity annotations are too sparse and unverified (in the widened dry run, male had 19 groups and juvenile 11). The code still accepts those columns; they are not configured.
- **No label or `crab_id` is to be described as verified** in any output: genus and species come from public identifications, and `crab_id` is a grouping key.
- **Source checks:** every run reports test accuracy and macro-F1 per source, and flags any class where one source supplies more than 80% of the images. Every genus is over 80% iNaturalist, so all are flagged; this is a known limitation.
- **One backbone for all attributes, scored on genus only.** Selection uses one pre-written rule, not per-attribute picks by hand. **Only the main attribute is scored** (`selection.attributes: [genus]`, decided 2026-10-26, before any results): exploratory results (species) are reported but must not affect the choice of backbone.
- **Selection score:** 50% macro-F1, 20% inference time (ONNX Runtime CPU latency, batch 1), 15% model size, 15% Grad-CAM compatibility. Min-max normalised. Weights fixed before results are seen.
- **Tie rule (on by default; the recommendation):** the task will probably be easy for all five models, so a model is **tied with the best** when it is not significantly worse by McNemar (Holm-corrected) and its mean macro-F1 is within one standard deviation of the best. Tied models are ranked by speed, size and Grad-CAM only (same weights); the others follow in plain-score order. `ranking.md` and `report.html` report both the plain weighted ranking and the tie-rule ranking and say which is the recommendation.
- **Fairness rules:** identical augmentation, optimizer, learning rate, batch size, epochs and early-stopping rule for every model. "Batch size" means the effective batch size (32): each model has its own micro-batch to fit the 4 GB GPU, and gradient accumulation brings every model to the same 32.
- **Mixed precision (AMP): off for all five models (`training.amp: false`), final (decided 2026-10-26).** One global setting, never per model. Reasons, measured on the GTX 1650 (4 GB), torch 2.14.1+cu126, cuDNN 9.10.2:
  - All ten configurations (five models, AMP on and off) fit in 4 GB at their configured batch sizes. The tightest is EfficientNet-B3 in fp32, at about 3.4 GB including the CUDA context.
  - AMP was slower on this GPU, which has no tensor cores. fp32 training steps were 1.3–3.8× faster depending on the model.
  - A cuDNN fp16 bug in ResNet-50's `layer4.0.downsample.0` (1×1 conv, 1024→2048, stride 2) produces NaNs that depend on batch size, not on the images. In eval it hits batch sizes 3–6, 11–13, 20 and larger, which leftover val/test batches can be. It is not an overflow: the fp32 output is below 4. The same layer is fine with cuDNN off or in fp32.
  - The AMP code path is kept. `run.py` prints a warning if `amp: true` is set.
- **Timing check on the real data (2026-10-26, `results/timing_20261026`, `scripts/timing_check.py`). Speed, memory and data loading only; its metrics are not results and are excluded from any analysis** (`run.py --analyze-only` refuses `timing_*` runs). Genus, fold 0, seed 0, 1 epoch, pretrained weights (downloaded then), configured micro-batches, effective batch 32, `amp: false`, GTX 1650. The data check passed: 1,321 images, 508 groups, 4 classes on the training side; no Philippine image or observer in any fold; train/val/test disjoint in every fold. 83 observers photographed more than one genus; each stays whole in one set, and the split stratifies them by their most common genus. Per model (train s / val s / test+external evaluation s / peak GPU reserved MiB): EfficientNet-B3 57 / 29 / 45 / 2576; ResNet-50 45 / 29 / 42 / 1880; DenseNet-121 46 / 31 / 63 / 2454; MobileNetV3 34 / 30 / 33 / 1832; ViT 90 / 33 / 72 / 2516. No out-of-memory errors, no non-finite losses or outputs. Validation takes about 30 s for every model on 197 images, which points to data-loader start-up and image decoding (2 workers on Windows) rather than the model; later epochs reuse persistent workers, so these first-epoch figures lean high. **Estimate for the full run** (5 models × 5 folds × 3 seeds × up to 50 epochs, plus evaluation): **upper bound about 89 h; likely about 45 h** if early stopping ends runs at about 25 epochs (ViT about 13 h of that, MobileNetV3 about 7 h).
- **Steady-state timing, workers and image cache (2026-10-26; `results/timing_steady_20261026`, `timing_workers{0,2,4}_20261026`, `timing_cache512_20261026`). Timing only, excluded from any analysis.** Fold 0, seed 0, pretrained, 3 epochs, same settings as the main run. Epochs 2 and 3 agree within 1 s, and the first-run numbers above were inflated by start-up. Per epoch in steady state (train s / val s): EfficientNet-B3 29.8 / 2.7, ResNet-50 16.5 / 1.2, DenseNet-121 18.0 / 1.3, MobileNetV3 4.8 / 0.7, ViT 63.3 / 5.1. Each run also pays a fixed cost of about 53–70 s to start data-loader workers on Windows (train and val loaders) plus 31–76 s of test and external evaluation (including the test loader's start-up and the CPU latency timing).
  - **`num_workers`** (ResNet-50, 2 epochs): 0 → 22.9 / 2.1 s per epoch, no start-up; 2 → 16.5 / 1.2 s, about 54 s start-up; 4 → 16.5 / 1.2 s, about 114 s start-up. No errors. Steady state is GPU-bound from 2 workers; 2 beats 0 for any run longer than about 7–8 epochs, and 4 only adds start-up. **Keep 2** (the default).
  - **Image cache** (`data.cache_longer_side`, default off; built once into `data/cache/longer_<N>/`, outside git, read by training, evaluation and Grad-CAM alike, so identical for every model; images above N are resized with Lanczos, the rest copied unchanged; augmentation unchanged). With 512: **no speedup** (ResNet-50 16.5 / 1.2 → 16.6 / 1.2 s; ViT 63.3 / 5.1 → 63.3 / 5.2 s). The downloaded images are already iNaturalist "medium" size (median longer side 500 px, 10 of 1,464 above 512) and decode in about 4 ms each, so there is nothing to save. Leave it off.
  - **Resumable runs:** `run.py` writes `results/<run>/completed/<attribute>/<arch>_fold<k>_seed<s>.json` after a run's rows are in the tables, skips any combination with a marker, and logs `Progress: k/N ... ETA`. A marker stores the run's settings (training, augmentation, split, model, labels, cache, external test); resuming with different settings is refused, so a resumed run can never mix settings. Tables merge on (attribute, arch, fold, seed), so a re-run combination replaces its rows instead of duplicating them. Tested with `tests/test_resume.py` and by hard-killing a real dummy run part-way (2 of 8 done), then resuming: 6 trained, 8 rows in `per_seed.csv`, no duplicates. To resume, run the same command with the same `--run-name`.
  - **Full-run estimate** (5 models × 5 folds; steady-state epochs, 2 workers, no cache; likely = early stop at half the maximum epochs; early stopping cannot stop before epoch 11 with patience 10):

    | seeds | max epochs | runs | upper bound | likely | of which ViT (likely) |
    |---|---|---|---|---|---|
    | 3 | 50 | 75 | 32.1 h | 17.1 h | 7.7 h |
    | 3 | 30 | 75 | 20.1 h | 11.2 h | 4.8 h |
    | 2 | 50 | 50 | 21.4 h | 11.4 h | 5.1 h |
    | 2 | 30 | 50 | 13.4 h | 7.4 h | 3.2 h |
    | 1 | 50 | 25 | 10.7 h | 5.7 h | 2.6 h |
    | 1 | 30 | 25 | 6.7 h | 3.7 h | 1.6 h |

    Current defaults: seeds [0, 1, 2], 50 epochs, patience 10. Not changed; the user will choose.
- **Full-run settings, PRE-DECLARED on 2026-10-26 before any results (run name `full_genus_v1`):** main attribute **genus** (species is not trained in this run; it stays exploratory); seeds **[0, 1, 2]**; maximum epochs **50**; early-stopping patience **10**; **`amp: false`**; **`data.num_workers: 2`**; **image cache off** (`data.cache_longer_side: null`); **no cap on *Portunus* photos and no balanced sampling** (imbalance is handled by the class-weighted loss and macro-F1, as decided earlier). Reasons: the steady-state timing estimate is **about 17 h likely and 32 h at most** for this setup, which is affordable, and the image cache gave **no speedup**. These are recorded in `config.yaml` too and must not be changed for this run.
  - **Run order: seed first, then fold, then model** (`run.run_order`), so each completed seed is a full five-model comparison on all five folds; a run stopped part-way still leaves complete seeds. The log says when a seed is complete.
  - **The final analysis covers genus only** (`selection.attributes`): statistics (`stats.json`, `mcnemar.csv`), Grad-CAM, ONNX and FLOPs (on a genus checkpoint), ranking and `report.html`. If species runs are in the same folder they stay in `per_seed.csv` and `summary.csv` as exploratory and the analysis does not fail (`tests/test_full_pipeline.py`).
  - **Commands** (in the project folder, virtual environment activated): full run `python run.py --config config.yaml --attribute genus --run-name full_genus_v1`; resume after an interruption: the same command (completed runs are skipped); analysis only: `python run.py --config config.yaml --run-name full_genus_v1 --analyze-only`.
- **Statistics:** Friedman test across models (folds as blocks), McNemar test for pairs (Holm-corrected).
- **No containers.** A virtual environment is sufficient for now.

## Open questions
1. **Taxonomy:** confirm the species names with a taxonomist, in particular the **blue swimming crab** (*Portunus pelagicus* in the sources; the complex has been split into several species) and the ***Scylla*** species, and confirm that excluding *S. paramamosain* is right.
2. **Scope:** confirm the scope with the adviser: genus-level main task, species as exploratory, sex, maturity and disease out of scope, *Atergatis floridus* as the toxic look-alike, and the mostly non-Philippine data with the Philippine external test as the local check.
3. **Contact-sheet review:** done for unusable photos (see Manual removals: 60 removed, 78 unsure kept). Wrong genera were deliberately not judged in that review; whether to check labels is open question 4.
4. **Label quality:** whether to spot-check a sample of genus (and species) labels against a reference, and report agreement (Cohen's kappa) on that sample.
5. **Duplicates across groups:** the same photo or animal under two observers or sources is only partly caught (GBIF copies of iNaturalist records and near-duplicate images are removed).
6. **Source confounding:** every genus is over 80% iNaturalist and concentrated in one or two countries; report it as a limitation, or rebalance if more sources are found.
7. **Image licences:** `ATTRIBUTION.md` lists each image's licence and credit; most are CC-BY-NC, so check that the dissertation's use is non-commercial, and register a GBIF derived dataset for the DOI.
8. **Mobile app:** not in this bench. Current recommendation is a basic on-device speed test in D2 and the full app in the final dissertation. Confirm with adviser.

Resolved on 2026-10-26: validation (5-fold by observer), selection attributes (genus only), *Atergatis* (fourth genus), genus-rank `needs_id` (off), attribute list (genus and species only). The GitHub account and private `crab-compare` repository exist, and git name/email are set by the user (Claude Code must still not invent or change them).

## Next steps, in order
Done: GitHub repo, environment setup and verification, build steps 1–7 tested on dummy data, dataset settings, image download.
1. Photo review done (2026-10-26): 60 removals applied after the group cap; every photo in `labels.csv` has a verdict.
2. Confirm the taxonomy and scope (open questions 1 and 2).
3. Run `python run.py --attribute all --splits-only` to see the external split, the usable attributes, the genus/species composition of every fold, and the source-balance warnings.
4. Full run `full_genus_v1` with the pre-declared settings (weights already downloaded); commands under Decisions.
5. Commit code changes after each step; never commit data or results.

## Expected layout
```
crab-compare/
  requirements.txt
  config.yaml
  data/              (not in git)  labels.csv, external/<source>/<genus or species>/, external/review/, external/runs/
  src/
    config.py  data.py  models.py  train.py  evaluate.py  stats.py  gradcam.py  export.py  select.py
    analyze.py  report.py
  scripts/           make_dummy_data.py, check_models.py, smoke_train.py, dataset_summary.py,
                     review_sheets.py, timing_check.py
  tools/dataset_builder/   public-data builder (see its README)
  tests/             python -m unittest discover -s tests -t .
  run.py
  results/           (not in git)
```

## Commands (target interface)
```
python run.py --config config.yaml --models all --seeds 3                      # main task: genus
python run.py --config config.yaml --attribute species --models all --seeds 3  # secondary, exploratory
python run.py --config config.yaml --attribute all --splits-only      # external split, usable attributes, folds
python run.py --config config.yaml --run-name <run> --analyze-only    # re-run statistics, ranking, report
python -m tools.dataset_builder --dry-run                             # counts only (widened, observer groups)
python -m tools.dataset_builder --rebuild-from download --run-name cleaned --overwrite-labels  # after removals
python scripts/dataset_summary.py --run cleaned                       # totals per genus, external split, disk
python scripts/timing_check.py --run-name timing_<date>               # data check + 1-epoch timing, NOT a result
```

## Rules for whoever continues this work
- Do not change the split, augmentation or hyperparameters for one model only.
- Do not choose the backbone by hand or per attribute.
- Do not promise exact age or hormone levels in any output.
- Do not force-push or commit data, checkpoints or results. The exceptions are `data/removals.csv` and `data/review_log.csv`, which are versioned.
- Ask the user before downloading large datasets or changing the model list.
- **Timing runs (`results/timing_*`) are excluded from any analysis.** They train 1 epoch on one fold only to measure speed, memory and data loading; never report, compare or analyse their metrics, and never point `--analyze-only` or a report at them.
- Log random seeds, GPU model, CUDA version and package versions with every result.
- Do not describe `crab_id` as a verified individual, or any label (genus, species, sex, maturity) as verified, in any output.

## Reference files
- `requirements.txt`
- `CLAUDE_CODE_HANDOUT.md`
- `CLAUDE_CODE_SETUP.md`
- `tools/dataset_builder/README.md`
