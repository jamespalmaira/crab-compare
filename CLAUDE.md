# Project instructions for Claude Code

This is the crab model comparison project (Dissertation 2). Read `CLAUDE_CODE_HANDOUT.md` for the build plan and `handoff.md` for current status.

## Setup
- Setup is automated by `bash setup.sh`. Run it from the project root. Use `TORCH_INDEX=cu126 bash setup.sh` (or the matching CUDA tag) on an NVIDIA GPU machine.
- After it runs, report the output to the user before starting any build work.
- The script never logs in to GitHub, sets git name/email, or pushes. Do not add code that does these things, and do not put tokens, passwords or personal emails in any file.

## Data
- The main task is genus-level classification (`data.main_attribute: genus`) of four genera: *Scylla*, *Portunus*, *Charybdis* and *Atergatis*. Species-level classification is secondary and exploratory: *Scylla* species are hard to tell apart from photos, so species labels are the least reliable, and *S. tranquebarica* has too little data to train on as a species. Report species results as exploratory.
- *Atergatis floridus* is included as a reported toxic look-alike (the safety angle). It is not an edible crab; say so wherever it appears.
- *Scylla paramamosain* is excluded (`exclude_species`): it has no Philippine photos, so its presence in the Philippines is uncertain.
- The project uses public datasets only: iNaturalist and GBIF, research grade only (genus-rank `needs_id` off). `labels.csv` records each image's `source`, `country`, `genus`, `species` (blank when identified only to genus) and `id_rank`.
- Countries are the widened list, because the Philippines alone supports only one trainable genus. The data is about 95% non-Philippine and genus is tied to country, so every Philippine photo is held out as the external test set (`data.external_test`), and observers who appear in it are removed from the training side. Describe it as the local check, and the mostly non-Philippine data as a limitation.
- `crab_id` is the observer (a grouping key), not a verified individual crab. At most 5 photos per observer within each genus go into `labels.csv`.
- Validation is 5-fold cross-validation by observer group, stratified on species within genus. The ranking scores genus only (`selection.attributes`), with the tie rule on.
- Sex, maturity and disease are out of scope for Dissertation 2.
- No label (genus, species, sex, maturity) and no `crab_id` may be described as verified in any output.
- Each attribute uses only the rows labeled for it, and is skipped with a message if it has too few classes or groups (see `config.yaml`).
- `CLAUDE_CODE_HANDOUT.md` predates these decisions; where it lists disease, treats species as the main label, or calls `crab_id` an individual, `handoff.md` wins.

## Rules
- Split data by `crab_id` group, never by image. (`crab_id` is the grouping key above; all rows with the same `crab_id` stay in one set.)
- Use the same split, augmentation, optimizer, epochs and seeds for all five models.
- Choose the backbone only with the pre-written weighted rule. No manual per-attribute picks.
- Do not promise exact age or hormone levels from photos.
- Do not commit `data/`, `results/`, checkpoints or model files.
- Do not force-push. Ask before any destructive git operation.
- Ask before downloading large datasets or changing the model list.
- Commit after each completed build step, with short descriptive messages.

## Build order
Follow section "Build order" in `CLAUDE_CODE_HANDOUT.md`. Test on dummy data before any real photos are used.
