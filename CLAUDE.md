# Project instructions for Claude Code

This is the crab model comparison project (Dissertation 2). Read `CLAUDE_CODE_HANDOUT.md` for the build plan and `handoff.md` for current status.

## Setup
- Setup is automated by `bash setup.sh`. Run it from the project root. Use `TORCH_INDEX=cu126 bash setup.sh` (or the matching CUDA tag) on an NVIDIA GPU machine.
- After it runs, report the output to the user before starting any build work.
- The script never logs in to GitHub, sets git name/email, or pushes. Do not add code that does these things, and do not put tokens, passwords or personal emails in any file.

## Data
- The main task is genus-level classification (`data.main_attribute: genus`). Species-level classification is secondary and exploratory: *Scylla* species are hard to tell apart from photos, so species labels are the least reliable, and *S. tranquebarica* has too little data to train on as a species. Report species results as exploratory.
- The project uses public datasets only: iNaturalist, GBIF, and manually imported sets. `labels.csv` records each image's `source`, `genus`, `species` (blank when identified only to genus) and `id_rank`.
- Disease is dropped from Dissertation 2: no public source provides it.
- Sex and maturity labels are weak and unverified (taken from the public sources, not checked by an expert). Report them as such; never present them as verified.
- `crab_id` is a grouping key (an observer, an observation, or an ID from the filename), not a verified individual crab.
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
