# Project instructions for Claude Code

This is the crab model comparison project (Dissertation 2). Read `CLAUDE_CODE_HANDOUT.md` for the build plan and `handoff.md` for current status.

## Setup
- Setup is automated by `bash setup.sh`. Run it from the project root. Use `TORCH_INDEX=cu126 bash setup.sh` (or the matching CUDA tag) on an NVIDIA GPU machine.
- After it runs, report the output to the user before starting any build work.
- The script never logs in to GitHub, sets git name/email, or pushes. Do not add code that does these things, and do not put tokens, passwords or personal emails in any file.

## Rules
- Split data by individual crab, never by image.
- Use the same split, augmentation, optimizer, epochs and seeds for all five models.
- Choose the backbone only with the pre-written weighted rule. No manual per-attribute picks.
- Do not promise exact age or hormone levels from photos.
- Do not commit `data/`, `results/`, checkpoints or model files.
- Do not force-push. Ask before any destructive git operation.
- Ask before downloading large datasets or changing the model list.
- Commit after each completed build step, with short descriptive messages.

## Build order
Follow section "Build order" in `CLAUDE_CODE_HANDOUT.md`. Test on dummy data before any real photos are used.
