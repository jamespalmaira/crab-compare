"""The final model for the app (a new step after the benchmark; the benchmark's results are only read).

    python scripts/final_model.py train       # all training-side groups, fixed epochs, external test once
    python scripts/final_model.py export      # model.onnx, model_meta.json, golden vectors
    python scripts/final_model.py confidence  # confidence_table.csv from the benchmark predictions, SHA-256s
    python scripts/final_model.py cutoffs     # atergatis_cutoff_table.csv from the same predictions, SHA-256s

`train` refuses to run twice for one run name, so the external test is only ever done once. Output goes
to results/<run>/ and to the export folder, both outside git.
"""
import argparse
import json
import logging
import sys
from datetime import date
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from run import hold_out_external  # noqa: E402
from src.config import load_config  # noqa: E402
from src.data import load_labels, prepare_attribute  # noqa: E402
from src.evaluate import evaluate_external  # noqa: E402
from src.final import (check_held_out, confidence_table, cutoff_table, export_final, external_table, median_best_epoch,  # noqa: E402
                       train_final, write_sha256s)
from src.train import limit_gpu_memory, resolve_device  # noqa: E402

log = logging.getLogger("final_model")


def held_out_rows(cfg, out_dir, attribute):
    """(training-side rows with a label, external rows), with the check that they share nothing."""
    internal, external = hold_out_external(cfg, load_labels(cfg["data"]["labels_csv"]), out_dir)
    rows, usage = prepare_attribute(internal, attribute, cfg["data"].get("min_groups_per_class", 30))
    if usage["skip_reason"]:
        raise SystemExit(f"{attribute}: {usage['skip_reason']}")
    check_held_out(rows, external)
    return rows, external


def cmd_train(args, cfg, out_dir, bench_dir):
    if (out_dir / "final_run.json").exists() or (out_dir / "external_metrics.csv").exists():
        raise SystemExit(f"{out_dir} already has a final run; the external test is done only once")
    device = resolve_device(cfg)
    limit_gpu_memory(cfg, device)
    per_seed = pd.read_csv(bench_dir / "per_seed.csv")
    epochs, best = median_best_epoch(per_seed, args.arch, args.attribute)
    log.info("Fixed epochs %d: median best epoch of %d benchmark runs of %s, %s", epochs, len(best), args.arch, best)
    rows, external = held_out_rows(cfg, out_dir, args.attribute)
    log.info("Training on %d images, %d groups, no validation split", len(rows), rows["crab_id"].nunique())
    model, data_cfg, info = train_final(cfg, args.arch, rows, args.attribute, epochs, args.seed, out_dir, device)

    metric_rows = evaluate_external(model, data_cfg, info, external, cfg, out_dir, device)
    pd.DataFrame(metric_rows).to_csv(out_dir / "external_metrics.csv", index=False)
    table = external_table(metric_rows)
    table.to_csv(out_dir / "external_per_genus.csv", index=False)
    print(table.to_string(index=False))
    summary = {"benchmark": str(bench_dir), "epochs": epochs, "benchmark_best_epochs": best,
               **{k: v for k, v in info.items() if k != "history"},
               "final_train_loss": info["history"][-1]["train_loss"], "external": metric_rows[0]}
    (out_dir / "final_run.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(f"epochs {epochs}, {info['n_train']} images, {info['n_groups']} groups, "
          f"{info['optimizer_steps']} optimizer steps, final train loss {summary['final_train_loss']:.4f}")


def cmd_export(args, cfg, out_dir, bench_dir):
    export_dir = Path(args.export_dir)
    if export_dir.exists() and not args.overwrite:
        raise SystemExit(f"{export_dir} already exists; pass --overwrite to replace it")
    _, external = held_out_rows(cfg, out_dir, args.attribute)
    metadata = Path(cfg["data"]["labels_csv"]).parent / "external" / "runs" / args.builder_run / "metadata.csv"
    if metadata.exists():  # the golden image's credit
        meta = pd.read_csv(metadata, dtype=str, keep_default_na=False)
        external = external.merge(meta[["image_path", "attribution"]].drop_duplicates("image_path"), on="image_path",
                                  how="left")
    ckpt = out_dir / "checkpoints" / f"{args.arch}_final_seed{args.seed}.pth"
    res = export_final(ckpt, external, cfg, export_dir, cfg["run_name"], date.today().isoformat())
    print(json.dumps({k: res[k] for k in ("onnx_outputs_match", "onnx_max_abs_diff", "onnx_size_mb", "golden_match",
                                          "golden_max_abs_diff")}, indent=1))
    print(json.dumps(res["golden"], indent=1))
    if not (res["onnx_outputs_match"] and res["golden_match"]):
        raise SystemExit("ONNX does not match PyTorch")
    print("\n".join(write_sha256s(export_dir)))


def benchmark_predictions(args, bench_dir):
    files = sorted((bench_dir / "predictions").glob(f"{args.arch}_{args.attribute}_*.csv"))
    if not files:
        raise SystemExit(f"No benchmark predictions of {args.arch} in {bench_dir}")
    return files, pd.concat([pd.read_csv(f) for f in files], ignore_index=True)


def cmd_cutoffs(args, cfg, out_dir, bench_dir):
    files, preds = benchmark_predictions(args, bench_dir)
    table = cutoff_table(preds, "Atergatis")
    export_dir = Path(args.export_dir)
    export_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(export_dir / "atergatis_cutoff_table.csv", index=False)
    print(f"{len(files)} prediction files, {len(preds)} predictions, {preds['image_path'].nunique()} distinct images")
    print(table.round(4).to_string(index=False))
    print("\n".join(write_sha256s(export_dir)))


def cmd_confidence(args, cfg, out_dir, bench_dir):
    files, preds = benchmark_predictions(args, bench_dir)
    classes = sorted(preds["true_label"].unique())
    table = confidence_table(preds, classes)
    export_dir = Path(args.export_dir)
    export_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(export_dir / "confidence_table.csv", index=False)
    print(f"{len(files)} prediction files, {len(preds)} predictions, {preds['image_path'].nunique()} distinct images")
    print(table[table["scope"] == "overall"].round(4).to_string(index=False))
    print("\n".join(write_sha256s(export_dir)))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["train", "export", "confidence", "cutoffs"])
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--run-name", default="final_densenet121_v1")
    ap.add_argument("--benchmark", default="full_genus_v1", help="benchmark run under results/ (only read)")
    ap.add_argument("--arch", default="densenet121")
    ap.add_argument("--attribute", default="genus")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--export-dir", default="exports/densenet121_genus_v1")
    ap.add_argument("--builder-run", default="cleaned", help="builder run whose metadata.csv credits the golden image")
    ap.add_argument("--overwrite", action="store_true", help="export: replace an existing export folder")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s", force=True)

    cfg = load_config(args.config)
    cfg["run_name"] = args.run_name
    results = Path(cfg.get("results_dir", "results"))
    out_dir, bench_dir = results / args.run_name, results / args.benchmark
    if args.run_name == args.benchmark or args.run_name.startswith("timing_"):
        raise SystemExit(f"{args.run_name} cannot be the final run's name")
    commands = {"train": cmd_train, "export": cmd_export, "confidence": cmd_confidence, "cutoffs": cmd_cutoffs}
    commands[args.cmd](args, cfg, out_dir, bench_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
