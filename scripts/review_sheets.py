"""Contact sheets for the manual photo review, and the audit sheet of a random sample of verdicts.

    python scripts/review_sheets.py sheets --run cleaned      # data/review/sheets/<genus>_NN.jpg + index.csv
    python scripts/review_sheets.py audit                      # data/review/audit/audit.jpg + audit_sample.csv
    python scripts/review_sheets.py summary --run cleaned      # verdict counts and training-side groups
    python scripts/review_sheets.py errors --results results/full_genus_v1 --arch resnet50
                                                               # misclassified CV test photos (read-only)

Every photo in data/labels.csv goes on exactly one sheet (per genus, in labels.csv order), at most
30 per sheet in 5 columns, with its photo_id (from the run's metadata.csv) printed under it.
The error sheets show one model's misclassified cross-validation test photos for the chosen
confusions, once per photo with how many seeds got it wrong, plus photo_id, country and the
review verdict; they go in results/<run>/error_analysis/<arch>/ and change no label or data.
The audit sample is 20 photos from data/removals_proposed.csv and 20 marked usable in
data/review_log.csv, drawn with a fixed seed, with each verdict printed under the photo.
"""
import argparse
import random
import sys
from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import load_config  # noqa: E402
from src.data import load_labels, split_external  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
REVIEW = DATA / "review"
THUMB, COLUMNS, PER_SHEET, CAPTION = 256, 5, 30, 36


def font(size=15):
    for name in ("arial.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default()


def labels_with_ids(run):
    """labels.csv plus photo_id, joined on image_path from the run's metadata.csv."""
    labels = pd.read_csv(DATA / "labels.csv", dtype=str, keep_default_na=False)
    meta = pd.read_csv(DATA / "external" / "runs" / run / "metadata.csv", dtype=str, keep_default_na=False)
    df = labels.merge(meta[["image_path", "photo_id"]].drop_duplicates("image_path"), on="image_path", how="left")
    if (df["photo_id"] == "").any() or df["photo_id"].isna().any():
        raise SystemExit(f"some labels.csv rows have no photo_id in run {run}")
    return df


def draw_sheet(rows, out, captions):
    """rows: dicts with image_path; captions: one list of text lines per row."""
    n_rows = (len(rows) + COLUMNS - 1) // COLUMNS
    cell = THUMB + CAPTION
    sheet = Image.new("RGB", (COLUMNS * THUMB, n_rows * cell), "white")
    draw, f = ImageDraw.Draw(sheet), font()
    for i, (r, lines) in enumerate(zip(rows, captions)):
        x, y = (i % COLUMNS) * THUMB, (i // COLUMNS) * cell
        try:
            with Image.open(DATA / r["image_path"]) as im:
                im = im.convert("RGB")
                im.thumbnail((THUMB - 4, THUMB - 4))
                sheet.paste(im, (x + (THUMB - im.width) // 2, y + (THUMB - im.height) // 2))
        except Exception as e:  # unreadable file: leave the cell blank and say so
            draw.text((x + 8, y + THUMB // 2), f"UNREADABLE: {type(e).__name__}", fill="red", font=f)
        for k, line in enumerate(lines):
            draw.text((x + 4, y + THUMB + 1 + k * 17), line, fill="black", font=f)
        draw.rectangle([x, y, x + THUMB - 1, y + cell - 1], outline="#cccccc")
    out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out, quality=92)


def cmd_sheets(args):
    df = labels_with_ids(args.run)
    out_dir = REVIEW / "sheets"
    for old in out_dir.glob("*.jpg") if out_dir.exists() else []:
        old.unlink()
    index = []
    for genus, g in df.groupby("genus", sort=True):
        rows = g.to_dict("records")
        for s in range(0, len(rows), PER_SHEET):
            chunk = rows[s:s + PER_SHEET]
            name = f"{genus.lower()}_{s // PER_SHEET + 1:02d}.jpg"
            draw_sheet(chunk, out_dir / name, [[f"{i + 1}. {r['photo_id']}"] for i, r in enumerate(chunk)])
            index += [{"sheet": name, "position": i + 1, "photo_id": r["photo_id"], "genus": genus,
                       "image_path": r["image_path"]} for i, r in enumerate(chunk)]
    pd.DataFrame(index).to_csv(out_dir / "index.csv", index=False)
    print(f"{len(index)} photos on {len({r['sheet'] for r in index})} sheets in {out_dir}")


def cmd_audit(args):
    log = pd.read_csv(DATA / "review_log.csv", dtype=str, keep_default_na=False)
    proposed = pd.read_csv(DATA / "removals_proposed.csv", dtype=str, keep_default_na=False)
    index = pd.read_csv(REVIEW / "sheets" / "index.csv", dtype=str, keep_default_na=False)
    log = log.merge(index[["photo_id", "image_path"]], on="photo_id", how="left")
    rng = random.Random(args.seed)
    removed = log[log["photo_id"].isin(proposed["id"])].to_dict("records")
    usable = log[log["verdict"] == "usable"].to_dict("records")
    sample = rng.sample(removed, min(20, len(removed))) + rng.sample(usable, min(20, len(usable)))
    out = REVIEW / "audit" / "audit.jpg"
    draw_sheet(sample, out, [[f"{i + 1}. {r['photo_id']}", f"{r['verdict']}: {r['reason'] or '-'}"[:30]]
                             for i, r in enumerate(sample)])
    pd.DataFrame(sample).drop(columns="image_path").to_csv(REVIEW / "audit" / "audit_sample.csv", index=False)
    print(f"audit sheet: {len(sample)} photos ({min(20, len(removed))} proposed removals, "
          f"{min(20, len(usable))} usable) -> {out}")


def cmd_summary(args):
    log = pd.read_csv(DATA / "review_log.csv", dtype=str, keep_default_na=False)
    proposed = pd.read_csv(DATA / "removals_proposed.csv", dtype=str, keep_default_na=False)
    df = labels_with_ids(args.run)
    missing = set(df["photo_id"]) - set(log["photo_id"])
    print(f"photos in labels.csv: {len(df)}; with a verdict: {len(set(df['photo_id']) & set(log['photo_id']))}"
          f"; missing: {len(missing)}")
    table = log.groupby(["genus", "verdict"]).size().unstack(fill_value=0)
    table.loc["ALL"] = table.sum()
    print("\nVerdicts per genus\n" + table.to_string())
    print(f"\nproposed removals (unusable, high confidence): {len(proposed)}")

    cfg = load_config(ROOT / "config.yaml")
    minimum = cfg["data"].get("min_groups_per_class", 30)
    labels = load_labels(DATA / "labels.csv")
    ext = cfg["data"].get("external_test")
    rows = []
    for name, frame in (("now", labels), ("after", labels[~labels["image_path"].isin(
            df.loc[df["photo_id"].isin(proposed["id"]), "image_path"])])):
        internal = split_external(frame, {**ext, "report_by": "genus"})[0] if ext else frame
        rows.append(internal.groupby("genus")["crab_id"].nunique().rename(name))
    groups = pd.concat(rows, axis=1).fillna(0).astype(int)
    print(f"\nTraining-side observer groups per genus (minimum {minimum})\n" + groups.to_string())
    low = groups.index[groups["after"] < minimum].tolist()
    print("\nBELOW MINIMUM after proposed removals: " + ", ".join(low) if low
          else f"\nNo genus falls below {minimum} training-side groups after the proposed removals.")
    return 1 if low else 0


ERROR_SETS = {  # sheet name -> (true genus, predicted genera or None for any other)
    "scylla_to_portunus": ("Scylla", {"Portunus"}),
    "charybdis_to_portunus": ("Charybdis", {"Portunus"}),
    "atergatis_to_other": ("Atergatis", None),
}


def cmd_errors(args):
    results = Path(args.results)
    preds = pd.concat([pd.read_csv(f).assign(run=f.stem) for f in sorted((results / "predictions").glob(
        f"{args.arch}_*.csv"))], ignore_index=True)
    n_seeds =preds["run"].str.split("_").str[-2].nunique()  # <arch>_<attribute>_<seed>_<fold>
    df = labels_with_ids(args.run)
    log = pd.read_csv(DATA / "review_log.csv", dtype=str, keep_default_na=False)[["photo_id", "verdict", "reason"]]
    info = df[["image_path", "photo_id", "country", "species", "source"]].merge(log, on="photo_id", how="left")
    out_dir = results / "error_analysis" / args.arch
    rows = []
    for name, (true, targets) in ERROR_SETS.items():
        wrong = preds[(preds["true_label"] == true) & (preds["pred_label"] != true)]
        if targets is not None:
            wrong = wrong[wrong["pred_label"].isin(targets)]
        tested = preds[preds["true_label"] == true].groupby("image_path").size()
        g = wrong.groupby("image_path").agg(crab_id=("crab_id", "first"), n_wrong=("pred_label", "size"),
                                             predicted=("pred_label", lambda v: "/".join(sorted(set(v)))))
        g = g.join(tested.rename("n_tested")).reset_index().merge(info, on="image_path", how="left")
        g = g.sort_values(["n_wrong", "country", "photo_id"], ascending=[False, True, True])
        recs = g.to_dict("records")
        for k in range(0, len(recs), PER_SHEET):
            chunk = recs[k:k + PER_SHEET]
            draw_sheet(chunk, out_dir / f"{name}_{k // PER_SHEET + 1:02d}.jpg",
                       [[f"{k + i + 1}. {r['photo_id']} {r['country']}",
                         f"{true[:4]}>{r['predicted']} {r['n_wrong']}/{r['n_tested']}"] for i, r in enumerate(chunk)])
        rows += [{"sheet": name, "position": i + 1, "true_label": true, **{c: r[c] for c in (
            "predicted", "n_wrong", "n_tested", "photo_id", "country", "species", "source", "crab_id", "verdict",
            "reason", "image_path")}} for i, r in enumerate(recs)]
        all_true = preds[preds["true_label"] == true].drop_duplicates("image_path").merge(info, on="image_path")
        print(f"{name}: {len(recs)} photos ({g['crab_id'].nunique()} observers) wrong in at least one of "
              f"{n_seeds} seeds, {int((g['n_wrong'] == g['n_tested']).sum())} in every seed")
        print("   country of errors:", g["country"].value_counts().to_dict())
        print("   country of all test photos of this genus:", all_true["country"].value_counts().to_dict())
    pd.DataFrame(rows).to_csv(out_dir / "index.csv", index=False)
    print(f"sheets and index.csv in {out_dir}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("sheets", "summary"):
        p = sub.add_parser(name)
        p.add_argument("--run", default="cleaned", help="builder run whose metadata.csv gives photo_id")
    sub.add_parser("audit").add_argument("--seed", type=int, default=0)
    p = sub.add_parser("errors")
    p.add_argument("--results", required=True, help="results folder, e.g. results/full_genus_v1")
    p.add_argument("--arch", default="resnet50")
    p.add_argument("--run", default="cleaned", help="builder run whose metadata.csv gives photo_id")
    args = ap.parse_args()
    sys.exit({"sheets": cmd_sheets, "audit": cmd_audit, "summary": cmd_summary, "errors": cmd_errors}[args.cmd](args)
             or 0)


if __name__ == "__main__":
    main()
