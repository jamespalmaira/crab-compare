"""Packet for a taxonomist: flagged photos, a random audit sample, contact sheets and the questions.

    python scripts/taxonomist_packet.py          # writes taxonomist_packet/ (outside git)

Reads data/labels.csv and the builder run's metadata.csv only, and changes neither. The audit sample
never reads results/ or data/review_log.csv, so no model result or review verdict can affect the
draw. The audit files leave out the genus and species labels, so the taxonomist can answer blind;
join on photo_id to compare (labels_with_ids in scripts/review_sheets.py).
"""
import argparse
import shutil
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.review_sheets import DATA, PER_SHEET, draw_sheet, labels_with_ids  # noqa: E402
from src.final import LABEL_NOTES  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
AUDIT_SEED = 20261102
AUDIT_PER_GENUS = 40
SHAREABLE = ("CC0", "CC-BY", "CC-BY-SA", "CC-BY-NC", "CC-BY-NC-SA")  # with attribution; NC only for non-commercial use
AUDIT_COLUMNS = ["photo_id", "sheet", "image_included", "observation_url", "license", "attribution", "taxonomist_label",
                 "confidence", "notes"]
FLAGGED_COLUMNS = ["photo_id", "genus_label", "source", "observation_url", "license", "observer_id", "country", "attribution",
                   "image_file", "reason_flagged"]

_SOURCE = "Noticed on the ResNet-50 error sheets; an impression, not an expert identification."
_SWIMMING = f"Labeled Scylla but looks like a swimming crab (paddle-shaped last legs, blue or purple colouring). {_SOURCE}"
FLAGGED = {  # the photos listed under "Error analysis" in handoff.md
    "gbif:3772633546:0": _SWIMMING,
    "inaturalist:photo:471525259": _SWIMMING,
    "inaturalist:photo:521146241": _SWIMMING,
    "inaturalist:photo:265230874": _SWIMMING,
    "inaturalist:photo:265230904": _SWIMMING,
    "inaturalist:photo:684540534": _SWIMMING,
    "inaturalist:photo:684540113": _SWIMMING,
    "inaturalist:photo:24051172": _SWIMMING,
    "inaturalist:photo:226379987": f"Labeled Atergatis; a dark crab in a hand that may be a different xanthid. {_SOURCE}",
    "inaturalist:photo:395641510": f"Labeled Charybdis; a market pile of mixed crabs that includes swimming crabs, so its "
                                   f"single label is ambiguous. {_SOURCE}",
}


def shareable(license_name):
    return license_name in SHAREABLE


def link_columns(rows):
    return rows["license"].map(lambda v: "yes" if shareable(v) else "no, link only")


def flagged_table(df, out_dir=None):
    """flagged.csv rows in FLAGGED order; with out_dir, also copies the shareable originals there."""
    missing = sorted(set(FLAGGED) - set(df["photo_id"]))
    if missing:
        raise SystemExit(f"flagged photos not in labels.csv: {missing}")
    rows = []
    for photo_id, reason in FLAGGED.items():
        r = df[df["photo_id"] == photo_id].iloc[0]
        image_file = ""
        if shareable(r["license"]):
            image_file = "flagged_images/" + photo_id.replace(":", "_") + Path(r["image_path"]).suffix.lower()
            if out_dir is not None:
                target = Path(out_dir) / image_file
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(DATA / r["image_path"], target)
        rows.append({"photo_id": photo_id, "genus_label": r["genus"], "source": r["source"],
                     "observation_url": r["observation_url"], "license": r["license"], "observer_id": r["observer_id"],
                     "country": r["country"], "attribution": r["attribution"], "image_file": image_file,
                     "reason_flagged": reason})
    return pd.DataFrame(rows, columns=FLAGGED_COLUMNS)


def draw_audit_sample(df, seed=AUDIT_SEED, per_genus=AUDIT_PER_GENUS):
    """per_genus photos per genus (all if fewer), in shuffled order, so sheets do not show the genus."""
    parts = []
    for genus in sorted(df["genus"].unique()):
        rows = df[df["genus"] == genus].sort_values("photo_id")
        parts.append(rows.sample(n=min(per_genus, len(rows)), random_state=seed))
    return pd.concat(parts).sample(frac=1, random_state=seed).reset_index(drop=True)


def audit_table(sample):
    """audit_sample.csv: where to see each photo and empty columns to fill in. No community label."""
    out = sample[["photo_id", "observation_url", "license", "attribution"]].copy()
    included = link_columns(sample)
    position = included.eq("yes").cumsum()
    out["sheet"] = [f"audit_sheets/sheet_{(p - 1) // PER_SHEET + 1:02d}.jpg" if i == "yes" else ""
                    for p, i in zip(position, included)]
    out["image_included"] = included
    for col in ("taxonomist_label", "confidence", "notes"):
        out[col] = ""
    return out[AUDIT_COLUMNS]


def write_sheets(sample, out_dir):
    rows = sample[sample["license"].map(shareable)].to_dict("records")
    for i in range(0, len(rows), PER_SHEET):
        part = rows[i:i + PER_SHEET]
        draw_sheet(part, Path(out_dir) / f"sheet_{i // PER_SHEET + 1:02d}.jpg", [[r["photo_id"]] for r in part])


def questions_text(df, sample, flagged):
    atergatis = LABEL_NOTES["Atergatis"]
    species = df[df["species"] != ""].assign(philippine=df["country"] == "PH")
    table = species.groupby("species")["philippine"].agg(photos="size", philippine="sum")
    counts = "\n".join(f"| *{s}* | {r['photos']} | {r['philippine']} |" for s, r in table.iterrows())
    per_genus = ", ".join(f"*{g}* {n}" for g, n in sample["genus"].value_counts().sort_index().items())
    n_scylla = int((flagged["genus_label"] == "Scylla").sum())
    return f"""# Questions for the taxonomist

**Project:** a dissertation comparing image classifiers on four crab genera: *Scylla*, *Portunus*, *Charybdis* and *Atergatis*. *Atergatis floridus* is included as a reported toxic look-alike; it is not an edible crab.

**About the labels:** they are community identifications from iNaturalist and GBIF, research grade only. None has been verified by an expert. Any correction you suggest will not change the labels of the finished study; it will be used only in the next stage (Dissertation 3) or in a separately labeled sensitivity analysis.

## Community species labels in the dataset

| Label | Photos | From the Philippines |
|---|---|---|
{counts}

Most of the photos are from outside the Philippines (other Southeast Asian countries and Taiwan). The Philippine photos are held out as a local test.

## Questions

1. **Philippine blue swimming crab.** The sources use *Portunus pelagicus*, but that complex has been split into several species. Which accepted name applies to the blue swimming crab in the Philippines?
2. ***Scylla* in the Philippines.** Which *Scylla* species occur in the Philippines, and can photos separate them? Does *S. paramamosain* occur there? It is excluded from this dataset because it has no Philippine photos, so its presence was uncertain.
3. **Other toxic crabs.** Should users be warned about other toxic crabs common at Philippine markets or landing sites? To the project author's knowledge, *Zosimus* and *Lophozozymus* occur in the Indo-Pacific.
4. ***Atergatis* wording.** Is this wording appropriate? "{atergatis}"
5. **Flagged photos.** Please check the {len(flagged)} photos in `flagged.csv` (originals in `flagged_images/`): {n_scylla} labeled *Scylla* that look like swimming crabs, one labeled *Atergatis* that may be a different xanthid, and one labeled *Charybdis* showing a mixed pile of crabs. They were noticed on a model's error sheets, so they are not a random sample.

## Audit sample

`audit_sample.csv` lists {len(sample)} photos, and `audit_sheets/` shows them with the `photo_id` under each one. For each photo, please fill in `taxonomist_label` (genus, or a lower rank if you are confident), `confidence` and `notes`.

- **How it was drawn:** at random with fixed seed {AUDIT_SEED}: up to {AUDIT_PER_GENUS} photos per community-labeled genus ({per_genus}), independently of any model result, then shuffled. It is not weighted by how common each genus is. A flagged photo may also appear here.
- **Blind review:** the labels are left out so that your answer is independent. The observation page in `observation_url` shows the community identification, so please record your answer from the photo before opening the link. The sheets are small; the link shows the larger original.

## Image use

The images are included for non-commercial review of this dissertation only, and are not to be redistributed. Each photo's licence and credit are in the CSV files. Sheet images are resized from the originals; files in `flagged_images/` are unmodified. A photo whose licence does not allow sharing is given as a link only.
"""


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", default="cleaned", help="builder run whose metadata.csv gives photo_id, URL and credit")
    ap.add_argument("--out", default=str(ROOT / "taxonomist_packet"))
    args = ap.parse_args(argv)
    meta = pd.read_csv(DATA / "external" / "runs" / args.run / "metadata.csv", dtype=str, keep_default_na=False)
    extra = meta[["image_path", "observation_url", "attribution"]].drop_duplicates("image_path")
    df = labels_with_ids(args.run).merge(extra, on="image_path", how="left")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    flagged = flagged_table(df, out)
    flagged.to_csv(out / "flagged.csv", index=False, encoding="utf-8-sig")
    sample = draw_audit_sample(df)
    audit_table(sample).to_csv(out / "audit_sample.csv", index=False, encoding="utf-8-sig")
    write_sheets(sample, out / "audit_sheets")
    (out / "questions.md").write_text(questions_text(df, sample, flagged), encoding="utf-8")

    print(f"flagged: {len(flagged)} photos, {(flagged['image_file'] != '').sum()} with images")
    print(f"audit (seed {AUDIT_SEED}): {sample['genus'].value_counts().sort_index().to_dict()}, "
          f"{sample['license'].map(shareable).sum()} of {len(sample)} on sheets, "
          f"{sample['crab_id'].nunique()} observers, {sample['photo_id'].isin(FLAGGED).sum()} also flagged")
    print(f"licences in the sample: {sample['license'].value_counts().to_dict()}")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
