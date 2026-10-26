"""Assemble a crab image dataset from public sources.

    python -m tools.dataset_builder --dry-run                     # Philippines, metadata only
    python -m tools.dataset_builder --dry-run --countries widened
    python -m tools.dataset_builder --countries PH                # downloads images (ask first)

See tools/dataset_builder/README.md.
"""
import argparse
import json
import logging
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yaml

from . import dedupe, gbif, grouping, inat, licenses, manual, outputs
from .cache import QueryCache
from .download import DownloadLog, contact_sheet, materialize
from .net import Client, MissingContact
from .records import skip, slug, url_ext

log = logging.getLogger("dataset_builder")

TOOL = Path(__file__).resolve().parent
ROOT = TOOL.parents[1]
API_SOURCES = ("inaturalist", "gbif")
INAT_OK = ("exact", "synonym")
GBIF_OK = ("exact", "fuzzy", "synonym")


def parse_args(argv=None):
    p = argparse.ArgumentParser(prog="python -m tools.dataset_builder", description=__doc__.split("\n")[1])
    p.add_argument("--config", default=TOOL / "config.yaml", type=Path)
    p.add_argument("--species-file", default=TOOL / "species.yaml", type=Path)
    p.add_argument("--dry-run", action="store_true", help="fetch metadata only; download and write no images")
    p.add_argument("--countries", default="default",
                   help="'default' (Philippines), 'widened' (config list), or ISO codes such as PH,ID,MY")
    p.add_argument("--quality-grade", default=None,
                   help="iNaturalist quality grade(s): research (default), needs_id, casual, comma list, or any")
    p.add_argument("--group-by", choices=grouping.LEVELS, default="observer",
                   help="crab_id grouping key: observer (default, strictest), observation, file_id")
    p.add_argument("--max-per-species", type=int, default=None, help="cap on kept photos per species")
    p.add_argument("--sources", default=",".join(API_SOURCES), help="API sources to query: inaturalist,gbif, or none")
    p.add_argument("--imports", nargs="*", type=Path, default=None,
                   help="manual-import YAML files (default: tools/dataset_builder/imports/*.yaml)")
    p.add_argument("--data-dir", default=ROOT / "data", type=Path)
    p.add_argument("--run-name", default=None, help="output folder name under data/external/runs/")
    p.add_argument("--overwrite-labels", action="store_true", help="replace an existing data/labels.csv")
    return p.parse_args(argv)


def load_species(path):
    with open(path, encoding="utf-8") as f:
        names = [str(s).strip() for s in (yaml.safe_load(f) or {}).get("species", []) if str(s).strip()]
    if len(set(n.lower() for n in names)) != len(names):
        raise ValueError(f"{path}: duplicate species names")
    return names


def countries_from_arg(arg, cfg):
    cc = cfg["countries"]
    if arg in ("default", "widened"):
        codes = list(cc[arg])
    else:
        codes = [c.strip().upper() for c in arg.split(",") if c.strip()]
    unknown = [c for c in codes if c not in cc["names"]]
    if unknown:
        raise ValueError(f"unknown country codes {unknown}; add them to countries.names in the config")
    return codes


def planned_path(rec):
    """data-relative path: external/<source>/<species>/<file>."""
    stem = re.sub(r"[^A-Za-z0-9-]+", "_", rec["photo_id"].split(":", 1)[1]).strip("_")
    ext = "jpg" if rec["crop_box"] else (url_ext(rec["image_url"]) if rec["image_url"]
                                         else Path(rec["local_path"]).suffix.lstrip(".").lower() or "jpg")
    return f"external/{rec['source']}/{slug(rec['species']) or 'unknown'}/{stem}.{ext}"


def resolve_taxa(client, cfg, species, sources):
    rows = []
    for name in species:
        row = {"species": name, "inat_taxon_id": "", "inat_name": "", "inat_rank": "", "inat_status": "not_queried",
               "inat_note": "", "gbif_usage_key": "", "gbif_species_key": "", "gbif_name": "", "gbif_rank": "",
               "gbif_match": "", "gbif_status": "not_queried", "gbif_note": ""}
        if "inaturalist" in sources:
            row.update(inat.resolve_taxon(client, cfg["inaturalist"]["base_url"], name))
        if "gbif" in sources:
            row.update(gbif.match_species(client, cfg["gbif"]["base_url"], name))
        rows.append(row)
    return pd.DataFrame(rows)


def collect_api(client, cfg, cache, taxa, countries, sources, quality_grade, limit):
    """Records from iNaturalist and GBIF, plus the places table and the species whose fetch was capped."""
    records, capped, places = [], set(), []
    inat_base = cfg["inaturalist"]["base_url"]
    place_ids = {}
    terms = {}
    if "inaturalist" in sources:
        for code in countries:
            pid, label = inat.resolve_place(client, inat_base, cfg["countries"]["names"][code])
            place_ids[code] = pid
            places.append({"country": code, "name": cfg["countries"]["names"][code],
                           "inat_place_id": pid or "", "inat_place": label if pid else "", "note": "" if pid else label})
        terms = inat.controlled_terms(client, inat_base)
        for key in ("sex", "maturity"):
            if key not in terms:
                log.warning("iNaturalist controlled term for %s not found; no %s labels from iNaturalist", key, key)
    for _, t in taxa.iterrows():
        sp = t["species"]
        if "inaturalist" in sources and t["inat_status"] in INAT_OK:
            for code in countries:
                if not place_ids.get(code):
                    continue
                key = f"inat__{t['inat_taxon_id']}__{place_ids[code]}__{quality_grade}"
                obs, state = inat.fetch_observations(client, cfg, cache, key, t["inat_taxon_id"], place_ids[code],
                                                     quality_grade, limit)
                capped |= {sp} if state.get("capped") else set()
                for o in obs:
                    records += inat.photo_records(o, sp, code, terms, cfg)
        if "gbif" in sources and t["gbif_status"] in GBIF_OK:
            for code in countries:
                key = f"gbif__{t['gbif_species_key']}__{code}"
                occs, state = gbif.fetch_occurrences(client, cfg, cache, key, t["gbif_species_key"], code, limit)
                capped |= {sp} if state.get("capped") else set()
                for o in occs:
                    records += gbif.photo_records(o, sp, cfg)
    return records, capped, pd.DataFrame(places, columns=["country", "name", "inat_place_id", "inat_place", "note"])


def collect_imports(specs, cfg, species):
    by_lower = {s.lower(): s for s in species}
    records = []
    for spec in specs:
        recs = manual.run_import(spec, cfg)
        for r in recs:
            name = by_lower.get(str(r["species"]).strip().lower())
            if name:
                r["species"] = name
            else:
                skip(r, f"species_not_in_list:{r['species'] or 'none'}")
        records += recs
    return records


def apply_cap(records, limit, priority):
    """Keep at most `limit` photos per species, preferring sources by priority. Returns capped species."""
    capped = set()
    if not limit:
        return capped
    per = Counter()
    for r in sorted(records, key=lambda r: dedupe.source_rank(r["source"], priority)):
        if r["skipped"]:
            continue
        if per[r["species"]] >= limit:
            skip(r, "cap")
            capped.add(r["species"])
        else:
            per[r["species"]] += 1
    return capped


def main(argv=None, client=None):
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = parse_args(argv)
    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    species = load_species(args.species_file)
    countries = countries_from_arg(args.countries, cfg)
    sources = [s for s in args.sources.split(",") if s in API_SOURCES]
    quality_grade = args.quality_grade or cfg["inaturalist"]["quality_grade"]
    run_name = args.run_name or (args.countries if args.countries in ("default", "widened")
                                 else "_".join(countries).lower())
    priority = list(cfg["source_priority"])
    data_dir = Path(args.data_dir)
    run_dir = data_dir / "external" / "runs" / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    started = datetime.now(timezone.utc).isoformat(timespec="seconds")

    import_files = args.imports if args.imports is not None else sorted((TOOL / "imports").glob("*.yaml"))
    specs = [manual.load_spec(p, ROOT) for p in import_files]

    if sources and client is None:
        try:
            client = Client.from_config(cfg)
        except MissingContact as e:
            print(f"error: {e}", file=sys.stderr)
            return 2
    cache = QueryCache(data_dir / "external" / "_cache")

    log.info("run %s: countries %s, sources %s, quality grade %s, group-by %s, %s",
             run_name, ",".join(countries), ",".join(sources) or "none", quality_grade, args.group_by,
             "dry run" if args.dry_run else "DOWNLOAD")
    taxa = resolve_taxa(client, cfg, species, sources)
    taxa.to_csv(run_dir / "taxa.csv", index=False)
    records, capped, places = collect_api(client, cfg, cache, taxa, countries, sources, quality_grade,
                                          args.max_per_species)
    places.to_csv(run_dir / "places.csv", index=False)
    records += collect_imports(specs, cfg, species)

    for r in records:
        r["image_path"] = planned_path(r)
        r["license"] = licenses.normalize(r["license_raw"])
    dedupe.cross_source(records, priority)
    allow = cfg["licenses"]["allowlist"]
    for r in records:
        if not r["skipped"]:
            _, reason = licenses.check(r["license_raw"], allow)
            if reason:
                skip(r, reason)
    capped |= apply_cap(records, args.max_per_species, priority)
    grouping.assign(records, args.group_by)
    outputs.write_metadata(records, run_dir / "metadata.csv")  # before any download, so progress is on disk

    if not args.dry_run:
        dlog = DownloadLog(run_dir / "downloads.jsonl")
        kept = [r for r in records if not r["skipped"]]
        for i, r in enumerate(kept, 1):
            ok, reason = materialize(r, data_dir, client, dlog)
            if not ok:
                skip(r, reason)
            if i % 100 == 0:
                log.info("images: %d of %d", i, len(kept))
                outputs.write_metadata(records, run_dir / "metadata.csv")
    for r in records:
        if r["skipped"]:
            continue
        path = data_dir / r["image_path"] if not args.dry_run else (r["local_path"] or None)
        if path:
            try:
                r["phash"] = dedupe.phash(path, "" if not args.dry_run else r["crop_box"])
            except Exception as e:
                log.warning("%s: cannot hash image: %s", r["photo_id"], e)
                if not args.dry_run:
                    skip(r, "invalid_image")
    kept_now = [r for r in records if not r["skipped"]]
    hashed = sum(1 for r in kept_now if r["phash"])
    near = dedupe.near_duplicates(records, cfg["near_duplicates"]["max_distance"], priority)
    pd.DataFrame(near, columns=["removed_photo_id", "removed_source", "removed_species", "kept_photo_id",
                                "kept_source", "kept_species", "distance"]).to_csv(run_dir / "near_duplicates.csv",
                                                                                    index=False)

    th = cfg["thresholds"]
    counts = outputs.counts_table(records, species, th, capped, near_dup_done=hashed > 0)
    attrs = outputs.attribute_table(records, th)
    verdicts = outputs.attribute_verdicts(attrs, th)
    warns = outputs.warnings_list(counts, attrs, taxa, th)
    finished = datetime.now(timezone.utc).isoformat(timespec="seconds")
    run_info = {"run_name": run_name, "started": started, "finished": finished, "dry_run": args.dry_run,
                "countries": countries, "sources": sources, "quality_grade": quality_grade,
                "group_by": args.group_by, "max_per_species": args.max_per_species,
                "imports": [s["spec_file"] for s in specs], "license_allowlist": allow,
                "near_duplicate_check": f"{hashed} of {len(kept_now)} kept photos hashed",
                "skip_reasons": dict(Counter(r["skip_reason"] for r in records if r["skipped"])),
                "attribute_verdicts": {k: {"trainable": v[0], "detail": v[1]} for k, v in verdicts.items()},
                "thresholds": th}

    outputs.write_metadata(records, run_dir / "metadata.csv")
    outputs.write_labels(records, run_dir / "labels.csv")
    counts.to_csv(run_dir / "counts.csv", index=False)
    attrs.to_csv(run_dir / "attribute_counts.csv", index=False)
    outputs.write_attribution(records, specs, run_dir / "ATTRIBUTION.md", run_info)
    (run_dir / "warnings.txt").write_text("\n".join(warns) + "\n", encoding="utf-8")
    (run_dir / "run.json").write_text(json.dumps(run_info, indent=2), encoding="utf-8")

    if not args.dry_run:
        labels_csv = data_dir / "labels.csv"
        if labels_csv.exists() and not args.overwrite_labels:
            log.warning("%s exists; not replaced (use --overwrite-labels). New labels are in %s",
                        labels_csv, run_dir / "labels.csv")
        else:
            outputs.write_labels(records, labels_csv)
        rc = cfg["review"]
        for sp in species:
            paths = [data_dir / r["image_path"] for r in records if not r["skipped"] and r["species"] == sp]
            contact_sheet(paths, data_dir / "external" / "review" / f"{slug(sp)}.jpg", rc["thumb_size"],
                          rc["columns"], rc["per_species"], rc["seed"])

    report = render_report(run_info, taxa, places, counts, attrs, verdicts, warns)
    (run_dir / "counts.txt").write_text(report, encoding="utf-8")
    print(report)
    return 0


def render_report(run_info, taxa, places, counts, attrs, verdicts, warns):
    th = run_info["thresholds"]
    out = [f"=== Run {run_info['run_name']} ({'dry run' if run_info['dry_run'] else 'download'}) ===",
           f"countries: {', '.join(run_info['countries'])} | sources: {', '.join(run_info['sources']) or 'none'} | "
           f"iNaturalist quality grade: {run_info['quality_grade']} | crab_id: {run_info['group_by']}",
           f"license allowlist: {', '.join(run_info['license_allowlist'])}",
           f"near-duplicate check: {run_info['near_duplicate_check']}"
           + (" (API images are hashed only after download)" if run_info["dry_run"] else ""), "",
           "--- Taxa ---",
           taxa[["species", "inat_status", "inat_taxon_id", "inat_name", "gbif_status", "gbif_species_key",
                 "gbif_name"]].to_string(index=False), ""]
    if len(places):
        out += ["--- iNaturalist places ---", places.to_string(index=False), ""]
    out += ["--- Counts per species (groups are over kept photos) ---", outputs.format_counts(counts), "",
            "--- Attribute classes (all species, weak and unverified labels) ---",
            attrs.drop(columns=["photos_by_species"]).to_string(index=False) if len(attrs) else "(no sex or maturity labels)",
            ""]
    out.append(f"--- Thresholds: species >= {th['min_species_groups']} observer groups; attribute >= "
               f"{th['min_attribute_classes']} classes x {th['min_class_groups']} groups; dominated > "
               f"{th['dominance']:.0%} from one source ---")
    sp = counts[counts["species"] != "ALL"]
    out.append("species trainable: " + (", ".join(sp[sp["trainable"] == True]["species"]) or "none"))  # noqa: E712
    out.append("species not trainable: " + (", ".join(sp[sp["trainable"] != True]["species"]) or "none"))  # noqa: E712
    for attr, (ok, why) in verdicts.items():
        out.append(f"attribute {attr}: {'trainable' if ok else 'not trainable'} - {why}")
    out.append("skip reasons: " + json.dumps(run_info["skip_reasons"]))
    out += ["", "--- Warnings ---"] + (warns or ["none"])
    return "\n".join(out) + "\n"


if __name__ == "__main__":
    sys.exit(main())
