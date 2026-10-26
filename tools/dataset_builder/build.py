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
from .records import skip, slug, taxon_label, url_ext

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
    p.add_argument("--include-genus-needs-id", action="store_true", default=None,
                   help="also keep iNaturalist genus-rank needs_id observations, flagged review_flag=genus_needs_id "
                        "(default: inaturalist.genus_needs_id in the config)")
    p.add_argument("--include-optional", action="store_true",
                   help="write optional candidates (optional: true in species.yaml) to labels.csv")
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
    """The taxa list: {'genera': [{'name', 'optional', 'note'}], 'species': [{'name', 'genus', 'optional', 'note'}]}.

    A species entry may be a plain name (its genus is the first word). Every species'
    genus is in the genus list; a species of an optional genus is optional.
    """
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    def entry(e, kind):
        e = {"name": e} if isinstance(e, str) else dict(e)
        e["name"] = str(e.get("name", "")).strip()
        if not e["name"]:
            raise ValueError(f"{path}: a {kind} entry has no name")
        e["optional"] = bool(e.get("optional", False))
        e["note"] = str(e.get("note") or "")
        return e

    genera = [entry(g, "genus") for g in raw.get("genera") or []]
    species = [entry(s, "species") for s in raw.get("species") or []]
    gmap = {g["name"].lower(): g for g in genera}
    for s in species:
        s["genus"] = str(s.get("genus") or s["name"].split()[0]).strip()
        g = gmap.get(s["genus"].lower())
        if g is None:  # plain list of species names: add the genus
            g = {"name": s["genus"], "optional": False, "note": ""}
            genera.append(g)
            gmap[g["name"].lower()] = g
        s["genus"] = g["name"]
        s["optional"] = s["optional"] or g["optional"]
        if s["name"].split()[0].lower() != g["name"].lower():
            log.warning("%s is listed under genus %s", s["name"], g["name"])
    for kind, items in (("genus", genera), ("species", species)):
        names = [i["name"].lower() for i in items]
        if len(set(names)) != len(names):
            raise ValueError(f"{path}: duplicate {kind} names")
    return {"genera": genera, "species": species}


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
    return f"external/{rec['source']}/{slug(taxon_label(rec)) or 'unknown'}/{stem}.{ext}"


def taxa_entries(taxa_list):
    """Genera, then species, as rows to resolve: name, rank, genus, optional."""
    return ([{"taxon": g["name"], "rank": "genus", "genus": g["name"], "optional": g["optional"]}
             for g in taxa_list["genera"]]
            + [{"taxon": s["name"], "rank": "species", "genus": s["genus"], "optional": s["optional"]}
               for s in taxa_list["species"]])


def resolve_taxa(client, cfg, taxa_list, sources):
    rows = []
    for t in taxa_entries(taxa_list):
        row = {**t, "inat_taxon_id": "", "inat_name": "", "inat_rank": "", "inat_status": "not_queried",
               "inat_note": "", "gbif_usage_key": "", "gbif_taxon_key": "", "gbif_name": "", "gbif_rank": "",
               "gbif_match": "", "gbif_status": "not_queried", "gbif_note": ""}
        if "inaturalist" in sources:
            row.update(inat.resolve_taxon(client, cfg["inaturalist"]["base_url"], t["taxon"], t["rank"]))
        if "gbif" in sources:
            row.update(gbif.match_species(client, cfg["gbif"]["base_url"], t["taxon"], t["rank"]))
        rows.append(row)
    return pd.DataFrame(rows)


def genus_needs_id_queries(quality_grade, include_needs_id):
    """True if genus-rank needs_id observations must be fetched separately (and flagged)."""
    grades = {g.strip() for g in str(quality_grade).split(",")}
    return bool(include_needs_id) and not ({"needs_id", "any"} & grades)


def collect_api(client, cfg, cache, taxa, countries, sources, quality_grade, limit, include_needs_id=False):
    """Records from iNaturalist and GBIF, plus the places table and the taxa whose fetch was capped.

    Species rows are queried as before. Genus rows are queried for observations identified
    only to the genus (iNaturalist rank=genus; GBIF has no rank filter, so its genus results
    are filtered to taxonRank GENUS here). With include_needs_id, iNaturalist genus-rank
    needs_id observations are added with review_flag = genus_needs_id.
    """
    records, capped, places = [], set(), []
    inat_base = cfg["inaturalist"]["base_url"]
    place_ids = {}
    terms = {}
    if "inaturalist" in sources:
        for code in countries:
            pid, label = inat.resolve_place(client, inat_base, cfg["countries"]["names"][code])
            place_ids[code] = pid
            names = cfg["countries"]["names"][code]
            places.append({"country": code, "name": names if isinstance(names, str) else " / ".join(names),
                           "inat_place_id": pid or "", "inat_place": label if pid else "", "note": "" if pid else label})
        terms = inat.controlled_terms(client, inat_base)
        for key in ("sex", "maturity"):
            if key not in terms:
                log.warning("iNaturalist controlled term for %s not found; no %s labels from iNaturalist", key, key)
    ic = cfg["inaturalist"]
    extra_needs_id = genus_needs_id_queries(quality_grade, include_needs_id)
    for _, t in taxa.iterrows():
        is_genus = t["rank"] == "genus"
        sp = "" if is_genus else t["taxon"]
        label = f"{t['taxon']} (genus only)" if is_genus else t["taxon"]
        if "inaturalist" in sources and t["inat_status"] in INAT_OK:
            queries = [(quality_grade, None, "")]
            if is_genus and extra_needs_id:
                queries.append(("needs_id", ic.get("genus_needs_id_identifications"), "genus_needs_id"))
            for code in countries:
                if not place_ids.get(code):
                    continue
                for qg, idents, flag in queries:
                    key = f"inat__{t['inat_taxon_id']}__{place_ids[code]}__{qg}"
                    if is_genus:
                        key += "__rank-genus" + (f"__ids-{idents}" if idents else "")
                    obs, state = inat.fetch_observations(client, cfg, cache, key, t["inat_taxon_id"], place_ids[code],
                                                         qg, limit, rank="genus" if is_genus else None,
                                                         identifications=idents)
                    capped |= {label} if state.get("capped") else set()
                    for o in obs:
                        records += inat.photo_records(o, sp, code, terms, cfg, genus=t["genus"], review_flag=flag,
                                                      optional=bool(t["optional"]))
        if "gbif" in sources and t["gbif_status"] in GBIF_OK:
            for code in countries:
                key = f"gbif__{t['gbif_taxon_key']}__{code}"
                occs, state = gbif.fetch_occurrences(client, cfg, cache, key, t["gbif_taxon_key"], code, limit)
                capped |= {label} if state.get("capped") else set()
                for o in occs:
                    if is_genus and gbif.id_rank(o.get("taxonRank")) != "genus":
                        continue  # species-level records come from the species queries
                    records += gbif.photo_records(o, sp, cfg, genus=t["genus"], optional=bool(t["optional"]))
    return records, capped, pd.DataFrame(places, columns=["country", "name", "inat_place_id", "inat_place", "note"])


def collect_imports(specs, cfg, taxa_list):
    """Manual-import records. The species column may hold a listed species, or a listed genus
    for an image identified only to genus."""
    by_species = {s["name"].lower(): s for s in taxa_list["species"]}
    by_genus = {g["name"].lower(): g for g in taxa_list["genera"]}
    records = []
    for spec in specs:
        recs = manual.run_import(spec, cfg)
        for r in recs:
            value = str(r["species"]).strip().lower()
            if value in by_species:
                s = by_species[value]
                r.update(species=s["name"], genus=s["genus"], id_rank="species", optional=s["optional"])
            elif value in by_genus:
                g = by_genus[value]
                r.update(species="", genus=g["name"], id_rank="genus", optional=g["optional"])
            else:
                skip(r, f"species_not_in_list:{r['species'] or 'none'}")
        records += recs
    return records


def apply_cap(records, limit, priority):
    """Keep at most `limit` photos per species (or genus-only group), preferring sources by priority.
    Returns the capped labels."""
    capped = set()
    if not limit:
        return capped
    per = Counter()
    for r in sorted(records, key=lambda r: dedupe.source_rank(r["source"], priority)):
        if r["skipped"]:
            continue
        label = taxon_label(r)
        if per[label] >= limit:
            skip(r, "cap")
            capped.add(label)
        else:
            per[label] += 1
    return capped


def main(argv=None, client=None):
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = parse_args(argv)
    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    taxa_list = load_species(args.species_file)
    species = [s["name"] for s in taxa_list["species"]]
    countries = countries_from_arg(args.countries, cfg)
    sources = [s for s in args.sources.split(",") if s in API_SOURCES]
    quality_grade = args.quality_grade or cfg["inaturalist"]["quality_grade"]
    include_needs_id = (args.include_genus_needs_id if args.include_genus_needs_id is not None
                        else bool(cfg["inaturalist"].get("genus_needs_id", False)))
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

    log.info("run %s: countries %s, sources %s, quality grade %s, genus-rank needs_id %s, group-by %s, %s",
             run_name, ",".join(countries), ",".join(sources) or "none", quality_grade,
             "included (flagged)" if include_needs_id else "excluded", args.group_by,
             "dry run" if args.dry_run else "DOWNLOAD")
    taxa = resolve_taxa(client, cfg, taxa_list, sources)
    taxa.to_csv(run_dir / "taxa.csv", index=False)
    records, capped, places = collect_api(client, cfg, cache, taxa, countries, sources, quality_grade,
                                          args.max_per_species, include_needs_id)
    places.to_csv(run_dir / "places.csv", index=False)
    records += collect_imports(specs, cfg, taxa_list)

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
    optional_species = {s["name"] for s in taxa_list["species"] if s["optional"]}
    counts = outputs.counts_table(records, species, th, capped, near_dup_done=hashed > 0, optional=optional_species)
    genus_counts = outputs.genus_table(records, taxa_list, th, capped)
    attrs = outputs.attribute_table(records, th)
    verdicts = outputs.attribute_verdicts(attrs, th)
    warns = outputs.warnings_list(counts, attrs, taxa, th, genus_counts)
    finished = datetime.now(timezone.utc).isoformat(timespec="seconds")
    run_info = {"run_name": run_name, "started": started, "finished": finished, "dry_run": args.dry_run,
                "countries": countries, "sources": sources, "quality_grade": quality_grade,
                "genus_needs_id": include_needs_id,
                "genus_needs_id_identifications": cfg["inaturalist"].get("genus_needs_id_identifications"),
                "include_optional": args.include_optional,
                "optional_taxa": sorted({g["name"] for g in taxa_list["genera"] if g["optional"]} | optional_species),
                "group_by": args.group_by, "max_per_species": args.max_per_species,
                "imports": [s["spec_file"] for s in specs], "license_allowlist": allow,
                "near_duplicate_check": f"{hashed} of {len(kept_now)} kept photos hashed",
                "skip_reasons": dict(Counter(r["skip_reason"] for r in records if r["skipped"])),
                "attribute_verdicts": {k: {"trainable": v[0], "detail": v[1]} for k, v in verdicts.items()},
                "thresholds": th}

    outputs.write_metadata(records, run_dir / "metadata.csv")
    outputs.write_labels(records, run_dir / "labels.csv", args.include_optional)
    counts.to_csv(run_dir / "counts.csv", index=False)
    genus_counts.to_csv(run_dir / "genus_counts.csv", index=False)
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
            outputs.write_labels(records, labels_csv, args.include_optional)
        rc = cfg["review"]
        for label in sorted({taxon_label(r) for r in records if not r["skipped"]}):
            paths = [data_dir / r["image_path"] for r in records if not r["skipped"] and taxon_label(r) == label]
            contact_sheet(paths, data_dir / "external" / "review" / f"{slug(label)}.jpg", rc["thumb_size"],
                          rc["columns"], rc["per_species"], rc["seed"])

    report = render_report(run_info, taxa, places, counts, attrs, verdicts, warns, genus_counts)
    (run_dir / "counts.txt").write_text(report, encoding="utf-8")
    print(report)
    return 0


def render_report(run_info, taxa, places, counts, attrs, verdicts, warns, genus_counts=None):
    th = run_info["thresholds"]
    needs_id = ("included, flagged review_flag=genus_needs_id"
                + (f", identifications={run_info['genus_needs_id_identifications']}"
                   if run_info.get("genus_needs_id_identifications") else "")
                if run_info.get("genus_needs_id") else "excluded")
    out = [f"=== Run {run_info['run_name']} ({'dry run' if run_info['dry_run'] else 'download'}) ===",
           f"countries: {', '.join(run_info['countries'])} | sources: {', '.join(run_info['sources']) or 'none'} | "
           f"iNaturalist quality grade: {run_info['quality_grade']} | crab_id: {run_info['group_by']}",
           f"genus-rank needs_id observations: {needs_id}",
           f"optional candidates (counted on their own lines, not in ALL"
           f"{'' if run_info.get('include_optional') else ' or labels.csv'}): "
           f"{', '.join(run_info.get('optional_taxa') or []) or 'none'}",
           f"license allowlist: {', '.join(run_info['license_allowlist'])}",
           f"near-duplicate check: {run_info['near_duplicate_check']}"
           + (" (API images are hashed only after download)" if run_info["dry_run"] else ""), "",
           "--- Taxa ---",
           taxa[["taxon", "rank", "optional", "inat_status", "inat_taxon_id", "inat_name", "gbif_status",
                 "gbif_taxon_key", "gbif_name"]].to_string(index=False), ""]
    if len(places):
        out += ["--- iNaturalist places ---", places.to_string(index=False), ""]
    if genus_counts is not None and len(genus_counts):
        out += ["--- Counts per genus: MAIN TASK (groups are over kept photos) ---",
                outputs.format_genus_counts(genus_counts), ""]
    out += ["--- Counts per species: secondary, exploratory (genus-only photos are not in any species) ---",
            outputs.format_counts(counts), "",
            "--- Attribute classes (all species, weak and unverified labels) ---",
            attrs.drop(columns=["photos_by_species"]).to_string(index=False) if len(attrs) else "(no sex or maturity labels)",
            ""]
    out.append(f"--- Thresholds: genus or species >= {th['min_species_groups']} observer groups; attribute >= "
               f"{th['min_attribute_classes']} classes x {th['min_class_groups']} groups; dominated > "
               f"{th['dominance']:.0%} from one source ---")
    if genus_counts is not None and len(genus_counts):
        gn = genus_counts[genus_counts["genus"] != "ALL"]
        name = lambda r: r["genus"] + (" (optional)" if r["optional"] else "")  # noqa: E731
        out.append("genus trainable: " + (", ".join(name(r) for _, r in gn[gn["trainable"] == True].iterrows()) or "none"))  # noqa: E712
        out.append("genus not trainable: " + (", ".join(name(r) for _, r in gn[gn["trainable"] != True].iterrows()) or "none"))  # noqa: E712
        out.append("genus dominated: " + (", ".join(f"{name(r)} ({r['top_source']} {r['top_source_share']:.0%})"
                                                     for _, r in gn[gn["dominated"] == True].iterrows()) or "none"))  # noqa: E712
    sp = counts[counts["species"] != "ALL"]
    label = lambda r: r["species"] + (" (optional)" if r["optional"] else "")  # noqa: E731
    out.append("species trainable: " + (", ".join(label(r) for _, r in sp[sp["trainable"] == True].iterrows()) or "none"))  # noqa: E712
    out.append("species not trainable: " + (", ".join(label(r) for _, r in sp[sp["trainable"] != True].iterrows()) or "none"))  # noqa: E712
    for attr, (ok, why) in verdicts.items():
        out.append(f"attribute {attr}: {'trainable' if ok else 'not trainable'} - {why}")
    out.append("skip reasons: " + json.dumps(run_info["skip_reasons"]))
    out += ["", "--- Warnings ---"] + (warns or ["none"])
    return "\n".join(out) + "\n"


if __name__ == "__main__":
    sys.exit(main())
