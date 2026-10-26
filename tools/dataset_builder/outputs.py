"""Counts, threshold flags and the output files (labels.csv, metadata.csv, ATTRIBUTION.md)."""
import json
from collections import Counter

import pandas as pd

from . import grouping
from .records import FIELDS

LABEL_COLUMNS = ("image_path", "crab_id", "species", "sex", "disease", "maturity", "source", "license",
                 "observer_id", "observation_id", "sex_source", "maturity_source")
ATTRIBUTES = ("sex", "maturity")


def frame(records):
    return pd.DataFrame(records, columns=list(FIELDS)) if records else pd.DataFrame(columns=list(FIELDS))


def _groups(rows, level):
    return len({grouping.group_key(r, level)[0] for r in rows})


def _counter_json(values):
    return json.dumps(dict(sorted(Counter(values).items())))


def _dominance(rows, threshold):
    if not rows:
        return "", 0.0, False
    src, n = Counter(r["source"] for r in rows).most_common(1)[0]
    share = n / len(rows)
    return src, round(share, 3), share > threshold


def _skip_count(rows, prefix):
    return sum(1 for r in rows if r["skipped"] and r["skip_reason"].startswith(prefix))


def counts_table(records, species, thresholds, capped=(), near_dup_done=True):
    """Per-species counts. Group counts are over kept photos; skip counts are over all fetched photos."""
    th = thresholds
    rows = []
    for name in list(species) + ["ALL"]:
        mine = [r for r in records if name == "ALL" or r["species"] == name]
        kept = [r for r in mine if not r["skipped"]]
        top, share, dominated = _dominance(kept, th["dominance"])
        g_obs = _groups(kept, "observer")
        rows.append({
            "species": name, "photos": len(kept),
            "groups_observer": g_obs, "groups_observation": _groups(kept, "observation"),
            "groups_file_id": _groups(kept, "file_id"),
            "sex_labeled": sum(1 for r in kept if r["sex"]), "maturity_labeled": sum(1 for r in kept if r["maturity"]),
            "skipped_license": _skip_count(mine, "license_"),
            "skipped_duplicate": _skip_count(mine, "duplicate_"),
            "skipped_near_duplicate": _skip_count(mine, "near_duplicate") if near_dup_done else "",
            "skipped_cap": _skip_count(mine, "cap"),
            "skipped_other": sum(1 for r in mine if r["skipped"] and not r["skip_reason"].startswith(
                ("license_", "duplicate_", "near_duplicate", "cap"))),
            "cap_hit": (name in capped) if name != "ALL" else bool(capped),
            "top_source": top, "top_source_share": share, "dominated": dominated,
            "trainable": g_obs >= th["min_species_groups"] if name != "ALL" else "",
            "photos_by_source": _counter_json(r["source"] for r in kept),
            "photos_by_license": _counter_json(r["license"] for r in kept),
            "photos_by_country": _counter_json(r["country"] or "unknown" for r in kept),
        })
    return pd.DataFrame(rows)


def attribute_table(records, thresholds):
    """Per attribute class (across all species): photos, groups and source dominance."""
    th = thresholds
    kept = [r for r in records if not r["skipped"]]
    rows = []
    for attr in ATTRIBUTES:
        for cls in sorted({r[attr] for r in kept if r[attr]}):
            mine = [r for r in kept if r[attr] == cls]
            top, share, dominated = _dominance(mine, th["dominance"])
            g = _groups(mine, "observer")
            rows.append({"attribute": attr, "class": cls, "photos": len(mine), "groups_observer": g,
                         "groups_observation": _groups(mine, "observation"), "class_ok": g >= th["min_class_groups"],
                         "top_source": top, "top_source_share": share, "dominated": dominated,
                         "photos_by_source": _counter_json(r["source"] for r in mine),
                         "photos_by_species": _counter_json(r["species"] for r in mine),
                         "label_sources": _counter_json(r[f"{attr}_source"] for r in mine)})
    return pd.DataFrame(rows, columns=["attribute", "class", "photos", "groups_observer", "groups_observation",
                                       "class_ok", "top_source", "top_source_share", "dominated",
                                       "photos_by_source", "photos_by_species", "label_sources"])


def attribute_verdicts(attr_table, thresholds):
    """{attribute: (trainable, reason)}. Trainable needs at least min_attribute_classes classes,
    each with at least min_class_groups observer-level groups (the bench skips the attribute otherwise)."""
    th = thresholds
    out = {}
    for attr in ATTRIBUTES:
        t = attr_table[attr_table["attribute"] == attr]
        weak = [f"{r['class']} ({r['groups_observer']} groups)" for _, r in t.iterrows() if not r["class_ok"]]
        if len(t) < th["min_attribute_classes"]:
            out[attr] = (False, f"{len(t)} class(es) labeled: {', '.join(t['class']) or 'none'}")
        elif weak:
            out[attr] = (False, f"below {th['min_class_groups']} groups: {', '.join(weak)}")
        else:
            out[attr] = (True, ", ".join(f"{r['class']} ({r['groups_observer']} groups)" for _, r in t.iterrows()))
    return out


def warnings_list(counts, attr_table, taxa, thresholds):
    th = thresholds
    out = []
    for _, t in taxa.iterrows():
        for src in ("inat", "gbif"):
            if t[f"{src}_status"] not in ("exact", "synonym", "fuzzy"):
                out.append(f"{t['species']}: {'iNaturalist' if src == 'inat' else 'GBIF'} {t[f'{src}_status']}"
                           f"{' - ' + t[f'{src}_note'] if t[f'{src}_note'] else ''} (not collected from this source)")
            elif t[f"{src}_note"]:
                out.append(f"{t['species']}: {'iNaturalist' if src == 'inat' else 'GBIF'} note - {t[f'{src}_note']}")
    for _, r in counts[counts["species"] != "ALL"].iterrows():
        if r["photos"] == 0:
            out.append(f"{r['species']}: zero photos kept")
        if not r["trainable"]:
            out.append(f"{r['species']}: {r['groups_observer']} observer-level groups, below {th['min_species_groups']}")
        if r["dominated"]:
            out.append(f"{r['species']}: dominated by {r['top_source']} ({r['top_source_share']:.0%} of photos)")
        if r["cap_hit"]:
            out.append(f"{r['species']}: --max-per-species cap hit; counts are truncated")
    for attr, (ok, why) in attribute_verdicts(attr_table, th).items():
        if not ok:
            out.append(f"attribute {attr}: not trainable - {why}")
    for _, r in attr_table.iterrows():
        if r["dominated"]:
            out.append(f"attribute {r['attribute']}={r['class']}: dominated by {r['top_source']} "
                       f"({r['top_source_share']:.0%} of photos)")
    return out


def write_labels(records, path):
    kept = frame([r for r in records if not r["skipped"]])
    kept["disease"] = ""  # no public source provides disease
    kept[list(LABEL_COLUMNS)].to_csv(path, index=False)


def write_metadata(records, path):
    frame(records).to_csv(path, index=False)


def write_attribution(records, imports, path, run_info):
    kept = [r for r in records if not r["skipped"]]
    lines = ["# Attribution", "",
             f"Photos selected by the crab dataset builder, run `{run_info['run_name']}` ({run_info['finished']}).",
             "Each photo keeps the license its author chose. Labels (species, and weak, unverified sex and",
             "maturity) come from the sources, not from an expert.", ""]
    by_source = {}
    for r in kept:
        by_source.setdefault(r["source"], []).append(r)
    if "inaturalist" in by_source:
        lines += ["## iNaturalist", "", "Source: iNaturalist (https://www.inaturalist.org), observations accessed "
                  f"through the API on {run_info['finished'][:10]}.", ""]
    if "gbif" in by_source:
        lines += ["## GBIF", "", "Source: GBIF.org occurrence search API (https://www.gbif.org), accessed "
                  f"{run_info['finished'][:10]}. Cite GBIF with a derived-dataset DOI (see README).", ""]
    for spec in imports:
        lines += [f"## {spec['name']} (manual import)", "", f"- URL: {spec['url']}", f"- License: {spec['license']}",
                  f"- Credit: {spec.get('attribution') or '(not given)'}", ""]
    lines += ["## Photos", "", "| source | species | photo | license | credit | record |", "|---|---|---|---|---|---|"]
    for src in sorted(by_source):
        for r in sorted(by_source[src], key=lambda r: (r["species"], r["photo_id"])):
            credit = str(r["attribution"]).replace("|", "/")
            lines.append(f"| {src} | {r['species']} | {r['photo_id']} | {r['license']} | {credit} | {r['observation_url']} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


MAIN_COLUMNS = ["species", "photos", "groups_observer", "groups_observation", "groups_file_id", "sex_labeled",
                "maturity_labeled", "skipped_license", "skipped_duplicate", "skipped_near_duplicate", "skipped_cap",
                "skipped_other", "cap_hit", "top_source", "top_source_share", "dominated", "trainable"]


def format_counts(counts):
    lines = [counts[MAIN_COLUMNS].to_string(index=False), ""]
    for col in ("photos_by_source", "photos_by_license", "photos_by_country"):
        lines.append(f"{col}:")
        for _, r in counts.iterrows():
            lines.append(f"  {r['species']:<24} {r[col]}")
    return "\n".join(lines)
