"""Counts, threshold flags and the output files (labels.csv, metadata.csv, ATTRIBUTION.md)."""
import json
from collections import Counter

import pandas as pd

from . import grouping
from .records import FIELDS, taxon_label

LABEL_COLUMNS = ("image_path", "crab_id", "genus", "species", "sex", "disease", "maturity", "source",
                 "id_rank", "quality_grade", "review_flag", "license",
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


def counts_table(records, species, thresholds, capped=(), near_dup_done=True, optional=()):
    """Per-species counts. Group counts are over kept photos; skip counts are over all fetched photos.
    ALL covers every photo except optional candidates, including photos identified only to genus."""
    th = thresholds
    rows = []
    for name in list(species) + ["ALL"]:
        mine = [r for r in records if (not r["optional"] if name == "ALL" else r["species"] == name)]
        kept = [r for r in mine if not r["skipped"]]
        top, share, dominated = _dominance(kept, th["dominance"])
        g_obs = _groups(kept, "observer")
        rows.append({
            "species": name, "optional": name in optional, "photos": len(kept),
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


def _share_json(values):
    c = Counter(values)
    total = sum(c.values())
    return json.dumps({k: round(v / total, 3) for k, v in sorted(c.items(), key=lambda kv: -kv[1])}) if total else "{}"


def genus_table(records, taxa_list, thresholds, capped=()):
    """Per-genus counts for the main (genus-level) task, over kept photos.

    Includes photos identified only to genus (id_rank = genus). ALL covers every genus
    except optional ones. A genus is trainable with at least min_species_groups
    observer-level groups, the same threshold as a species.
    """
    th = thresholds
    kept_all = [r for r in records if not r["skipped"]]
    rows = []
    entries = [(g["name"], g["optional"]) for g in taxa_list["genera"]] + [("ALL", False)]
    for name, optional in entries:
        kept = [r for r in kept_all if (not r["optional"] if name == "ALL" else r["genus"] == name)]
        top, share, dominated = _dominance(kept, th["dominance"])
        g_obs = _groups(kept, "observer")
        by_species = {}
        for r in kept:
            by_species.setdefault(taxon_label(r), []).append(r)
        rows.append({
            "genus": name, "optional": optional, "photos": len(kept),
            "groups_observer": g_obs, "groups_observation": _groups(kept, "observation"),
            "photos_genus_only": sum(1 for r in kept if r["id_rank"] == "genus"),
            "photos_needs_id_flagged": sum(1 for r in kept if r["review_flag"] == "genus_needs_id"),
            "n_species": len({r["species"] for r in kept if r["species"]}),
            "top_source": top, "top_source_share": share, "dominated": dominated,
            "trainable": g_obs >= th["min_species_groups"] if name != "ALL" else "",
            # capped labels are species names or "<Genus> (genus only)": both start with the genus
            "cap_hit": any(str(c).split()[0] == name for c in capped) if name != "ALL" else bool(capped),
            "photos_by_species": json.dumps({k: len(v) for k, v in sorted(by_species.items())}),
            "groups_observer_by_species": json.dumps({k: _groups(v, "observer") for k, v in sorted(by_species.items())}),
            "groups_observation_by_species": json.dumps({k: _groups(v, "observation") for k, v in sorted(by_species.items())}),
            "photos_by_id_rank": _counter_json(r["id_rank"] or "unknown" for r in kept),
            "photos_by_quality_grade": _counter_json(r["quality_grade"] or f"none ({r['source']})" for r in kept),
            "photos_by_source": _counter_json(r["source"] for r in kept),
            "photos_by_country": _counter_json(r["country"] or "unknown" for r in kept),
            "photo_share_by_country": _share_json(r["country"] or "unknown" for r in kept),
        })
    return pd.DataFrame(rows)


GENUS_COLUMNS = ["genus", "optional", "photos", "groups_observer", "groups_observation", "photos_genus_only",
                 "photos_needs_id_flagged", "n_species", "top_source", "top_source_share", "dominated", "trainable"]


def format_genus_counts(genus_counts):
    lines = [genus_counts[GENUS_COLUMNS].to_string(index=False), ""]
    for col in ("photos_by_species", "groups_observer_by_species", "groups_observation_by_species",
                "photos_by_id_rank", "photos_by_quality_grade", "photos_by_source", "photo_share_by_country"):
        lines.append(f"{col}:")
        for _, r in genus_counts.iterrows():
            name = r["genus"] + (" (optional)" if r["optional"] else "")
            lines.append(f"  {name:<22} {r[col]}")
    return "\n".join(lines)


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


def warnings_list(counts, attr_table, taxa, thresholds, genus_counts=None):
    th = thresholds
    out = []
    for _, t in taxa.iterrows():
        for src in ("inat", "gbif"):
            if t[f"{src}_status"] not in ("exact", "synonym", "fuzzy"):
                out.append(f"{t['taxon']} ({t['rank']}): {'iNaturalist' if src == 'inat' else 'GBIF'} {t[f'{src}_status']}"
                           f"{' - ' + t[f'{src}_note'] if t[f'{src}_note'] else ''} (not collected from this source)")
            elif t[f"{src}_note"]:
                out.append(f"{t['taxon']} ({t['rank']}): {'iNaturalist' if src == 'inat' else 'GBIF'} note - {t[f'{src}_note']}")
    if genus_counts is not None:
        for _, r in genus_counts[genus_counts["genus"] != "ALL"].iterrows():
            tag = f"genus {r['genus']}" + (" (optional)" if r["optional"] else "")
            if not r["trainable"]:
                out.append(f"{tag}: {r['groups_observer']} observer-level groups, below {th['min_species_groups']}")
            if r["dominated"]:
                out.append(f"{tag}: dominated by {r['top_source']} ({r['top_source_share']:.0%} of photos)")
            if r["photos_needs_id_flagged"]:
                out.append(f"{tag}: {r['photos_needs_id_flagged']} photos are genus-rank needs_id (review_flag=genus_needs_id)")
    for _, r in counts[counts["species"] != "ALL"].iterrows():
        tag = r["species"] + (" (optional)" if r["optional"] else "")
        if r["photos"] == 0:
            out.append(f"{tag}: zero photos kept")
        if not r["trainable"]:
            out.append(f"{tag}: {r['groups_observer']} observer-level groups, below {th['min_species_groups']}")
        if r["dominated"]:
            out.append(f"{tag}: dominated by {r['top_source']} ({r['top_source_share']:.0%} of photos)")
        if r["cap_hit"]:
            out.append(f"{tag}: --max-per-species cap hit; counts are truncated")
    for attr, (ok, why) in attribute_verdicts(attr_table, th).items():
        if not ok:
            out.append(f"attribute {attr}: not trainable - {why}")
    for _, r in attr_table.iterrows():
        if r["dominated"]:
            out.append(f"attribute {r['attribute']}={r['class']}: dominated by {r['top_source']} "
                       f"({r['top_source_share']:.0%} of photos)")
    return out


def write_labels(records, path, include_optional=False):
    """Kept photos in the bench's format. Optional candidates are left out unless include_optional."""
    kept = frame([r for r in records if not r["skipped"] and (include_optional or not r["optional"])])
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


MAIN_COLUMNS = ["species", "optional", "photos", "groups_observer", "groups_observation", "groups_file_id", "sex_labeled",
                "maturity_labeled", "skipped_license", "skipped_duplicate", "skipped_near_duplicate", "skipped_cap",
                "skipped_other", "cap_hit", "top_source", "top_source_share", "dominated", "trainable"]


def format_counts(counts):
    lines = [counts[MAIN_COLUMNS].to_string(index=False), ""]
    for col in ("photos_by_source", "photos_by_license", "photos_by_country"):
        lines.append(f"{col}:")
        for _, r in counts.iterrows():
            name = r["species"] + (" (optional)" if r.get("optional") is True else "")
            lines.append(f"  {name:<35} {r[col]}")
    return "\n".join(lines)
