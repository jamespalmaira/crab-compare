"""GBIF API: species match and occurrences with still images."""
import logging
import re

from . import labels
from .records import new_record

log = logging.getLogger(__name__)

INAT_OBS = re.compile(r"inaturalist\.org/observations/(\d+)")
INAT_PHOTO = re.compile(r"inaturalist[^/]*/photos/(\d+)/")


def match_species(client, base, name):
    """{'gbif_usage_key', 'gbif_species_key', 'gbif_name', 'gbif_rank', 'gbif_match', 'gbif_status', 'gbif_note'}.

    Status: exact, fuzzy, synonym, higher_rank or not_found. Only exact, fuzzy and
    synonym matches with a species key are collected.
    """
    m = client.get_json(f"{base}/species/match", {"name": name, "kingdom": "Animalia", "strict": "true"})
    out = {"gbif_usage_key": m.get("usageKey", ""), "gbif_species_key": "", "gbif_name": m.get("canonicalName", ""),
           "gbif_rank": str(m.get("rank", "")).lower(), "gbif_match": m.get("matchType", "NONE"),
           "gbif_status": "not_found", "gbif_note": ""}
    match = out["gbif_match"]
    if match == "NONE":
        out["gbif_note"] = m.get("note", "")
    elif match == "HIGHERRANK" or out["gbif_rank"] not in ("species", "subspecies"):
        out["gbif_status"] = "higher_rank"
        out["gbif_note"] = f"only {out['gbif_name']} ({out['gbif_rank']}) matched"
    else:
        out["gbif_species_key"] = m.get("speciesKey") or m.get("usageKey")
        out["gbif_status"] = "synonym" if m.get("synonym") else match.lower()
        if m.get("synonym"):
            out["gbif_note"] = f"{name} is a synonym of {m.get('species', '')}"
        elif match == "FUZZY":
            out["gbif_note"] = f"fuzzy match to {out['gbif_name']}"
    return out


def fetch_occurrences(client, cfg, cache, key, species_key, country, limit=None):
    """Raw occurrences with still images for one species key and country, resumed from the cache."""
    gc = cfg["gbif"]
    records, state = cache.load(key, "key")
    if state.get("done"):
        return records, state
    params = {"taxonKey": species_key, "country": country, "mediaType": "StillImage",
              "occurrenceStatus": "PRESENT", "limit": gc["page_size"]}
    state["capped"] = False
    while True:
        if limit and sum(_n_images(o) for o in records) >= limit:
            state["capped"] = True
            break
        offset = state.get("offset", 0)
        if offset >= gc["max_offset"]:
            log.warning("GBIF %s: stopped at the %d-record paging limit", key, gc["max_offset"])
            state["done"] = True
            state["truncated"] = True
            cache.append(key, [], state)
            break
        params["offset"] = offset
        data = client.get_json(f"{gc['base_url']}/occurrence/search", params)
        if offset == 0:
            state["total"] = data.get("count", 0)
        page = data.get("results", [])
        state["offset"] = offset + len(page)
        state["done"] = bool(data.get("endOfRecords", True)) or not page
        cache.append(key, page, state)
        records.extend(page)
        if state["done"]:
            break
    return records, state


def _n_images(occ):
    return sum(1 for m in occ.get("media") or [] if m.get("type") == "StillImage")


def inat_observation_id(occ, inat_dataset_key):
    """The iNaturalist observation ID if this occurrence came from iNaturalist, else ''."""
    for field in ("references", "occurrenceID"):
        m = INAT_OBS.search(str(occ.get(field) or ""))
        if m:
            return m.group(1)
    if occ.get("datasetKey") == inat_dataset_key and str(occ.get("catalogNumber", "")).isdigit():
        return str(occ["catalogNumber"])
    return ""


def photo_records(occ, species, cfg):
    """One record per still image of an occurrence. The image license is the media license."""
    maps = cfg["labels"]
    inat_obs = inat_observation_id(occ, cfg["gbif"]["inaturalist_dataset_key"])
    sex = labels.map_value(occ.get("sex"), maps["sex"])
    mat = labels.map_value(occ.get("lifeStage"), maps["maturity"])
    observer = str(occ.get("recordedBy") or "").strip()
    lat, lon = occ.get("decimalLatitude", ""), occ.get("decimalLongitude", "")
    out = []
    for i, media in enumerate(m for m in occ.get("media") or [] if m.get("type") == "StillImage"):
        url = media.get("identifier") or ""
        photo = INAT_PHOTO.search(url)
        credit = media.get("rightsHolder") or media.get("creator") or observer
        out.append(new_record(
            source="gbif", species=species, photo_id=f"gbif:{occ['key']}:{i}", image_url=url,
            observation_id=f"inaturalist:{inat_obs}" if inat_obs else f"gbif:{occ['key']}",
            observation_url=f"https://www.gbif.org/occurrence/{occ['key']}",
            observer_id=f"gbif:{observer.lower()}" if observer else "", observer_name=observer,
            inat_observation_id=inat_obs, inat_photo_id=photo.group(1) if photo else "",
            taxon_id=occ.get("taxonKey", ""), taxon_name=occ.get("scientificName", ""),
            taxon_rank=str(occ.get("taxonRank", "")).lower(), basis_of_record=occ.get("basisOfRecord", ""),
            dataset=occ.get("datasetKey", ""), date=str(occ.get("eventDate") or "")[:10],
            country=occ.get("countryCode", ""), latitude="" if lat is None else lat,
            longitude="" if lon is None else lon, location_note="" if lat not in ("", None) else "no location",
            license_raw=media.get("license") or "", attribution=f"{credit} via GBIF" if credit else "via GBIF",
            sex=sex, sex_source="gbif:sex" if sex else "", sex_raw=occ.get("sex") or "",
            maturity=mat, maturity_source="gbif:lifeStage" if mat else "", maturity_raw=occ.get("lifeStage") or "",
        ))
    return out
