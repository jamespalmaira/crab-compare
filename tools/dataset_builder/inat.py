"""iNaturalist API v1: taxa, places, controlled terms and observations with photos."""
import logging
import re

from . import labels
from .records import new_record

log = logging.getLogger(__name__)


RANKS = {"species": ("species", "subspecies"), "genus": ("genus",)}


def id_rank(taxon_rank):
    """'genus' for an observation identified only to genus, 'species' for species or below, else ''."""
    rank = str(taxon_rank or "").lower()
    if rank == "genus":
        return "genus"
    if rank in ("species", "subspecies", "variety", "form", "hybrid"):
        return "species"
    return ""


def resolve_taxon(client, base, name, rank="species"):
    """{'inat_taxon_id', 'inat_name', 'inat_rank', 'inat_status', 'inat_note'} for one scientific name.

    Status: exact, synonym (the name points to another accepted taxon), higher_rank
    (only a higher rank than `rank` matched) or not_found.
    """
    data = client.get_json(f"{base}/taxa", {"q": name, "is_active": "true", "per_page": 30})
    results = data.get("results", [])
    target = name.strip().lower()
    out = {"inat_taxon_id": "", "inat_name": "", "inat_rank": "", "inat_status": "not_found", "inat_note": ""}

    def preferred(hits):  # homonyms exist (Charybdis is also a plant section): wanted rank and animals first
        return sorted(hits, key=lambda t: (t.get("rank") not in RANKS[rank],
                                           t.get("iconic_taxon_name", "Animalia") in ("Plantae", "Fungi")))
    exact = preferred([t for t in results if str(t.get("name", "")).lower() == target])
    synonym = preferred([t for t in results if str(t.get("matched_term", "")).lower() == target and t not in exact])
    for status, hits in (("exact", exact), ("synonym", synonym)):
        if hits:
            t = hits[0]
            out.update(inat_taxon_id=t["id"], inat_name=t["name"], inat_rank=t.get("rank", ""), inat_status=status)
            if t.get("rank") not in RANKS[rank]:
                out["inat_status"] = "higher_rank"
                out["inat_note"] = f"matched {t['name']} at rank {t.get('rank')}"
            elif status == "synonym":
                out["inat_note"] = f"{name} is treated as {t['name']}"
            return out
    if rank == "species":
        genus = name.split()[0].lower()
        higher = [t for t in results if str(t.get("name", "")).lower() == genus]
        if higher:
            t = higher[0]
            out.update(inat_status="higher_rank", inat_note=f"only {t['name']} ({t.get('rank')}) matched",
                       inat_rank=t.get("rank", ""))
    return out


def resolve_place(client, base, country_names):
    """(place_id, display name) of a country-level place, or (None, reason).

    `country_names` is one name or a list tried in order (iNaturalist may use another name).
    """
    names = [country_names] if isinstance(country_names, str) else list(country_names)
    for country_name in names:
        data = client.get_json(f"{base}/places/autocomplete", {"q": country_name, "per_page": 20})
        countries = [p for p in data.get("results", [])
                     if p.get("admin_level") == 0 or p.get("place_type") == 12]
        target = country_name.strip().lower()
        exact = [p for p in countries
                 if target in (str(p.get("name", "")).lower(), str(p.get("display_name", "")).lower())]
        hits = exact or countries
        if hits:
            return hits[0]["id"], hits[0].get("display_name") or hits[0].get("name")
    return None, f"no country-level place named {' or '.join(repr(n) for n in names)}"


def controlled_terms(client, base):
    return labels.inat_terms(client.get_json(f"{base}/controlled_terms"))


def fetch_observations(client, cfg, cache, key, taxon_id, place_id, quality_grade, limit=None,
                       rank=None, identifications=None):
    """Raw observations with photos for one taxon and place, resumed from the cache.

    taxon_id includes descendants; `rank` (e.g. 'genus') keeps only observations whose
    taxon has exactly that rank, so a genus query returns the records identified only to
    the genus. `identifications` (most_agree, some_agree) is the API's agreement filter.
    Pages by id (id_above) so results past 10,000 can be read. Stops early once the
    query has `limit` photos (the state then stays not-done, so a later run continues).
    """
    ic = cfg["inaturalist"]
    records, state = cache.load(key, "id")
    if state.get("done"):
        return records, state
    params = {"taxon_id": taxon_id, "place_id": place_id, "photos": "true", "per_page": ic["per_page"],
              "order_by": "id", "order": "asc"}
    if quality_grade and quality_grade != "any":
        params["quality_grade"] = quality_grade
    if rank:
        params["rank"] = rank
    if identifications:
        params["identifications"] = identifications
    state["capped"] = False
    while True:
        if limit and sum(len(o.get("photos") or []) for o in records) >= limit:
            state["capped"] = True
            break
        params["id_above"] = state.get("id_above", 0)
        data = client.get_json(f"{ic['base_url']}/observations", params)
        if params["id_above"] == 0:
            state["total"] = data.get("total_results", 0)
        page = data.get("results", [])
        if page:
            state["id_above"] = max(o["id"] for o in page)
        state["done"] = len(page) < ic["per_page"]
        cache.append(key, page, state)
        records.extend(page)
        if state["done"]:
            break
    return records, state


def _public_location(obs):
    private = (obs.get("obscured") or obs.get("geoprivacy") not in (None, "", "open")
               or obs.get("taxon_geoprivacy") not in (None, "", "open"))
    if private:
        return "", "", "obscured"
    coords = (obs.get("geojson") or {}).get("coordinates")
    if coords and len(coords) == 2:
        return coords[1], coords[0], ""
    return "", "", "no location"


def photo_records(obs, species, country, terms, cfg, genus="", review_flag="", optional=False):
    """One record per photo of an observation. `species` is '' for a genus-level query."""
    maps = cfg["labels"]
    sex, sex_raw = labels.inat_annotation(obs.get("annotations"), terms.get("sex"), maps["sex"])
    mat, mat_raw = labels.inat_annotation(obs.get("annotations"), terms.get("maturity"), maps["maturity"])
    lat, lon, note = _public_location(obs)
    user = obs.get("user") or {}
    taxon = obs.get("taxon") or {}
    size = cfg["inaturalist"]["photo_size"]
    out = []
    for p in obs.get("photos") or []:
        url = re.sub(r"/square\.", f"/{size}.", p.get("url") or "")
        dims = p.get("original_dimensions") or {}
        out.append(new_record(
            source="inaturalist", genus=genus, species=species,
            id_rank=id_rank(taxon.get("rank")) or ("species" if species else "genus"),
            optional=optional, review_flag=review_flag,
            photo_id=f"inaturalist:photo:{p['id']}", image_url=url,
            observation_id=f"inaturalist:{obs['id']}", inat_observation_id=str(obs["id"]), inat_photo_id=str(p["id"]),
            observation_url=obs.get("uri") or f"https://www.inaturalist.org/observations/{obs['id']}",
            observer_id=f"inaturalist:{user['id']}" if user.get("id") else "", observer_name=user.get("login", ""),
            taxon_id=taxon.get("id", ""), taxon_name=taxon.get("name", ""), taxon_rank=taxon.get("rank", ""),
            quality_grade=obs.get("quality_grade", ""), date=obs.get("observed_on") or "", country=country,
            latitude=lat, longitude=lon, location_note=note,
            license_raw=p.get("license_code") or "", attribution=p.get("attribution", ""),
            width=dims.get("width", ""), height=dims.get("height", ""),
            sex=sex, sex_source="inaturalist:annotation" if sex else "", sex_raw=sex_raw,
            maturity=mat, maturity_source="inaturalist:annotation" if mat else "", maturity_raw=mat_raw,
        ))
    return out
