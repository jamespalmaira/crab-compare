"""Canned API responses and a fake client for the dataset builder tests (no network)."""
import copy
from pathlib import Path

import yaml

TOOL = Path(__file__).resolve().parents[1]
INAT = "https://api.inaturalist.org/v1"
GBIF = "https://api.gbif.org/v1"


def load_cfg():
    with open(TOOL / "config.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


CONTROLLED_TERMS = {"results": [
    {"id": 1, "label": "Life Stage", "values": [{"id": 2, "label": "Adult"}, {"id": 8, "label": "Juvenile"},
                                                {"id": 6, "label": "Larva"}]},
    {"id": 9, "label": "Sex", "values": [{"id": 11, "label": "Female"}, {"id": 10, "label": "Male"},
                                         {"id": 20, "label": "Cannot Be Determined"}]},
    {"id": 12, "label": "Plant Phenology", "values": []},
]}
TERMS = {"sex": (9, {11: "Female", 10: "Male", 20: "Cannot Be Determined"}),
         "maturity": (1, {2: "Adult", 8: "Juvenile", 6: "Larva"})}


def inat_obs(obs_id, user_id, photos, annotations=(), **extra):
    """photos: [(photo_id, license_code)]"""
    o = {"id": obs_id, "uri": f"https://www.inaturalist.org/observations/{obs_id}", "quality_grade": "research",
         "observed_on": "2024-05-01", "user": {"id": user_id, "login": f"user{user_id}"},
         "taxon": {"id": 47185, "name": "Scylla serrata", "rank": "species"},
         "geojson": {"type": "Point", "coordinates": [121.0, 14.5]}, "obscured": False, "geoprivacy": None,
         "taxon_geoprivacy": None, "annotations": list(annotations),
         "photos": [{"id": pid, "license_code": lic, "attribution": f"(c) user{user_id}",
                     "url": f"https://inaturalist-open-data.s3.amazonaws.com/photos/{pid}/square.jpg",
                     "original_dimensions": {"width": 2048, "height": 1536}} for pid, lic in photos]}
    o.update(extra)
    return o


def gbif_occ(key, media, inat_obs_id=None, recorded_by="Jane Doe", **extra):
    """media: [(url, license)]"""
    o = {"key": key, "datasetKey": "other-dataset", "recordedBy": recorded_by, "countryCode": "PH",
         "eventDate": "2023-02-03T00:00:00", "decimalLatitude": 10.0, "decimalLongitude": 123.0,
         "scientificName": "Scylla serrata (Forskål, 1775)", "taxonKey": 2225776, "taxonRank": "SPECIES",
         "basisOfRecord": "HUMAN_OBSERVATION",
         "media": [{"type": "StillImage", "format": "image/jpeg", "identifier": url, "license": lic,
                    "rightsHolder": recorded_by} for url, lic in media]}
    if inat_obs_id:
        o.update(datasetKey="50c9509d-22c7-4a22-a47d-8c48425ef4a7",
                 references=f"https://www.inaturalist.org/observations/{inat_obs_id}",
                 occurrenceID=f"https://www.inaturalist.org/observations/{inat_obs_id}")
    o.update(extra)
    return o


class FakeResponse:
    def __init__(self, data=None, content=b"", status=200):
        self.data, self.content, self.status_code, self.headers = data, content, status, {}

    def json(self):
        return copy.deepcopy(self.data)

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


class FakeClient:
    """Routes get_json(url, params) to a handler by URL suffix; records every call."""

    def __init__(self, routes):
        self.routes = routes
        self.calls = []

    def get_json(self, url, params=None):
        self.calls.append((url, dict(params or {})))
        for suffix, handler in self.routes.items():
            if url.endswith(suffix):
                return copy.deepcopy(handler(params or {}) if callable(handler) else handler)
        raise AssertionError(f"unexpected request {url} {params}")

    def get(self, url, params=None):
        self.calls.append((url, dict(params or {})))
        raise AssertionError(f"unexpected download {url}")


def default_routes(observations=(), occurrences=()):
    """A small world: Scylla serrata resolves; Scylla fakeus matches only the genus."""
    def taxa(p):
        if p["q"] == "Scylla serrata":
            return {"results": [{"id": 47185, "name": "Scylla serrata", "rank": "species", "matched_term": "Scylla serrata"}]}
        return {"results": [{"id": 47184, "name": "Scylla", "rank": "genus", "matched_term": "Scylla"}]}

    def match(p):
        if p["name"] == "Scylla serrata":
            return {"usageKey": 2225776, "speciesKey": 2225776, "genusKey": 2225770, "canonicalName": "Scylla serrata",
                    "rank": "SPECIES", "matchType": "EXACT", "synonym": False}
        if p["name"] == "Scylla" and p.get("rank") == "GENUS":
            return {"usageKey": 2225770, "genusKey": 2225770, "canonicalName": "Scylla", "rank": "GENUS",
                    "matchType": "EXACT", "synonym": False}
        return {"usageKey": 2225770, "canonicalName": "Scylla", "rank": "GENUS", "matchType": "HIGHERRANK"}

    def places(p):
        return {"results": [{"id": 6888, "name": "Philippines", "display_name": "Philippines", "admin_level": 0},
                            {"id": 1, "name": "Philippines Sea", "admin_level": None}]}

    def obs(p):
        # like the API: rank keeps only observations whose taxon has that rank; quality_grade filters too
        match = [o for o in observations
                 if (not p.get("rank") or o["taxon"]["rank"] == p["rank"])
                 and (not p.get("quality_grade") or o["quality_grade"] in p["quality_grade"].split(","))]
        page = [o for o in match if o["id"] > p.get("id_above", 0)][: p["per_page"]]
        return {"total_results": len(match), "results": page}

    def occ(p):
        page = list(occurrences)[p["offset"]: p["offset"] + p["limit"]]
        return {"count": len(occurrences), "results": page, "endOfRecords": p["offset"] + p["limit"] >= len(occurrences)}

    return {"/taxa": taxa, "/species/match": match, "/places/autocomplete": places,
            "/controlled_terms": CONTROLLED_TERMS, "/observations": obs, "/occurrence/search": occ}
