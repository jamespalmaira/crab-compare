"""One record per photo, shared by every source."""
import re

FIELDS = (
    "source", "genus", "species", "id_rank", "optional", "review_flag",
    "photo_id", "image_url", "local_path", "crop_box", "image_path",
    "observation_id", "observation_url", "observer_id", "observer_name", "file_id",
    "inat_observation_id", "inat_photo_id",
    "taxon_id", "taxon_name", "taxon_rank", "quality_grade", "basis_of_record", "dataset",
    "date", "country", "latitude", "longitude", "location_note",
    "license", "license_raw", "attribution", "width", "height",
    "sex", "sex_source", "sex_raw", "maturity", "maturity_source", "maturity_raw",
    "skipped", "skip_reason", "crab_id", "crab_id_field", "phash",
)


def new_record(**values):
    """One photo. id_rank is 'species' or 'genus' (identified only to genus: species is empty).
    review_flag is set on records kept only because of an opt-in looser filter."""
    rec = dict.fromkeys(FIELDS, "")
    rec["skipped"] = False
    rec["optional"] = False
    unknown = set(values) - set(FIELDS)
    if unknown:
        raise KeyError(f"unknown record fields: {sorted(unknown)}")
    rec.update(values)
    return rec


def skip(rec, reason):
    """Mark a record skipped; the first reason wins."""
    if not rec["skipped"]:
        rec["skipped"] = True
        rec["skip_reason"] = reason


def taxon_label(rec):
    """Species name, or '<Genus> (genus only)' for a record identified only to genus."""
    return rec["species"] or f"{rec['genus']} (genus only)"


def slug(name):
    return re.sub(r"[^a-z0-9]+", "_", str(name).lower()).strip("_")


def url_ext(url):
    m = re.search(r"\.(jpe?g|png|webp)(?:$|\?)", str(url).lower())
    return m.group(1) if m else "jpg"
