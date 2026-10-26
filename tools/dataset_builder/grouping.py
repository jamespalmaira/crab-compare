"""Group keys for crab_id (leakage control). A key is a grouping, not a verified individual crab.

Each level falls back to the next finer key when a record lacks the field, and the
level used is recorded so the fallback is visible in metadata.csv.
"""
LEVELS = ("observer", "observation", "file_id")
CHAINS = {
    "observer": ("observer_id", "observation_id", "photo_id"),
    "observation": ("observation_id", "photo_id"),
    "file_id": ("file_id", "observer_id", "observation_id", "photo_id"),
}
_PREFIX = {"observer_id": "observer", "observation_id": "observation", "file_id": "file", "photo_id": "photo"}


def group_key(record, level):
    """(crab_id, field used) for one record at a grouping level."""
    for field in CHAINS[level]:
        value = record.get(field)
        if value:
            return f"{_PREFIX[field]}:{value}", field
    raise ValueError(f"record has no grouping field: {record.get('photo_id')!r}")


def assign(records, level):
    for r in records:
        r["crab_id"], r["crab_id_field"] = group_key(r, level)
