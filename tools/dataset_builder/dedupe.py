"""Duplicate removal: GBIF copies of iNaturalist observations, exact repeats, and near-duplicate images."""
import json

import imagehash
import numpy as np
from PIL import Image

from .records import skip


def source_rank(source, priority):
    """Lower is preferred; sources not in the priority list (manual imports) come last."""
    return priority.index(source) if source in priority else len(priority)


def cross_source(records, priority):
    """Mark GBIF records whose iNaturalist observation or photo is also in the iNaturalist records,
    then exact repeats of one photo (same iNaturalist photo ID or image URL). The preferred copy is kept.

    All iNaturalist records count as the original, including ones later skipped for their license.
    Returns the number marked.
    """
    inat = [r for r in records if r["source"] == "inaturalist"]
    obs = {r["inat_observation_id"] for r in inat if r["inat_observation_id"]}
    photos = {r["inat_photo_id"] for r in inat if r["inat_photo_id"]}
    n = 0
    for r in records:
        if r["skipped"] or r["source"] != "gbif":
            continue
        if r["inat_observation_id"] in obs or r["inat_photo_id"] in photos:
            skip(r, "duplicate_of_inaturalist")
            n += 1
    seen = set()
    for r in sorted(records, key=lambda r: source_rank(r["source"], priority)):
        if r["skipped"]:
            continue
        keys = {k for k in (f"inatphoto:{r['inat_photo_id']}" if r["inat_photo_id"] else "",
                            f"url:{r['image_url']}" if r["image_url"] else "") if k}
        if keys & seen:
            skip(r, "duplicate_photo")
            n += 1
        seen |= keys
    return n


def open_image(path, crop_box=""):
    im = Image.open(path)
    im = im.convert("RGB")
    if crop_box:
        im = im.crop(tuple(json.loads(crop_box)))
    return im


def phash(path, crop_box=""):
    """64-bit perceptual hash as a 16-character hex string."""
    with open_image(path, crop_box) as im:
        return str(imagehash.phash(im, hash_size=8))


def _popcount(x):
    return np.unpackbits(x.view(np.uint8).reshape(-1, 8), axis=1).sum(axis=1)


def near_duplicates(records, max_distance, priority):
    """Mark near-duplicate images among records that have a 'phash'. The preferred source, then
    the earlier record, is kept. Returns rows for near_duplicates.csv."""
    candidates = [r for r in records if not r["skipped"] and r["phash"]]
    candidates.sort(key=lambda r: source_rank(r["source"], priority))
    kept_hashes = np.zeros(len(candidates), dtype=np.uint64)
    kept = []
    log = []
    for r in candidates:
        h = np.uint64(int(r["phash"], 16))
        if kept:
            dist = _popcount(np.bitwise_xor(kept_hashes[:len(kept)], h))
            j = int(np.argmin(dist))
            if dist[j] <= max_distance:
                skip(r, "near_duplicate")
                k = kept[j]
                log.append({"removed_photo_id": r["photo_id"], "removed_source": r["source"],
                            "removed_species": r["species"], "kept_photo_id": k["photo_id"],
                            "kept_source": k["source"], "kept_species": k["species"], "distance": int(dist[j])})
                continue
        kept_hashes[len(kept)] = h
        kept.append(r)
    return log
