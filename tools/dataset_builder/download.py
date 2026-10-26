"""Image download (only when not a dry run), validity check, and review contact sheets."""
import json
import logging
import os
import random
import shutil
from pathlib import Path

from PIL import Image, ImageDraw

from .dedupe import open_image

log = logging.getLogger(__name__)


def verify_image(path):
    """(width, height) of a valid image file; raises if it is not one."""
    with Image.open(path) as im:
        im.verify()
    with Image.open(path) as im:
        im.load()
        return im.size


class DownloadLog:
    """Append-only log of finished downloads, so an interrupted run resumes where it stopped."""

    def __init__(self, path):
        self.path = Path(path)
        self.done = {}
        if self.path.exists():
            for line in self.path.read_text(encoding="utf-8").splitlines():
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue
                self.done[entry["photo_id"]] = entry

    def add(self, entry):
        self.done[entry["photo_id"]] = entry
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")


def materialize(rec, data_dir, client, dlog):
    """Download (API sources), copy or crop (manual imports) one record's image to data_dir/image_path.

    Returns (ok, reason). Sets width and height from the saved file.
    """
    dest = Path(data_dir) / rec["image_path"]
    prev = dlog.done.get(rec["photo_id"])
    if prev and prev.get("ok") and dest.exists():
        rec["width"], rec["height"] = prev["width"], prev["height"]
        return True, ""
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".part")
    try:
        if rec["image_url"]:
            tmp.write_bytes(client.get(rec["image_url"]).content)
        elif rec["crop_box"]:
            with open_image(rec["local_path"], rec["crop_box"]) as im:
                im.save(tmp, format="JPEG", quality=95)
        else:
            shutil.copyfile(rec["local_path"], tmp)
    except Exception as e:  # network error after retries, or an unreadable source file
        return _fail(rec, tmp, dlog, "download_failed", e)
    try:
        w, h = verify_image(tmp)
    except Exception as e:  # not an image, or truncated
        return _fail(rec, tmp, dlog, "invalid_image", e)
    os.replace(tmp, dest)
    rec["width"], rec["height"] = w, h
    dlog.add({"photo_id": rec["photo_id"], "ok": True, "width": w, "height": h})
    return True, ""


def _fail(rec, tmp, dlog, reason, err):
    tmp.unlink(missing_ok=True)
    log.warning("%s: %s: %s", rec["photo_id"], reason, err)
    dlog.add({"photo_id": rec["photo_id"], "ok": False, "reason": reason})
    return False, reason


def contact_sheet(paths, out, thumb=160, columns=10, n=50, seed=0):
    """A grid of up to n randomly chosen images, labeled with their file names."""
    paths = list(paths)
    random.Random(seed).shuffle(paths)
    paths = paths[:n]
    if not paths:
        return None
    rows = (len(paths) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * thumb, rows * (thumb + 14)), "white")
    draw = ImageDraw.Draw(sheet)
    for i, p in enumerate(paths):
        x, y = (i % columns) * thumb, (i // columns) * (thumb + 14)
        with Image.open(p) as im:
            im = im.convert("RGB")
            im.thumbnail((thumb, thumb))
            sheet.paste(im, (x + (thumb - im.width) // 2, y + (thumb - im.height) // 2))
        draw.text((x + 2, y + thumb), Path(p).name[:24], fill="black")
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out, quality=90)
    return out
