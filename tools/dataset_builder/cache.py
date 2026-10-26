"""Resumable on-disk cache of raw API pages: records are appended as each page arrives."""
import json
import os
import re
from pathlib import Path


class QueryCache:
    def __init__(self, root):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _paths(self, key):
        safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", key)
        return self.root / f"{safe}.jsonl", self.root / f"{safe}.state.json"

    def load(self, key, id_field):
        """(records, state). Records repeated by an interrupted write are kept once."""
        data, state = self._paths(key)
        records, seen = [], set()
        if data.exists():
            with open(data, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:  # half-written last line
                        continue
                    if rec.get(id_field) not in seen:
                        seen.add(rec.get(id_field))
                        records.append(rec)
        st = json.loads(state.read_text(encoding="utf-8")) if state.exists() else {}
        return records, st

    def append(self, key, records, state):
        """Append one page, then save the cursor (so a crash at worst repeats a page)."""
        data, st = self._paths(key)
        with open(data, "a", encoding="utf-8") as f:
            for rec in records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        tmp = st.with_suffix(".tmp")
        tmp.write_text(json.dumps(state), encoding="utf-8")
        os.replace(tmp, st)
