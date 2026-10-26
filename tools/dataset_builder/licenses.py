"""Normalise license strings from iNaturalist, GBIF and manual imports, and check them against the allowlist."""
import re

ARR = "ARR"  # all rights reserved
OTHER = "OTHER"
_PARTS = ("by", "nc", "sa", "nd")


def normalize(raw):
    """A license code such as CC0 or CC-BY-NC-SA, '' if none given, ARR or OTHER.

    Accepts iNaturalist codes (cc-by-nc), GBIF URLs (http://creativecommons.org/licenses/by-nc/4.0/)
    and free text (CC BY-SA 4.0, CC0: Public Domain). Version numbers are dropped.
    """
    if raw is None:
        return ""
    s = str(raw).strip().lower()
    if not s:
        return ""
    if "all rights reserved" in s or s in ("arr", "c", "copyright"):
        return ARR
    if "publicdomain/zero" in s or re.search(r"\bcc[-_ ]?0\b", s) or "cc0" in s:
        return "CC0"
    m = re.search(r"creativecommons\.org/licenses/([a-z-]+)", s)
    if m:
        parts = m.group(1).split("-")
    else:
        m = re.search(r"\bcc[-_ ]+((?:by|nc|sa|nd)(?:[-_ ]+(?:by|nc|sa|nd))*)\b", s)
        if not m:
            return OTHER
        parts = re.split(r"[-_ ]+", m.group(1))
    if "by" not in parts or any(p not in _PARTS for p in parts):
        return OTHER
    return "CC-" + "-".join(p.upper() for p in _PARTS if p in parts)


def check(raw, allowlist):
    """(code, skip_reason); skip_reason is '' when the license is allowed."""
    code = normalize(raw)
    allowed = {a.upper() for a in allowlist}
    if code == "":
        return code, "license_missing"
    if code == ARR:
        return code, "license_all_rights_reserved"
    if code in allowed or (code == OTHER and str(raw).strip().upper() in allowed):
        return code, ""
    return code, f"license_not_allowed:{code}"
