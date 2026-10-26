"""Map sex and life-stage values from the sources to the label set. Never guesses from images."""


def map_value(raw, mapping):
    """The mapped label for one raw value, or '' if it is missing or not in the mapping."""
    if raw is None:
        return ""
    return mapping.get(str(raw).strip().lower(), "")


def inat_terms(controlled_terms):
    """{'sex': (attribute_id, {value_id: label}), 'maturity': ...} from /v1/controlled_terms.

    The terms are found by label ('Sex', 'Life Stage'), not by hardcoded IDs.
    """
    wanted = {"sex": "sex", "life stage": "maturity"}
    out = {}
    for term in controlled_terms.get("results", []):
        key = wanted.get(str(term.get("label", "")).strip().lower())
        if key:
            out[key] = (term["id"], {v["id"]: v["label"] for v in term.get("values", [])})
    return out


def inat_annotation(annotations, term, mapping):
    """(label, raw) for one controlled term from an observation's annotations.

    Annotations voted down (vote_score < 0) are ignored. Conflicting values give an empty label.
    """
    if term is None:
        return "", ""
    attr_id, values = term
    raw = sorted({values.get(a.get("controlled_value_id"), f"value:{a.get('controlled_value_id')}")
                  for a in annotations or []
                  if a.get("controlled_attribute_id") == attr_id and (a.get("vote_score") or 0) >= 0})
    mapped = {map_value(v, mapping) for v in raw} - {""}
    return (mapped.pop() if len(mapped) == 1 else ""), ";".join(raw)
