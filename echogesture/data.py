import csv
from pathlib import Path


def read_manifest(path):
    path = Path(path)
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    ids = [r.get("clip_id", "") for r in rows]
    if not rows or any(not i for i in ids) or len(ids) != len(set(ids)):
        raise ValueError("Manifest must contain unique, nonempty clip_id values")
    for r in rows:
        if r.get("split") not in ("train", "val", "test"):
            raise ValueError("Every row needs split=train, val or test")
        if not r.get("path"):
            raise ValueError("Every row needs path")
        p = Path(r["path"])
        r["path"] = str(p if p.is_absolute() else (path.parent/p).resolve())
        if r.get("label"):
            r["label"] = r["label"].upper()
    return rows


def audit(rows):
    """Fail on duplicate file paths and known cross-split signer leakage."""
    paths, signers = {}, {}
    for r in rows:
        p = str(Path(r["path"]).resolve())
        if p in paths:
            raise ValueError(f"Duplicate video path: {r['clip_id']} and {paths[p]}")
        paths[p] = r["clip_id"]
        if r.get("signer_id"):
            signers.setdefault(r["signer_id"], set()).add(r["split"])
    overlap = {s: sorted(v) for s, v in signers.items() if len(v) > 1}
    return {"signer_overlap": overlap,
            "signer_ids_missing": sum(not r.get("signer_id") for r in rows)}
