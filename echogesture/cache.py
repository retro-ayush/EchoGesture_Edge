"""Extract landmarks once per clip into local NPZ files.

Resumable: existing NPZ files are reused unless --overwrite is given.
Clips that fail extraction are excluded from the output manifest and listed
in failures.json so they can be disclosed, never silently dropped.
--workers N extracts clips in N parallel processes (one extractor per clip).
"""
import argparse
import csv
import hashlib
import json
from multiprocessing import get_context
from pathlib import Path
import numpy as np
from .data import read_manifest
from .extract import video_features


def _extract(job):
    """Returns (row, cached_name, status, error)."""
    row, target, models, overwrite = job
    target = Path(target)
    if target.exists() and not overwrite:
        return row, target.name, "Reused", None
    try:
        features = video_features(row["path"], models)
    except Exception as error:  # reported, never silently dropped
        return row, target.name, "FAILED", str(error)
    temporary = target.with_suffix(".partial.npz")
    np.savez_compressed(temporary, features=features); temporary.replace(target)
    return row, target.name, "Cached", None


def main():
    p = argparse.ArgumentParser(); p.add_argument("manifest")
    p.add_argument("--out", default="data/features"); p.add_argument("--models", default="models")
    p.add_argument("--overwrite", action="store_true", help="Re-extract clips already cached")
    p.add_argument("--strict", action="store_true", help="Stop on the first failed clip")
    p.add_argument("--workers", type=int, default=1, help="Parallel extraction processes")
    a = p.parse_args(); out = Path(a.out).resolve(); out.mkdir(parents=True, exist_ok=True)
    rows = read_manifest(a.manifest)
    models = str(Path(a.models).resolve())
    # Avoid treating externally supplied clip IDs as filesystem paths.
    jobs = [(row, str(out/(hashlib.sha256(row["clip_id"].encode()).hexdigest()[:24]+".npz")),
             models, a.overwrite) for row in rows]
    results = {}
    failures = []

    def handle(n, result):
        row, name, status, error = result
        print(f"[{n}/{len(rows)}] {status}", row["clip_id"], error or "", flush=True)
        if error is not None:
            if a.strict:
                raise SystemExit(f"Extraction failed for {row['clip_id']}: {error}")
            failures.append({"clip_id": row["clip_id"], "split": row["split"],
                             "label": row.get("label", ""), "error": error})
        else:
            results[row["clip_id"]] = dict(row, path=name)

    if a.workers > 1:
        with get_context("spawn").Pool(a.workers, maxtasksperchild=50) as pool:
            for n, result in enumerate(pool.imap_unordered(_extract, jobs), 1):
                handle(n, result)
    else:
        for n, job in enumerate(jobs, 1):
            handle(n, _extract(job))
    kept = [results[r["clip_id"]] for r in rows if r["clip_id"] in results]  # manifest order
    if not kept:
        raise SystemExit("No clips extracted")
    with (out/"manifest.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(kept[0])); writer.writeheader(); writer.writerows(kept)
    (out/"failures.json").write_text(json.dumps(failures, indent=2))
    print(json.dumps({"cached": len(kept), "failed": len(failures),
                      "failures_file": str(out/"failures.json")}, indent=2))


if __name__ == "__main__": main()
