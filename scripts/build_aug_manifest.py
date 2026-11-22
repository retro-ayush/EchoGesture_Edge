"""Combine the clean features manifest with corrupted *training* copies.

Corrupted rows keep their label and the train split, get clip_id
'<id>__<kind>', and only train rows are taken from the extra manifests,
so validation and test stay clean and untouched.

  python scripts/build_aug_manifest.py data/features/manifest.csv \
      low_light=data/features_aug/low_light/manifest.csv \
      occlusion=data/features_aug/occlusion/manifest.csv \
      --out data/features_aug/manifest.csv
"""
import argparse
import csv
import json
from collections import Counter
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from echogesture.data import read_manifest  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("base"); p.add_argument("extra", nargs="+", help="kind=path/to/manifest.csv")
    p.add_argument("--out", required=True)
    a = p.parse_args()
    rows = read_manifest(a.base)  # paths resolved to absolute
    for spec in a.extra:
        kind, path = spec.split("=", 1)
        for r in read_manifest(path):
            if r["split"] == "train":
                rows.append(dict(r, clip_id=f"{r['clip_id']}__{kind}"))
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["clip_id", "path", "label", "split", "signer_id"], extrasaction="ignore")
        w.writeheader(); w.writerows(rows)
    print(json.dumps({"rows": len(rows), "splits": Counter(r["split"] for r in rows)}))


if __name__ == "__main__":
    main()
