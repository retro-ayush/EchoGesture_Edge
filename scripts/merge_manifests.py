"""Append extra trimmed clips (e.g. from trim_wlasl_sources.py) to a manifest.

Official splits/labels/signers come from the annotations; clips already in the
manifest are skipped; clips shorter than 8 or longer than 400 frames are rejected.
"""
import argparse
import csv
import json
from pathlib import Path
import cv2


def main():
    p = argparse.ArgumentParser()
    p.add_argument("manifest"); p.add_argument("annotations"); p.add_argument("clips")
    p.add_argument("--vocab", required=True)
    a = p.parse_args()
    vocab = set(json.loads(Path(a.vocab).read_text()))
    inst = {i["video_id"]: (e["gloss"].upper(), i) for e in json.loads(Path(a.annotations).read_text())
            if e["gloss"].upper() in vocab for i in e["instances"]}
    rows = list(csv.DictReader(open(a.manifest, newline="")))
    have = {r["clip_id"] for r in rows}
    added = rejected = 0
    for f in sorted(Path(a.clips).glob("*.mp4")):
        if f.name.startswith("._") or f.stem in have or f.stem not in inst:
            continue
        frames = int(cv2.VideoCapture(str(f)).get(cv2.CAP_PROP_FRAME_COUNT))
        if not 8 <= frames <= 400:
            rejected += 1; continue
        gloss, i = inst[f.stem]
        rows.append({"clip_id": f.stem, "path": str(f.resolve()), "label": gloss,
                     "split": i["split"], "signer_id": str(i.get("signer_id", ""))})
        added += 1
    with open(a.manifest, "w", newline="") as out:
        w = csv.DictWriter(out, fieldnames=["clip_id", "path", "label", "split", "signer_id"])
        w.writeheader(); w.writerows(rows)
    print(json.dumps({"added": added, "rejected_length": rejected, "total": len(rows)}))


if __name__ == "__main__":
    main()
