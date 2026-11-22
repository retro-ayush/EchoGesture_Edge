"""Write paired corrupted copies of one split for robustness evaluation.

Example:
  python scripts/make_corruptions.py data/videos.csv --split test --kind low_light \
      --out data/corrupted/low_light
Then pass data/corrupted/low_light/videos.csv to evaluate --corrupted-manifest.
Generated clips derive from dataset videos: keep them out of public repos.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from echogesture.corrupt import KINDS, apply, clip_rng, params_for  # noqa: E402
from echogesture.data import read_manifest  # noqa: E402


def main():
    import cv2
    p = argparse.ArgumentParser()
    p.add_argument("manifest"); p.add_argument("--split", default="test", choices=["train", "val", "test"])
    p.add_argument("--kind", required=True, choices=KINDS)
    p.add_argument("--out", required=True); p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    rows = [r for r in read_manifest(a.manifest) if r["split"] == a.split]
    if not rows:
        raise ValueError("No clips in selected split")
    out = Path(a.out).resolve(); (out/"clips").mkdir(parents=True, exist_ok=True)
    written, record = [], []
    for n, row in enumerate(rows, 1):
        rng = clip_rng(row["clip_id"], a.kind, a.seed)
        params = params_for(a.kind, rng)
        cap = cv2.VideoCapture(row["path"])
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {row['path']}")
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        name = hashlib.sha256(row["clip_id"].encode()).hexdigest()[:24]+".mp4"
        writer = None
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                if writer is None:
                    h, w = frame.shape[:2]
                    writer = cv2.VideoWriter(str(out/"clips"/name), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
                writer.write(apply(a.kind, frame, params, rng))
        finally:
            cap.release()
            if writer is not None:
                writer.release()
        if writer is None:
            raise ValueError(f"No frames decoded: {row['path']}")
        written.append({**row, "path": f"clips/{name}"})
        record.append({"clip_id": row["clip_id"], **params})
        print(f"[{n}/{len(rows)}] {a.kind}", row["clip_id"])
    with (out/"videos.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(written[0])); w.writeheader(); w.writerows(written)
    (out/"corruption_params.json").write_text(json.dumps(
        {"kind": a.kind, "seed": a.seed, "source": str(Path(a.manifest).resolve()),
         "split": a.split, "clips": record}, indent=2))
    print("Manifest:", out/"videos.csv")


if __name__ == "__main__":
    main()
