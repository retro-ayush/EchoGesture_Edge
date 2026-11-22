"""Trim downloaded WLASL source videos to official frame_start/frame_end.

Mirrors the WLASL start_kit preprocessing: 1-indexed frame ranges on the
decoded source frames; frame_end == -1 keeps the whole video. Writes
<video_id>.mp4 for every instance whose source was downloaded, ready for
scripts/prepare_wlasl.py --clips-are-trimmed.
"""
import argparse
import json
import re
from pathlib import Path
import cv2


def youtube_id(url):
    m = re.search(r"(?:v=|youtu\.be/|embed/)([\w-]{11})", url)
    return m.group(1) if m else None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("annotations"); p.add_argument("sources"); p.add_argument("out")
    p.add_argument("--vocab", required=True)
    a = p.parse_args()
    allowed = set(json.loads(Path(a.vocab).read_text()))
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    sources = {f.stem: f for f in Path(a.sources).iterdir() if not f.name.startswith("._")}
    written = skipped = 0
    for entry in json.loads(Path(a.annotations).read_text()):
        if entry["gloss"].upper() not in allowed:
            continue
        for inst in entry["instances"]:
            src = sources.get(youtube_id(inst["url"]) or "")
            dest = out/f"{inst['video_id']}.mp4"
            if src is None or dest.exists():
                skipped += src is None; continue
            cap = cv2.VideoCapture(str(src)); fps = cap.get(cv2.CAP_PROP_FPS) or 25
            start, end = inst["frame_start"], inst["frame_end"]
            writer, n = None, 0
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                n += 1
                if n < start or (end != -1 and n > end):
                    if end != -1 and n > end:
                        break
                    continue
                if writer is None:
                    h, w = frame.shape[:2]
                    writer = cv2.VideoWriter(str(dest), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
                writer.write(frame)
            cap.release()
            if writer is not None:
                writer.release(); written += 1
    print(json.dumps({"written": written, "no_source": skipped}))


if __name__ == "__main__":
    main()
