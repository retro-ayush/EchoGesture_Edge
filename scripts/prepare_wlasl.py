"""Adapt official annotations and already trimmed local clips; no video scraping.

The caller must supply clips trimmed to official frame_start/frame_end.
Using untrimmed source videos without annotation-based cropping is invalid.
"""
import argparse
import csv
import json
from pathlib import Path


def main():
    p=argparse.ArgumentParser();p.add_argument("annotations");p.add_argument("clips")
    p.add_argument("--vocab",required=True);p.add_argument("--out",default="data/videos.csv")
    p.add_argument("--clips-are-trimmed",action="store_true",required=True)
    a=p.parse_args()
    labels=json.loads(Path(a.vocab).read_text()); allowed=set(labels)
    entries=json.loads(Path(a.annotations).read_text()); rows=[]; missing=[]
    for entry in entries:
        label=entry["gloss"].upper()
        if label not in allowed:continue
        for item in entry["instances"]:
            clip_id=str(item["video_id"])
            path=(Path(a.clips)/(clip_id+".mp4")).resolve()
            if not path.exists():
                missing.append({"clip_id":clip_id,"label":label,"split":item["split"]});continue
            rows.append({"clip_id":clip_id,"path":str(path),"label":label,
                         "split":item["split"],"signer_id":str(item.get("signer_id",""))})
    out=Path(a.out);out.parent.mkdir(parents=True,exist_ok=True)
    with out.open("w",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=["clip_id","path","label","split","signer_id"])
        writer.writeheader();writer.writerows(rows)
    out.with_suffix(".missing.json").write_text(json.dumps(missing,indent=2))
    absent=allowed-{r["label"] for r in rows if r["split"]=="train"}
    print(json.dumps({"available_clips":len(rows),"missing_clips":len(missing),"classes_without_training":sorted(absent)},indent=2))
    if absent:raise SystemExit("Vocabulary coverage incomplete. Obtain missing data before training.")


if __name__ == "__main__":main()
