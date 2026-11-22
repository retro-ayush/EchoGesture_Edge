"""Run once during setup; inference never downloads assets."""
import hashlib
import json
from pathlib import Path
from urllib.request import urlopen

URLS = {
    "hand_landmarker.task": "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
    "pose_landmarker_lite.task": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task",
}


def main():
    root = Path(__file__).resolve().parents[1]/"models"; root.mkdir(exist_ok=True)
    records = []
    for name, url in URLS.items():
        target = root/name
        if not target.exists():
            print("Downloading setup asset:", name)
            with urlopen(url, timeout=90) as response:
                data = response.read()
            if len(data) < 100_000:
                raise ValueError("Unexpectedly small model download")
            temporary = target.with_suffix(".partial")
            temporary.write_bytes(data); temporary.replace(target)
        data = target.read_bytes()
        records.append({"name":name,"bytes":len(data),"sha256":hashlib.sha256(data).hexdigest(),"url":url,"precision":"FP16 asset"})
    (root/"extractor_assets.json").write_text(json.dumps(records,indent=2))
    print("Assets ready. These detector assets are FP16, not INT8.")


if __name__ == "__main__": main()
