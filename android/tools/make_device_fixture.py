"""Write app/src/androidTest/assets/device_fixture.json: landmark sequences of real
WLASL100 test-split clips plus the Python ONNX Runtime logits/prediction for each.
ClassifierDeviceTest runs the same sequences through the Android classifier.
Run from the repo root: PYTHONPATH=. .venv/bin/python android/tools/make_device_fixture.py
"""
import csv
import json
from pathlib import Path
import numpy as np
from echogesture.inference import Predictor

rows = [r for r in csv.DictReader(open("data/features/manifest.csv")) if r["split"] == "test"]
rows = [rows[i] for i in np.linspace(0, len(rows) - 1, 12).astype(int)]
p = Predictor("models/echogesture.int8.onnx")
clips = []
for r in rows:
    path = Path(r["path"])
    if not path.is_absolute():
        path = Path("data/features") / path
    with np.load(path, allow_pickle=False) as f:
        base = f["features"].astype(np.float32).round(6)  # exactly what the JSON stores
    logits, _, _ = p.logits(base)
    res = p.classify(base)
    clips.append({"clip_id": r["clip_id"], "label": r["label"], "base": base.tolist(),
                  "logits": logits.tolist(), "prediction": res["prediction"],
                  "confidence": res["confidence"], "top5": res["top5"]})
    print(r["clip_id"], r["label"], "->", res["prediction"], round(res["confidence"], 3))
out = Path("android/app/src/androidTest/assets/device_fixture.json")
out.write_text(json.dumps({"clips": clips}))
print(out, out.stat().st_size, "bytes")
