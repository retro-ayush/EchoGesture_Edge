"""Environment self-check: python version, packages, model assets, and a
smoke test of the full predict() path on a generated video (no dataset needed).

  python scripts/doctor.py            # runtime checks + predict() smoke test
  python scripts/doctor.py --train    # also check training packages
Exit code 0 means the installation can run predict(), evaluation and the demo.
"""
import argparse
import importlib
import json
import platform
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
PINS = {"numpy": "1.26.4", "cv2": "4.10.0", "mediapipe": "0.10.21", "onnxruntime": "1.19.2"}
TRAIN = {"torch": None, "onnx": "1.16.2"}
ASSETS = ["echogesture.int8.onnx", "echogesture.int8.json", "hand_landmarker.task", "pose_landmarker_lite.task"]
problems = []


def check(ok, label, fix=""):
    print(("  ok   " if ok else "  FAIL ") + label + ("" if ok else f"\n         fix: {fix}"))
    if not ok:
        problems.append(label)
    return ok


def main():
    p = argparse.ArgumentParser(); p.add_argument("--train", action="store_true")
    p.add_argument("--no-smoke", action="store_true"); a = p.parse_args()
    print(f"EchoGesture doctor - Python {platform.python_version()} on {platform.system()} {platform.machine()}")
    v = sys.version_info[:2]
    check((3, 10) <= v <= (3, 12), f"Python {v[0]}.{v[1]} in 3.10-3.12",
          "run bash setup.sh (macOS/Linux) or setup.bat (Windows); they provision Python 3.11")
    for mod, want in {**PINS, **(TRAIN if a.train else {})}.items():
        try:
            m = importlib.import_module(mod)
            have = getattr(m, "__version__", "?")
            check(want is None or have.startswith(want), f"{mod} {have}" + (f" (expected {want})" if want else ""),
                  "reinstall with bash setup.sh or setup.bat")
        except Exception as e:  # import errors are the most common install problem
            check(False, f"import {mod}: {e}", "reinstall with bash setup.sh or setup.bat" +
                  (" --train" if mod in TRAIN else ""))
    for name in ASSETS:
        f = ROOT/"models"/name
        check(f.is_file() and f.stat().st_size > 0, f"models/{name}",
              "python scripts/download_models.py" if name.endswith(".task") else
              "git pull the repository (the trained model is committed) or retrain")
    if not a.no_smoke and not problems:
        import cv2
        import numpy as np
        from predict import predict
        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp)/"doctor_smoke.mp4")
            w = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), 25, (320, 240))
            for i in range(30):
                frame = np.full((240, 320, 3), 40, np.uint8)
                cv2.circle(frame, (100 + 4*i, 120), 20, (200, 180, 160), -1)
                w.write(frame)
            w.release()
            try:
                out = predict(path)
                check(set(out) >= {"clip_id", "prediction", "top5", "confidence", "latency_ms"} and
                      out["prediction"].isupper(), f"predict() smoke test -> {json.dumps(out)[:120]}")
            except Exception as e:
                check(False, f"predict() smoke test: {e}", "see the error above; run with the venv's python")
    print("\nAll checks passed." if not problems else f"\n{len(problems)} problem(s) found.")
    sys.exit(1 if problems else 0)


if __name__ == "__main__":
    main()
