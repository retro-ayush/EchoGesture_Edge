"""Sustained-load latency test for the thermal-drift metric.

Repeats the same workload for --minutes and compares p95 latency in the first
and last 20% of the run (equal-length windows). Run on the actual device with
the screen/charging state recorded; pass the JSON to evaluate --thermal-json.

  full:       predict-style video processing (decode + MediaPipe + classifier)
  classifier: cached features through the ONNX classifier only

--realtime (full mode) paces clips at their native frame rate, like a live
camera: after processing a clip the run idles until the clip's real duration
has elapsed. Use it to measure conversational power draw; without it the run
is throughput-bound (back-to-back clips), which measures maximum load.

Examples:
  python scripts/sustained_benchmark.py data/videos.csv --mode full --minutes 10
  python scripts/sustained_benchmark.py data/features/manifest.csv --mode classifier
"""
import argparse
import json
import platform
from pathlib import Path
import sys
from time import perf_counter, time
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from echogesture.data import read_manifest  # noqa: E402
from echogesture.inference import Predictor  # noqa: E402
from echogesture.metrics import drift_percent, latency_summary  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("manifest", help="Video manifest for full mode, features manifest for classifier mode")
    p.add_argument("--mode", default="full", choices=["full", "classifier"])
    p.add_argument("--split", default="val", choices=["train", "val", "test"])
    p.add_argument("--clips", type=int, default=20, help="Distinct clips cycled during the run")
    p.add_argument("--minutes", type=float, default=10.0)
    p.add_argument("--model", default="models/echogesture.int8.onnx")
    p.add_argument("--models", default="models")
    p.add_argument("--device", required=True, help="Actual hardware name")
    p.add_argument("--out", default="outputs/sustained_benchmark.json")
    p.add_argument("--realtime", action="store_true", help="Pace clips at native frame rate (full mode)")
    a = p.parse_args()
    rows = [r for r in read_manifest(a.manifest) if r["split"] == a.split][:a.clips]
    if not rows:
        raise ValueError("No clips in selected split")
    predictor = Predictor(a.model)
    if a.mode == "classifier":
        cached = []
        for r in rows:
            with np.load(r["path"], allow_pickle=False) as f:
                cached.append(f["features"])
        work = lambda i: predictor.classify(cached[i % len(cached)])  # noqa: E731
    else:
        work = lambda i: predictor.video(rows[i % len(rows)]["path"], a.models)  # noqa: E731
    durations = []
    if a.realtime:
        if a.mode != "full":
            p.error("--realtime applies to --mode full")
        import cv2
        for r in rows:
            cap = cv2.VideoCapture(r["path"])
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            durations.append(cap.get(cv2.CAP_PROP_FRAME_COUNT) / fps)
            cap.release()
    work(0)  # warm-up, excluded
    stamps, latencies, start, i = [], [], perf_counter(), 0
    deadline = start + a.minutes*60
    while perf_counter() < deadline:
        t = perf_counter(); work(i)
        latencies.append((perf_counter()-t)*1000); stamps.append(t-start)
        if a.realtime:  # idle until the clip's real duration has passed
            from time import sleep
            sleep(max(0.0, durations[i % len(durations)] - (perf_counter()-t)))
        i += 1
        if i % 50 == 0:
            print(f"{stamps[-1]/60:5.1f} min  {i} runs  last {latencies[-1]:.1f} ms")
    busy = sum(latencies)/1000/max(1e-9, perf_counter()-start)
    report = {"device": a.device, "platform": platform.platform(), "mode": a.mode,
              "realtime": a.realtime, "busy_fraction": round(busy, 4),
              "minutes": a.minutes, "runs": len(latencies), "started_unix": time() - (perf_counter()-start),
              "latency_ms": latency_summary(latencies), **drift_percent(stamps, latencies),
              "definition": "100*(late p95 - early p95)/early p95, first vs last 20% of run",
              "samples": [[round(s, 3), round(v, 3)] for s, v in zip(stamps, latencies)]}
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(f"Drift {report['drift_percent']:.1f}%  (early p95 {report['early_p95_ms']:.1f} ms, "
          f"late p95 {report['late_p95_ms']:.1f} ms)  -> {out}")


if __name__ == "__main__":
    main()
