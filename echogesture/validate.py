"""Compare exported classifiers on cached validation features; calibrate one.

Runs on the cached-features manifest (no video decoding), so latency here is
classifier-only. The test split is refused: it stays untouched until the
final `echogesture.evaluate` run.

Calibration fits a softmax temperature on validation logits and suggests the
lowest HUD threshold whose accepted predictions reach --target-precision.
Both are written into the calibrated model's JSON metadata.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from .data import read_manifest
from .inference import Predictor
from .metrics import (accuracy_report, fit_temperature, latency_summary, softmax,
                      suggest_threshold, threshold_table)


def run_model(path, rows):
    predictor = Predictor(path)
    labels = predictor.labels
    index = {label: i for i, label in enumerate(labels)}
    unknown = {r["label"] for r in rows} - set(labels)
    if unknown:
        raise ValueError(f"Labels missing from model vocabulary: {sorted(unknown)}")
    logits, times = [], []
    for row in rows:
        with np.load(row["path"], allow_pickle=False) as f:
            z, _, ms = predictor.logits(f["features"])
        logits.append(z); times.append(ms)
    logits = np.stack(logits)
    targets = np.array([index[r["label"]] for r in rows])
    ranked = [[labels[i] for i in np.argsort(-z)[:5]] for z in logits]
    report = accuracy_report([r["label"] for r in rows], ranked)
    report.pop("per_class_top1")
    return {"model": str(path), "size_mb": Path(path).stat().st_size/1_000_000,
            **report, "classifier_latency_ms": latency_summary(times[1:] or times),
            "note": "First call excluded from latency (warm-up)."}, logits, targets


def main():
    p = argparse.ArgumentParser()
    p.add_argument("manifest", help="Cached features manifest (data/features/manifest.csv)")
    p.add_argument("--models", nargs="+",
                   default=["models/echogesture.int8.onnx", "models/echogesture.int8.fp32.onnx"])
    p.add_argument("--split", default="val", choices=["train", "val"])
    p.add_argument("--calibrate", help="One of --models: write temperature/threshold into its JSON")
    p.add_argument("--target-precision", type=float, default=.9)
    p.add_argument("--out", default="outputs/validation.json")
    a = p.parse_args()
    rows = [r for r in read_manifest(a.manifest) if r["split"] == a.split]
    if not rows:
        raise ValueError(f"No {a.split} clips in manifest")
    if any(not r.get("label") for r in rows):
        raise ValueError("Validation requires labels")
    results, calibration = [], None
    for path in a.models:
        summary, logits, targets = run_model(path, rows)
        results.append(summary)
        if a.calibrate and Path(path).resolve() == Path(a.calibrate).resolve():
            fit = fit_temperature(logits, targets)
            probs = softmax(logits, fit["temperature"])
            table = threshold_table(probs.max(1), probs.argmax(1) == targets)
            threshold = suggest_threshold(table, a.target_precision)
            meta_path = Path(path).with_suffix(".json")
            meta = json.loads(meta_path.read_text())
            meta["temperature"] = fit["temperature"]
            meta["confidence_threshold"] = threshold if threshold is not None else .6
            meta["calibration"] = {"split": a.split, "clips": len(rows), **fit,
                                   "target_precision": a.target_precision,
                                   "threshold_found": threshold is not None,
                                   "threshold_table": table}
            meta_path.write_text(json.dumps(meta, indent=2))
            calibration = meta["calibration"] | {"confidence_threshold": meta["confidence_threshold"],
                                                 "written_to": str(meta_path)}
    if a.calibrate and calibration is None:
        raise ValueError("--calibrate must name one of the --models paths")
    report = {"split": a.split, "models": results, "calibration": calibration}
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    for r in results:
        print(f"{r['model']}: top1 {r['top1_percent']:.2f}%  top5 {r['top5_percent']:.2f}%  "
              f"macro {r['macro_top1_percent']:.2f}%  size {r['size_mb']:.3f} MB  "
              f"p95 {r['classifier_latency_ms']['p95']:.2f} ms")
    if calibration:
        print(f"Calibrated temperature {calibration['temperature']:.3f}; "
              f"threshold {calibration['confidence_threshold']} "
              f"({'found' if calibration['threshold_found'] else 'target not reached, kept 0.6'})")
    print("Report:", out)


if __name__ == "__main__":
    main()
