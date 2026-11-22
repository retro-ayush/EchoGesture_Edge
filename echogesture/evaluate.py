"""One record per manifest ID; missing measurements remain NOT_MEASURED."""
import argparse
import csv
import json
from pathlib import Path
from .data import read_manifest
from .inference import Predictor
from .metrics import accuracy_report, latency_summary


def run_split(predictor, rows, models):
    details = []
    for n, row in enumerate(rows, 1):
        # Video manifest, never cached features, for end-to-end measurements.
        details.append(predictor.video(row["path"], models, row["clip_id"]))
        print(f"[{n}/{len(rows)}] {row['clip_id']} -> {details[-1]['prediction']}")
    return details


def main():
    p = argparse.ArgumentParser()
    p.add_argument("manifest"); p.add_argument("--split", default="test", choices=["train","val","test"])
    p.add_argument("--model", default="models/echogesture.int8.onnx")
    p.add_argument("--models", default="models"); p.add_argument("--out", default="outputs")
    p.add_argument("--device", required=True, help="Actual hardware name")
    p.add_argument("--corrupted-manifest", nargs="*", default=[],
                   help="Paired manifests from scripts/make_corruptions.py (same clip_ids/labels)")
    p.add_argument("--power-mw", type=float, help="Measured average power in mW")
    p.add_argument("--power-method", default="", help="Tool and scope, e.g. 'powermetrics, CPU+GPU, screen excluded'")
    p.add_argument("--thermal-json", help="Output of scripts/sustained_benchmark.py")
    a = p.parse_args()
    if a.power_mw is not None and not a.power_method:
        p.error("--power-method is required with --power-mw")
    # Validate every input before the slow video pass.
    thermal = json.loads(Path(a.thermal_json).read_text()) if a.thermal_json else None
    rows = [r for r in read_manifest(a.manifest) if r["split"] == a.split]
    if not rows: raise ValueError("No clips in selected split")
    for path in a.corrupted_manifest:
        read_manifest(path)
    predictor = Predictor(a.model)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    details = run_split(predictor, rows, a.models)
    predictions = [{k:d[k] for k in ("clip_id","prediction","top5","confidence","latency_ms")}
                   for d in details]
    with (out/"predictions.jsonl").open("w") as f:
        for record in predictions: f.write(json.dumps(record)+"\n")
    (out/"timing_details.json").write_text(json.dumps(details, indent=2))

    labeled = all(r.get("label") for r in rows)
    acc = accuracy_report([r["label"] for r in rows], [d["top5"] for d in details]) if labeled else None
    robustness = {}
    if a.corrupted_manifest and not labeled:
        raise ValueError("Robustness drop requires labeled clips")
    for path in a.corrupted_manifest:
        corrupted = {r["clip_id"]: r for r in read_manifest(path) if r["split"] == a.split}
        if set(corrupted) != {r["clip_id"] for r in rows}:
            raise ValueError(f"{path} is not paired with the clean {a.split} split")
        if any(corrupted[r["clip_id"]].get("label") != r["label"] for r in rows):
            raise ValueError(f"{path} changes labels; corruptions must be paired")
        c_rows = [corrupted[r["clip_id"]] for r in rows]
        c_details = run_split(predictor, c_rows, a.models)
        c_acc = accuracy_report([r["label"] for r in c_rows], [d["top5"] for d in c_details])
        robustness[path] = {"top1_percent": c_acc["top1_percent"],
                            "drop_pt": acc["top1_percent"] - c_acc["top1_percent"]}
    runtime = "ONNX Runtime CPU 1 thread, classifier INT8 QDQ; MediaPipe FP16 models"
    total = latency_summary([d["latency_ms"] for d in details])
    classifier = latency_summary([d["classifier_latency_ms"] for d in details])
    metrics = []
    def add(name, target, value, unit, op=None, limit=None, notes=""):
        passed = "NOT_MEASURED" if value is None else (
            "INFO" if op is None else str(bool(op(value, limit))).upper())
        metrics.append(dict(metric=name,target=target,measured="" if value is None else round(value, 4),
                            unit=unit,device=a.device,runtime=runtime,**{"pass":passed},notes=notes))
    ge, le, lt = (lambda v,l:v>=l), (lambda v,l:v<=l), (lambda v,l:v<l)
    add("top1_accuracy", ">=70", acc and acc["top1_percent"], "%", ge, 70,
        f"{len(rows)} clips, {acc['classes_present'] if acc else 0} classes present")
    add("top5_accuracy", "report", acc and acc["top5_percent"], "%")
    add("macro_top1_accuracy", "report", acc and acc["macro_top1_percent"], "%")
    add("latency_p95", "<=50", total["p95"], "ms", le, 50,
        "Full clip: decode + MediaPipe init/tracking + classifier; see timing_details.json")
    add("latency_p50", "report", total["p50"], "ms")
    add("latency_p99", "report", total["p99"], "ms")
    per_frame = latency_summary([(d["extraction_latency_ms"]-(d["extractor_init_ms"] or 0))/d["frames_processed"]
                                 for d in details])
    add("latency_per_frame_p95", "<=50 (streaming)", per_frame["p95"], "ms", le, 50,
        "Per-clip mean of decode + MediaPipe hand/pose tracking per frame (init excluded); "
        "streaming budget, NOT full-clip latency")
    add("classifier_latency_p95", "report separately", classifier["p95"], "ms", notes="Model invocation only")
    add("extractor_init_p50", "report separately",
        latency_summary([d["extractor_init_ms"] for d in details])["p50"], "ms",
        notes="MediaPipe setup per clip, included in latency_p95")
    add("model_size", "<20", Path(a.model).stat().st_size/1_000_000, "MB", lt, 20, "Classifier ONNX only")
    assets = [Path(a.model),Path(a.models)/"hand_landmarker.task",Path(a.models)/"pose_landmarker_lite.task"]
    add("total_ml_assets", "report separately", sum(x.stat().st_size for x in assets)/1_000_000, "MB",
        notes="Classifier + MediaPipe hand and pose (FP16)")
    worst = max(robustness.items(), key=lambda kv: kv[1]["drop_pt"]) if robustness else None
    add("accuracy_drop_robustness", "<=5", worst and worst[1]["drop_pt"], "pt", le, 5,
        "; ".join(f"{Path(k).parent.name}: {v['top1_percent']:.2f}% ({v['drop_pt']:+.2f} pt)"
                  for k, v in robustness.items()) + (" | worst case reported" if worst else ""))
    add("avg_power", "<=500", a.power_mw, "mW", le, 500, a.power_method)
    add("thermal_drift_10min", "<=20", thermal and thermal["drift_percent"], "%", le, 20,
        f"{thermal['mode']} mode, {thermal['minutes']} min, {thermal['runs']} runs" if thermal else "")
    with (out/"results.csv").open("w",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=list(metrics[0])); writer.writeheader(); writer.writerows(metrics)
    summary = {"clips": len(rows), "accuracy": acc, "latency_ms": total, "per_frame_latency_ms": per_frame,
               "classifier_latency_ms": classifier, "robustness": robustness}
    (out/"summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if k != "accuracy"} |
                     {"top1_percent": acc and acc["top1_percent"], "output": str(out)}, indent=2))


if __name__ == "__main__": main()
