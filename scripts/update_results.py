"""Fill thermal and/or power rows of an existing results.csv without re-running
the (slow) video evaluation.

  python scripts/update_results.py --thermal-json outputs/sustained_benchmark.json
  python scripts/update_results.py --power-mw 3200 --power-method "powermetrics CPU+GPU, ..."
  python scripts/update_results.py --powermetrics-log ~/power.txt --power-method "..."
"""
import argparse
import csv
import json
import re
from pathlib import Path


def mean_power_from_powermetrics(path):
    """Average of 'Combined Power (CPU + GPU + ANE): N mW' samples."""
    vals = [float(m) for m in re.findall(r"Combined Power \(CPU \+ GPU \+ ANE\):\s*([\d.]+)\s*mW",
                                         Path(path).read_text())]
    if not vals:
        raise ValueError("No 'Combined Power' samples found in the powermetrics log")
    return sum(vals)/len(vals), len(vals)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results", default="outputs/results.csv")
    p.add_argument("--thermal-json")
    p.add_argument("--power-mw", type=float)
    p.add_argument("--powermetrics-log")
    p.add_argument("--power-method", default="")
    a = p.parse_args()
    path = Path(a.results)
    rows = list(csv.DictReader(path.open(newline="")))
    by = {r["metric"]: r for r in rows}
    if a.thermal_json:
        t = json.loads(Path(a.thermal_json).read_text())
        r = by["thermal_drift_10min"]
        r.update(measured=round(t["drift_percent"], 4), **{"pass": str(t["drift_percent"] <= 20).upper()},
                 notes=f"{t['mode']} mode, {t['minutes']} min, {t['runs']} runs; late vs early p95")
    if a.powermetrics_log:
        mw, n = mean_power_from_powermetrics(a.powermetrics_log)
        a.power_mw = mw
        a.power_method = a.power_method or f"macOS powermetrics Combined CPU+GPU+ANE, {n} x 1 s samples"
    if a.power_mw is not None:
        if not a.power_method:
            p.error("--power-method is required")
        r = by["avg_power"]
        r.update(measured=round(a.power_mw, 1), **{"pass": str(a.power_mw <= 500).upper()}, notes=a.power_method)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print(path.read_text())


if __name__ == "__main__":
    main()
