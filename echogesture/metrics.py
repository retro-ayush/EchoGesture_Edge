"""Accuracy, latency and calibration summaries shared by evaluation tools."""
import numpy as np


def accuracy_report(truth, ranked):
    """truth: true labels; ranked: per-clip label lists, best first."""
    if not truth or len(truth) != len(ranked):
        raise ValueError("Need one ranked prediction list per labeled clip")
    hit1 = np.array([t == r[0] for t, r in zip(truth, ranked)])
    hit5 = np.array([t in r[:5] for t, r in zip(truth, ranked)])
    truth_arr = np.array(truth)
    per_class = {c: float(hit1[truth_arr == c].mean()) for c in sorted(set(truth))}
    return {"clips": len(truth), "classes_present": len(per_class),
            "top1_percent": 100*float(hit1.mean()),
            "top5_percent": 100*float(hit5.mean()),
            "macro_top1_percent": 100*float(np.mean(list(per_class.values()))),
            "per_class_top1": per_class}


def latency_summary(values_ms):
    v = np.asarray(values_ms, float)
    if v.size == 0:
        raise ValueError("No latency samples")
    return {"n": int(v.size), "mean": float(v.mean()),
            **{f"p{q}": float(np.percentile(v, q)) for q in (50, 95, 99)}}


def softmax(logits, temperature=1.0):
    z = np.asarray(logits, np.float64)/temperature
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e/e.sum(axis=-1, keepdims=True)


def nll(logits, targets, temperature=1.0):
    p = softmax(logits, temperature)
    return float(-np.log(np.clip(p[np.arange(len(p)), targets], 1e-12, 1)).mean())


def fit_temperature(logits, targets):
    """Grid search for the NLL-minimizing softmax temperature (no SciPy)."""
    logits, targets = np.asarray(logits, np.float64), np.asarray(targets, int)
    grid = np.exp(np.linspace(np.log(.05), np.log(20), 400))
    losses = [nll(logits, targets, t) for t in grid]
    best = int(np.argmin(losses))
    return {"temperature": float(grid[best]), "nll_before": nll(logits, targets),
            "nll_after": float(losses[best])}


def threshold_table(confidences, correct, thresholds=None):
    """Coverage and accepted-prediction accuracy for each HUD threshold."""
    c, ok = np.asarray(confidences, float), np.asarray(correct, bool)
    if thresholds is None:
        thresholds = np.round(np.arange(.3, .951, .05), 2)
    rows = []
    for t in thresholds:
        keep = c >= t
        rows.append({"threshold": float(t), "coverage": float(keep.mean()),
                     "accepted_accuracy": float(ok[keep].mean()) if keep.any() else None})
    return rows


def suggest_threshold(table, target_precision):
    """Lowest threshold whose accepted predictions reach the target accuracy."""
    for row in table:
        if row["accepted_accuracy"] is not None and row["accepted_accuracy"] >= target_precision:
            return row["threshold"]
    return None


def drift_percent(timestamps_s, latencies_ms, window_fraction=.2):
    """Percent increase of late versus early p95 latency over equal windows."""
    t, v = np.asarray(timestamps_s, float), np.asarray(latencies_ms, float)
    if len(t) < 10:
        raise ValueError("Too few samples for a drift estimate")
    span = t[-1] - t[0]
    early = v[t <= t[0] + span*window_fraction]
    late = v[t >= t[-1] - span*window_fraction]
    e95, l95 = float(np.percentile(early, 95)), float(np.percentile(late, 95))
    return {"early_p95_ms": e95, "late_p95_ms": l95, "early_n": int(early.size),
            "late_n": int(late.size), "drift_percent": 100*(l95 - e95)/e95}
