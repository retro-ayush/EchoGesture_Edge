"""Evaluator entry point; full video processing is timed honestly."""
import argparse
import json
import os
from pathlib import Path
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from echogesture.inference import Predictor  # noqa: E402

_predictor = None


def predict(video_path: str) -> dict:
    global _predictor
    start = perf_counter()
    if _predictor is None:
        _predictor = Predictor(os.environ.get("ECHOGESTURE_MODEL", str(ROOT/"models/echogesture.int8.onnx")))
    result = _predictor.video(video_path, os.environ.get("ECHOGESTURE_MODELS", str(ROOT/"models")))
    result["latency_ms"] = (perf_counter()-start)*1000
    return {k:result[k] for k in ("clip_id", "prediction", "top5", "confidence", "latency_ms")}


if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("video")
    args = p.parse_args(); print(json.dumps(predict(args.video)))
