"""Static INT8 QDQ conversion of the temporal classifier only."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from .data import read_manifest
from .features import encode_sequence, FEATURE_DIM, FRAMES
from .model import build_model, DeployedEnsemble


def main():
    from onnxruntime.quantization import CalibrationDataReader, QuantFormat, QuantType, quantize_static
    from onnxruntime.quantization.shape_inference import quant_pre_process
    p = argparse.ArgumentParser()
    p.add_argument("checkpoint", nargs="+", help="One or more checkpoints (same labels); logits are averaged")
    p.add_argument("manifest")
    p.add_argument("--out", default="models/echogesture.int8.onnx")
    p.add_argument("--no-mirror-tta", action="store_true", help="Disable in-graph mirrored-view averaging")
    a = p.parse_args()
    checkpoints = [torch.load(c, map_location="cpu", weights_only=True) for c in a.checkpoint]
    if any(c.get("ablation", "none") != "none" for c in checkpoints):
        raise ValueError("Ablation checkpoints are for experiments only; export the full model")
    if any(c["labels"] != checkpoints[0]["labels"] for c in checkpoints):
        raise ValueError("All checkpoints must share the same label order")
    model = DeployedEnsemble([build_model(c).eval() for c in checkpoints],
                             mirror_tta=not a.no_mirror_tta).eval()
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    fp32, prepared = out.with_suffix(".fp32.onnx"), out.with_suffix(".prepared.onnx")
    torch.onnx.export(model, torch.zeros(1, FEATURE_DIM, FRAMES), str(fp32),
                      input_names=["landmarks"], output_names=["logits"], opset_version=17)
    quant_pre_process(str(fp32), str(prepared))
    rows = [r for r in read_manifest(a.manifest) if r["split"] == "train"]
    if not rows:
        raise ValueError("Training-only calibration data required")
    # Deterministic spread across rows; no validation or test calibration.
    ids = np.unique(np.linspace(0, len(rows)-1, min(256, len(rows))).astype(int))
    class Reader(CalibrationDataReader):
        def __init__(self):
            self.items = iter(ids)
        def get_next(self):
            i = next(self.items, None)
            if i is None:
                return None
            with np.load(rows[i]["path"], allow_pickle=False) as f:
                x = encode_sequence(f["features"])[None]
            return {"landmarks":x}
    quantize_static(str(prepared), str(out), Reader(), quant_format=QuantFormat.QDQ,
                    activation_type=QuantType.QInt8, weight_type=QuantType.QInt8,
                    per_channel=True, op_types_to_quantize=["Conv", "Gemm", "MatMul"])
    meta = json.loads(Path(a.checkpoint[0]).with_suffix(".json").read_text())
    meta["ensemble"] = {"members": [Path(c).name for c in a.checkpoint],
                        "member_best_val_accuracy": [json.loads(Path(c).with_suffix(".json").read_text())["best_val_accuracy"] for c in a.checkpoint],
                        "mirror_tta": not a.no_mirror_tta}
    meta.update({"quantization":"INT8 QDQ classifier; floating-point IO and other operations",
                 "model_size_mb":out.stat().st_size/1_000_000,
                 "calibration_clips":len(ids), "extractors_quantized":False})
    for path in (out, fp32):
        path.with_suffix(".json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
