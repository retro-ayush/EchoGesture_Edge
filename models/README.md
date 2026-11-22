# Model assets

| File | What it is |
|---|---|
| `echogesture.int8.onnx` | **Deployed classifier** (0.98 MB): 4-member temporal-CNN ensemble, trained on clean + synthetic low-light/occluded training clips, with in-graph mirror test-time augmentation, static INT8 QDQ. Used by `predict.py`, the evaluator and the demo. |
| `echogesture.int8.json` | Its metadata: label order (WLASL100), calibrated temperature (0.80), HUD threshold (0.50), ensemble members, split audit. Required next to the `.onnx`. |
| `echogesture.int8.fp32.onnx` / `.json` | FP32 reference of the same graph, for INT8-vs-FP32 comparison (`echogesture.validate`). |
| `rw192.json`, `rs1.json`, `rs2.json`, `rs3.json` | Training metadata of each ensemble member (config, seed, best validation accuracy). |
| `abl_no_masks.json`, `abl_no_motion.json` | Ablation runs (experiments only, not deployable). |
| `hand_landmarker.task`, `pose_landmarker_lite.task` | Public MediaPipe extractor assets (FP16), 13.6 MB; re-download with `scripts/download_models.py`. |

PyTorch checkpoints (`*.pt`) and training curves (`*.history.json`) are not committed; rebuild them with the recipe in the main README. The classifier is INT8; the MediaPipe extractors are FP16, so the full pipeline is mixed precision. Classifier bytes and total ML assets (14.6 MB) are reported separately in `outputs/results.csv`.
