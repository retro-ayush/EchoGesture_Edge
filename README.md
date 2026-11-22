# EchoGesture Edge

An offline, research-grounded ASL word-recognition prototype for **XR Hackathon 2026, Track 2**.

**Status:** trained, exported and evaluated on WLASL100 (22 November 2026). **Held-out test: 74.5% top-1, 90.0% top-5** (200 clips, 100 classes) with a **0.98 MB INT8** classifier and **21 ms p95 per-frame** processing (MediaPipe on the GPU). Measurements are in **Results** below and in `outputs/results.csv`.

The project includes local video feature extraction, a small temporal CNN, training, INT8 classifier conversion, `predict(video_path)`, manifest evaluation and a local camera caption HUD. The HUD uses **SPACE to start and finish one sign**. It is not automatic continuous sign translation. No frames are saved or sent to a server during camera inference.

## Results (WLASL100, held-out test split)

Measured 22 November 2026 on an Apple M3 Mac (Mac15,12, macOS). MediaPipe hand/pose tracking runs on the GPU (Metal delegate); the INT8 classifier runs in ONNX Runtime on 1 CPU thread. Full sheet: `outputs/results.csv`; per-clip predictions: `outputs/predictions.jsonl`.

| Metric | Target | Measured | Pass |
|---|---|---|---|
| Top-1 accuracy (200 test clips, 100 classes) | ≥ 70 % | **74.5 %** | ✅ |
| Top-5 / macro top-1 | report | 90.0 % / 75.3 % | — |
| Classifier size (INT8 ONNX) | < 20 MB | **0.98 MB** | ✅ |
| All ML assets (classifier + MediaPipe FP16 hand/pose) | report | 14.6 MB | — |
| Per-frame processing p95 (decode + GPU hand/pose tracking) | ≤ 50 ms streaming | **21.0 ms** (p50 10.3 ms) | ✅ |
| Classifier latency p95 (once per sign) | report | 3.4 ms | — |
| Full-clip `predict()` p95 (whole video, incl. tracker setup) | report | 2.0 s (p50 1.2 s) | — |
| Robustness: angle / low light / occlusion (synthetic, paired) | ≤ 5 pt drop | −1.0 / −12.0 / −16.0 pt | ❌ (worst case) |
| Thermal drift, 10 min sustained (back-to-back clips) | ≤ 20 % | **−2.7 %** (early p95 2.33 s → late 2.27 s, 428 runs) | ✅ |
| Average power at camera frame rate (CPU+GPU+ANE) | ≤ 500 mW | 941 mW (CPU 662 + GPU 279); 1,716 mW at max load | ❌ |

Validation split (242 clips, used for all model selection and calibration): **78.9 % top-1** INT8 (FP32 79.3 %). Confidence calibration: temperature 0.80; HUD threshold 0.50 gives ≥ 90 % precision on accepted captions.

**What changed to get here** (each change accepted on validation before the single test run):

| Change | Effect (validation unless noted) |
|---|---|
| MediaPipe GPU delegate (RGBA input), CPU fallback where unsupported | per-frame p95 57.6 → **21.0 ms** (test); accuracy unchanged (78.51 % both) |
| Train on clean + synthetic low-light and occluded copies of the *training* clips (3,051 rows; corruption seed differs from the test copies) | low light 58.7 → **68.2 %**, occlusion 52.5 → **64.9 %**, clean 78.5 → 78.9 % |
| Result on test | clean 74.0 → 74.5 %, low light 53.0 → 62.5 %, occlusion 53.5 → 58.5 % |

**Honest reading.** Accuracy, size, per-frame latency and thermal drift pass. Power fell from 2.2 W (CPU tracking) to 0.94 W at camera pace but is still above 0.5 W on a laptop; most remaining draw is CPU-side video decode, Python and per-clip tracker setup, which a native phone app with one persistent tracker would avoid. Whole-clip latency includes decoding and tracking every frame of a multi-second video plus ~0.5 s GPU tracker setup, so it is reported separately and never relabelled; in streaming use a caption arrives one classifier call (~3 ms) after the sign ends. Robustness improved by 5-9 points but still misses ≤ 5 pt for heavy synthetic darkness and a 25 % black box over the signing area, which hide the hands from MediaPipe; the test corruptions use the same generator as the training copies (different seed and clips), so real-world gains may be smaller. A tested exposure-correction front end made low light *worse* on validation (58.7 % → 49.6 %), so it ships disabled (`ECHOGESTURE_ENHANCE=1` to experiment). The first model (clean-only training, CPU tracking: 74.0 % test, 57.6 ms/frame, 2.2 W) is in git history.

**Ablations** (validation top-1, same seed and recipe, width 192): full 78.1 %, no presence masks 76.0 % (−2.1), no motion channels 77.3 % (−0.8). Single seed on 242 clips (1 clip ≈ 0.4 pt): small effects, near noise.

**Data actually used.** WLASL100 (official v0.3 splits, 100 glosses in official order in `data/vocab.json`). Of 2,038 annotated instances, 1,459 could be obtained (1,013 from the Kaggle `wlasl-processed` mirror + 446 recovered from still-live YouTube sources and trimmed to the official frame ranges); 579 sources are dead and listed in `data/videos.missing.json`. Splits: 1,017 train / 242 val / 200 test, with 98 of 100 classes present in test. WLASL splits are not signer-independent; the split audit is stored in the model JSON. No organizer-specific vocabulary or test manifest was available: if one is published, retrain on it.

## Quick setup (any machine)

One command creates `.venv`, installs pinned dependencies, checks the bundled model and runs a `predict()` smoke test:

```bash
bash setup.sh            # macOS / Linux   (add --train to also install PyTorch for retraining)
setup.bat             # Windows         (setup.bat --train)
source .venv/bin/activate        # Windows: .venv\Scripts\activate
python predict.py path/to/one_sign.mp4
```

The trained model (`models/echogesture.int8.onnx` + `.json`) and the MediaPipe assets are committed, so prediction needs no dataset and no download. If anything breaks, `python scripts/doctor.py` says exactly what is wrong and how to fix it.

**Supported platforms** (verified on 22 Nov 2026 with `scripts/check_install_matrix.py`, which dry-runs pip's resolver against each platform's wheel tags):

| | Python 3.10 | Python 3.11 (recommended) | Python 3.12 |
|---|---|---|---|
| macOS Apple Silicon (12+) | ✅ | ✅ exact pins, **clean install tested** | ✅ exact pins |
| macOS Intel (13) | ✅ | ✅ (torch 2.2.2 for training) | ✅ |
| Windows x64 | ✅ | ✅ exact pins | ✅ exact pins |
| Linux x86_64 (glibc ≥ 2.28) | ✅ | ✅ exact pins | ✅ exact pins |

- **Python 3.13+ is not supported**: mediapipe 0.10.21, numpy 1.26 and onnxruntime 1.19 publish no wheels for it. The setup scripts look for 3.11, 3.12 or 3.10 automatically; if none is installed they use [uv](https://docs.astral.sh/uv/) (when present) to download Python 3.11, otherwise they tell you what to install.
- **Linux ARM (e.g. Raspberry Pi) is not supported**: mediapipe has no aarch64 wheel. Use x86_64 or the Docker image under emulation.
- `requirements.txt` holds the runtime only (numpy, OpenCV, MediaPipe, ONNX Runtime: no PyTorch). `requirements-train.txt` adds PyTorch and ONNX. `constraints.txt` pins every transitive package for Python 3.11; on any platform where an exact pin has no wheel, the setup script retries with the direct pins only.
- **Docker** (Linux x86_64, identical everywhere): `docker build --platform linux/amd64 -t echogesture .` (Dockerfile provided; not tested on this machine, where Docker is not installed).

Manual install, if you prefer: `python3.11 -m venv .venv && .venv/bin/pip install -r requirements.txt -c constraints.txt && .venv/bin/python scripts/doctor.py`.

Tests: `python -m unittest discover -s tests` (18 tests; the 4 training tests are skipped without PyTorch).

## 1. Obtain the official inputs

You need the organizer's fixed vocabulary and evaluation manifest, permitted labeled training videos, and the target phone/runtime specification. Copy the vocabulary into `data/vocab.json` as an uppercase JSON array in the supplied order. Do not substitute an arbitrary set of easier words.

For WLASL, obtain annotations and video access through the [official repository](https://github.com/dxli94/WLASL). Read its data-use agreement. Use correctly trimmed isolated clips, with the original splits preserved. Source videos must be trimmed to the annotation boundaries: `scripts/trim_wlasl_sources.py` applies the official `frame_start`/`frame_end` to downloaded sources, and `scripts/fetch_wlasl_kaggle.py` fetches the already-processed clips (see **How this model was built**). Missing videos must be disclosed.

```bash
python scripts/prepare_wlasl.py /path/to/WLASL_v0.3.json /path/to/trimmed_clips \
  --vocab data/vocab.json --out data/videos.csv --clips-are-trimmed
```

Alternatively make a CSV with `clip_id,path,label,split,signer_id`. Paths are relative to that CSV. IDs must exactly match the organizer manifest, including leading zeros. `split` is `train`, `val` or `test`. `data/manifest.example.csv` shows the schema only; the listed files do not exist. Check signer overlap separately; retaining official splits does not by itself prove signer independence.

## 2. Extract and train

```bash
python -m echogesture.cache data/videos.csv --out data/features
python -m echogesture.train data/features/manifest.csv --vocab data/vocab.json \
  --out models/student.pt --epochs 80 --patience 20 --device cpu
```

Extraction is resumable: rerunning skips clips already cached (`--overwrite` re-extracts). Clips that fail to decode are excluded from `data/features/manifest.csv` and listed in `data/features/failures.json`; disclose them. Use `--strict` to stop on the first failure instead.

Set `--device cuda` for supported GPU training or `--device mps` on supported Macs. Training uses AdamW with a cosine learning-rate schedule (`--lr`, default 1e-3); `--patience N` stops after N epochs without validation improvement. Train exclusively on `train`; choose checkpoints on `val`; reserve `test` for final reporting. Duplicate paths fail the audit. Signer overlap is reported, not silently removed. Do not change official splits to improve scores.

Outputs: `models/student.pt` (best-val checkpoint), `models/student.json` (metadata) and `models/student.history.json` (per-epoch loss, val top-1/top-5).

### Ablations for the research question

Same data, same seed, separate output paths. Only the full model can be exported.

```bash
python -m echogesture.train data/features/manifest.csv --vocab data/vocab.json --out models/abl_no_masks.pt  --ablation no_masks
python -m echogesture.train data/features/manifest.csv --vocab data/vocab.json --out models/abl_no_motion.pt --ablation no_motion
```

`no_masks` zeroes the presence channels and computes motion without presence gating, so detection dropouts create false jumps, as in an unmasked baseline. `no_motion` zeroes the motion channels. Compare `best_val_accuracy` in each `.json` against `models/student.json`.

## 3. Export the classifier

```bash
python -m echogesture.export models/student.pt data/features/manifest.csv \
  --out models/echogesture.int8.onnx
```

Conversion uses representative **training** sequences for static QDQ calibration. It writes `models/echogesture.int8.onnx` and the FP32 reference `models/echogesture.int8.fp32.onnx`. Compare them on the same validation clips and calibrate the deployed model:

```bash
python -m echogesture.validate data/features/manifest.csv \
  --models models/echogesture.int8.onnx models/echogesture.int8.fp32.onnx \
  --calibrate models/echogesture.int8.onnx --target-precision 0.9
```

This reports top-1/top-5/macro accuracy, size and classifier-only p50/p95/p99 for each model in `outputs/validation.json`. It fits a softmax temperature on validation logits and stores the lowest HUD threshold whose accepted predictions reach the target precision in `models/echogesture.int8.json`; `predict()`, the evaluator and the demo then use them. The test split is refused. Quantization can reduce accuracy and is not guaranteed to improve latency on every device. The classifier's main Conv/Gemm weights and activations use INT8 QDQ; graph IO and some operators remain floating point.

**Precision scope:** MediaPipe's provided detector assets are FP16. Thus the application is mixed precision. The classifier conversion does not establish full-pipeline INT8 compliance. Confirm the organizer's interpretation. Report classifier size and total ML assets separately. Do not claim sub-20 MB for the complete application from the classifier alone.

## 4. Predict and evaluate

```bash
python predict.py /path/to/one_sign.mp4
python -m echogesture.evaluate data/videos.csv --split test \
  --device 'ACTUAL DEVICE AND CHIP' --out outputs
```

For direct evaluator use:

```python
from predict import predict
result = predict('/path/to/one_sign.mp4')
```

The direct function derives `clip_id` from the filename stem. When IDs differ from filenames, use the manifest evaluator: it preserves the exact `clip_id` explicitly. All vocabulary labels are uppercase. Errors stop evaluation rather than fabricate outputs. Do not rename an error or missing prediction as a successful run.

The evaluator writes `predictions.jsonl`, `results.csv` and detailed timing. It reports total clip processing in `latency_ms`, including decoding, feature extraction and classification; a first direct `predict()` call also includes classifier initialization. This number will generally exceed 50 ms for a whole clip. Model-only timing is reported separately. Agree the organizer's exact latency boundary before submission; never relabel model-only time as full-pipeline latency.

`latency_ms` includes MediaPipe setup for each clip (a fresh tracker per clip prevents tracking state leaking between clips). `results.csv` reports that setup separately as `extractor_init_p50`, plus top-5, macro accuracy and p50/p99.

Power, robustness and 10-minute thermal drift stay `NOT_MEASURED` until you supply them:

```bash
# Paired corruptions of the test split (same clip_ids and labels)
for k in low_light occlusion angle; do
  python scripts/make_corruptions.py data/videos.csv --split test --kind $k --out data/corrupted/$k
done
# 10-minute sustained run on the target device (full pipeline)
python scripts/sustained_benchmark.py data/videos.csv --mode full --split val --minutes 10 \
  --device 'ACTUAL DEVICE AND CHIP'
# Final evaluation with everything filled
python -m echogesture.evaluate data/videos.csv --split test --device 'ACTUAL DEVICE AND CHIP' \
  --corrupted-manifest data/corrupted/*/videos.csv \
  --thermal-json outputs/sustained_benchmark.json \
  --power-mw MEASURED --power-method 'TOOL, what is included'
```

The robustness row reports the worst-case drop, with every corruption listed in `notes`. The corruptions are synthetic proxies; record real low-light and off-angle clips if you can. Power must come from a real measurement. On a Mac, `sudo powermetrics --samplers cpu_power,gpu_power -i 1000` running alongside the sustained benchmark works; on Android, use Perfetto or the vendor power rail tools. Never infer it from battery percentage.

## 5. Camera demo

```bash
python demo.py --camera 0
```

Press SPACE, sign one word, then press SPACE again. C clears; Q exits. Inputs stay unmirrored, while the preview is mirrored. The confidence threshold comes from `echogesture.validate --calibrate` (0.6 if not calibrated); override with `--threshold`. On macOS, allow camera access for your terminal or editor in System Settings → Privacy & Security → Camera. Softmax confidence is not a guarantee of correctness. This demo processes a single signer and uses manually delimited clips. Face expressions, automatic sign boundaries, speech output and a native Android app are future work.

## Features and architecture

- 13 body landmarks provide 26 shoulder-centered, shoulder-scaled image coordinates.
- Two hands provide 126 wrist-relative 3D coordinates, each scaled by palm length. Separate wrist locations retain where signs occur relative to the body.
- 17 presence masks distinguish an undetected point from a valid zero coordinate.
- 156 masked coordinate differences encode motion. Missing observations never create false jumps.
- A sequence becomes a `329 x 64` tensor. Uniform nearest sampling preserves missing observations.
- A temporal CNN (width 128-192) uses depthwise separable residual blocks with dilation 1, 2, 4 and 8, concatenated mean+max temporal pooling, dropout and a vocabulary head.
- The deployed ONNX graph averages 4 such members, each applied to the input and its exact left/right mirror (a fixed channel permutation), so left-handed signing is handled in-graph.

This normalization handles translation and scale; it is not full viewpoint invariance. Independently estimated hand world origins are never combined as one physical coordinate system. Hand orientation is retained because it can distinguish signs. The model has no SPOTER weights or copied SPOTER architecture. It is an original compact baseline inspired by the research in the brief.

## How this model was built (exact recipe)

```bash
# Data (≈ 0.5 GB of the 5 GB archive, fetched by HTTP range requests; needs ~/.kaggle/kaggle.json)
python scripts/fetch_wlasl_kaggle.py ../WLASL/start_kit/WLASL_v0.3.json --vocab data/vocab.json --out ../wlasl_raw
python scripts/prepare_wlasl.py ../WLASL/start_kit/WLASL_v0.3.json ../wlasl_raw/videos \
  --vocab data/vocab.json --out data/videos.csv --clips-are-trimmed
# Optional: recover still-live YouTube sources (ids from videos.missing.json), then trim + merge
#   yt-dlp -f "bv*[height<=480]" -o "../wlasl_recover/yt/%(id)s.%(ext)s" <urls>
python scripts/trim_wlasl_sources.py ../WLASL/start_kit/WLASL_v0.3.json ../wlasl_recover/yt ../wlasl_recover/clips --vocab data/vocab.json
python scripts/merge_manifests.py data/videos.csv ../WLASL/start_kit/WLASL_v0.3.json ../wlasl_recover/clips --vocab data/vocab.json

# Landmarks (parallel, resumable): ~25 min for 1,459 clips on an M3
python -m echogesture.cache data/videos.csv --out data/features --workers 7

# Robustness data: synthetic low-light and occluded copies of the TRAINING clips (seed 1;
# test copies use seed 0), extracted with the GPU delegate, merged as extra train rows
for k in low_light occlusion; do
  python scripts/make_corruptions.py data/videos.csv --split train --kind $k --seed 1 --out data/corrupted_train/$k --workers 6
  python -m echogesture.cache data/corrupted_train/$k/videos.csv --out data/features_aug/$k --workers 4
done
python scripts/build_aug_manifest.py data/features/manifest.csv \
  low_light=data/features_aug/low_light/manifest.csv occlusion=data/features_aug/occlusion/manifest.csv \
  --out data/features_aug/manifest.csv

# Four members, strong augmentation (mirror/handedness, speed, 2D affine, 3D hand rotation); ~10 min on an M3
C="--aug strong --dropout 0.2 --pool meanmax --label-smoothing 0.1 --epochs 70 --patience 25 --device mps"
python -m echogesture.train data/features_aug/manifest.csv --vocab data/vocab.json --out models/rw192.pt $C --seed 4 --width 192
for s in 1 2 3; do python -m echogesture.train data/features_aug/manifest.csv --vocab data/vocab.json --out models/rs$s.pt $C --seed $s --width 128; done

# Export one INT8 graph = 4-member ensemble x {original, mirrored} views, then calibrate on val
python -m echogesture.export models/rw192.pt models/rs1.pt models/rs2.pt models/rs3.pt data/features/manifest.csv --out models/echogesture.int8.onnx
python -m echogesture.validate data/features/manifest.csv --models models/echogesture.int8.onnx models/echogesture.int8.fp32.onnx \
  --calibrate models/echogesture.int8.onnx --target-precision 0.9

# Final test evaluation (sequential, honest per-clip timing) + 10-minute thermal run
for k in low_light occlusion angle; do python scripts/make_corruptions.py data/videos.csv --split test --kind $k --out data/corrupted/$k; done
python -m echogesture.evaluate data/videos.csv --split test --device "YOUR DEVICE" --out outputs \
  --corrupted-manifest data/corrupted/low_light/videos.csv data/corrupted/occlusion/videos.csv data/corrupted/angle/videos.csv
python scripts/sustained_benchmark.py data/videos.csv --mode full --split val --minutes 10 --device "YOUR DEVICE"
python scripts/update_results.py --thermal-json outputs/sustained_benchmark.json
```

Use `--device cuda` or `cpu` instead of `mps` off Apple Silicon. MediaPipe uses the GPU when available (`ECHOGESTURE_DELEGATE=auto`, the default; `gpu` or `cpu` to force). All model selection used validation only; the test split was evaluated once per deployed model. Training curves: `models/*.history.json` (not committed); metadata of every member: `models/*.json`.

## Next steps

1. **Power**: a native app with one persistent tracker (no per-clip setup), hardware video decode, and tracking at 15 fps would cut most of the remaining 0.94 W. Re-measure with `sudo powermetrics --samplers cpu_power,gpu_power -i 1000 -n 300 > ~/power.txt` during `scripts/sustained_benchmark.py --realtime`, then `python scripts/update_results.py --powermetrics-log ~/power.txt`.
2. **Robustness**: replace the synthetic training corruptions with real low-light and partially occluded recordings.
3. **Other vocabularies**: rebuild `data/vocab.json`, retrain with the commands in **How this model was built**, and re-evaluate. WLASL100 results do not transfer to a different vocabulary.

## Training guide

```bash
source .venv/bin/activate
# 1. Extraction: the slow part (about 30-60 ms per frame on a laptop CPU). Resumable.
python -m echogesture.cache data/videos.csv --out data/features
cat data/features/failures.json            # disclose these
# 2. Training: members are tiny (138k params at width 128, 257k at width 192); CPU works, mps/cuda faster
python -m echogesture.train data/features/manifest.csv --vocab data/vocab.json \
  --out models/student.pt --epochs 80 --patience 20 --device mps
```

What to watch in the per-epoch JSON lines:
- `train_loss` should fall steadily. If it stays near `ln(vocab size)` (4.6 for 100 words), check that labels match `vocab.json` and that features are not all zero. Inspect one NPZ: hand masks `features[:, 169:171]` should be mostly 1.
- `val_accuracy` rising, then flattening, is normal. A large train/val gap means overfitting: try more epochs with `--patience`, or fewer classes only if the organizer allows it.
- If val accuracy is noisy and low, try `--lr 3e-3` or `--batch-size 16`. Keep the `--seed` fixed when comparing runs.
- The best checkpoint and its `.json` metadata are saved on every improvement, so stopping with Ctrl-C after a good epoch is safe; the result can be exported directly.

Then run section 3 (export → validate). Don't touch the test split until the final evaluation.

## Data and licensing

WLASL videos are distributed under the WLASL data-use agreement (research use). They are not included in this repository and must never be committed: `.gitignore` excludes `data/` except the vocabulary, the example manifest and the list of unavailable clip IDs. The bundled MediaPipe `.task` files are Google's public assets.
