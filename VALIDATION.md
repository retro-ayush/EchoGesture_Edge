# Validation record

Originally prepared 21 November 2026; trained, evaluated and updated 22 November 2026.

## Trained model (22 Nov 2026)

- **Data:** WLASL100, official v0.3 splits. 1,459 of 2,038 instances obtainable (1,013 from the Kaggle `wlasl-processed` mirror, 446 recovered from live YouTube sources and trimmed to official frame ranges). 1,017 train / 242 val / 200 test. Landmark extraction: 1,459 cached, 0 failures. Missing IDs: `data/videos.missing.json`.
- **Model:** 4-member ensemble (one width-192, three width-128 temporal CNNs; strong augmentation, dropout 0.2, label smoothing 0.1, mean+max pooling), averaged with in-graph mirror test-time augmentation; static INT8 QDQ, 0.98 MB.
- **Selection used validation only.** Candidate INT8 exports on val: single w192 77.7 %, 3-member 78.1 %, 4-member 78.5 % (FP32 78.5 %). Calibrated temperature 0.79, HUD threshold 0.55 (≥ 90 % precision on val).
- **Test split, evaluated once** (`outputs/results.csv`): top-1 74.0 %, top-5 89.0 %, macro 73.6 %. Robustness (paired synthetic copies): angle 73.0 % (−1.0 pt), low light 53.0 % (−21.0), occlusion 53.5 % (−20.5).
- **Latency on Apple M3 CPU:** classifier p95 2.8 ms; per-frame decode + tracking p50 48.6 / p95 57.6 ms; full clip p50 3.2 s / p95 5.3 s; MediaPipe setup 107 ms per clip.
- **Ablations (val, width 192, seed 4):** full 78.1 %, no masks 76.0 %, no motion 77.3 %.
- **Negative result:** an exposure-correction front end (gamma LUT on frames whose 99th-percentile luma < 100, ± median filter) lowered low-light **validation** accuracy from 58.7 % to 49.6 % (LUT + median) and 54.5 % (LUT only). It is disabled by default.
- **Thermal:** 10-minute full-pipeline run (Mac kept awake with `caffeinate`): drift 8.3 % (early p95 4.95 s, late p95 5.36 s). A first attempt was discarded because the Mac idle-slept during it.

## Reproducibility checks (22 Nov 2026)

- 18 unit tests pass (`python -m unittest discover -s tests`); without PyTorch the 4 training tests are skipped and the rest pass.
- **Clean install:** a copy of the repository without `.venv`, data or outputs ran `bash setup.sh` end to end on macOS arm64 (system bash 3.2): fresh venv, pinned install, all doctor checks and the `predict()` smoke test passed.
- **Install matrix:** `scripts/check_install_matrix.py` dry-runs pip's resolver with each platform's wheel tags for Python 3.10, 3.11 and 3.12 on macOS Apple Silicon (12 and 14), macOS Intel, Linux x86_64 and Windows x64, for runtime and training sets: all resolve (exact pins on 3.11/3.12 except Intel Mac, which uses the direct-pin fallback). Python 3.13 and Linux aarch64 have no mediapipe wheels and are unsupported.
- Not run here: actual installs on Windows/Linux/Intel hardware, and the Dockerfile (no Docker on this machine).

- **Power:** macOS `powermetrics` (CPU+GPU+ANE combined), 149 × 1 s samples from 17:46:02 to 17:48:30, inside the full-pipeline sustained run (17:45:50-17:55:50): mean 2,209 mW (min 1,900, max 3,524). The sampler stopped after 2.5 of 10 minutes; readings were steady. Screen, camera and the rest of the system are excluded.

## Not completed

- The camera HUD (`demo.py`) needs an interactive camera and display; record it for the demo.
- On-phone (Android/iOS) packaging and measurement.

## Honest interpretation

Accuracy (≥ 70 %) and size (< 20 MB) targets are met on WLASL100's held-out test split. The 50 ms latency target is met by the classifier but not by MediaPipe tracking on a laptop CPU, and whole-clip latency is seconds. Synthetic low light and central occlusion cause large drops. WLASL splits are not signer-independent. These results apply to the WLASL100 vocabulary only; an organizer vocabulary would need retraining.
