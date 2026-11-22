# Validation record

Originally prepared 21 November 2026; trained, evaluated and updated 22 November 2026.

## Deployed model (22 Nov 2026, second iteration)

- **Data:** WLASL100, official v0.3 splits. 1,459 of 2,038 instances obtainable (1,013 from the Kaggle `wlasl-processed` mirror, 446 recovered from live YouTube sources and trimmed to official frame ranges). 1,017 train / 242 val / 200 test; 0 extraction failures. Missing IDs: `data/videos.missing.json`.
- **Training data for the deployed model:** the 1,017 clean training clips plus synthetic low-light and occluded copies of the same training clips (corruption seed 1; the test copies use seed 0), 3,051 training rows. Validation and test stay clean.
- **Model:** 4-member ensemble (one width-192, three width-128 temporal CNNs; strong augmentation, dropout 0.2, label smoothing 0.1, mean+max pooling), averaged with in-graph mirror test-time augmentation; static INT8 QDQ, 0.98 MB. Member val top-1: 80.2, 76.0, 78.5, 75.6 %.
- **MediaPipe on the GPU** (Metal delegate; RGBA input; automatic CPU fallback). The previous CPU-trained model scored 78.51 % on validation with both CPU- and GPU-extracted landmarks, so GPU tracking does not change accuracy.
- **Selection used validation only.** Deployed vs previous model on val: clean 78.9 vs 78.5 %, synthetic low light 68.2 vs 58.7 %, occlusion 64.9 vs 52.5 %. Calibrated temperature 0.80, HUD threshold 0.50 (≥ 90 % precision on val).
- **Test split** (`outputs/results.csv`): top-1 74.5 %, top-5 90.0 %, macro 75.3 %. Robustness: angle 73.5 % (−1.0 pt), low light 62.5 % (−12.0), occlusion 58.5 % (−16.0).
- **Latency (Apple M3, GPU tracking):** per-frame decode + tracking p50 10.3 / p95 21.0 ms; classifier p95 3.4 ms; full clip p50 1.2 / p95 2.0 s; tracker setup ~0.46 s per clip.
- **Thermal:** 10-minute back-to-back run, drift −2.7 % (early p95 2.33 s, late 2.27 s, 428 runs; Mac kept awake with `caffeinate`).
- **Power** (macOS `powermetrics`, CPU+GPU+ANE): 941 mW average over 300 × 1 s samples while processing clips at their native frame rate (`--realtime`, busy 58 % including per-clip tracker setup); 1,716 mW with clips back to back. Screen and camera excluded.
- **Ablations (val, first data recipe, width 192, seed 4):** full 78.1 %, no masks 76.0 %, no motion 77.3 %.
- **Negative result:** an exposure-correction front end lowered low-light **validation** accuracy (58.7 → 49.6 % with LUT + median, 54.5 % LUT only). Disabled by default.

## First model (superseded, in git history)

Clean-only training, CPU tracking: test top-1 74.0 %, per-frame p95 57.6 ms, low light −21.0 pt, occlusion −20.5 pt, thermal drift 8.3 %, power 2,209 mW (149 samples, back-to-back load).

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

Accuracy (≥ 70 %), size (< 20 MB), per-frame latency (≤ 50 ms) and thermal drift (≤ 20 %) targets are met on WLASL100's held-out test split on an Apple M3. Power (0.94 W at camera pace) and robustness to heavy synthetic darkness and central occlusion (−12 / −16 pt) still miss their targets. Training and test corruptions come from the same generator (different clips and seeds), so real-world robustness gains may be smaller. WLASL splits are not signer-independent. Results apply to the WLASL100 vocabulary only.
