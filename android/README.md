# EchoGesture Edge — Android app

Native Android port of the camera caption HUD (`demo.py`). Everything runs on the phone:
MediaPipe hand + pose landmarks → the same 64-frame features as `echogesture/features.py`
→ the same INT8 ONNX classifier (`models/echogesture.int8.onnx`). No video is stored or sent.

## Install on a phone

1. Get `dist/EchoGesture-Edge.apk` (build it below, or download it from the GitHub release).
2. Copy it to the phone (USB, Drive, WhatsApp/Telegram to yourself…) and open it.
   Allow **Install unknown apps** for the app you opened it from when Android asks.
3. Open **EchoGesture**, allow the camera.

Or over USB with developer mode on: `adb install -r dist/EchoGesture-Edge.apk`.

**Works on:** any ARM Android phone, Android 7.0 (API 24) or newer, 64-bit and 32-bit,
including Android 15+ devices with 16 KB memory pages. Not x86 emulators (MediaPipe ships
no x86 libraries); use an arm64 emulator image on Apple Silicon.

## Use

Tap **Start sign**, sign one word from the 100-word WLASL vocabulary (`data/vocab.json`), tap
**Finish**. The caption shows when confidence ≥ 0.50 (calibrated threshold from the model JSON),
otherwise "Uncertain – please repeat"; the top-5 alternatives are listed below. The top bar shows
hands/body detection, the MediaPipe delegate (GPU, or CPU when the GPU fails), fps and per-frame time.
Use the button at the top right to switch between front and back camera. Sign with your upper body
and both hands in frame, in good light.

## Build

Needs JDK 17+ and the Android SDK (platform 35, build-tools 35; Android Studio installs both).

```bash
cd android
./build_apk.sh          # runs the Python-parity unit tests, writes ../dist/EchoGesture-Edge.apk
```

The model files are copied from `../models` at build time, so a retrained model only needs a rebuild.
The APK is signed with the committed demo key `app/echogesture-demo.jks` (password `echogesture`)
so every build installs over the previous one. It is for sideloading/demos only: use a private key
for a Play Store release.

## Tests

- `./gradlew testDebugUnitTest` — `FeaturesParityTest`: Kotlin features vs Python
  (`tools/make_parity_fixture.py`), 120 frames + 5 sequences, tolerance 1e-4.
- `./gradlew connectedDebugAndroidTest` (device/emulator) — `ClassifierDeviceTest`: 12 WLASL100
  test clips through the on-device classifier must give Python's prediction, top-5 and confidence
  (±0.01); and the MediaPipe landmarkers must load and run. The clip fixture is derived from WLASL,
  so it is not committed: create it with `PYTHONPATH=. .venv/bin/python android/tools/make_device_fixture.py`.

If the repo is on an exFAT/FAT drive, pass `-PbuildRoot=$HOME/.cache/echogesture-android`
(the build script does this) because macOS `._*` files there break resource packaging.
