#!/usr/bin/env bash
# Build the EchoGesture Android APK (universal arm64 + armv7, Android 7.0+).
# Output: dist/EchoGesture-Edge.apk at the repo root.
# Needs: JDK 17+ and the Android SDK (platform 35, build-tools 35). See android/README.md.
set -euo pipefail
cd "$(dirname "$0")"

if [[ -z "${JAVA_HOME:-}" ]] && [[ -x /usr/libexec/java_home ]]; then
  JAVA_HOME=$(/usr/libexec/java_home -v 17 2>/dev/null || /usr/libexec/java_home 2>/dev/null || true)
  export JAVA_HOME
fi
SDK="${ANDROID_HOME:-${ANDROID_SDK_ROOT:-$HOME/Library/Android/sdk}}"
[[ -d "$SDK" ]] || SDK="$HOME/Android/Sdk"
if [[ ! -d "$SDK/platforms" ]]; then
  echo "Android SDK not found. Install Android Studio, or set ANDROID_HOME to an SDK with platforms;android-35." >&2
  exit 1
fi
[[ -f local.properties ]] || echo "sdk.dir=$SDK" > local.properties

# Build outside the source tree (exFAT/FAT drives get macOS "._*" files that break packaging).
BUILD_ROOT="${BUILD_ROOT:-$HOME/.cache/echogesture-android}"
find . -name '._*' -not -path './.gradle/*' -delete 2>/dev/null || true

./gradlew --console=plain testDebugUnitTest assembleRelease -PbuildRoot="$BUILD_ROOT"
mkdir -p ../dist
cp "$BUILD_ROOT/app/outputs/apk/release/app-release.apk" ../dist/EchoGesture-Edge.apk
echo
echo "APK: $(cd ../dist && pwd)/EchoGesture-Edge.apk ($(du -h ../dist/EchoGesture-Edge.apk | cut -f1))"
echo "Install on a USB-connected phone:  adb install -r ../dist/EchoGesture-Edge.apk"
