#!/usr/bin/env bash
# One-command setup for macOS and Linux.   bash setup.sh [--train]
# Creates .venv with Python 3.11 (3.10/3.12 accepted), installs pinned
# dependencies, checks model assets, and runs scripts/doctor.py.
set -euo pipefail
cd "$(dirname "$0")"
TRAIN=0; [[ "${1:-}" == "--train" ]] && TRAIN=1
REQ=requirements.txt; [[ $TRAIN == 1 ]] && REQ=requirements-train.txt

find_python() {
  for c in python3.11 python3.12 python3.10 python3 python; do
    if command -v "$c" >/dev/null 2>&1; then
      v=$("$c" -c 'import sys;print("%d.%d"%sys.version_info[:2])' 2>/dev/null || true)
      case "$v" in 3.10|3.11|3.12) echo "$c"; return 0;; esac
    fi
  done
  return 1
}

if [[ ! -x .venv/bin/python ]] || ! .venv/bin/python -c 'import sys;assert (3,10)<=sys.version_info[:2]<=(3,12)' 2>/dev/null; then
  rm -rf .venv
  if PY=$(find_python); then
    echo "Using $PY ($($PY --version))"
    "$PY" -m venv .venv
  elif command -v uv >/dev/null 2>&1; then
    echo "No Python 3.10-3.12 found; uv will download Python 3.11"
    uv venv --python 3.11 --seed .venv
  else
    echo "No Python 3.10-3.12 found (Python 3.13+ is not supported by mediapipe 0.10.21)."
    echo "Install one of:  Python 3.11 from https://www.python.org/downloads/"
    echo "            or:  uv (https://docs.astral.sh/uv/) then re-run bash setup.sh"
    exit 1
  fi
fi

PIP=(.venv/bin/python -m pip)
"${PIP[@]}" install --quiet --upgrade pip
echo "Installing $REQ (pinned with constraints.txt)..."
if ! "${PIP[@]}" install -r "$REQ" -c constraints.txt; then
  echo "Exact transitive pins unavailable on this platform; retrying with direct pins only..."
  "${PIP[@]}" install -r "$REQ"
fi

if [[ ! -f models/hand_landmarker.task || ! -f models/pose_landmarker_lite.task ]]; then
  .venv/bin/python scripts/download_models.py
fi

if [[ $TRAIN == 1 ]]; then .venv/bin/python scripts/doctor.py --train; else .venv/bin/python scripts/doctor.py; fi
echo
echo "Done. Activate with:  source .venv/bin/activate"
echo "Then:  python predict.py path/to/clip.mp4    or    python demo.py"
