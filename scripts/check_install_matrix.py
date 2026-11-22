"""Dry-run dependency resolution for each supported OS/CPU (metadata only).

  python scripts/check_install_matrix.py           # all targets, Python 3.11
Uses `pip install --dry-run --only-binary=:all: --platform ...`, so it proves
binary wheels exist and versions are mutually compatible on each target
without downloading full wheels or needing that machine. Markers in the
requirement files are evaluated for the *host*, so Intel-Mac torch pins are
checked by passing the matching requirement explicitly.
"""
import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def mac(arch, newest):
    tags = []
    for v in [f"{m}_0" for m in range(newest, 10, -1)] + ["10_15", "10_14", "10_13", "10_12", "10_9"]:
        tags += [f"macosx_{v}_{arch}", f"macosx_{v}_universal2"]
    return tags


LINUX = [f"manylinux_2_{m}_x86_64" for m in range(28, 4, -1)] + ["manylinux2014_x86_64", "manylinux2010_x86_64", "manylinux1_x86_64", "linux_x86_64"]
TARGETS = {
    "macOS 14+ Apple Silicon": mac("arm64", 14),
    "macOS 12 Apple Silicon": mac("arm64", 12),
    "macOS 13 Intel": mac("x86_64", 13),
    "Linux x86_64 (glibc 2.28+)": LINUX,
    "Windows x64": ["win_amd64"],
}


def resolve(platforms, reqs, constraints, python):
    with tempfile.TemporaryDirectory() as tmp:
        cmd = [sys.executable, "-m", "pip", "install", "--dry-run", "--quiet", "--ignore-installed",
               "--only-binary=:all:", "--python-version", python, "--target", tmp,
               "--report", f"{tmp}/r.json"]
        for p in platforms:
            cmd += ["--platform", p]
        for r in reqs:
            cmd += ["-r", r] if r.endswith(".txt") else [r]
        if constraints:
            cmd += ["-c", constraints]
        out = subprocess.run(cmd, capture_output=True, text=True)
        if out.returncode:
            why = [l.strip() for l in out.stderr.splitlines() if "requested" in l or "depends on" in l or "No matching" in l]
            return False, "; ".join(why[:3]) or out.stderr.strip().splitlines()[-1]
        report = json.loads(Path(f"{tmp}/r.json").read_text())
        pkgs = {i["metadata"]["name"].lower(): i["metadata"]["version"] for i in report["install"]}
        return True, f"{len(pkgs)} pkgs, torch {pkgs.get('torch', '-')}"


def main():
    python = sys.argv[1] if len(sys.argv) > 1 else "3.11"
    req, cons = str(ROOT/"requirements.txt"), str(ROOT/"constraints.txt")
    failed = False
    for name, plats in TARGETS.items():
        intel = "Intel" in name
        train = [req, "onnx==1.16.2", "torch==2.2.2" if intel else "torch==2.5.1"]
        for label, reqs in (("runtime", [req]), ("train", train)):
            for c in (cons, None):
                ok, info = resolve(plats, reqs, c, python)
                tag = "pinned" if c else "fallback"
                print(f"{'OK  ' if ok else 'FAIL'} py{python} {name:28s} {label:7s} {tag:8s} {info}", flush=True)
                if ok:
                    break
            failed |= not ok
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
