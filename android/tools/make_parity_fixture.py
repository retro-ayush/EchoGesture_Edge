"""Write app/src/test/resources/parity.json: Python reference outputs of
frame_features and encode_sequence on random inputs (with missing pose/hands,
low visibility and degenerate cases). FeaturesParityTest checks the Kotlin port.
Run from the repo root: PYTHONPATH=. .venv/bin/python android/tools/make_parity_fixture.py
"""
import json
from pathlib import Path
import numpy as np
from echogesture.features import frame_features, encode_sequence

rng = np.random.default_rng(7)
frames = []
for i in range(120):
    pose_xy = vis = None
    if rng.random() > .15:
        pose_xy = rng.uniform(0, 1, (33, 2)).astype(np.float32)
        vis = rng.uniform(.2, 1, 33).astype(np.float32)
        if i % 10 == 0:
            vis[11] = .3  # shoulders hidden -> no body frame
    hands = {}
    for name in ("Left", "Right"):
        if rng.random() > .3:
            world = rng.normal(0, .05, (21, 3)).astype(np.float32)
            if i % 17 == 0:
                world[:] = 0  # degenerate palm -> skipped
            hands[name] = (world, rng.uniform(0, 1, (21, 2)).astype(np.float32))
    aspect = float(rng.choice([.75, 4/3, 16/9, 9/16]))
    out = frame_features(pose_xy, vis, hands, aspect)
    frames.append({
        "pose_xy": None if pose_xy is None else pose_xy.reshape(-1).tolist(),
        "vis": None if vis is None else vis.tolist(),
        "hands": {k: {"world": w.reshape(-1).tolist(), "image": im.reshape(-1).tolist()} for k, (w, im) in hands.items()},
        "aspect": aspect,
        "expected": out.tolist(),
    })
base = np.stack([np.array(f["expected"], np.float32) for f in frames])
sequences = []
for n in (8, 23, 64, 97, 120):
    sequences.append({"length": n, "expected": encode_sequence(base[:n]).reshape(-1).tolist()})
out = Path("android/app/src/test/resources/parity.json")
out.write_text(json.dumps({"frames": frames, "sequences": sequences}))
print(out, out.stat().st_size, "bytes")
