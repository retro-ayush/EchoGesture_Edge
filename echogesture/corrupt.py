"""Deterministic paired corruptions for robustness testing.

Each clip gets one parameter set (seeded by clip_id) applied to every frame,
so the corrupted clip keeps the same label and timing as the clean clip.
These are synthetic proxies; also record real low-light/angle clips if possible.
"""
import hashlib
import numpy as np

KINDS = ("low_light", "occlusion", "angle")


def clip_rng(clip_id, kind, seed=0):
    digest = hashlib.sha256(f"{seed}:{kind}:{clip_id}".encode()).digest()
    return np.random.default_rng(int.from_bytes(digest[:8], "little"))


def params_for(kind, rng):
    if kind == "low_light":
        return {"gain": float(rng.uniform(.25, .4)), "gamma": float(rng.uniform(1.4, 1.8)),
                "noise_sigma": float(rng.uniform(4, 8))}
    if kind == "occlusion":
        # Box covering ~6% of the frame, placed in the central signing area.
        return {"w": .25, "h": .25, "x": float(rng.uniform(.2, .55)), "y": float(rng.uniform(.2, .55))}
    if kind == "angle":
        return {"squeeze": float(rng.uniform(.12, .2)), "side": int(rng.integers(2))}
    raise ValueError(f"kind must be one of {KINDS}")


def low_light(frame, gain, gamma, noise_sigma, rng):
    x = (frame.astype(np.float32)/255.0)**gamma * gain * 255.0
    x += rng.normal(0, noise_sigma, frame.shape)
    return np.clip(x, 0, 255).astype(np.uint8)


def occlusion(frame, w, h, x, y):
    out = frame.copy()
    H, W = frame.shape[:2]
    x0, y0 = int(x*W), int(y*H)
    out[y0:y0+int(h*H), x0:x0+int(w*W)] = 0
    return out


def angle(frame, squeeze, side):
    """Perspective warp approximating a sideways (yaw) camera offset."""
    import cv2
    H, W = frame.shape[:2]
    src = np.float32([[0, 0], [W, 0], [W, H], [0, H]])
    d = squeeze*H/2
    dst = src.copy()
    if side:
        dst[1, 1] += d; dst[2, 1] -= d
    else:
        dst[0, 1] += d; dst[3, 1] -= d
    M = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(frame, M, (W, H), borderMode=cv2.BORDER_CONSTANT)


def apply(kind, frame, params, rng):
    if kind == "low_light":
        return low_light(frame, rng=rng, **params)
    if kind == "occlusion":
        return occlusion(frame, **params)
    if kind == "angle":
        return angle(frame, **params)
    raise ValueError(f"kind must be one of {KINDS}")
