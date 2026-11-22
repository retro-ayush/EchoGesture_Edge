"""Separate hand-local 3D from body-relative image coordinates.

World coordinates from independent hand detections do NOT share an origin.
Keep wrist position in the image/body frame to preserve signing location.
"""
import numpy as np

BODY = [0, 2, 5, 7, 8, 11, 12, 13, 14, 15, 16, 23, 24]
COORD_DIM = 156  # body xy (26), local hand xyz (126), wrist xy (4)
MASK_DIM = 17  # body (13), hands (2), wrist positions (2)
BASE_DIM = COORD_DIM + MASK_DIM
FEATURE_DIM = COORD_DIM * 2 + MASK_DIM
FRAMES = 64


def frame_features(pose_xy, pose_visibility, hands, aspect=1.0):
    """hands maps Left/Right to (21x3 world xyz, 21x2 image xy).

    Input image coordinates must be unmirrored. Labels are MediaPipe's
    handedness, which assumes mirrored input, so on unmirrored frames "Left"
    is the signer's right hand. This is harmless because training and
    inference use the same convention; never mirror only one of them.
    Hand orientation is retained; this is not full viewpoint invariance.
    """
    coords = np.zeros(COORD_DIM, np.float32)
    masks = np.zeros(MASK_DIM, np.float32)
    center, scale = None, None
    if pose_xy is not None and pose_visibility is not None:
        xy = np.asarray(pose_xy, np.float32) * [aspect, 1.0]
        vis = np.asarray(pose_visibility)
        if vis[11] >= .5 and vis[12] >= .5:
            center = (xy[11] + xy[12]) / 2
            scale = float(np.linalg.norm(xy[11] - xy[12]))
            if scale > 1e-4:
                for j, idx in enumerate(BODY):
                    if vis[idx] >= .5:
                        coords[j*2:j*2+2] = (xy[idx] - center) / scale
                        masks[j] = 1
            else:
                center, scale = None, None
    for h, name in enumerate(("Left", "Right")):
        if name not in hands:
            continue
        world, image = (np.asarray(x, np.float32) for x in hands[name])
        if world.shape != (21, 3) or image.shape != (21, 2):
            raise ValueError("A hand requires 21 world and image landmarks")
        palm = float(np.linalg.norm(world[9] - world[0]))
        if palm < 1e-6 or not np.isfinite(world).all():
            continue
        start = 26 + 63*h
        coords[start:start+63] = ((world-world[0])/palm).reshape(-1)
        masks[13+h] = 1
        if center is not None:
            coords[152+2*h:154+2*h] = (image[0]*[aspect, 1.0]-center)/scale
            masks[15+h] = 1
    return np.concatenate([np.clip(coords, -10, 10), masks]).astype(np.float32)


def coord_mask(masks):
    return np.concatenate([
        np.repeat(masks[:, :13], 2, axis=1),
        np.repeat(masks[:, 13:15], 63, axis=1),
        np.repeat(masks[:, 15:17], 2, axis=1),
    ], axis=1)


ABLATIONS = ("none", "no_masks", "no_motion")


def encode_sequence(base, frames=FRAMES, ablation="none"):
    """Uniform nearest sampling preserves missing observations as missing.

    Velocity is per resampled step, not metres/second. A complete isolated
    clip is required; 64 observations are not a 50 ms observation window.

    ablation (research experiments only; tensor shape is unchanged):
      no_masks  - mask channels zeroed and motion NOT gated by presence, so
                  dropouts produce false jumps as in an unmasked baseline.
      no_motion - motion channels zeroed.
    """
    if ablation not in ABLATIONS:
        raise ValueError(f"ablation must be one of {ABLATIONS}")
    base = np.asarray(base, np.float32)
    if base.ndim != 2 or base.shape[1] != BASE_DIM or len(base) == 0:
        raise ValueError(f"Expected a nonempty (T,{BASE_DIM}) array")
    if not np.isfinite(base).all():
        raise ValueError("Landmarks contain NaN or infinity")
    if frames < 2:
        raise ValueError("At least two frames required")
    ids = np.rint(np.linspace(0, len(base)-1, frames)).astype(int)
    sampled = base[ids]
    masks = sampled[:, COORD_DIM:]
    valid = coord_mask(masks)
    xyz = sampled[:, :COORD_DIM] * valid
    velocity = np.zeros_like(xyz)
    if ablation == "no_masks":
        velocity[1:] = xyz[1:]-xyz[:-1]
        masks = np.zeros_like(masks)
    elif ablation == "none":
        velocity[1:] = (xyz[1:]-xyz[:-1]) * valid[1:] * valid[:-1]
    return np.concatenate([xyz, velocity, masks], axis=1).T.copy()


def augment(base, rng):
    """Modest local noise, time crop, and short hand detection dropout."""
    x = np.asarray(base, np.float32).copy()
    if len(x) > 12 and rng.random() < .5:
        trim = max(1, len(x)//12)
        left, right = int(rng.integers(trim)), int(rng.integers(trim))
        x = x[left:len(x)-right]
    mask = coord_mask(x[:, COORD_DIM:])
    x[:, :COORD_DIM] += rng.normal(0, .008, x[:, :COORD_DIM].shape)*mask
    if rng.random() < .3:
        hand = int(rng.integers(2))
        start = int(rng.integers(len(x)))
        end = min(len(x), start + max(1, len(x)//10))
        x[start:end, 26+hand*63:26+(hand+1)*63] = 0
        x[start:end, 152+hand*2:154+hand*2] = 0
        x[start:end, COORD_DIM+13+hand] = 0
        x[start:end, COORD_DIM+15+hand] = 0
    return x


# Body joint slots (index into BODY) that swap under a left/right mirror.
_BODY_SWAP = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11]


def mirror(base):
    """Horizontal mirror of a (T, BASE_DIM) sequence: flips x and swaps
    left/right body joints and the two hand slots. Makes the classifier
    tolerant of left-handed signers; the label is unchanged."""
    x = np.asarray(base, np.float32).copy()
    body = x[:, :26].reshape(len(x), 13, 2)[:, _BODY_SWAP]
    body[..., 0] *= -1
    hands = x[:, 26:152].reshape(len(x), 2, 21, 3)[:, ::-1].copy()
    hands[..., 0] *= -1
    wrists = x[:, 152:156].reshape(len(x), 2, 2)[:, ::-1].copy()
    wrists[..., 0] *= -1
    m = x[:, COORD_DIM:]
    masks = np.concatenate([m[:, :13][:, _BODY_SWAP], m[:, 13:15][:, ::-1], m[:, 15:17][:, ::-1]], 1)
    return np.concatenate([body.reshape(len(x), 26), hands.reshape(len(x), 126),
                           wrists.reshape(len(x), 4), masks], 1).astype(np.float32)


def _rotation(rng, degrees):
    ax, ay, az = np.deg2rad(rng.uniform(-degrees, degrees, 3))
    rx = np.array([[1, 0, 0], [0, np.cos(ax), -np.sin(ax)], [0, np.sin(ax), np.cos(ax)]])
    ry = np.array([[np.cos(ay), 0, np.sin(ay)], [0, 1, 0], [-np.sin(ay), 0, np.cos(ay)]])
    rz = np.array([[np.cos(az), -np.sin(az), 0], [np.sin(az), np.cos(az), 0], [0, 0, 1]])
    return (rx @ ry @ rz).astype(np.float32)


def augment_strong(base, rng):
    """Training-only augmentation for real signer variation: mirror
    (handedness), speed change, small 2D affine on the body frame, small 3D
    rotation of each hand, then the modest `augment` noise/crop/dropout.
    Missing points stay exactly zero (masks are respected)."""
    x = np.asarray(base, np.float32)
    if rng.random() < .5:
        x = mirror(x)
    if len(x) > 4 and rng.random() < .8:  # speed change via nearest resampling
        n = max(4, int(round(len(x) * rng.uniform(.75, 1.3))))
        x = x[np.rint(np.linspace(0, len(x)-1, n)).astype(int)]
    x = x.copy()
    valid = coord_mask(x[:, COORD_DIM:])
    if rng.random() < .8:  # body-frame 2D affine (shoulder-centred units)
        a = np.deg2rad(rng.uniform(-12, 12))
        s = rng.uniform(.85, 1.15)
        shear = rng.uniform(-.15, .15)
        m = (np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]]) @
             np.array([[1, shear], [0, 1]]) * s * [[rng.uniform(.9, 1.1)], [1]]).astype(np.float32)
        for lo, hi in ((0, 26), (152, 156)):
            pts = x[:, lo:hi].reshape(len(x), -1, 2) @ m.T
            x[:, lo:hi] = pts.reshape(len(x), -1)
    for h in range(2):
        if rng.random() < .8:
            lo = 26 + 63*h
            pts = x[:, lo:lo+63].reshape(len(x), 21, 3) @ _rotation(rng, 15).T
            pts *= rng.uniform(.9, 1.1, 3).astype(np.float32)
            x[:, lo:lo+63] = pts.reshape(len(x), 63)
    x[:, :COORD_DIM] *= valid
    return augment(x, rng)
