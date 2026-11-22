"""Local MediaPipe Tasks extraction. No network operations at inference."""
import os
from pathlib import Path
import numpy as np
from .features import frame_features

# MediaPipe delegate: "auto" (GPU, CPU fallback; GPU landmarks gave identical val top-1,
# 78.51 %, to the CPU-trained model), "gpu" or "cpu". Override: ECHOGESTURE_DELEGATE.
DEFAULT_DELEGATE = "auto"

DARK_P99 = 100  # 99th-percentile luma below which a frame is exposure-corrected
TARGET_P99 = float(os.environ.get("ECHOGESTURE_ENHANCE_TARGET", 200))


def enhance_low_light(bgr):
    """Automatic exposure correction for underexposed frames.

    Uses the 99th-percentile luma (the brightest skin/face), not the mean:
    many clean WLASL clips have black backgrounds with a well-lit signer
    (clean p99 >= 105 on our data), whereas a dim scene compresses everything
    (synthetic low light p99 <= 98). A gamma LUT lifts p99 to TARGET_P99,
    then a 3x3 median filter suppresses amplified sensor noise. ~1 ms/frame.
    Frames that are not dark pass through unchanged.
    """
    import cv2
    small = cv2.resize(bgr, (160, 90), interpolation=cv2.INTER_AREA)
    p99 = float(np.percentile(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY), 99))
    if p99 >= DARK_P99 or p99 < 4:
        return bgr
    gamma = np.log(TARGET_P99 / 255.0) / np.log(p99 / 255.0)
    lut = np.clip(((np.arange(256) / 255.0) ** gamma) * 255.0, 0, 255).astype(np.uint8)
    out = cv2.LUT(bgr, lut)
    return cv2.medianBlur(out, 3) if os.environ.get("ECHOGESTURE_ENHANCE_BLUR", "1") == "1" else out


class Extractor:
    def __init__(self, models="models", enhance=None, delegate=None):
        import mediapipe as mp
        self.mp = mp
        # Experimental exposure correction, OFF by default: on low-light val copies it
        # lowered top-1 (58.7% off vs 49.6% LUT+median, 54.5% LUT-only), so opt in only
        # with ECHOGESTURE_ENHANCE=1 for experiments.
        self.enhance = os.environ.get("ECHOGESTURE_ENHANCE", "0") == "1" if enhance is None else enhance
        root = Path(models)
        for name in ("hand_landmarker.task", "pose_landmarker_lite.task"):
            if not (root/name).is_file():
                raise FileNotFoundError(f"Missing {root/name}; run scripts/download_models.py")
        self.delegate = self._create(mp, root, delegate or os.environ.get("ECHOGESTURE_DELEGATE", DEFAULT_DELEGATE))
        self.last_timestamp = -1

    def _create(self, mp, root, delegate):
        """delegate: "gpu", "cpu" or "auto" (GPU, falling back to CPU where
        MediaPipe has no GPU support, e.g. Windows Python wheels)."""
        if delegate not in ("auto", "gpu", "cpu"):
            raise ValueError("delegate must be auto, gpu or cpu")
        order = ["gpu", "cpu"] if delegate == "auto" else [delegate]
        error = None
        for name in order:
            try:
                d = mp.tasks.BaseOptions.Delegate.GPU if name == "gpu" else mp.tasks.BaseOptions.Delegate.CPU
                self._build(mp, root, d)
                return name
            except Exception as e:  # no GPU support on this platform
                error = e
        raise RuntimeError(f"Could not create MediaPipe landmarkers: {error}")

    def _build(self, mp, root, d):
        vision = mp.tasks.vision
        base = lambda f: mp.tasks.BaseOptions(model_asset_path=str(root/f), delegate=d)  # noqa: E731
        self.hand = vision.HandLandmarker.create_from_options(
            vision.HandLandmarkerOptions(base_options=base("hand_landmarker.task"),
                                         running_mode=vision.RunningMode.VIDEO, num_hands=2))
        self.pose = vision.PoseLandmarker.create_from_options(
            vision.PoseLandmarkerOptions(base_options=base("pose_landmarker_lite.task"),
                                         running_mode=vision.RunningMode.VIDEO, num_poses=1,
                                         output_segmentation_masks=False))

    def frame(self, bgr, timestamp_ms):
        import cv2
        timestamp_ms = max(self.last_timestamp+1, int(timestamp_ms))
        self.last_timestamp = timestamp_ms
        if self.enhance:
            bgr = enhance_low_light(bgr)
        # GPU delegate needs 4-channel input; the pixels are identical.
        if self.delegate == "gpu":
            image = self.mp.Image(image_format=self.mp.ImageFormat.SRGBA,
                                  data=cv2.cvtColor(bgr, cv2.COLOR_BGR2RGBA))
        else:
            image = self.mp.Image(image_format=self.mp.ImageFormat.SRGB,
                                  data=cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        p = self.pose.detect_for_video(image, timestamp_ms)
        h = self.hand.detect_for_video(image, timestamp_ms)
        pose_xy, vis = None, None
        if p.pose_landmarks:
            pose_xy = [[v.x, v.y] for v in p.pose_landmarks[0]]
            vis = [v.visibility for v in p.pose_landmarks[0]]
        hands, scores = {}, {}
        for label, world, landmarks in zip(h.handedness, h.hand_world_landmarks, h.hand_landmarks):
            name, score = label[0].category_name, label[0].score
            if score <= scores.get(name, -1):
                continue
            scores[name] = score
            hands[name] = (np.array([[v.x, v.y, v.z] for v in world]),
                           np.array([[v.x, v.y] for v in landmarks]))
        return frame_features(pose_xy, vis, hands, bgr.shape[1]/bgr.shape[0])

    def close(self):
        self.hand.close()
        self.pose.close()


def video_features(video_path, models="models", max_frames=1800, timing=None):
    """A fresh Extractor per clip: VIDEO-mode tracking state must not leak
    between clips. Pass a dict as `timing` to receive extractor_init_ms."""
    import cv2
    from time import perf_counter
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    extractor = None
    frames = []
    try:
        start = perf_counter()
        extractor = Extractor(models)
        if timing is not None:
            timing["extractor_init_ms"] = (perf_counter()-start)*1000
            timing["delegate"] = extractor.delegate
        fps = cap.get(cv2.CAP_PROP_FPS)
        if not np.isfinite(fps) or fps <= 0:
            fps = 30.0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if len(frames) >= max_frames:
                raise ValueError("Clip exceeds frame limit; provide an isolated-sign clip")
            frames.append(extractor.frame(frame, 1000*len(frames)/fps))
    finally:
        cap.release()
        if extractor is not None:
            extractor.close()
    if not frames:
        raise ValueError("Video contained no decodable frames")
    return np.stack(frames)
