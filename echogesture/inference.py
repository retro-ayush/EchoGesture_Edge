import json
from pathlib import Path
from time import perf_counter
import numpy as np
from .features import encode_sequence, FEATURE_DIM, FRAMES


class Predictor:
    def __init__(self, model="models/echogesture.int8.onnx"):
        import onnxruntime as ort
        path = Path(model)
        if not path.is_file():
            raise FileNotFoundError(f"Train and export a model first: missing {path}")
        self.meta = json.loads(path.with_suffix(".json").read_text())
        self.labels = self.meta["labels"]
        if self.meta["feature_dim"] != FEATURE_DIM or self.meta["frames"] != FRAMES:
            raise ValueError("Model/preprocessing schema mismatch")
        if self.meta.get("ablation", "none") != "none":
            raise ValueError("Ablation models cannot be deployed")
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 1; opts.inter_op_num_threads = 1
        self.session = ort.InferenceSession(str(path), sess_options=opts,
                                            providers=["CPUExecutionProvider"])

    def logits(self, base):
        """Raw logits plus preprocessing and model-only timings."""
        t = perf_counter()
        x = encode_sequence(base)[None]
        pre_ms = (perf_counter()-t)*1000
        t = perf_counter()
        logits = self.session.run(["logits"], {"landmarks":x})[0][0]
        return logits, pre_ms, (perf_counter()-t)*1000

    def classify(self, base):
        logits, pre_ms, inference_ms = self.logits(base)
        temperature = float(self.meta.get("temperature", 1.0))
        if temperature <= 0:
            raise ValueError("Temperature must be positive")
        logits = logits/temperature
        probs = np.exp(logits-logits.max()); probs /= probs.sum()
        order = np.argsort(-probs)[:min(5, len(probs))]
        return {"prediction":self.labels[int(order[0])],
                "top5":[self.labels[int(i)] for i in order],
                "confidence":float(probs[order[0]]),
                "classifier_latency_ms":inference_ms,
                "preprocess_latency_ms":pre_ms}

    def video(self, video_path, models="models", clip_id=None):
        from .extract import video_features
        start = perf_counter()
        timing = {}
        base = video_features(video_path, models, timing=timing)
        extraction_ms = (perf_counter()-start)*1000
        result = self.classify(base)
        total_ms = (perf_counter()-start)*1000
        result.update({"clip_id":clip_id if clip_id is not None else Path(video_path).stem,
                       "latency_ms":total_ms,
                       "extraction_latency_ms":extraction_ms,
                       "extractor_init_ms":timing.get("extractor_init_ms"),
                       "frames_processed":len(base)})
        return result
