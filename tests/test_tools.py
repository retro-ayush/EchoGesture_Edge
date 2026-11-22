import unittest
import numpy as np
from echogesture.features import frame_features, encode_sequence, BASE_DIM, COORD_DIM, MASK_DIM
from echogesture.metrics import (accuracy_report, drift_percent, fit_temperature, softmax,
                                 suggest_threshold, threshold_table)
from echogesture import corrupt


class MetricTests(unittest.TestCase):
    def test_accuracy_report(self):
        r = accuracy_report(["A", "A", "B", "B"],
                            [["A", "B"], ["B", "A"], ["B", "A"], ["A", "C"]])
        self.assertEqual(r["top1_percent"], 50)
        self.assertEqual(r["top5_percent"], 75)
        self.assertEqual(r["macro_top1_percent"], 50)
        r = accuracy_report(["A", "A", "A", "B"], [["A"], ["A"], ["A"], ["A"]])
        self.assertEqual(r["top1_percent"], 75)
        self.assertEqual(r["macro_top1_percent"], 50)

    def test_temperature_reduces_overconfidence(self):
        rng = np.random.default_rng(0)
        targets = rng.integers(0, 5, 400)
        logits = rng.normal(0, 1, (400, 5))
        logits[np.arange(400), targets] += 1.0
        fit = fit_temperature(logits*8, targets)  # artificially overconfident
        self.assertGreater(fit["temperature"], 2)
        self.assertLess(fit["nll_after"], fit["nll_before"])
        np.testing.assert_allclose(softmax(logits).sum(1), 1)

    def test_threshold_selection(self):
        table = threshold_table([.2, .5, .7, .9], [False, False, True, True], [.3, .6, .8])
        self.assertEqual(table[0]["coverage"], .75)
        self.assertEqual(suggest_threshold(table, .9), .6)
        self.assertIsNone(suggest_threshold(threshold_table([.9], [False], [.5]), .9))

    def test_drift(self):
        t = np.linspace(0, 600, 600)
        v = np.where(t > 480, 12.0, 10.0)
        self.assertAlmostEqual(drift_percent(t, v)["drift_percent"], 20.0)


class AblationTests(unittest.TestCase):
    def setUp(self):
        pose = np.zeros((33, 2), np.float32); pose[11] = [.4, .4]; pose[12] = [.6, .4]
        hand = np.stack([np.linspace(0, .1, 21), np.linspace(0, .2, 21), np.linspace(0, .03, 21)], axis=1)
        a = frame_features(pose, np.ones(33), {"Left": (hand, np.tile([.5, .5], (21, 1)))})
        self.seq = np.stack([a, np.zeros(BASE_DIM, np.float32), a])

    def test_no_masks_exposes_false_jumps(self):
        full = encode_sequence(self.seq, frames=3)
        unmasked = encode_sequence(self.seq, frames=3, ablation="no_masks")
        np.testing.assert_array_equal(unmasked[2*COORD_DIM:], 0)
        self.assertTrue(np.abs(unmasked[COORD_DIM:2*COORD_DIM]).sum() > 0)
        np.testing.assert_array_equal(full[:COORD_DIM], unmasked[:COORD_DIM])

    def test_no_motion(self):
        x = encode_sequence(self.seq, frames=3, ablation="no_motion")
        np.testing.assert_array_equal(x[COORD_DIM:2*COORD_DIM], 0)
        self.assertTrue(x[2*COORD_DIM:].sum() > 0)
        self.assertEqual(x.shape[0], 2*COORD_DIM+MASK_DIM)
        with self.assertRaises(ValueError):
            encode_sequence(self.seq, ablation="bogus")


class CorruptionTests(unittest.TestCase):
    def test_deterministic_and_shape_preserving(self):
        frame = np.full((48, 64, 3), 200, np.uint8)
        for kind in ("low_light", "occlusion"):
            p1 = corrupt.params_for(kind, corrupt.clip_rng("clip7", kind))
            p2 = corrupt.params_for(kind, corrupt.clip_rng("clip7", kind))
            self.assertEqual(p1, p2)
            out = corrupt.apply(kind, frame, p1, corrupt.clip_rng("clip7", kind))
            self.assertEqual(out.shape, frame.shape); self.assertEqual(out.dtype, np.uint8)
            self.assertLess(out.mean(), frame.mean())

    def test_angle(self):
        try:
            import cv2  # noqa: F401
        except ImportError:
            self.skipTest("OpenCV not installed")
        frame = np.full((48, 64, 3), 200, np.uint8)
        out = corrupt.apply("angle", frame, {"squeeze": .2, "side": 1}, None)
        self.assertEqual(out.shape, frame.shape)
        self.assertEqual(out[0, -1].sum(), 0)  # warped corner falls outside


if __name__ == "__main__":
    unittest.main()
