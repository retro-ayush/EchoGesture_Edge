import unittest
import numpy as np
from echogesture.features import BASE_DIM, COORD_DIM, encode_sequence, mirror, augment_strong, coord_mask
try:
    import torch
    from echogesture.model import TinyTemporalNet, DeployedEnsemble, build_model, mirror_permutation
except ImportError:  # runtime-only install
    torch = None


def random_base(t=40, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(t, BASE_DIM)).astype(np.float32)
    x[:, COORD_DIM:] = rng.random((t, BASE_DIM-COORD_DIM)) > .3
    x[:, :COORD_DIM] *= coord_mask(x[:, COORD_DIM:])
    return x


class UpgradeTests(unittest.TestCase):
    def test_mirror_is_an_involution(self):
        x = random_base()
        np.testing.assert_allclose(mirror(mirror(x)), x, atol=1e-6)

    @unittest.skipIf(torch is None, "needs torch")
    def test_in_graph_mirror_matches_reference(self):
        x = random_base(seed=1)
        index, sign = mirror_permutation()
        a = encode_sequence(mirror(x))
        b = (torch.tensor(encode_sequence(x))[index] * sign[:, None]).numpy()
        np.testing.assert_allclose(a, b, atol=1e-6)

    def test_strong_augmentation_keeps_missing_points_zero(self):
        x = random_base(seed=2)
        for s in range(5):
            y = augment_strong(x, np.random.default_rng(s))
            self.assertTrue(np.isfinite(y).all())
            self.assertEqual(y.shape[1], BASE_DIM)
            self.assertTrue((y[:, :COORD_DIM][coord_mask(y[:, COORD_DIM:]) == 0] == 0).all())

    @unittest.skipIf(torch is None, "needs torch")
    def test_checkpoint_rebuild_and_ensemble(self):
        torch.manual_seed(0)
        config = {"width": 32, "dropout": .1, "pool": "meanmax"}
        net = TinyTemporalNet(7, **config).eval()
        ck = {"state_dict": net.state_dict(), "labels": list("ABCDEFG"), "config": config}
        rebuilt = build_model(ck).eval()
        x = torch.tensor(encode_sequence(random_base(seed=3)))[None]
        torch.testing.assert_close(rebuilt(x), net(x))
        ens = DeployedEnsemble([net, rebuilt]).eval()
        self.assertEqual(tuple(ens(x).shape), (1, 7))
        plain = DeployedEnsemble([net], mirror_tta=False).eval()
        torch.testing.assert_close(plain(x), net(x))

    @unittest.skipIf(torch is None, "needs torch")
    def test_v1_checkpoint_without_config_still_loads(self):
        net = TinyTemporalNet(3)
        model = build_model({"state_dict": net.state_dict(), "labels": ["A", "B", "C"]})
        self.assertEqual(model.pool, "mean")


if __name__ == "__main__":
    unittest.main()
