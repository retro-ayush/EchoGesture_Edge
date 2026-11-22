import torch
from torch import nn
from .features import FEATURE_DIM


class TemporalBlock(nn.Module):
    def __init__(self, channels, dilation, dropout=0.0):
        super().__init__()
        self.branch = nn.Sequential(
            nn.Conv1d(channels, channels, 3, padding=dilation,
                      dilation=dilation, groups=channels),
            nn.BatchNorm1d(channels), nn.ReLU(),
            nn.Conv1d(channels, channels, 1),
            nn.BatchNorm1d(channels),
            nn.Dropout(dropout),
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(x + self.branch(x))


class TinyTemporalNet(nn.Module):
    """An original small CNN baseline, not a reproduction of SPOTER.

    Mean and max temporal pooling are concatenated when `pool="meanmax"`
    (the default for new checkpoints); `pool="mean"` matches v1 checkpoints.
    """
    def __init__(self, classes, width=96, dilations=(1, 2, 4, 8), dropout=0.0, pool="mean"):
        super().__init__()
        self.pool = pool
        self.encoder = nn.Sequential(
            nn.Conv1d(FEATURE_DIM, width, 1), nn.BatchNorm1d(width), nn.ReLU(),
            *[TemporalBlock(width, d, dropout) for d in dilations],
        )
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(width * (2 if pool == "meanmax" else 1), classes)

    def forward(self, x):
        h = self.encoder(x)
        z = torch.cat([h.mean(dim=2), h.amax(dim=2)], 1) if self.pool == "meanmax" else h.mean(dim=2)
        return self.head(self.drop(z))


def build_model(checkpoint):
    """Rebuild a model from a saved checkpoint's architecture config."""
    config = checkpoint.get("config", {})
    model = TinyTemporalNet(len(checkpoint["labels"]), **config)
    model.load_state_dict(checkpoint["state_dict"])
    return model


def mirror_permutation():
    """(index, sign) such that x[:, index] * sign mirrors an encoded
    (FEATURE_DIM, T) tensor exactly as features.mirror does on base frames."""
    import numpy as np
    from .features import BASE_DIM, COORD_DIM, mirror
    probe = mirror(np.arange(1, BASE_DIM+1, dtype=np.float32)[None])[0]
    src = np.abs(probe).astype(int) - 1
    sign = np.sign(probe).astype(np.float32)
    index = np.concatenate([src[:COORD_DIM], COORD_DIM + src[:COORD_DIM], src[COORD_DIM:] + COORD_DIM])
    signs = np.concatenate([sign[:COORD_DIM], sign[:COORD_DIM], np.ones(BASE_DIM-COORD_DIM, np.float32)])
    return torch.tensor(index), torch.tensor(signs)


class DeployedEnsemble(nn.Module):
    """Averages logits of several members, optionally also on the mirrored input."""
    def __init__(self, members, mirror_tta=True):
        super().__init__()
        self.members = nn.ModuleList(members)
        self.mirror_tta = mirror_tta
        index, sign = mirror_permutation()
        self.register_buffer("index", index)
        self.register_buffer("sign", sign[None, :, None])

    def forward(self, x):
        views = [x, x[:, self.index] * self.sign] if self.mirror_tta else [x]
        outs = [m(v) for m in self.members for v in views]
        return torch.stack(outs).mean(0)
