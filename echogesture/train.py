"""Train on cached landmarks. Validation selects checkpoints; test is untouched."""
import argparse
import hashlib
import json
from pathlib import Path
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from .data import read_manifest, audit
from .features import ABLATIONS, augment, augment_strong, encode_sequence, FEATURE_DIM, FRAMES
from .model import TinyTemporalNet


class Landmarks(Dataset):
    def __init__(self, rows, labels, training=False, seed=42, ablation="none", strong=False):
        self.rows, self.labels, self.training = rows, labels, training
        self.index = {label: i for i, label in enumerate(labels)}
        self.ablation, self.strong = ablation, strong
        self.rng = np.random.default_rng(seed)
        self.cache = {}

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        row = self.rows[i]
        if i not in self.cache:  # small dataset: keep base landmarks in memory
            with np.load(row["path"], allow_pickle=False) as f:
                self.cache[i] = f["features"]
        x = self.cache[i]
        if self.training:
            x = augment_strong(x, self.rng) if self.strong else augment(x, self.rng)
        return torch.from_numpy(encode_sequence(x, ablation=self.ablation)), self.index[row["label"]]


def load_vocab(path):
    labels = json.loads(Path(path).read_text())
    if (not isinstance(labels, list) or not labels or
        any(not isinstance(s, str) or s != s.upper() or not s for s in labels) or
        len(set(labels)) != len(labels)):
        raise ValueError("Vocabulary must be a unique nonempty uppercase JSON list")
    return labels


def main():
    p = argparse.ArgumentParser()
    p.add_argument("manifest")
    p.add_argument("--vocab", required=True, help="Organizer's uppercase JSON list, fixed order")
    p.add_argument("--out", default="models/student.pt")
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--patience", type=int, default=0,
                   help="Stop after N epochs without val improvement (0 = run all epochs)")
    p.add_argument("--ablation", default="none", choices=ABLATIONS,
                   help="Research ablation; use a separate --out. Only 'none' can be exported.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cpu")
    p.add_argument("--width", type=int, default=96)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--pool", default="mean", choices=["mean", "meanmax"])
    p.add_argument("--label-smoothing", type=float, default=0.0)
    p.add_argument("--weight-decay", type=float, default=.01)
    p.add_argument("--aug", default="basic", choices=["basic", "strong"])
    a = p.parse_args()
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    torch.set_num_threads(2)
    rows = read_manifest(a.manifest)
    labels = load_vocab(a.vocab)
    train = [r for r in rows if r["split"] == "train"]
    val = [r for r in rows if r["split"] == "val"]
    if not train or not val:
        raise ValueError("Separate train and validation partitions are required")
    unknown = {r.get("label") for r in train+val} - set(labels)
    missing = set(labels) - {r["label"] for r in train}
    if unknown or missing:
        raise ValueError(f"Unknown labels {unknown}; no training samples for {missing}")
    audit_result = audit(rows)
    print("Split audit:", json.dumps(audit_result))
    loaders = [DataLoader(Landmarks(rs, labels, training=(i == 0), seed=a.seed, ablation=a.ablation,
                                    strong=a.aug == "strong"),
                          batch_size=a.batch_size, shuffle=(i == 0), num_workers=0)
               for i, rs in enumerate((train, val))]
    config = {"width": a.width, "dropout": a.dropout, "pool": a.pool}
    model = TinyTemporalNet(len(labels), **config).to(a.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, a.epochs))
    counts = np.bincount([labels.index(r["label"]) for r in train], minlength=len(labels))
    weights = torch.tensor(1/np.sqrt(counts), dtype=torch.float32, device=a.device)
    criterion = torch.nn.CrossEntropyLoss(weight=weights/weights.mean(), label_smoothing=a.label_smoothing)
    output = Path(a.out); output.parent.mkdir(parents=True, exist_ok=True)
    best, best_epoch, history = -1.0, 0, []
    manifest_sha = hashlib.sha256(Path(a.manifest).read_bytes()).hexdigest()

    def write_metadata():
        # Written on every new best, so an interrupted run stays exportable.
        metadata = {"labels": labels, "feature_dim": FEATURE_DIM, "frames": FRAMES,
                    "feature_version": "body2d-handlocal3d-v1", "ablation": a.ablation,
                    "best_val_accuracy": best, "best_epoch": best_epoch,
                    "epochs_run": len(history), "lr": a.lr, "config": config, "augmentation": a.aug,
                    "label_smoothing": a.label_smoothing, "weight_decay": a.weight_decay,
                    "seed": a.seed, "split_audit": audit_result, "training_clips": len(train),
                    "validation_clips": len(val), "temperature": 1.0,
                    "manifest_sha256": manifest_sha,
                    "parameters": sum(v.numel() for v in model.parameters()),
                    "note": "Validation result only; no test, mobile or robustness claim."}
        output.with_suffix(".json").write_text(json.dumps(metadata, indent=2))
        output.with_suffix(".history.json").write_text(json.dumps(history, indent=2))

    for epoch in range(a.epochs):
        model.train(); loss_sum = 0
        for x, y in loaders[0]:
            x, y = x.to(a.device), y.to(a.device)
            optimizer.zero_grad(); loss = criterion(model(x), y)
            loss.backward(); optimizer.step(); loss_sum += loss.item()*len(y)
        scheduler.step()
        model.eval(); correct, correct5, total = 0, 0, 0
        with torch.no_grad():
            for x, y in loaders[1]:
                logits = model(x.to(a.device)).cpu()
                top = logits.topk(min(5, logits.shape[1]), dim=1).indices
                correct += int((top[:, 0] == y).sum())
                correct5 += int((top == y[:, None]).any(1).sum()); total += len(y)
        acc = correct/total
        record = {"epoch": epoch+1, "train_loss": loss_sum/len(train), "val_accuracy": acc,
                  "val_top5": correct5/total, "lr": scheduler.get_last_lr()[0]}
        history.append(record); print(json.dumps(record))
        if acc > best:
            best, best_epoch = acc, epoch+1
            state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            torch.save({"state_dict": state, "labels": labels, "ablation": a.ablation,
                        "config": config}, output)
            write_metadata()
        elif a.patience and epoch+1 - best_epoch >= a.patience:
            print(f"Early stop: no val improvement for {a.patience} epochs")
            break
    write_metadata()
    print(f"Best val accuracy {best:.4f} at epoch {best_epoch} -> {output}")


if __name__ == "__main__":
    main()
