"""Replay training/train.py's loop (same seeds, split, windows, optimiser) and save per-epoch curves.

Writes docs/data/training_log.json only. It does NOT save weights or touch models/.
Requires training/data/ (see training/download_data.py). Run from repo root:
    .venv/Scripts/python docs/make_training_log.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "training"))
import train as T  # noqa: E402


def main() -> None:
    torch.manual_seed(T.SEED)
    rng = np.random.default_rng(T.SEED)
    recs = T.load_recordings()
    train_r, val_r, test_r = T.split_recordings(recs, rng)
    xtr, btr, ytr = T.make_windows(train_r, stride=45)
    xva, bva, yva = T.make_windows(val_r, stride=T.WIN)
    net = T.SpikeCNN()
    weights = torch.tensor([1.0, (ytr == 0).sum().item() / (ytr == 1).sum().item()])
    loss_fn = nn.CrossEntropyLoss(weight=weights)
    opt = torch.optim.Adam(net.parameters(), lr=2e-3, weight_decay=1e-4)
    hist = []
    for epoch in range(1, 31):
        net.train()
        perm = torch.randperm(len(ytr)); total = 0.0
        for i in range(0, len(perm), 128):
            idx = perm[i:i + 128]
            opt.zero_grad()
            loss = loss_fn(net(xtr[idx], btr[idx]), ytr[idx]); loss.backward(); opt.step()
            total += loss.item() * len(idx)
        pv = T.predict(net, xva, bva)
        hist.append({"epoch": epoch, "train_loss": total / len(ytr),
                     "val_auc": T.auc(yva.numpy(), pv), "val_acc": T.metrics(yva.numpy(), pv, 0.5)["accuracy"]})
        print(f"epoch {epoch:2d} loss={hist[-1]['train_loss']:.4f} val_auc={hist[-1]['val_auc']:.4f}")
    out = {"history": hist, "class_weight": float(weights[1]),
           "counts": {"train_windows": len(ytr), "val_windows": len(yva), "train_seizure_share": float(ytr.float().mean())}}
    (ROOT / "docs" / "data").mkdir(parents=True, exist_ok=True)
    (ROOT / "docs" / "data" / "training_log.json").write_text(json.dumps(out, indent=1))
    print("saved docs/data/training_log.json")


if __name__ == "__main__":
    main()
