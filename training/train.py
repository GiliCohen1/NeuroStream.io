"""Train the seizure-detection CNN on the University of Bonn EEG dataset.

Run from repo root:   .venv\\Scripts\\python training/train.py

Data:   training/data/{z,o,n,f,s}/  (100 recordings each, 4097 samples @ 173.61 Hz)
Label:  set S (recorded during seizures) = 1, sets Z, O, N, F = 0
Split:  by RECORDING (70/15/15 per set) so no recording leaks across train/val/test.
Input:  the exact same worker.preprocessing.preprocess() used at serving time.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "services" / "ml-server"), str(ROOT / "services" / "worker")]
from app.model import SpikeCNN  # noqa: E402
from worker.preprocessing import preprocess  # noqa: E402

FS = 173.61
WIN = 178          # 1 second of signal
NORM_WINDOW = 64   # must match the worker's NORM_WINDOW
SEED = 42
DATA = ROOT / "training" / "data"
OUT = ROOT / "models"


def load_recordings() -> list[tuple[np.ndarray, int, str]]:
    recs = []
    for set_name in "zonfs":
        files = sorted(p for p in (DATA / set_name).rglob("*.txt") if not p.name.startswith("._"))
        assert len(files) == 100, f"{set_name}: expected 100 files, got {len(files)}"
        for f in files:
            recs.append((np.loadtxt(f), int(set_name == "s"), set_name, f.relative_to(DATA).as_posix()))
    return recs


def split_recordings(recs, rng):
    train, val, test = [], [], []
    for set_name in "zonfs":
        group = [r for r in recs if r[2] == set_name]
        idx = rng.permutation(len(group))
        for k, i in enumerate(idx):
            (train if k < 70 else val if k < 85 else test).append(group[i])
    return train, val, test


def make_windows(recs, stride: int):
    xs, bs, ys = [], [], []
    for sig, label, _, _ in recs:
        for start in range(0, len(sig) - WIN + 1, stride):
            x, b = preprocess(sig[start:start + WIN].tolist(), FS, NORM_WINDOW)
            xs.append(x); bs.append(b); ys.append(label)
    return (torch.tensor(np.stack(xs)).unsqueeze(1), torch.tensor(np.stack(bs)), torch.tensor(ys))


def predict(net, x, b):
    net.eval()
    with torch.inference_mode():
        return torch.softmax(net(x, b), dim=1)[:, 1].numpy()


def auc(y, p):  # rank-based ROC AUC
    order = np.argsort(p); ranks = np.empty(len(p)); ranks[order] = np.arange(1, len(p) + 1)
    pos = y == 1
    return float((ranks[pos].sum() - pos.sum() * (pos.sum() + 1) / 2) / (pos.sum() * (~pos).sum()))


def metrics(y, p, thr):
    pred = (p >= thr).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum()); fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum()); tn = int(((pred == 0) & (y == 0)).sum())
    prec = tp / max(tp + fp, 1); rec = tp / max(tp + fn, 1)
    return {"accuracy": (tp + tn) / len(y), "precision": prec, "recall": rec,
            "f1": 2 * prec * rec / max(prec + rec, 1e-9), "auc": auc(y, p),
            "confusion": {"tp": tp, "fp": fp, "fn": fn, "tn": tn}}


def main() -> None:
    torch.manual_seed(SEED); rng = np.random.default_rng(SEED)
    recs = load_recordings()
    train_r, val_r, test_r = split_recordings(recs, rng)
    print(f"recordings  train={len(train_r)} val={len(val_r)} test={len(test_r)}")

    xtr, btr, ytr = make_windows(train_r, stride=45)   # overlapping windows = augmentation
    xva, bva, yva = make_windows(val_r, stride=WIN)
    xte, bte, yte = make_windows(test_r, stride=WIN)
    print(f"windows     train={len(ytr)} val={len(yva)} test={len(yte)}  seizure share(train)={ytr.float().mean():.2f}")

    net = SpikeCNN()
    weights = torch.tensor([1.0, (ytr == 0).sum().item() / (ytr == 1).sum().item()])
    loss_fn = nn.CrossEntropyLoss(weight=weights)
    opt = torch.optim.Adam(net.parameters(), lr=2e-3, weight_decay=1e-4)
    best_state, best_auc = None, -1.0

    for epoch in range(1, 31):
        net.train()
        perm = torch.randperm(len(ytr)); total = 0.0
        for i in range(0, len(perm), 128):
            idx = perm[i:i + 128]
            opt.zero_grad()
            loss = loss_fn(net(xtr[idx], btr[idx]), ytr[idx])
            loss.backward(); opt.step(); total += loss.item() * len(idx)
        pv = predict(net, xva, bva)
        a = auc(yva.numpy(), pv)
        if a > best_auc:
            best_auc, best_state = a, {k: v.clone() for k, v in net.state_dict().items()}
        if epoch % 3 == 0 or epoch == 1:
            print(f"epoch {epoch:2d}  train_loss={total / len(ytr):.4f}  val_auc={a:.4f}  val_acc={metrics(yva.numpy(), pv, 0.5)['accuracy']:.4f}")

    net.load_state_dict(best_state)
    pv = predict(net, xva, bva)
    thr = max(np.linspace(0.05, 0.95, 91), key=lambda t: metrics(yva.numpy(), pv, t)["f1"])
    m_val = metrics(yva.numpy(), pv, thr)
    m_test = metrics(yte.numpy(), predict(net, xte, bte), thr)   # touched exactly once
    print(f"\nthreshold (picked on validation) = {thr:.2f}")
    print("VALIDATION", json.dumps(m_val, indent=1))
    print("TEST      ", json.dumps(m_test, indent=1))

    OUT.mkdir(exist_ok=True)
    # Lets simulators/demos replay ONLY recordings the model has never seen.
    (OUT / "heldout_test_files.json").write_text(json.dumps(
        [{"file": r[3], "seizure": bool(r[1])} for r in test_r], indent=1))
    torch.save(net.state_dict(), OUT / "spike_cnn.pt")
    (OUT / "meta.json").write_text(json.dumps({
        "version": "spike-cnn-bonn-0.1", "input_length": WIN, "sampling_rate_hz": FS, "norm_window": NORM_WINDOW,
        "threshold": float(thr), "dataset": "University of Bonn EEG (Andrzejak et al. 2001)",
        "metrics": {"validation": m_val, "test": m_test},
    }, indent=2))
    print(f"saved {OUT / 'spike_cnn.pt'}")


if __name__ == "__main__":
    main()
