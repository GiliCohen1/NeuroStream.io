"""Data for the site's animated explainer (the front end lives in site/js/explainer.js).

Everything shown is computed from the real held-out data and the trained model:
  * 5 consecutive seconds of a real held-out recording per class (seizure-free / seizure)
  * the epochs' normalized waveforms, band powers and model scores
  * score distribution over all 1,725 held-out epochs (drives the threshold explorer)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

FS = 173.61
WIN = 178
N_EPOCHS = 5


def build_data(root: Path) -> dict:
    sys.path[:0] = [str(root / "services" / "ml-server"), str(root / "services" / "worker")]
    from app.model import SpikeDetector
    from worker.preprocessing import preprocess

    det = SpikeDetector()
    thr = det.threshold
    files = json.loads((root / "models" / "heldout_test_files.json").read_text())
    scores, bands_by_class = [], {0: [], 1: []}
    per_file: dict[str, tuple[bool, np.ndarray, np.ndarray, list, list]] = {}
    for f in sorted(files, key=lambda f: f["file"]):
        sig = np.loadtxt(root / "data" / "heldout" / f["file"])
        eps = [preprocess(sig[s:s + WIN].tolist(), FS, 64) for s in range(0, len(sig) - WIN + 1, WIN)]
        p = det.score([e[0].tolist() for e in eps], [e[1].tolist() for e in eps])
        label = int(f["seizure"])
        for pp, (_, b) in zip(p, eps):
            scores.append([round(float(pp), 3), label])
            bands_by_class[label].append(b)
        per_file[f["file"]] = (f["seizure"], sig, np.array(p), [e[0] for e in eps], [e[1] for e in eps])

    def pick(seizure: bool, prefix: str | None):
        for name, (s, sig, p, norms, bands) in per_file.items():
            if s != seizure or (prefix and not name.lower().startswith(prefix)):
                continue
            ok = p[:N_EPOCHS] >= thr if seizure else p[:N_EPOCHS] < thr
            if ok.all():
                n = N_EPOCHS * WIN
                return {
                    "file": name,
                    "raw": [int(round(v)) for v in sig[:n]],
                    "norm": [round(float(v), 2) for e in norms[:N_EPOCHS] for v in e],
                    "bands": [[round(float(v), 4) for v in b] for b in bands[:N_EPOCHS]],
                    "p": [round(float(v), 4) for v in p[:N_EPOCHS]],
                    "amp": float(np.percentile(np.abs(sig[:n]), 99)),
                }
        raise RuntimeError("no suitable example recording")

    data = {
        "fs": FS, "thr": thr, "win": WIN,
        "ex": {"normal": pick(False, "n/"), "seizure": pick(True, "s/")},
        "bandMean": {"normal": [round(float(v), 4) for v in np.mean(bands_by_class[0], axis=0)],
                     "seizure": [round(float(v), 4) for v in np.mean(bands_by_class[1], axis=0)]},
        "scores": scores,
    }
    # Sanity: the explorer must reproduce the training-time metrics at the deployed threshold.
    meta = json.loads((root / "models" / "meta.json").read_text())["metrics"]["test"]["confusion"]
    tp = sum(1 for p, y in scores if y == 1 and p >= thr); fp = sum(1 for p, y in scores if y == 0 and p >= thr)
    assert (tp, fp) == (meta["tp"], meta["fp"]), f"explorer mismatch {(tp, fp)} vs {(meta['tp'], meta['fp'])}"
    return data


