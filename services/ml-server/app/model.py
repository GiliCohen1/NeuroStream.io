"""Seizure detector: a small 1-D convolutional neural network trained on real EEG.

Training lives in `training/train.py`; this module holds the architecture (shared by both)
and the loader used at serving time. Weights + decision threshold are read from `models/`.
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

log = logging.getLogger("model")

N_BANDS = 5


def default_model_dir() -> Path:
    """MODEL_DIR env var (set in Docker), else <repo>/models when running from a checkout."""
    if os.getenv("MODEL_DIR"):
        return Path(os.environ["MODEL_DIR"])
    return Path(__file__).resolve().parents[3] / "models"


class SpikeCNN(nn.Module):
    """Input: normalized waveform (B, 1, L) + relative EEG band powers (B, 5). Output: 2 logits."""

    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(1, 16, 7, padding=3), nn.BatchNorm1d(16), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(16, 32, 5, padding=2), nn.BatchNorm1d(32), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(32, 64, 3, padding=1), nn.BatchNorm1d(64), nn.ReLU(),
        )
        # 64 avg-pooled + 64 max-pooled conv features + 5 spectral features
        self.head = nn.Sequential(nn.Linear(128 + N_BANDS, 64), nn.ReLU(), nn.Dropout(0.3), nn.Linear(64, 2))

    def forward(self, x: torch.Tensor, bands: torch.Tensor) -> torch.Tensor:
        h = self.features(x)
        pooled = torch.cat([h.mean(dim=2), h.amax(dim=2)], dim=1)
        return self.head(torch.cat([pooled, bands], dim=1))


class SpikeDetector:
    """Serving wrapper: resamples input to the training length, runs the net, applies the tuned threshold."""

    def __init__(self, model_dir: Path | None = None) -> None:
        model_dir = model_dir or default_model_dir()
        meta_path, weights_path = model_dir / "meta.json", model_dir / "spike_cnn.pt"
        if not weights_path.exists():
            raise FileNotFoundError(f"{weights_path} missing: run `python training/train.py` first")
        self.meta = json.loads(meta_path.read_text())
        self.input_length: int = self.meta["input_length"]
        self.threshold: float = self.meta["threshold"]
        self.version: str = self.meta["version"]
        self.net = SpikeCNN()
        self.net.load_state_dict(torch.load(weights_path, map_location="cpu"))
        self.net.eval()
        log.info("loaded %s (threshold=%.3f, test F1=%.3f)", self.version, self.threshold,
                 self.meta["metrics"]["test"]["f1"])

    def _prepare(self, signals: list[Sequence[float]]) -> torch.Tensor:
        x = torch.tensor(np.stack([np.asarray(s, dtype=np.float32) for s in signals]) if len(
            {len(s) for s in signals}) == 1 else np.stack([
                np.interp(np.linspace(0, len(s) - 1, self.input_length), np.arange(len(s)), s).astype(np.float32)
                for s in signals]))
        if x.shape[1] != self.input_length:
            x = F.interpolate(x.unsqueeze(1), size=self.input_length, mode="linear", align_corners=False).squeeze(1)
        return x.unsqueeze(1)

    @torch.inference_mode()
    def score(self, signals: list[Sequence[float]], band_powers: list[Sequence[float]]) -> list[float]:
        """Return P(seizure) for each sample."""
        bands = torch.tensor(np.asarray(band_powers, dtype=np.float32))
        logits = self.net(self._prepare(signals), bands)
        return F.softmax(logits, dim=1)[:, 1].tolist()
