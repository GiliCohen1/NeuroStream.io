"""Signal preprocessing: sliding-window normalization + FFT band powers."""
from __future__ import annotations

import numpy as np
from scipy.ndimage import uniform_filter1d

# Classical EEG bands (Hz): delta, theta, alpha, beta, gamma
EEG_BANDS = ((0.5, 4.0), (4.0, 8.0), (8.0, 13.0), (13.0, 30.0), (30.0, 100.0))
_EPS = 1e-8


def sliding_window_normalize(signal: np.ndarray, window: int = 64) -> np.ndarray:
    """Local z-score using a centred moving mean/std (vectorised, O(n)).

    Note: the window is clamped to the signal length. A spike inflates its own
    local std, so the window must be wide enough for the spike to stand out.
    """
    x = np.asarray(signal, dtype=np.float64)
    if x.ndim != 1 or x.size == 0:
        raise ValueError("signal must be a non-empty 1-D array")
    if not np.all(np.isfinite(x)):
        raise ValueError("signal contains NaN or infinite values")
    w = max(2, min(window, x.size))
    mean = uniform_filter1d(x, size=w, mode="reflect")
    var = np.maximum(uniform_filter1d(x * x, size=w, mode="reflect") - mean**2, 0.0)
    return ((x - mean) / np.sqrt(var + _EPS)).astype(np.float32)


def band_powers(signal: np.ndarray, fs: float) -> np.ndarray:
    """Relative power per EEG band from a real FFT of the Hann-windowed signal."""
    x = np.asarray(signal, dtype=np.float64)
    x = (x - x.mean()) * np.hanning(x.size)
    power = np.abs(np.fft.rfft(x)) ** 2
    freqs = np.fft.rfftfreq(x.size, d=1.0 / fs)
    total = power.sum() + _EPS
    out = [power[(freqs >= lo) & (freqs < hi)].sum() / total for lo, hi in EEG_BANDS]
    return np.asarray(out, dtype=np.float32)


def preprocess(raw: list[float], fs: float, window: int = 64) -> tuple[np.ndarray, np.ndarray]:
    """Return (normalized signal, band powers) for one telemetry chunk."""
    x = np.asarray(raw, dtype=np.float64)
    return sliding_window_normalize(x, window), band_powers(x, fs)
