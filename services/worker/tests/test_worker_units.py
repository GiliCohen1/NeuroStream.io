import numpy as np
import pytest

from worker.batcher import Batcher
from worker.preprocessing import band_powers, preprocess, sliding_window_normalize

FS = 256.0


def _sine(n=256, hz=10.0):
    t = np.arange(n) / FS
    return np.sin(2 * np.pi * hz * t)


def test_spike_stands_out_after_normalization():
    x = _sine()
    x[100] += 8.0
    z = sliding_window_normalize(x, window=64)
    assert np.abs(z).argmax() == 100
    assert np.abs(z[100]) > 4.0


def test_clean_sine_has_no_extreme_zscores():
    z = sliding_window_normalize(_sine(), window=64)
    assert np.abs(z).max() < 3.0


def test_band_powers_peak_in_alpha_for_10hz():
    bp = band_powers(_sine(hz=10.0), FS)
    assert bp.argmax() == 2
    assert bp.sum() == pytest.approx(1.0, abs=0.05)


@pytest.mark.parametrize("bad", [[], [float("nan")] * 32])
def test_invalid_signal_rejected(bad):
    with pytest.raises(ValueError):
        preprocess(bad, FS)


def test_batcher_flushes_on_size():
    b = Batcher(max_size=2, max_wait_ms=1000)
    b.add(1)
    assert not b.should_flush()
    b.add(2)
    assert b.should_flush()
    assert b.drain() == [1, 2]
    assert len(b) == 0


def test_batcher_flushes_on_age():
    now = [0.0]
    b = Batcher(max_size=10, max_wait_ms=500, clock=lambda: now[0])
    b.add("a")
    now[0] = 0.4
    assert not b.should_flush()
    now[0] = 0.6
    assert b.should_flush()
