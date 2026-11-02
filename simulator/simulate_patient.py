"""Synthetic EEG-like telemetry simulator.

Streams 1-second chunks (default 256 samples @ 256 Hz) of a 10 Hz alpha-like sine wave
plus noise to the gateway. With probability --spike-prob a chunk gets a high-amplitude
spike to mimic anomalous neurological activity.
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
import requests

log = logging.getLogger("simulator")


def make_chunk(rng: np.random.Generator, fs: int, seconds: float, spike: bool, t0: float) -> list[float]:
    n = int(fs * seconds)
    t = t0 + np.arange(n) / fs
    x = 20.0 * np.sin(2 * np.pi * 10.0 * t) + 5.0 * np.sin(2 * np.pi * 4.0 * t) + rng.normal(0, 2.0, n)
    if spike:
        start = int(rng.integers(n // 4, 3 * n // 4))
        x[start:start + 3] += rng.choice([-1, 1]) * rng.uniform(150, 250)
    return np.round(x, 3).tolist()


class Replay:
    """Replays real Bonn EEG recordings the model was never trained on (see models/heldout_test_files.json)."""

    WIN = 178  # 1 second at 173.61 Hz

    def __init__(self, root: Path, rng: np.random.Generator):
        self.rng = rng
        files = json.loads((root / "models" / "heldout_test_files.json").read_text())
        load = lambda f: np.loadtxt(root / "training" / "data" / f["file"])
        self.normal = [load(f) for f in files if not f["seizure"]]
        self.seizure = [load(f) for f in files if f["seizure"]]

    def next_chunk(self, seizure: bool) -> list[float]:
        rec = (self.seizure if seizure else self.normal)[int(self.rng.integers(len(self.seizure if seizure else self.normal)))]
        start = int(self.rng.integers(0, len(rec) - self.WIN))
        return rec[start:start + self.WIN].tolist()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:3000/api/v1/telemetry/stream")
    ap.add_argument("--patient-id", default="patient-001")
    ap.add_argument("--fs", type=int, default=256)
    ap.add_argument("--chunk-seconds", type=float, default=1.0)
    ap.add_argument("--interval", type=float, default=1.0, help="seconds between POSTs")
    ap.add_argument("--spike-prob", type=float, default=0.2)
    ap.add_argument("--count", type=int, default=0, help="0 = run forever")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--replay", action="store_true", help="replay real held-out Bonn EEG instead of synthetic waves")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    rng = np.random.default_rng(args.seed)
    session = requests.Session()
    replay = Replay(Path(__file__).resolve().parents[1], rng) if args.replay else None
    sent, t0 = 0, 0.0
    try:
        while args.count == 0 or sent < args.count:
            spike = bool(rng.random() < args.spike_prob)
            payload = {
                "patientId": args.patient_id,
                "timestamp": int(time.time() * 1000),
                "signalVector": replay.next_chunk(spike) if replay else make_chunk(rng, args.fs, args.chunk_seconds, spike, t0),
            }
            t0 += args.chunk_seconds
            try:
                r = session.post(args.url, json=payload, timeout=5)
                log.info("sent chunk #%d seizure_ground_truth=%s -> HTTP %d", sent + 1, spike, r.status_code)
            except requests.RequestException as exc:
                log.error("gateway unreachable: %s", exc)
            sent += 1
            time.sleep(args.interval)
    except KeyboardInterrupt:
        log.info("stopped after %d chunks", sent)


if __name__ == "__main__":
    main()
