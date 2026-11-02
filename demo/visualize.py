"""Run held-out EEG through the real worker code + trained CNN, then rebuild the website in site/.

Usage (repo root):  .venv/Scripts/python demo/visualize.py [--chunks 16] [--seed 5] [--open]
Real: preprocessing (normalize + FFT), batching, gRPC client/server, CNN, alerting.
Stand-ins (no Docker): in-memory queue for Kafka, a validation function for the NestJS gateway.
"""
from __future__ import annotations

import argparse
import logging
import queue
import sys
import time
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "demo"), str(ROOT / "services" / "ml-server"), str(ROOT / "services" / "worker"), str(ROOT / "simulator")]

from run_demo import generate_stubs, validate  # noqa: E402
import build_site  # noqa: E402
import motion  # noqa: E402

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks", type=int, default=16)
    ap.add_argument("--seed", type=int, default=5)
    ap.add_argument("--open", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.WARNING)
    generate_stubs()

    import numpy as np
    from app.server import build_server
    from simulate_patient import Replay
    from worker.alerting import build_alert, dispatch_alert
    from worker.batcher import Batcher
    from worker.config import Config
    from worker.inference_client import InferenceClient
    from worker.preprocessing import preprocess

    cfg, rng = Config(), np.random.default_rng(args.seed)
    server = build_server("50552"); server.start()
    client = InferenceClient("localhost:50552")
    replay, kafka, batcher = Replay(ROOT, rng), queue.Queue(), Batcher(4, 300)
    cards: dict[int, dict] = {}
    model_version = None

    def flush() -> None:
        nonlocal model_version
        resp = client.predict_batch(batcher.drain())
        model_version = resp.model_version
        for p in resp.predictions:
            cards[p.timestamp].update(conf=p.confidence, alert=p.anomaly_detected)
            if p.anomaly_detected:
                dispatch_alert(build_alert(p.patient_id, p.timestamp, p.confidence, resp.model_version))

    base = int(time.time() * 1000)
    seizure_flags = [True, False, False, True] + [bool(rng.random() < 0.3) for _ in range(args.chunks - 4)]
    for i, seizure in enumerate(seizure_flags):
        ts = base + i
        payload = {"patientId": "patient-001", "timestamp": ts, "signalVector": replay.next_chunk(seizure)}
        validate(payload); kafka.put(payload); raw = kafka.get()
        sig, bands = preprocess(raw["signalVector"], cfg.sampling_rate_hz, cfg.norm_window)
        cards[ts] = {"i": i + 1, "truth": seizure, "raw": raw["signalVector"], "bands": bands.tolist()}
        batcher.add((raw["patientId"], ts, sig, bands))
        if batcher.should_flush() or i == len(seizure_flags) - 1:
            flush()
    import json
    meta = json.loads((ROOT / "models" / "meta.json").read_text())
    client.close(); server.stop(0)

    ordered = sorted(cards.values(), key=lambda c: c["i"])
    page = build_site.write_site(ROOT, ordered, meta, model_version, motion.build_data(ROOT))
    flagged = sum(c["truth"] and c["alert"] for c in ordered); seizures = sum(c["truth"] for c in ordered)
    false_alarms = sum(not c["truth"] and c["alert"] for c in ordered)
    print(f"wrote {page}  ({flagged}/{seizures} seizure epochs flagged, {false_alarms} false alarms)")
    if args.open:
        webbrowser.open(page.as_uri())


if __name__ == "__main__":
    main()
