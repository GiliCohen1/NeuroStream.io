"""Run held-out EEG through the real worker code + trained CNN and write demo/report.html.

Usage (repo root):  .venv/Scripts/python demo/visualize.py [--chunks 16] [--seed 5] [--open]
Real: preprocessing (normalize + FFT), batching, gRPC client/server, CNN, alerting.
Stand-ins (no Docker): in-memory queue for Kafka, a validation function for the NestJS gateway.
"""
from __future__ import annotations

import argparse
import html
import logging
import queue
import sys
import time
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "demo"), str(ROOT / "services" / "ml-server"), str(ROOT / "services" / "worker"), str(ROOT / "simulator")]

from run_demo import generate_stubs, validate  # noqa: E402
import story  # noqa: E402

BANDS = ["δ", "θ", "α", "β", "γ"]


def svg_wave(sig: list[float], color: str, w: int = 360, h: int = 70) -> str:
    lo, hi = min(sig), max(sig)
    span = (hi - lo) or 1.0
    pts = " ".join(f"{i / (len(sig) - 1) * w:.1f},{h - 4 - (v - lo) / span * (h - 8):.1f}" for i, v in enumerate(sig))
    return f'<svg class="w" viewBox="0 0 {w} {h}" preserveAspectRatio="none"><polyline points="{pts}" fill="none" stroke="{color}" stroke-width="1.4"/></svg>'


def svg_bands(b: list[float], color: str) -> str:
    m = max(max(b), 1e-6)
    bars = "".join(
        f'<rect x="{i * 22 + 2}" y="{34 - b[i] / m * 28:.1f}" width="16" height="{b[i] / m * 28:.1f}" fill="{color}" opacity=".75"/>'
        f'<text x="{i * 22 + 10}" y="46" text-anchor="middle" font-size="9" fill="#5b6b7b">{BANDS[i]}</text>'
        for i in range(5))
    return f'<svg viewBox="0 0 112 50" width="112" height="50">{bars}</svg>'


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
    tp = sum(c["truth"] and c["alert"] for c in ordered); fn = sum(c["truth"] and not c["alert"] for c in ordered)
    fp = sum(not c["truth"] and c["alert"] for c in ordered); tn = sum(not c["truth"] and not c["alert"] for c in ordered)

    card_html = []
    for c in ordered:
        ok = c["truth"] == c["alert"]
        color = "#a8322d" if c["alert"] else "#13808b"
        verdict = f'<div class="verdict a">{story.icon("bell", 18)}ALERT' if c["alert"] else '<div class="verdict n">No alert'
        note = "" if ok else ('<span class="flag">Missed seizure</span>' if c["truth"] else '<span class="flag">False alarm</span>')
        card_html.append(f'''<div class="ep {'alert' if c['alert'] else ''}">
 <div class="row"><b>Epoch {c['i']}</b><span class="truth {'s' if c['truth'] else ''}">Ground truth: {'seizure' if c['truth'] else 'seizure-free'}</span></div>
 {svg_wave(c['raw'], color)}
 <div class="row"><div>{svg_bands(c['bands'], color)}</div>
  {verdict}<small>P(seizure) {c['conf']:.3f}</small>{note}</div></div></div>''')

    stages = [("Simulated bedside monitor", "replays held-out EEG"), ("Ingestion gateway", "validates input"), ("Message queue", "Kafka"),
              ("Preprocessing worker", "normalize, FFT, batch"), ("Inference service", "1D CNN over gRPC"), ("Alert", "notify care team")]
    chev = story.icon("chevron", 18)
    flow = chev.join(f'<div class="stage">{html.escape(a)}<small>{html.escape(b)}</small></div>' for a, b in stages)
    page = f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>NeuroStream seizure detection</title><style>{story.CSS}</style></head><body>
{story.render_top_html()}
<div class="wrap"><section class="block"><h2 class="live">See it working</h2>
<p class="sub">{len(ordered)} one-second epochs of real held-out EEG, never seen in training, through the worker and the trained model ({html.escape(str(model_version))}). Alert threshold {meta['threshold']:.2f}. Kafka and the gateway are stand-ins in this local view; the automated pipeline runs the real ones.</p>
<div class="flow">{flow}</div>
<div class="sumrow"><div class="sum"><b>{tp}/{tp + fn}</b>seizures flagged</div><div class="sum"><b>{fp}</b>false alarms</div><div class="sum"><b>{tn}</b>seizure-free epochs passed</div></div>
<div class="grid">{''.join(card_html)}</div>
<p class="note">Waveform: raw EEG epoch. Bars: relative power in the delta, theta, alpha, beta and gamma bands, which the model also receives.</p></section></div>
{story.render_footer_html()}
</body></html>'''
    out = ROOT / "demo" / "report.html"
    out.write_text(page, encoding="utf-8")
    print(f"wrote {out}  (caught {tp}/{tp + fn} seizures, {fp} false alarms)")
    if args.open:
        webbrowser.open(out.as_uri())


if __name__ == "__main__":
    main()
