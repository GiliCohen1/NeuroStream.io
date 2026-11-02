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
    return f'<svg viewBox="0 0 {w} {h}" preserveAspectRatio="none"><polyline points="{pts}" fill="none" stroke="{color}" stroke-width="1.4"/></svg>'


def svg_bands(b: list[float], color: str) -> str:
    m = max(max(b), 1e-6)
    bars = "".join(
        f'<rect x="{i * 22 + 2}" y="{34 - b[i] / m * 28:.1f}" width="16" height="{b[i] / m * 28:.1f}" fill="{color}" opacity=".75"/>'
        f'<text x="{i * 22 + 10}" y="46" text-anchor="middle" font-size="9" fill="currentColor" opacity=".6">{BANDS[i]}</text>'
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
        color = "#d6334a" if c["alert"] else "#2f9e6b"
        verdict = "🚨 ALERT" if c["alert"] else "normal"
        note = "" if ok else ('<span class="flag">missed seizure</span>' if c["truth"] else '<span class="flag">false alarm</span>')
        card_html.append(f'''<div class="card {'alert' if c['alert'] else ''}">
 <div class="row"><b>Chunk {c['i']}</b><span class="truth {'s' if c['truth'] else ''}">ground truth: {'SEIZURE' if c['truth'] else 'normal'}</span></div>
 {svg_wave(c['raw'], color)}
 <div class="row"><div>{svg_bands(c['bands'], color)}</div>
  <div class="verdict" style="color:{color}">{verdict}<small>seizure probability {c['conf']:.3f}</small>{note}</div></div></div>''')

    stages = ["Simulator<small>replays real EEG</small>", "NestJS gateway<small>validates, 202</small>", "Kafka<small>neuro-raw-signals</small>",
              "Worker<small>normalize + FFT, batch</small>", "gRPC", "1-D CNN<small>trained on Bonn EEG</small>", "Alert<small>on-call neurologist</small>"]
    flow = '<span class="arrow">→</span>'.join(f'<div class="stage">{s}</div>' for s in stages)
    page = f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>NeuroStream.IO demo</title><style>
:root{{--bg:#f6f7f9;--card:#fff;--fg:#1b2430;--mut:#6b7686;--bd:#e2e6ec}}
@media(prefers-color-scheme:dark){{:root{{--bg:#12161c;--card:#1a2029;--fg:#e6ebf2;--mut:#93a0b2;--bd:#2a3340}}}}
body{{margin:0;padding:24px 16px;background:var(--bg);color:var(--fg);font:15px/1.45 system-ui,sans-serif}}
main{{max-width:1100px;margin:auto}}h1{{margin:0 0 4px}}p.sub{{color:var(--mut);margin:0 0 20px}}
.flow{{display:flex;flex-wrap:wrap;align-items:center;gap:6px;margin-bottom:20px}}
.stage{{background:var(--card);border:1px solid var(--bd);border-radius:8px;padding:8px 12px;font-weight:600}}
.stage small{{display:block;font-weight:400;color:var(--mut);font-size:12px}}.arrow{{color:var(--mut)}}
.stats{{display:flex;flex-wrap:wrap;gap:12px;margin-bottom:20px}}
.stat{{background:var(--card);border:1px solid var(--bd);border-radius:8px;padding:10px 16px}}.stat b{{font-size:22px;display:block}}
.grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(300px,1fr));gap:12px}}
.card{{background:var(--card);border:1px solid var(--bd);border-radius:10px;padding:12px}}
.card.alert{{border-color:#d6334a;box-shadow:0 0 0 1px #d6334a33}}
.card svg{{width:100%}}.row{{display:flex;justify-content:space-between;align-items:center;gap:8px}}
.truth{{font-size:12px;color:var(--mut)}}.truth.s{{color:#d6334a;font-weight:600}}
.verdict{{text-align:right;font-weight:700}}.verdict small{{display:block;font-weight:400;color:var(--mut)}}
.flag{{display:inline-block;background:#f5a62333;color:#b36b00;border-radius:4px;padding:0 6px;font-size:12px;font-weight:600}}
{story.CSS}
footer{{color:var(--mut);font-size:13px;margin-top:20px}}</style></head><body><main>
<h1>NeuroStream.IO</h1><p class="sub">A digital watcher that listens to brain signals and rings an alarm when it sees a seizure.</p>
{story.render_html()}
<h2 class="live">See it working</h2><p class="sub">{len(ordered)} one-second chunks of <b>real held-out EEG</b> (never seen in training) through the real worker + trained model {html.escape(str(model_version))}. Alert threshold {meta['threshold']:.2f}.</p>
<div class="flow">{flow}</div>
<div class="stats"><div class="stat"><b>{tp}/{tp + fn}</b>seizures caught</div><div class="stat"><b>{fp}</b>false alarms</div><div class="stat"><b>{tn}</b>normals passed</div>
<div class="stat"><b>{meta['metrics']['test']['recall']:.1%}</b>recall on full held-out test</div><div class="stat"><b>{meta['metrics']['test']['precision']:.1%}</b>precision on full held-out test</div></div>
<div class="grid">{''.join(card_html)}</div>
<footer>Waveform = raw EEG chunk. Bars = relative power in δ θ α β γ bands (the FFT features the model also sees). In this no-Docker view Kafka and the gateway are stand-ins; the Docker/CI run uses the real ones. Research benchmark data, not a clinical device.</footer>
</main></body></html>'''
    out = ROOT / "demo" / "report.html"
    out.write_text(page, encoding="utf-8")
    print(f"wrote {out}  (caught {tp}/{tp + fn} seizures, {fp} false alarms)")
    if args.open:
        webbrowser.open(out.as_uri())


if __name__ == "__main__":
    main()
