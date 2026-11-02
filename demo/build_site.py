"""Generate the dynamic parts of the website (site/): data.js, headline figures and the live-epoch cards.

site/index.html, css/ and js/ are hand-authored source. This script only fills the regions between
<!--GEN:name--> ... <!--/GEN:name--> markers and writes site/js/data.js from the real model outputs.
"""
from __future__ import annotations

import html
import json
import re
from pathlib import Path

BANDS = ["δ", "θ", "α", "β", "γ"]
TEAL, CORAL, MUTED = "#0f7c86", "#b8402e", "#5b6b7b"


def svg_wave(sig: list[float], color: str, w: int = 360, h: int = 64) -> str:
    lo, hi = min(sig), max(sig)
    span = (hi - lo) or 1.0
    pts = " ".join(f"{i / (len(sig) - 1) * w:.1f},{h - 4 - (v - lo) / span * (h - 8):.1f}" for i, v in enumerate(sig))
    return (f'<svg class="w" viewBox="0 0 {w} {h}" preserveAspectRatio="none" role="img" aria-label="EEG epoch waveform">'
            f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="1.4"/></svg>')


def svg_bands(b: list[float], color: str) -> str:
    m = max(max(b), 1e-6)
    bars = "".join(
        f'<rect x="{i * 22 + 2}" y="{34 - b[i] / m * 28:.1f}" width="16" height="{b[i] / m * 28:.1f}" fill="{color}" opacity=".8"/>'
        f'<text x="{i * 22 + 10}" y="46" text-anchor="middle" font-size="9" fill="{MUTED}">{BANDS[i]}</text>' for i in range(5))
    return f'<svg viewBox="0 0 112 50" width="112" height="50" role="img" aria-label="Relative band power">{bars}</svg>'


def epoch_cards(cards: list[dict]) -> str:
    out = []
    for c in cards:
        color = CORAL if c["alert"] else TEAL
        if c["alert"]:
            verdict = '<div class="verdict a"><svg class="i sm ic" aria-hidden="true"><use href="#i-bell"/></svg>ALERT'
        else:
            verdict = '<div class="verdict n">No alert'
        flag = "" if c["truth"] == c["alert"] else (
            '<span class="flag">Missed seizure</span>' if c["truth"] else '<span class="flag">False alarm</span>')
        out.append(
            f'<div class="ep{" alert" if c["alert"] else ""}">'
            f'<div class="row"><b>Epoch {c["i"]}</b><span class="truth{" s" if c["truth"] else ""}">Ground truth: {"seizure" if c["truth"] else "seizure-free"}</span></div>'
            f'{svg_wave(c["raw"], color)}'
            f'<div class="row"><div>{svg_bands(c["bands"], color)}</div>{verdict}<small>P(seizure) {c["conf"]:.3f}</small>{flag}</div></div></div>')
    return '<div class="epochs">' + "".join(out) + "</div>"


def figures(meta: dict) -> str:
    t = meta["metrics"]["test"]
    n = sum(t["confusion"].values())
    items = [
        (t["accuracy"] * 100, 1, "%", "Accuracy", f"{n:,} held-out epochs"),
        (t["recall"] * 100, 1, "%", "Sensitivity", "seizure epochs flagged"),
        (t["precision"] * 100, 1, "%", "Precision", "alerts that were real"),
        (t["auc"], 3, "", "ROC AUC", "separation of the two classes"),
    ]
    cells = "".join(
        f'<div class="fig"><b data-count="{v:.{d}f}" data-dec="{d}" data-suffix="{suf}">{v:.{d}f}{suf}</b><span>{html.escape(lab)}</span><small>{html.escape(sub)}</small></div>'
        for v, d, suf, lab, sub in items)
    return (f'<div class="figures"><div class="wrap">{cells}</div>'
            '<div class="wrap figures-note">Scored once on 75 recordings the model never saw during training. Benchmark data, not clinical evidence.</div></div>')


def inject(page: str, name: str, content: str) -> str:
    pat = re.compile(rf"(<!--GEN:{name}-->)(.*?)(<!--/GEN:{name}-->)", re.S)
    if not pat.search(page):
        raise RuntimeError(f"marker GEN:{name} missing from site/index.html")
    return pat.sub(lambda m: f"{m.group(1)}\n{content}\n{m.group(3)}", page)


def write_site(root: Path, cards: list[dict], meta: dict, model_version: str, data: dict) -> Path:
    site = root / "site"
    (site / "js" / "data.js").write_text("window.NS_DATA=" + json.dumps(data, separators=(",", ":")) + ";\n", encoding="utf-8")
    tp = sum(c["truth"] and c["alert"] for c in cards); fn = sum(c["truth"] and not c["alert"] for c in cards)
    fp = sum(not c["truth"] and c["alert"] for c in cards); tn = sum(not c["truth"] and not c["alert"] for c in cards)
    live_meta = (f'{len(cards)} one-second epochs of real held-out EEG, never seen in training, pass through the worker&rsquo;s preprocessing '
                 f'and the trained model ({html.escape(model_version)}) at alert threshold {meta["threshold"]:.2f}. In this local run Kafka and the '
                 'gateway are stand-ins; the CI pipeline runs the real ones.')
    sums = (f'<div class="sumrow"><div class="sum"><b>{tp}/{tp + fn}</b>seizure epochs flagged</div>'
            f'<div class="sum"><b>{fp}</b>false alarms</div><div class="sum"><b>{tn}</b>seizure-free epochs passed</div></div>')
    page_path = site / "index.html"
    page = page_path.read_text(encoding="utf-8")
    page = inject(page, "figures", figures(meta))
    page = inject(page, "livemeta", live_meta)
    page = inject(page, "epochs", sums + epoch_cards(cards))
    page_path.write_text(page, encoding="utf-8", newline="\n")
    return page_path
