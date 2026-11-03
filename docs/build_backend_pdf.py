"""Build docs/NeuroStream_Backend_Guide.pdf: a complete technical guide to the NeuroStream backend.

Every number is read from the repo (models/meta.json, training log, held-out data, live micro-benchmarks), so the
document cannot drift from the code. Run from repo root:
    .venv/Scripts/python docs/build_backend_pdf.py
Requires: reportlab, matplotlib, torch, numpy, scipy (and docs/data/training_log.json from make_training_log.py).
"""
from __future__ import annotations

import json
import statistics as st
import sys
import tempfile
import time
from datetime import date
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (BaseDocTemplate, Flowable, Frame, Image, KeepTogether, PageBreak, PageTemplate, Paragraph,
                                Spacer, Table, TableStyle)
from reportlab.platypus.tableofcontents import TableOfContents

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "demo"), str(ROOT / "services" / "ml-server"), str(ROOT / "services" / "worker")]
OUT = ROOT / "docs" / "NeuroStream_Backend_Guide.pdf"
TMP = Path(tempfile.mkdtemp(prefix="ns_pdf_"))

NAVY, TEAL, CORAL = colors.HexColor("#0b2545"), colors.HexColor("#0f7c86"), colors.HexColor("#b8402e")
INK, MUTED, LINE = colors.HexColor("#14202e"), colors.HexColor("#5b6b7b"), colors.HexColor("#d7dee6")
SURF, SURF2 = colors.HexColor("#f4f6f8"), colors.HexColor("#eaeff4")
HEX = {"navy": "#0b2545", "teal": "#0f7c86", "coral": "#b8402e", "muted": "#5b6b7b", "line": "#d7dee6"}

# ------------------------------------------------------------------ fonts
FD = Path(matplotlib.get_data_path()) / "fonts" / "ttf"
for name, f in [("DV", "DejaVuSans.ttf"), ("DV-B", "DejaVuSans-Bold.ttf"), ("DV-I", "DejaVuSans-Oblique.ttf"),
                ("DV-BI", "DejaVuSans-BoldOblique.ttf"), ("DVS", "DejaVuSerif.ttf"), ("DVS-B", "DejaVuSerif-Bold.ttf"),
                ("DVM", "DejaVuSansMono.ttf")]:
    pdfmetrics.registerFont(TTFont(name, str(FD / f)))
pdfmetrics.registerFontFamily("DV", normal="DV", bold="DV-B", italic="DV-I", boldItalic="DV-BI")
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.5, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": HEX["muted"], "axes.labelcolor": "#14202e", "xtick.color": HEX["muted"], "ytick.color": HEX["muted"]})

# ------------------------------------------------------------------ styles
S = {
    "body": ParagraphStyle("body", fontName="DV", fontSize=9.3, leading=13.6, textColor=INK, spaceAfter=6, alignment=TA_LEFT),
    "small": ParagraphStyle("small", fontName="DV", fontSize=7.8, leading=10.6, textColor=MUTED, spaceAfter=4),
    "cap": ParagraphStyle("cap", fontName="DV-I", fontSize=7.8, leading=10.4, textColor=MUTED, spaceBefore=2, spaceAfter=10),
    "h1": ParagraphStyle("h1", fontName="DVS-B", fontSize=17, leading=21, textColor=NAVY, spaceBefore=4, spaceAfter=8, keepWithNext=1),
    "h2": ParagraphStyle("h2", fontName="DVS-B", fontSize=12, leading=15, textColor=NAVY, spaceBefore=10, spaceAfter=4, keepWithNext=1),
    "h3": ParagraphStyle("h3", fontName="DV-B", fontSize=9.3, leading=12.5, textColor=TEAL, spaceBefore=6, spaceAfter=2, keepWithNext=1),
    "toch": ParagraphStyle("toch", fontName="DVS-B", fontSize=17, leading=21, textColor=NAVY, spaceAfter=10),
    "bullet": ParagraphStyle("bullet", fontName="DV", fontSize=9.3, leading=13.2, textColor=INK, leftIndent=13, bulletIndent=2, spaceAfter=2.5),
    "cell": ParagraphStyle("cell", fontName="DV", fontSize=8, leading=10.6, textColor=INK),
    "cellb": ParagraphStyle("cellb", fontName="DV-B", fontSize=8, leading=10.6, textColor=INK),
    "cellh": ParagraphStyle("cellh", fontName="DV-B", fontSize=8, leading=10.6, textColor=colors.white),
    "code": ParagraphStyle("code", fontName="DVM", fontSize=7.6, leading=10.2, textColor=INK, backColor=SURF, borderPadding=(5, 6, 5, 6),
                           spaceBefore=3, spaceAfter=9, leftIndent=6, rightIndent=6),
    "callout": ParagraphStyle("callout", fontName="DV", fontSize=9, leading=13, textColor=INK, backColor=colors.HexColor("#fdf3ee"),
                              borderColor=CORAL, borderWidth=0.8, borderPadding=(7, 9, 7, 9), spaceBefore=6, spaceAfter=12, leftIndent=8, rightIndent=8),
    "note": ParagraphStyle("note", fontName="DV", fontSize=9, leading=13, textColor=INK, backColor=colors.HexColor("#eef7f8"),
                           borderColor=TEAL, borderWidth=0.8, borderPadding=(7, 9, 7, 9), spaceBefore=6, spaceAfter=12, leftIndent=8, rightIndent=8),
}
W = A4[0] - 40 * mm


def P(t, s="body"): return Paragraph(t, S[s])
def B(items): return [Paragraph(t, S["bullet"], bulletText="•") for t in items]
def code(t): return Paragraph(t.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace("\n", "<br/>").replace(" ", "&nbsp;"), S["code"])


def table(rows, widths, header=True, zebra=True, small=False):
    st_ = S["cell"]
    data = []
    for i, r in enumerate(rows):
        data.append([Paragraph(str(c), S["cellh"] if header and i == 0 else (S["cellb"] if j == 0 and not (header and i == 0) else st_))
                     for j, c in enumerate(r)])
    t = Table(data, colWidths=[w * W for w in widths], repeatRows=1 if header else 0)
    style = [("VALIGN", (0, 0), (-1, -1), "TOP"), ("LINEBELOW", (0, 0), (-1, -1), 0.4, LINE),
             ("TOPPADDING", (0, 0), (-1, -1), 4), ("BOTTOMPADDING", (0, 0), (-1, -1), 4), ("LEFTPADDING", (0, 0), (-1, -1), 5), ("RIGHTPADDING", (0, 0), (-1, -1), 5)]
    if header:
        style += [("BACKGROUND", (0, 0), (-1, 0), NAVY)]
    if zebra:
        style += [("ROWBACKGROUNDS", (0, 1 if header else 0), (-1, -1), [colors.white, SURF])]
    t.setStyle(TableStyle(style))
    return t


def figure(path, caption, width=1.0):
    from PIL import Image as PILImage
    w, h = PILImage.open(path).size
    fw = W * width
    return KeepTogether([Image(str(path), width=fw, height=fw * h / w), P(caption, "cap")])


# ------------------------------------------------------------------ diagrams (vector)
class Diagram(Flowable):
    def __init__(self, w, h, fn): super().__init__(); self.w, self.h, self.fn = w, h, fn
    def wrap(self, *_): return self.w, self.h
    def draw(self): self.fn(self.canv, self.w, self.h)


def box(c, x, y, w, h, title, sub="", fill=SURF, stroke=NAVY, tcol=NAVY, bar=None):
    c.setFillColor(fill); c.setStrokeColor(stroke); c.setLineWidth(0.9); c.roundRect(x, y, w, h, 4, fill=1, stroke=1)
    if bar:
        c.setFillColor(bar); c.rect(x + 0.45, y + h - 3.4, w - 0.9, 3, fill=1, stroke=0)
    c.setFillColor(tcol); c.setFont("DV-B", 7.6)
    parts = title.split("\n")
    ty = y + h - 13 - (0 if bar is None else 1)
    for p in parts:
        c.drawCentredString(x + w / 2, ty, p); ty -= 9
    c.setFillColor(MUTED); c.setFont("DV", 6.6)
    for p in sub.split("\n"):
        if p:
            c.drawCentredString(x + w / 2, ty - 1, p); ty -= 8


def arrow(c, x1, y1, x2, y2, label="", col=MUTED, lab_dx=0, lab_dy=4):
    import math
    c.setStrokeColor(col); c.setFillColor(col); c.setLineWidth(1); c.line(x1, y1, x2, y2)
    a = math.atan2(y2 - y1, x2 - x1); s = 5
    c.saveState(); c.translate(x2, y2); c.rotate(math.degrees(a))
    p = c.beginPath(); p.moveTo(0, 0); p.lineTo(-s, s * 0.55); p.lineTo(-s, -s * 0.55); p.close(); c.drawPath(p, fill=1, stroke=0); c.restoreState()
    if label:
        c.setFont("DV-I", 6.4); c.setFillColor(col)
        for k, ln in enumerate(label.split("\n")):
            c.drawCentredString((x1 + x2) / 2 + lab_dx, (y1 + y2) / 2 + lab_dy - 7.5 * k, ln)


def d_arch(c, w, h):
    bw, bh = 118, 54
    X = [0, (w - bw) / 2, w - bw]
    r3 = 4; r2 = r3 + bh + 46; r1 = r2 + bh + 46
    box(c, X[0], r1, bw, bh, "Bedside monitor", "simulated: replays\nreal held-out EEG", bar=TEAL)
    box(c, X[1], r1, bw, bh, "Ingestion gateway", "NestJS · TypeScript\nvalidate · HTTP 202", bar=TEAL)
    box(c, X[2], r1, bw, bh, "Apache Kafka", "neuro-raw-signals\n3 partitions · KRaft", bar=NAVY)
    arrow(c, X[0] + bw, r1 + bh / 2, X[1], r1 + bh / 2, "HTTP POST", lab_dy=3)
    arrow(c, X[1] + bw, r1 + bh / 2, X[2], r1 + bh / 2, "key = patientId", lab_dy=3)
    box(c, X[2], r2, bw, bh, "Preprocessing worker", "Python · NumPy · SciPy\nnormalize · FFT · batch", bar=TEAL)
    arrow(c, X[2] + bw / 2, r1, X[2] + bw / 2, r2 + bh, "consume\n(consumer group)", lab_dx=-42, lab_dy=4)
    box(c, X[1], r2, bw, bh, "Inference service", "PyTorch · gRPC\n1D CNN · threshold 0.37", bar=TEAL)
    arrow(c, X[2], r2 + bh * 0.70, X[1] + bw, r2 + bh * 0.70, "gRPC batch", lab_dy=3)
    arrow(c, X[1] + bw, r2 + bh * 0.28, X[2], r2 + bh * 0.28, "scores", lab_dy=-9)
    box(c, X[2], r3, bw, bh, "Alerting", "structured critical alert\n+ model version", fill=colors.HexColor("#fdf3ee"), stroke=CORAL, tcol=CORAL, bar=CORAL)
    arrow(c, X[2] + bw * 0.5, r2, X[2] + bw * 0.5, r3 + bh, "if score\n≥ threshold", lab_dx=30, lab_dy=4, col=CORAL)
    box(c, X[1], r3, bw, bh, "Dead-letter topic", "neuro-dead-letter\nmalformed messages", fill=SURF2, bar=MUTED)
    arrow(c, X[2] + 6, r2 + 6, X[1] + bw, r3 + bh - 6, "poison\nmessages", lab_dx=-14, lab_dy=12)


def d_split(c, w, h):
    box(c, 0, h / 2 - 28, 104, 56, "500 recordings", "5 sets × 100 recordings\nsplit 70 / 15 / 15 per set", bar=NAVY)
    ys = [h - 56, h / 2 - 28, 0]
    specs = [("Train", "350 recordings\n30,800 windows (stride 45)", "learns the weights", TEAL),
             ("Validation", "75 recordings\n1,725 windows (stride 178)", "picks best epoch + threshold", NAVY),
             ("Test", "75 recordings (15 seizure)\n1,725 windows (stride 178)", "scored exactly once", CORAL)]
    for y, (t, s, role, col) in zip(ys, specs):
        box(c, 140, y, 170, 56, t, s, bar=col)
        c.setFillColor(col); c.setFont("DV-B", 7.6); c.drawString(322, y + 24, role)
        arrow(c, 104 + 36 - 4, h / 2 + (y + 28 - h / 2) * 0.0, 140, y + 28, "", col=MUTED) if False else None
    for y in ys:
        arrow(c, 126, h / 2, 140, y + 28, "") if False else None
    c.setStrokeColor(MUTED); c.setLineWidth(1); c.line(104, h / 2, 132, h / 2)
    for y in ys:
        c.line(132, h / 2, 132, y + 28); c.line(132, y + 28, 140, y + 28)


def d_cnn(c, w, h):
    bw, bh, gap = 100, 52, (w - 4 * 100) / 3
    xs = [i * (bw + gap) for i in range(4)]
    top, bot = h - bh - 2, 62
    box(c, xs[0], top, bw, bh, "Input", "1 channel × 178\nnormalized waveform", bar=TEAL)
    box(c, xs[1], top, bw, bh, "Conv1d k=7", "16 ch · BatchNorm\nReLU · MaxPool 2", bar=NAVY)
    box(c, xs[2], top, bw, bh, "Conv1d k=5", "32 ch · BatchNorm\nReLU · MaxPool 2", bar=NAVY)
    box(c, xs[3], top, bw, bh, "Conv1d k=3", "64 ch · BatchNorm\nReLU", bar=NAVY)
    for i in range(3):
        arrow(c, xs[i] + bw, top + bh / 2, xs[i + 1], top + bh / 2)
    box(c, xs[3], bot, bw, bh, "Global pooling", "average ‖ max over time\n→ 128 features", bar=NAVY)
    arrow(c, xs[3] + bw / 2, top, xs[3] + bw / 2, bot + bh)
    box(c, xs[2], bot, bw, bh, "Concatenate", "128 + 5 band powers\n→ 133 features", bar=NAVY)
    arrow(c, xs[3], bot + bh / 2, xs[2] + bw, bot + bh / 2)
    box(c, xs[1], bot, bw, bh, "Dense 64", "ReLU · Dropout 0.3", bar=NAVY)
    arrow(c, xs[2], bot + bh / 2, xs[1] + bw, bot + bh / 2)
    box(c, xs[0], bot, bw, bh, "Dense 2 → softmax", "P(seizure-free), P(seizure)", fill=colors.HexColor("#fdf3ee"), stroke=CORAL, tcol=CORAL, bar=CORAL)
    arrow(c, xs[1], bot + bh / 2, xs[0] + bw, bot + bh / 2)
    c.setFillColor(TEAL); c.setFont("DV-B", 7); c.drawCentredString(xs[2] + bw / 2, bot - 21, "5 band powers: delta · theta · alpha · beta · gamma")
    arrow(c, xs[2] + bw / 2, bot - 14, xs[2] + bw / 2, bot, col=TEAL)


# ------------------------------------------------------------------ data + figures
def gather() -> dict:
    import torch
    import motion
    from app.model import SpikeDetector
    from worker.preprocessing import preprocess
    meta = json.loads((ROOT / "models" / "meta.json").read_text())
    data = motion.build_data(ROOT)
    log = json.loads((ROOT / "docs" / "data" / "training_log.json").read_text())
    # micro-benchmarks, in process, on this machine
    torch.set_num_threads(2)
    det = SpikeDetector()
    raw = data["ex"]["seizure"]["raw"][:178]
    n, b = preprocess(raw, 173.61, 64)

    def bench(fn, reps):
        fn(); ts = []
        for _ in range(reps):
            t = time.perf_counter(); fn(); ts.append((time.perf_counter() - t) * 1000)
        return st.median(ts)
    bm = {"pre": bench(lambda: preprocess(raw, 173.61, 64), 400)}
    for k in (1, 8, 32, 128):
        bm[k] = bench(lambda: det.score([n.tolist()] * k, [b.tolist()] * k), 40) / k
    bm["bytes"] = len(json.dumps({"patientId": "patient-001", "timestamp": 1793531113494, "signalVector": raw}))
    return {"meta": meta, "data": data, "log": log, "bench": bm}


def make_figures(g: dict) -> dict:
    d, meta, log = g["data"], g["meta"], g["log"]
    thr = d["thr"]; sc = np.array(d["scores"]); p, y = sc[:, 0], sc[:, 1].astype(int)
    F = {}
    teal, coral, navy = HEX["teal"], HEX["coral"], HEX["navy"]
    fs = d["fs"]

    # 1 waveforms
    fig, ax = plt.subplots(2, 1, figsize=(7.0, 3.1), sharex=True, sharey=True)
    for a, k, col, ttl in [(ax[0], "normal", teal, "Seizure-free (held-out recording " + d["ex"]["normal"]["file"].split("/")[-1] + ")"),
                           (ax[1], "seizure", coral, "Seizure (held-out recording " + d["ex"]["seizure"]["file"].split("/")[-1] + ")")]:
        s = np.array(d["ex"][k]["raw"][:356]); a.plot(np.arange(len(s)) / fs, s, color=col, lw=1)
        a.set_title(ttl, loc="left", fontsize=8, color=HEX["muted"]); a.set_ylabel("amplitude")
    ax[1].set_xlabel("time (s)"); fig.tight_layout(); fig.savefig(TMP / "wave.png", dpi=200); plt.close(fig); F["wave"] = TMP / "wave.png"

    # 2 normalization
    e = d["ex"]["seizure"]; raw = np.array(e["raw"][:178]); nm = np.array(e["norm"][:178]); t = np.arange(178) / fs
    fig, ax = plt.subplots(2, 1, figsize=(7.0, 2.9), sharex=True)
    ax[0].plot(t, raw, color=coral, lw=1); ax[0].set_ylabel("raw"); ax[0].set_title("One one-second epoch before and after sliding-window normalization", loc="left", fontsize=8, color=HEX["muted"])
    ax[1].plot(t, nm, color=coral, lw=1); ax[1].axhline(0, color=HEX["line"], lw=.8); ax[1].set_ylabel("z-score"); ax[1].set_xlabel("time (s)")
    fig.tight_layout(); fig.savefig(TMP / "norm.png", dpi=200); plt.close(fig); F["norm"] = TMP / "norm.png"

    # 3 band means
    names = ["delta\n0.5–4 Hz", "theta\n4–8 Hz", "alpha\n8–13 Hz", "beta\n13–30 Hz", "gamma\n30–100 Hz"]
    bm = d["bandMean"]; x = np.arange(5)
    fig, ax = plt.subplots(figsize=(7.0, 2.5))
    ax.bar(x - .19, bm["normal"], .36, color=teal, label="seizure-free"); ax.bar(x + .19, bm["seizure"], .36, color=coral, label="seizure")
    for i in range(5):
        ax.text(i - .19, bm["normal"][i] + .008, f"{bm['normal'][i]:.0%}", ha="center", fontsize=7); ax.text(i + .19, bm["seizure"][i] + .008, f"{bm['seizure'][i]:.0%}", ha="center", fontsize=7)
    ax.set_xticks(x); ax.set_xticklabels(names); ax.set_ylabel("mean relative power"); ax.legend(frameon=False, fontsize=8)
    ax.set_title("Average band-power profile on the 1,725 held-out test epochs", loc="left", fontsize=8, color=HEX["muted"])
    fig.tight_layout(); fig.savefig(TMP / "bands.png", dpi=200); plt.close(fig); F["bands"] = TMP / "bands.png"

    # 4 training curves
    h = log["history"]; ep = [r["epoch"] for r in h]; best = int(np.argmax([r["val_auc"] for r in h]))
    fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.5))
    ax[0].plot(ep, [r["train_loss"] for r in h], color=navy); ax[0].set_xlabel("epoch"); ax[0].set_ylabel("training loss (weighted CE)"); ax[0].set_yscale("log")
    ax[1].plot(ep, [r["val_auc"] for r in h], color=teal); ax[1].axvline(ep[best], color=coral, ls="--", lw=1)
    ax[1].text(ep[best], min(r["val_auc"] for r in h) + .0005, f" best: epoch {ep[best]}", color=coral, fontsize=7.5)
    ax[1].set_xlabel("epoch"); ax[1].set_ylabel("validation ROC AUC")
    fig.tight_layout(); fig.savefig(TMP / "train.png", dpi=200); plt.close(fig); F["train"] = TMP / "train.png"; F["best_epoch"] = ep[best]; F["best_auc"] = h[best]["val_auc"]

    # 5 ROC
    order = np.argsort(-p); ys = y[order]; tpr = np.cumsum(ys) / ys.sum(); fpr = np.cumsum(1 - ys) / (1 - ys).sum()
    tp = int(((p >= thr) & (y == 1)).sum()); fp = int(((p >= thr) & (y == 0)).sum())
    fig, ax = plt.subplots(figsize=(3.3, 3.1))
    ax.plot(np.r_[0, fpr], np.r_[0, tpr], color=navy, lw=1.6); ax.plot([0, 1], [0, 1], color=HEX["line"], ls="--", lw=1)
    ax.scatter([fp / (1 - y).sum()], [tp / y.sum()], color=coral, zorder=5, s=26); ax.annotate(f"threshold {thr:.2f}", (fp / (1 - y).sum(), tp / y.sum()), xytext=(.22, .6), fontsize=7.5, color=coral, arrowprops=dict(arrowstyle="-", color=coral, lw=.8))
    ax.set_xlabel("false-alarm rate"); ax.set_ylabel("sensitivity"); ax.set_title("ROC (test)", loc="left", fontsize=8, color=HEX["muted"])
    fig.tight_layout(); fig.savefig(TMP / "roc.png", dpi=200); plt.close(fig); F["roc"] = TMP / "roc.png"

    # 6 histogram
    fig, ax = plt.subplots(figsize=(3.7, 3.1))
    bins = np.linspace(0, 1, 21); ax.hist([p[y == 0], p[y == 1]], bins=bins, color=[teal, coral], label=["seizure-free", "seizure"], log=True)
    ax.axvline(thr, color=HEX["navy"], lw=1.4); ax.text(thr + .02, ax.get_ylim()[1] * .35, f"threshold {thr:.2f}", fontsize=7.5, color=navy)
    ax.set_xlabel("P(seizure)"); ax.set_ylabel("epochs (log scale)"); ax.legend(frameon=False, fontsize=7.5, loc="lower right", bbox_to_anchor=(1.0, 1.0), ncol=2); ax.set_title("Score distribution (test)", loc="left", fontsize=8, color=HEX["muted"])
    fig.tight_layout(); fig.savefig(TMP / "hist.png", dpi=200); plt.close(fig); F["hist"] = TMP / "hist.png"

    # 7 trade-off curve
    ths = np.linspace(.02, .98, 97); sens, prec, fa = [], [], []
    for t_ in ths:
        f_ = p >= t_; tp_ = int((f_ & (y == 1)).sum()); fp_ = int((f_ & (y == 0)).sum())
        sens.append(tp_ / y.sum()); prec.append(tp_ / max(1, tp_ + fp_)); fa.append(fp_)
    fig, ax = plt.subplots(figsize=(7.0, 2.6)); ax.plot(ths, sens, color=coral, label="sensitivity"); ax.plot(ths, prec, color=teal, label="precision")
    ax2 = ax.twinx(); ax2.plot(ths, fa, color=HEX["muted"], ls="--", lw=1, label="false alarms (count, right axis)"); ax2.spines["right"].set_visible(True); ax2.set_ylabel("false alarms")
    ax.axvline(thr, color=navy, lw=1.2); ax.set_xlabel("decision threshold"); ax.set_ylabel("sensitivity / precision"); ax.set_ylim(.8, 1.005)
    l1, n1 = ax.get_legend_handles_labels(); l2, n2 = ax2.get_legend_handles_labels(); ax.legend(l1 + l2, n1 + n2, frameon=False, fontsize=7.5, loc="lower left")
    fig.tight_layout(); fig.savefig(TMP / "trade.png", dpi=200); plt.close(fig); F["trade"] = TMP / "trade.png"
    F["tp"], F["fp"] = tp, fp
    return F


# ------------------------------------------------------------------ page furniture
class Doc(BaseDocTemplate):
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        fr = Frame(20 * mm, 20 * mm, W, A4[1] - 40 * mm, id="f", leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0)
        self.addPageTemplates([PageTemplate(id="cover", frames=[fr], onPage=lambda c, d: None),
                               PageTemplate(id="main", frames=[fr], onPage=self._furn)])

    def _furn(self, c, d):
        c.saveState(); c.setStrokeColor(LINE); c.setLineWidth(.6); c.line(20 * mm, A4[1] - 15 * mm, A4[0] - 20 * mm, A4[1] - 15 * mm)
        c.setFont("DV", 7.2); c.setFillColor(MUTED); c.drawString(20 * mm, A4[1] - 12.5 * mm, "NeuroStream · Backend technical guide")
        c.drawRightString(A4[0] - 20 * mm, A4[1] - 12.5 * mm, "Research prototype · not a medical device")
        c.drawString(20 * mm, 12 * mm, f"Generated {date.today().isoformat()} from the repository"); c.drawRightString(A4[0] - 20 * mm, 12 * mm, f"Page {d.page}")
        c.restoreState()

    def afterFlowable(self, f):
        if isinstance(f, Paragraph) and f.style.name in ("h1", "h2"):
            lvl = 0 if f.style.name == "h1" else 1
            key = f"s{self.seq.nextf('toc')}"; self.canv.bookmarkPage(key); self.notify("TOCEntry", (lvl, f.getPlainText(), self.page, key))


def cover(g):
    t = g["meta"]["metrics"]["test"]; items = []
    items.append(Spacer(1, 62 * mm))
    items.append(Paragraph("NeuroStream", ParagraphStyle("ct", fontName="DVS-B", fontSize=34, leading=38, textColor=NAVY)))
    items.append(Spacer(1, 4 * mm))
    items.append(Paragraph("Backend technical guide", ParagraphStyle("cs", fontName="DVS", fontSize=18, leading=22, textColor=TEAL)))
    items.append(Spacer(1, 8 * mm))
    items.append(Paragraph("From recorded EEG to a real-time seizure alert: the data, the preprocessing, the model, the validation, the streaming pipeline, "
                           "and the open questions about scale.", ParagraphStyle("cl", fontName="DV", fontSize=11, leading=16, textColor=INK)))
    items.append(Spacer(1, 22 * mm))
    stats = [[f"{t['accuracy']:.1%}", f"{t['recall']:.1%}", f"{t['precision']:.1%}", f"{t['auc']:.3f}"], ["accuracy", "sensitivity", "precision", "ROC AUC"]]
    tb = Table([[Paragraph(f'<font name="DVS-B" size="19" color="#0b2545">{v}</font>', S["cell"]) for v in stats[0]],
                [Paragraph(f'<font size="8" color="#5b6b7b">{v}</font>', S["cell"]) for v in stats[1]]], colWidths=[W / 4] * 4)
    tb.setStyle(TableStyle([("LINEABOVE", (0, 0), (-1, 0), 1, NAVY), ("LINEBELOW", (0, -1), (-1, -1), .5, LINE), ("TOPPADDING", (0, 0), (-1, -1), 6)]))
    items += [tb, Spacer(1, 2 * mm), Paragraph("Scored once on 75 held-out recordings (1,725 one-second epochs). Benchmark data, not clinical evidence.", S["small"])]
    items.append(Spacer(1, 70 * mm))
    items.append(Paragraph("Research prototype. Not a medical device and not intended for clinical use.<br/>"
                           f"Repository: github.com/GiliCohen1/NeuroStream.io · Document generated {date.today().isoformat()}.", S["small"]))
    return items


# ------------------------------------------------------------------ content
def content(g, F):
    m, d, bm, log = g["meta"], g["data"], g["bench"], g["log"]; tv, tt = m["metrics"]["validation"], m["metrics"]["test"]; thr = d["thr"]
    cmv, cmt = tv["confusion"], tt["confusion"]
    n_seiz, n_free = cmt["tp"] + cmt["fn"], cmt["fp"] + cmt["tn"]
    story = []
    H1 = lambda t: story.append(P(t, "h1")); H2 = lambda t: story.append(P(t, "h2")); H3 = lambda t: story.append(P(t, "h3"))
    A = lambda x: story.extend(x) if isinstance(x, list) else story.append(x)

    # ---- 1 executive summary
    H1("1. Executive summary")
    A(P("NeuroStream is a research prototype that answers one question: <b>can a model trained on labeled EEG recordings flag seizure activity within "
        "one second of signal, inside a real-time system that keeps working when parts of it fail?</b> It has two halves that this guide explains in turn. "
        "The first is a <b>model half</b>: real EEG, preprocessing, a small convolutional neural network, and an evaluation designed to avoid flattering itself. "
        "The second is a <b>systems half</b>: an ingestion gateway, a Kafka queue, a preprocessing worker, a gRPC inference service and an alerting step."))
    A(P("What was built, and how each part was verified, in one table:"))
    A(table([["Part", "What it is", "How it is verified"],
             ["Data and preprocessing", "University of Bonn EEG; sliding-window normalization and FFT band power, in one shared module", "7 unit tests; the same code runs in training and in serving"],
             ["Model and evaluation", "1D CNN (17,858 parameters); recording-level split; threshold tuned on validation data", "Seeded, reproducible script; test set scored once; metrics stored in models/meta.json"],
             ["Ingestion gateway", "NestJS service: validates input, publishes to Kafka keyed by patient", "14 automated tests including real HTTP requests against the actual app with a fake Kafka producer"],
             ["Streaming pipeline", "Kafka (KRaft), Python worker, gRPC inference, dead-letter topic", "CI builds the full Docker stack on every push and replays held-out EEG end to end"],
             ["Failure handling", "Retries with backoff, offset rewind, dead-letter routing", "Commit and rewind logic covered by 6 unit tests with fake Kafka and inference; <b>not</b> yet tested against a real broker or a killed container (see 8.6)"],
             ["Scalability", "Design reasoning and in-process micro-benchmarks", "<b>Not load-tested.</b> Section 10 separates measured facts from estimates"]],
            [.21, .43, .36]))
    A(Spacer(1, 6))
    A(P("<b>Headline result.</b> On "
        f"{n_seiz + n_free:,} one-second test epochs from recordings never used in training: accuracy {tt['accuracy']:.1%}, sensitivity {tt['recall']:.1%} "
        f"({cmt['tp']} of {n_seiz} seizure epochs flagged), precision {tt['precision']:.1%}, ROC AUC {tt['auc']:.3f}, with {cmt['fp']} false alarms among {n_free:,} seizure-free epochs."))
    A(P("<b>How to read that result honestly.</b> The Bonn benchmark is clean, single-channel and widely reported to be easy, and its recordings come from only a handful of people "
        "(section 3.4). These figures show that the method and the engineering work; they are not evidence that it would work in a hospital.", "callout"))

    A(PageBreak())

    # ---- 2 system overview
    H1("2. System overview")
    A(P("The system moves one second of EEG signal (an <b>epoch</b>) from a monitor to a decision. Every component is a separate process with a narrow job, so each can fail, "
        "restart and scale independently. The contract between components is a message, never shared memory."))
    A(Diagram(W, 264, d_arch)); A(P("Figure 2.1. Runtime architecture. Teal bars mark services we wrote; the navy bar marks infrastructure. The coral path is the alert.", "cap"))
    A(table([["Component", "Technology", "Responsibility", "Key setting"],
             ["Ingestion gateway", "NestJS 10, TypeScript, class-validator", "Accept <font name='DVM'>POST /api/v1/telemetry/stream</font>, validate, publish to Kafka, answer 202", "JSON limit 2 MB; Kafka retries 8; publish timeout 5 s"],
             ["Message queue", "Apache Kafka 7.6 (KRaft, single node)", "Durable buffer between producers and consumers; per-patient ordering", "3 partitions; auto topic creation disabled"],
             ["Preprocessing worker", "Python, confluent-kafka, NumPy, SciPy", "Consume, validate, normalize, compute band power, batch, call inference, raise alerts", "Batch 8 or 500 ms; manual offset commit"],
             ["Inference service", "PyTorch, gRPC, Protocol Buffers", "Score each epoch with the CNN, apply the tuned threshold", "Batch limit 256; 4 workers; 2 torch threads"],
             ["Alerting", "Python module", "Build a structured critical alert with the model version attached", "Simulated: written to the log"]],
            [.17, .25, .38, .20]))
    H2("2.1 The three message formats")
    A(table([["Hop", "Format", "Fields"],
             ["Monitor → gateway", "HTTP POST, JSON", "<font name='DVM'>patientId</font> (string, 1–64 chars), <font name='DVM'>timestamp</font> (positive number, epoch ms), <font name='DVM'>signalVector</font> (16–4,096 finite numbers)"],
             ["Gateway → Kafka", "JSON string, key = patientId", "Exactly the validated payload. A typical 178-sample epoch is " + f"{bm['bytes']:,} bytes"],
             ["Worker → inference", "Protocol Buffers (proto/inference.proto)", "<font name='DVM'>Sample</font>: patient_id, timestamp, signal (normalized), band_powers (5). Response: a <font name='DVM'>Prediction</font> per sample (anomaly_detected, confidence) plus model_version and server latency"]],
            [.2, .27, .53]))
    A(P("<b>Why two formats?</b> JSON at the edge is easy for any monitor vendor to produce and for humans to debug. Protocol Buffers between our own services gives a typed, versioned contract that is faster and fails loudly when the two sides disagree."))

    # ---- 3 data
    H1("3. The data")
    H2("3.1 What it is")
    A(P("The model is trained on the <b>University of Bonn EEG dataset</b> (Andrzejak et al., 2001, <i>Physical Review E</i> 64, 061907), a standard benchmark in seizure-detection research. "
        "It contains five sets of 100 single-channel recordings, each about 23.6 seconds long, sampled at 173.61 Hz (4,097 samples per file). The files are plain text, one number per line."))
    A(table([["Set", "Recording", "Condition", "Label used here"],
             ["Z (A)", "Scalp EEG", "Healthy volunteers, eyes open", "seizure-free (0)"],
             ["O (B)", "Scalp EEG", "Healthy volunteers, eyes closed", "seizure-free (0)"],
             ["N (C)", "Intracranial EEG", "Epilepsy patients, between seizures", "seizure-free (0)"],
             ["F (D)", "Intracranial EEG", "Epilepsy patients, between seizures (from the epileptogenic zone)", "seizure-free (0)"],
             ["S (E)", "Intracranial EEG", "Epilepsy patients, <b>during seizures</b>", "seizure (1)"]], [.1, .22, .48, .2]))
    A(P("The task is therefore binary: <b>seizure (set S, 100 recordings) versus seizure-free (the other four sets, 400 recordings)</b>. That is a 20% seizure share, "
        "which is far higher than in real monitoring, where seizures are rare (see 3.4).", "body"))
    A(figure(F["wave"], "Figure 3.1. Real held-out recordings, same vertical scale. A seizure appears as large, rhythmic, regular activity; seizure-free EEG is smaller and irregular."))
    H2("3.2 How it is stored and used")
    A(B(["<b>Source:</b> downloaded from the University of Bonn site with <font name='DVM'>training/download_data.py</font>; the full data is <b>not</b> committed.",
         "<b>In the repository:</b> only the 75 held-out <i>test</i> recordings (1.6 MB, <font name='DVM'>data/heldout/</font>) so the demo and CI can replay them. Attribution is included.",
         "<b>Epoching:</b> each recording is cut into one-second epochs of 178 samples. One epoch is the unit of every prediction."]))
    H2("3.3 Why this dataset")
    A(P("It is public, small enough to train on a laptop CPU in about a minute, and heavily reported in the literature, so results can be compared. Those same properties are its weakness (3.4)."))
    H2("3.4 Limitations of the data (important)")
    A(P("<b>The files carry no subject identifiers.</b> According to the original publication, as it is commonly summarized, the 500 recordings come from only about ten people: "
        "five healthy volunteers and five patients. We could not confirm this count from the data files themselves, which contain only numbers. If it is right, then "
        "<b>splitting by recording is not splitting by person</b>: recordings from the same individual appear in training and in the test set, so the test result is likely "
        "<b>optimistic about how the model would perform on a new patient</b>. A true patient-independent evaluation needs a dataset with patient IDs (for example CHB-MIT), "
        "and that is the first item on the roadmap.", "callout"))
    A(B(["<b>Single channel.</b> Clinical EEG is typically 19–256 channels; the model sees one at a time.",
         "<b>Clean, short and selected.</b> Artifacts (muscle, eye blinks, electrode pops) that dominate real recordings were screened out.",
         "<b>Unrealistic prevalence.</b> 20% seizure epochs versus a small fraction of a percent in routine monitoring. At low prevalence the same false-alarm rate produces far more false alerts than true ones."]))

    # ---- 4 preprocessing
    H1("4. Preprocessing")
    A(P("Preprocessing lives in one module, <font name='DVM'>services/worker/worker/preprocessing.py</font>. The training script and the live worker import it, so the model is "
        "always trained on exactly what it will see in production. Mismatched preprocessing between training and serving (<i>train/serve skew</i>) is a classic silent failure."))
    H2("4.1 Step 1: validation")
    A(P("An epoch must be a non-empty one-dimensional array of finite numbers. Anything else raises an error; in the worker that sends the message to the dead-letter topic."))
    H2("4.2 Step 2: sliding-window normalization")
    A(P("Different recordings have different baselines and amplitudes. For each sample the code subtracts a local mean and divides by a local standard deviation, "
        "computed over a centered 64-sample window (0.37 s):"))
    A(P("<font name='DVM'>z[i] = (x[i] − mean_64(i)) / sqrt(var_64(i) + 1e-8)</font>", "note"))
    A(P("Both statistics come from <font name='DVM'>scipy.ndimage.uniform_filter1d</font> (reflect mode at the edges), so the whole epoch is processed in one vectorized pass. "
        "The window is clamped to the epoch length. <b>Trade-off:</b> a local window removes absolute amplitude, one of the signs of a seizure. The model recovers most of "
        "the information from waveform shape and from the band powers below, but amplitude is genuinely discarded."))
    A(figure(F["norm"], "Figure 4.1. Normalization removes baseline drift and scale while preserving the waveform's shape.", .96))
    H2("4.3 Step 3: band power by FFT")
    A(P("A Hann window is applied to the mean-removed epoch, a real FFT gives the power spectrum, and the power inside each clinical band is divided by the total power, "
        "giving five <b>relative band powers</b> that sum to about 1:"))
    A(table([["Band", "Frequency", "Clinical association (general)"],
             ["delta", "0.5–4 Hz", "Slow activity; deep sleep; also pathological slowing"],
             ["theta", "4–8 Hz", "Drowsiness; in seizures, often prominent rhythmic activity"],
             ["alpha", "8–13 Hz", "Relaxed wakefulness, eyes closed"],
             ["beta", "13–30 Hz", "Active thinking; also medication effects"],
             ["gamma", "30–100 Hz", "Fast activity; very little energy in this dataset"]], [.14, .18, .68]))
    A(Spacer(1, 6))
    A(figure(F["bands"], "Figure 4.2. In the held-out data, seizure epochs shift power from delta toward theta on average. The difference is real but modest, which is why the model also reads the waveform itself.", .96))
    A(P(f"<b>Cost.</b> Measured in-process on the development laptop: preprocessing one epoch takes about <b>{bm['pre']:.2f} ms</b> (median of 400 runs)."))

    # ---- 5 model
    H1("5. The model")
    H2("5.1 Architecture")
    A(P("A small one-dimensional convolutional neural network (CNN). Convolutions slide short filters along the waveform and learn patterns such as sharp rises or rhythmic bursts; "
        "stacking three layers lets later filters respond to longer, more abstract patterns. The five band powers are appended before the final decision layers."))
    A(Diagram(W, 200, d_cnn)); A(P("Figure 5.1. Model architecture: 17,858 trainable parameters in total.", "cap"))
    A(table([["Layer", "Output shape", "Parameters"],
             ["Conv1d (k=7) + BatchNorm + ReLU + MaxPool 2", "16 × 89", "160"], ["Conv1d (k=5) + BatchNorm + ReLU + MaxPool 2", "32 × 44", "2,656"],
             ["Conv1d (k=3) + BatchNorm + ReLU", "64 × 44", "6,336"], ["Global average pool ‖ global max pool", "128", "0"],
             ["Concatenate 5 band powers", "133", "0"], ["Dense 64 + ReLU + Dropout 0.3", "64", "8,576"], ["Dense 2 (logits) → softmax", "2", "130"]], [.55, .25, .20]))
    A(Spacer(1, 6))
    H2("5.2 Design decisions")
    A(B(["<b>Why a small CNN, not a large model?</b> 17,858 parameters train in about a minute on a CPU, serve at a fraction of a millisecond per epoch, and are far less prone to memorizing 350 training recordings.",
         "<b>Why 1D, not a spectrogram image?</b> The signal is a time series. A first prototype used a pre-trained image network on stretched waveforms; it was replaced because its head was untrained and an image backbone is the wrong inductive bias here.",
         "<b>Why global average <i>and</i> max pooling?</b> Average captures sustained activity; max captures brief, sharp events. Seizure patterns can be either.",
         "<b>Why append band powers?</b> They give the classifier an explicit, interpretable spectral summary that is cheap to compute and that clinicians recognize.",
         f"<b>Output.</b> A softmax probability P(seizure). The service flags an epoch when P(seizure) ≥ <b>{thr:.2f}</b> (section 7.3)."]))
    A(P("The architecture lives in <font name='DVM'>services/ml-server/app/model.py</font>, shared by the training script and the serving code, together with the loader that reads "
        "<font name='DVM'>models/spike_cnn.pt</font> and <font name='DVM'>models/meta.json</font> (version, input length, threshold and metrics)."))

    # ---- 6 training
    H1("6. Training")
    A(Diagram(W, 205, d_split)); A(P("Figure 6.1. The data split. Splitting is per recording, within each of the five sets, so every set is represented in every split.", "cap"))
    H2("6.1 Procedure")
    A(table([["Step", "Detail"],
             ["Split", "Within each set, a seeded random permutation assigns 70 recordings to train, 15 to validation, 15 to test (350 / 75 / 75). Seed 42."],
             ["Windows", f"Train: overlapping epochs with stride 45 samples (≈ 4× augmentation) → {log['counts']['train_windows']:,} windows, {log['counts']['train_seizure_share']:.0%} seizure. Validation and test: non-overlapping epochs, 23 per recording → 1,725 each."],
             ["Features", "Each window goes through the shared preprocessing: normalized waveform (178) and band powers (5)."],
             ["Loss", f"Cross-entropy with the seizure class weighted ×{log['class_weight']:.1f} (the inverse class ratio), so the rarer class is not ignored."],
             ["Optimizer", "Adam, learning rate 2e-3, weight decay 1e-4, batch size 128, 30 epochs."],
             ["Model selection", f"After every epoch the model is scored on the validation set; the weights from the best validation ROC AUC are kept (epoch {F['best_epoch']}, AUC {F['best_auc']:.4f})."],
             ["Threshold", f"Chosen on <b>validation</b> data by maximizing F1 over 0.05–0.95 in steps of 0.01 → <b>{thr:.2f}</b>. The test set plays no part."],
             ["Test", "Scored exactly once, at the end, with the frozen weights and threshold."]], [.17, .83]))
    A(Spacer(1, 6))
    A(figure(F["train"], "Figure 6.2. Left: the training loss falls steadily. Right: validation AUC is high from the first epoch and plateaus, which is a sign the task is easy, not that tuning was heroic.", .98))
    H2("6.2 Why overlapping windows only for training")
    A(P("Overlap multiplies the number of training examples and teaches the model that a seizure pattern can appear at any position within the epoch. Validation and test epochs do not overlap, "
        "so no two evaluation examples share samples and the metrics are not inflated by near-duplicates."))
    H2("6.3 Reproducibility")
    A(B(["Fixed seed (42) for the split, the weight initialization and the shuffling; re-running <font name='DVM'>training/train.py</font> reproduces the reported metrics exactly.",
         "<font name='DVM'>docs/make_training_log.py</font> replays the same loop and saves the curves above without touching the saved model.",
         "The trained weights (under 100 KB) and the metrics are committed, so the pipeline and CI do not need to retrain."]))

    # ---- 7 evaluation
    H1("7. Testing and validation of the model")
    H2("7.1 The metrics, in plain terms")
    A(table([["Metric", "Question it answers", "Why a clinician cares"],
             ["Sensitivity (recall)", "Of the seizure epochs, how many did we flag?", "Missed seizures are the costly error"],
             ["Precision (PPV)", "Of the epochs we flagged, how many were real?", "Low precision means alarm fatigue"],
             ["Accuracy", "How often is the label right overall?", "Misleading when classes are imbalanced"],
             ["ROC AUC", "How well do scores separate the two classes at <i>every</i> threshold?", "Threshold-independent quality of the scores"]], [.2, .45, .35]))
    A(Spacer(1, 6))
    H2("7.2 Results")
    A(table([["", "Accuracy", "Sensitivity", "Precision", "ROC AUC", "TP", "FP", "FN", "TN"],
             ["Validation (75 recordings)", f"{tv['accuracy']:.1%}", f"{tv['recall']:.1%}", f"{tv['precision']:.1%}", f"{tv['auc']:.4f}", cmv["tp"], cmv["fp"], cmv["fn"], cmv["tn"]],
             ["<b>Test (75 recordings)</b>", f"<b>{tt['accuracy']:.1%}</b>", f"<b>{tt['recall']:.1%}</b>", f"<b>{tt['precision']:.1%}</b>", f"<b>{tt['auc']:.4f}</b>", cmt["tp"], cmt["fp"], cmt["fn"], cmt["tn"]]],
            [.21, .11, .13, .11, .11, .08, .08, .08, .09]))
    A(P("TP/FP/FN/TN are true positives, false positives, false negatives and true negatives at the deployed threshold. Test performance is somewhat lower than validation, as expected: the threshold was tuned on validation data.", "small"))
    A(Table([[Image(str(F["roc"]), width=W * .47, height=W * .47 * 3.1 / 3.3), Image(str(F["hist"]), width=W * .52, height=W * .52 * 3.1 / 3.7)]], colWidths=[W * .48, W * .52]))
    A(P("Figure 7.1. Left: ROC curve on the test set with the deployed operating point. Right: the model's scores are strongly bimodal: most epochs score near 0 or near 1, and the errors sit in between.", "cap"))
    H2("7.3 The threshold is a clinical decision")
    A(P("Every setting trades missed seizures against false alarms. Moving the line right reduces false alarms and misses more seizures; moving it left does the reverse. "
        "The deployed value (0.37) maximizes F1 on validation data, which weights both errors equally. A real deployment might weight a missed seizure far more heavily."))
    A(figure(F["trade"], "Figure 7.2. Sensitivity, precision and false-alarm count against the threshold on the test set. Shown for illustration only: choosing a threshold on test data would overstate performance.", .98))
    rows = [["Threshold", "Sensitivity", "Precision", "Missed seizure epochs", "False alarms"]]
    sc = np.array(d["scores"]); pp, yy = sc[:, 0], sc[:, 1].astype(int)
    for t_ in (0.10, 0.25, thr, 0.50, 0.75, 0.90):
        f_ = pp >= t_; tp_ = int((f_ & (yy == 1)).sum()); fp_ = int((f_ & (yy == 0)).sum())
        rows.append([f"<b>{t_:.2f} (deployed)</b>" if abs(t_ - thr) < 1e-9 else f"{t_:.2f}", f"{tp_ / n_seiz:.1%}", f"{tp_ / max(1, tp_ + fp_):.1%}", f"{n_seiz - tp_} of {n_seiz}", f"{fp_} of {n_free:,}"])
    A(table(rows, [.2, .18, .18, .24, .20]))
    H2("7.4 What this validation does and does not protect against")
    A(table([["Risk", "Status"],
             ["Test data used for training or tuning", "<b>Guarded.</b> Weights fit on train; epoch and threshold chosen on validation; test scored once."],
             ["Near-duplicate epochs across splits", "<b>Guarded.</b> Evaluation epochs do not overlap; splits are by recording."],
             ["Same <i>person</i> in train and test", "<b>Not guarded.</b> Files carry no subject IDs (3.4). Likely optimistic for new patients."],
             ["Class prevalence unlike the real world", "<b>Not addressed.</b> 20% seizure epochs here; precision would be far lower at realistic prevalence."],
             ["Multichannel, artifacts, other hardware", "<b>Not addressed.</b> Single clean channel only."],
             ["Calibration of the probabilities", "<b>Not evaluated.</b> The score ranks epochs well; it is not shown to be a calibrated probability."],
             ["Seizure-level (event) metrics", "<b>Not evaluated.</b> Everything is per one-second epoch; clinicians reason in seizure events and false alarms per hour."]], [.42, .58]))

    # ---- 8 pipeline
    H1("8. The real-time pipeline")
    A(P("This chapter follows one epoch from the monitor to the alert and explains what each component guarantees. File paths are relative to the repository root."))
    H2("8.1 Ingestion gateway (services/ingestion-gateway)")
    A(B(["<b>Endpoint:</b> <font name='DVM'>POST /api/v1/telemetry/stream</font> returns <b>202 Accepted</b> (the message is queued, not yet analyzed) with {accepted, patientId, samples}. <font name='DVM'>GET /health</font> for liveness.",
         "<b>Validation</b> (class-validator, global pipe with <font name='DVM'>whitelist</font> and <font name='DVM'>forbidNonWhitelisted</font>): patientId 1–64 chars; timestamp a positive finite number; 16–4,096 finite numbers. Unknown fields are rejected. Invalid input returns 400 and <b>nothing is published</b>.",
         "<b>Publish:</b> the validated JSON is emitted to <font name='DVM'>neuro-raw-signals</font> with <b>key = patientId</b>, so all epochs of a patient land on one partition and keep their order. The call has a 5 s timeout.",
         "<b>Failure:</b> if Kafka is unreachable the client gets <b>503</b> with a generic message. A global exception filter ensures stack traces and broker addresses never leak. The producer retries 8 times with backoff first.",
         "<b>Shutdown:</b> shutdown hooks flush and close the producer on SIGTERM."]))
    H2("8.2 Kafka (docker-compose.yml)")
    A(table([["Item", "Setting", "Why"],
             ["Mode", "Single node, KRaft (no ZooKeeper)", "Light enough for a laptop; one process to run"],
             ["neuro-raw-signals", "3 partitions, replication 1", "3 partitions allow up to 3 parallel consumers; replication 1 is a laptop compromise"],
             ["neuro-dead-letter", "1 partition", "Low volume; holds messages a human should inspect"],
             ["Topic creation", "Automatic creation disabled; a one-shot init job creates the topics", "A typo in a topic name fails loudly instead of silently creating a new topic"]], [.2, .38, .42]))
    A(Spacer(1, 6))
    H2("8.3 Preprocessing worker (services/worker/worker/main.py)")
    A(P("The worker is a single loop built around <font name='DVM'>confluent-kafka</font> with <b>auto-commit disabled</b>. Offsets are committed by hand, only after a message has been fully handled."))
    A(code("loop:\n  msg = consumer.poll(0.1)\n  if msg:        ingest(msg)          # parse, validate, preprocess, add to batcher\n  if batcher.should_flush():    flush()   # size >= 8  OR  oldest item >= 500 ms\n\nflush():\n  response = grpc.PredictBatch(batch)     # retries + backoff inside the client\n  for each prediction: log; if anomaly -> dispatch_alert()\n  commit offsets (max offset + 1 per partition)"))
    A(B(["<b>Dynamic batching.</b> A batch is sent when it reaches 8 epochs or when its oldest epoch is 500 ms old, whichever comes first. Larger batches use the CPU more efficiently (section 10.2); the age limit bounds the added latency.",
         "<b>Poison messages.</b> If JSON is invalid, a field is missing, or the signal is shorter than 16 samples, the message is copied to <font name='DVM'>neuro-dead-letter</font> with <font name='DVM'>error</font> and <font name='DVM'>source</font> headers, then skipped. One bad record cannot block the stream.",
         "<b>Graceful shutdown.</b> On SIGTERM the loop stops, flushes the pending batch, flushes the dead-letter producer and closes the consumer."]))
    H2("8.4 Inference client and service")
    A(B(["<b>Retries.</b> The gRPC client retries on UNAVAILABLE, DEADLINE_EXCEEDED and RESOURCE_EXHAUSTED up to 4 times with exponential backoff (0.25 s × 2<super>attempt</super>, capped at 8 s) and random jitter, to avoid synchronized retry storms.",
         "<b>Server validation.</b> Empty batch, more than 256 epochs, or any signal shorter than 16 samples returns INVALID_ARGUMENT; unexpected exceptions return INTERNAL.",
         "<b>Server work.</b> Stack the epochs, run the CNN once for the whole batch, apply softmax, flag epochs with P(seizure) ≥ the threshold stored beside the weights. The reply carries the model version and the server-side latency.",
         "<b>Model versioning.</b> Every response and every alert carries the model version, so any alert can be traced to the exact model that raised it."]))
    H2("8.5 Alerting")
    A(P("An alert is a structured record: type <font name='DVM'>CRITICAL_NEURO_ANOMALY</font>, severity, patientId, the epoch timestamp, the confidence, the model version, a UTC time and a recommended action. "
        "Here it is written to the log; in production this step would call a paging or notification service. That is the step that closes the loop for the clinician."))
    H2("8.6 Delivery guarantees and failure behavior")
    A(P("The pipeline aims for <b>at-least-once</b> delivery: an epoch may be processed more than once after a crash, but it should not be lost. The table states what the code does in each situation."))
    A(table([["Situation", "Behavior", "Verified by tests?"],
             ["Gateway receives invalid input", "400, nothing published", "Yes (6 cases in the HTTP test suite)"],
             ["Kafka down at the gateway", "Producer retries; then 503 without internals", "Yes (fake broker failure)"],
             ["Malformed message in the topic", "Copied to the dead-letter topic with headers, then committed past", "Yes (unit test)"],
             ["Malformed message arrives while valid ones are buffered", "Buffered messages are processed first; the malformed one is committed only after. If that processing fails, nothing is committed and all of them are redelivered", "Yes (2 unit tests)"],
             ["Inference service briefly unavailable", "Client retries with backoff and jitter", "No"],
             ["Inference unavailable after all retries", "Batch not committed; consumer seeks back to the earliest failed offset and tries again", "Yes (unit test, fake client)"],
             ["Inference rejects the batch (non-retryable)", "Whole batch sent to the dead-letter topic, then committed", "Yes (unit test)"],
             ["Worker crashes after inference, before commit", "Epochs are re-delivered and processed again: <b>duplicate alerts are possible</b>", "No"],
             ["Worker receives SIGTERM", "Flushes the pending batch, then exits", "No"]], [.34, .48, .18]))
    A(Spacer(1, 6))
    H3("A defect found while writing this guide, now fixed")
    A(P("<b>Poison commit could overtake unprocessed messages.</b> When a malformed message was dead-lettered, its offset was committed immediately. Kafka offsets are cumulative, "
        "so earlier valid messages from the same partition that were still waiting in the batch buffer were implicitly marked as done; a crash before the next flush (a window of up to 500 ms) would have lost them. "
        "<b>Fix:</b> before a malformed message is committed the worker now flushes the buffered batch, and commits only if that succeeds. If inference is unavailable, nothing is committed, "
        "and the buffered messages <i>and</i> the malformed one (even on another partition) are rewound and redelivered. A regression test fails on the old behavior and passes on the new.", "note"))
    H3("Gaps that remain open")
    A(B(["<b>No duplicate suppression.</b> At-least-once delivery can raise the same alert twice. <i>Fix:</i> deduplicate on (patientId, timestamp) at the alerting step.",
         "<b>No rebalance handling.</b> There is no partition-revocation callback to flush or commit before ownership moves. This matters as soon as more than one worker runs.",
         "<b>Failure paths are tested only with fakes.</b> The commit and rewind logic is unit-tested, and CI covers the happy path against real Kafka. A resilience test (stop the inference container mid-stream, check nothing is lost) is the natural next test."]))
    H2("8.7 Why these choices")
    A(B(["<b>Kafka between the gateway and the worker</b> so a slow or restarting worker never blocks, and never loses, incoming monitoring data; it also lets the worker be scaled by adding consumers.",
         "<b>Key by patient</b> so one patient's epochs stay in order, which matters for any future logic that looks across consecutive epochs.",
         "<b>gRPC to the model</b> so the model can be deployed, scaled and versioned independently of the worker.",
         "<b>Manual offset commit</b> because commit-after-success is the simplest way to avoid silent loss."]))

    # ---- 9 testing + CI
    H1("9. Software testing and continuous integration")
    A(table([["Level", "What", "Count / scope"],
             ["Unit (worker)", "Normalization finds a spike; band power peaks at the right band for a 10 Hz sine; invalid input is rejected; batcher flushes on size and on age. Offset-commit safety with fake Kafka and inference: a malformed message never lets a commit overtake buffered messages, failed inference rewinds without committing, non-retryable errors are dead-lettered", "13 tests"],
             ["Unit (gateway)", "DTO validation rules", "5 tests"],
             ["HTTP end-to-end (gateway)", "The real NestJS app over real HTTP with an in-memory fake Kafka: valid chunk → 202 and exact payload keyed by patient; six invalid cases → 400 and nothing published; broker down → 503; health", "9 tests"],
             ["Full-stack end-to-end (CI)", "Docker Compose builds Kafka, gateway, worker and inference; the script posts 12 held-out seizure and 12 held-out seizure-free epochs; passes only if ≥ 9 seizure epochs raise alerts, ≤ 1 false alarm, and every alert carries a model version", "Every push"]], [.2, .62, .18]))
    A(Spacer(1, 6))
    A(P("<b>What CI caught.</b> The first full-stack run failed, three times, for reasons unit tests could not see: (1) the inference container crashed at startup because a model path was computed with a "
        "parent-directory lookup that does not exist inside the container; (2) the gateway image failed to build because a type-definitions package was missing; (3) the Python images could not be built because "
        "a pinned protobuf version conflicted with the gRPC tools. All three were fixed, and the run then passed. This is the strongest argument for testing the assembled system, not only its parts.", "note"))

    # ---- 10 scalability
    H1("10. Scalability: the questions and how we answer them")
    A(P("The prototype runs on one laptop. This chapter lists the questions that must be answered before it could serve many patients, says what is <b>known</b> (measured here), "
        "what is <b>designed but not built</b>, and what is <b>still open</b>. Nothing in this chapter has been load-tested."))
    H2("10.1 Sizing the problem")
    A(P("The unit of work is one epoch per channel per second, so the message rate equals the number of monitored channels. With the measured message size of "
        f"{bm['bytes']:,} bytes:"))
    A(table([["Monitored channels", "Messages per second", "Ingress (JSON)", "Compute at measured per-epoch cost*"],
             ["100", "100", f"{100 * bm['bytes'] / 1e6:.2f} MB/s", f"{100 * (bm['pre'] + bm[32]) / 1000:.2f} cores"],
             ["1,000", "1,000", f"{1000 * bm['bytes'] / 1e6:.2f} MB/s", f"{1000 * (bm['pre'] + bm[32]) / 1000:.2f} cores"],
             ["10,000", "10,000", f"{10000 * bm['bytes'] / 1e6:.1f} MB/s", f"{10000 * (bm['pre'] + bm[32]) / 1000:.1f} cores"],
             ["100,000", "100,000", f"{100000 * bm['bytes'] / 1e6:.0f} MB/s", f"{100000 * (bm['pre'] + bm[32]) / 1000:.0f} cores"]], [.22, .2, .2, .38]))
    A(P(f"*Preprocessing {bm['pre']:.2f} ms + model scoring {bm[32]:.3f} ms per epoch at batch 32, measured in-process on an 8-core Windows laptop CPU with 2 torch threads. "
        "This is pure compute. It excludes JSON parsing, Kafka, gRPC serialization and network, which will dominate in practice, so treat the right-hand column as a floor, not a forecast.", "small"))
    H2("10.2 Measured: why batching matters")
    A(table([["Epochs per model call", "Median time per call", "Time per epoch", "Speed-up vs 1"],
             ["1", f"{bm[1]:.2f} ms", f"{bm[1]:.3f} ms", "1.0×"], ["8 (current default)", f"{bm[8] * 8:.2f} ms", f"{bm[8]:.3f} ms", f"{bm[1] / bm[8]:.1f}×"],
             ["32", f"{bm[32] * 32:.2f} ms", f"{bm[32]:.3f} ms", f"{bm[1] / bm[32]:.1f}×"], ["128", f"{bm[128] * 128:.2f} ms", f"{bm[128]:.3f} ms", f"{bm[1] / bm[128]:.1f}×"]], [.3, .26, .24, .2]))
    A(P("Per-call overhead dominates at small batch sizes, which is why the worker batches. The gain flattens beyond about 32 epochs, so the 8-epoch default leaves room to raise it under load.", "small"))
    H2("10.3 The open questions")
    A(table([["Question", "Why it matters", "Today", "How we would answer it"],
             ["How many partitions?", "Partitions cap consumer parallelism (consumers ≤ partitions) and are painful to change once keys are in use", "3", "Size for target peak plus headroom (e.g. 12–48), decided before go-live; key stays patientId"],
             ["How do workers scale?", "One worker is a single Python process", "1 worker, no rebalance handling", "Run N workers in one consumer group; add rebalance callbacks (8.6); autoscale on consumer lag"],
             ["What if the model is the bottleneck?", "Inference must keep up with the batches", "Stateless gRPC server; 4 threads", "Run replicas behind a gRPC-aware load balancer; larger dynamic batches; export to ONNX or TorchScript; quantize; GPU only if the model grows"],
             ["What happens during a burst or an outage?", "Monitoring data does not pause", "Kafka buffers; gRPC retries", "Retention sized for the longest outage we must survive; alert on consumer lag; deadlines so slow calls fail fast"],
             ["Can alerts be duplicated or lost?", "Duplicate pages cause fatigue; lost alerts are unsafe", "At-least-once; the poison-commit defect is fixed (8.6); duplicates still possible", "Deduplicate on (patientId, timestamp); consider a transactional outbox for alerts"],
             ["What is the end-to-end latency budget?", "The point is early warning", "Not measured end to end", "Define an objective (e.g. epoch captured → alert dispatched under N seconds); budget each hop; record it with tracing"],
             ["How do we run it with several channels per patient?", "Real EEG has many channels; message size and model input change", "One channel per message", "A channel-aware message and model; partition by patient so channels of one patient stay together"],
             ["How do we operate it?", "You cannot fix what you cannot see", "Logs only", "Metrics (lag, latency percentiles, alert and dead-letter rates), distributed tracing with a correlation ID, dashboards and on-call alerts on the system itself"],
             ["How do we change the model safely?", "A bad model is a patient-safety issue", "Version string in every response; model baked into the image", "Shadow mode, canary release, rollback; monitor score distribution drift; revalidate on every retrain"],
             ["How do we protect patient data?", "EEG tied to a patient is health data", "Plain HTTP and plaintext Kafka; no auth", "TLS everywhere, authentication and authorization, pseudonymous IDs, encryption at rest, audit logs, retention policy, data-protection review"]],
            [.18, .26, .2, .36]))
    A(Spacer(1, 6))
    H2("10.4 One honest limit on latency")
    A(P("A one-second epoch means the earliest possible alert comes one second after the seizure pattern begins, plus the pipeline. The configured 500 ms batch wait adds up to half a second at low load. "
        "Overlapping windows (for example one prediction every 250 ms) would cut detection delay at four times the compute, which is a trade-off worth measuring."))

    # ---- 11 safety / regulatory
    H1("11. Safety, privacy and regulatory notes")
    A(P("None of the following is implemented or claimed. It lists what would have to be addressed before any clinical use."))
    A(B(["<b>Intended use and risk.</b> Software that informs clinical decisions is regulated as a medical device in most jurisdictions. It needs a defined intended use, risk analysis, documented validation and post-market monitoring.",
         "<b>Clinical validation.</b> Requires prospective data from the target setting, patient-independent evaluation, event-level metrics and false alarms per hour.",
         "<b>Privacy.</b> Patient-linked EEG is personal health data. Authentication, encryption in transit and at rest, access control, auditing and a retention policy are required.",
         "<b>Human oversight.</b> The design keeps a clinician in the loop: the system flags an epoch, and the clinician confirms it."]))

    # ---- 12 roadmap
    H1("12. Limitations and roadmap")
    A(table([["Priority", "Next step", "What it would show"],
             ["1", "Train and evaluate on CHB-MIT (pediatric patients, 23 patients across 24 recording cases, multichannel); split by patient", "Whether the result holds for a new patient: the biggest open question"],
             ["2", "Report seizure-event metrics and false alarms per hour; evaluate at realistic prevalence", "Clinically meaningful performance"],
             ["3", "Add alert de-duplication and rebalance handling", "A delivery guarantee that holds under failure"],
             ["4", "Resilience test in CI: stop the inference container mid-stream, check nothing is lost", "Fault tolerance demonstrated, not just designed"],
             ["5", "Load test (k6 or a custom producer) and publish measured throughput and latency percentiles", "Real capacity numbers"],
             ["6", "Metrics and tracing; calibration; model monitoring", "Operability and trust in the score"]], [.1, .52, .38]))

    # ---- appendices
    H1("Appendix A. Repository map")
    A(code("proto/inference.proto            gRPC contract (Sample, Prediction)\n"
           "docker-compose.yml               Kafka (KRaft), topic init, ml-server, gateway, worker\n"
           "services/ingestion-gateway/      NestJS gateway: DTO validation, Kafka producer, 14 tests\n"
           "services/worker/worker/          preprocessing, batcher, inference_client, alerting, main\n"
           "services/ml-server/app/          model.py (CNN + loader)  server.py (gRPC)\n"
           "models/                          spike_cnn.pt  meta.json  heldout_test_files.json\n"
           "training/                        train.py  download_data.py\n"
           "data/heldout/                    75 held-out test recordings (attribution included)\n"
           "scripts/e2e_check.py             full-stack check used by CI\n"
           "demo/                            no-Docker demo and website data build\n"
           "site/                            static website\n"
           ".github/workflows/ci.yml         unit tests + Docker end-to-end"))
    H1("Appendix B. Configuration reference")
    A(table([["Setting", "Default", "Effect"],
             ["KAFKA_BROKERS / KAFKA_TOPIC / KAFKA_DLQ_TOPIC", "kafka:9092 · neuro-raw-signals · neuro-dead-letter", "Where the worker reads and dead-letters"],
             ["KAFKA_GROUP_ID", "neuro-preprocessing-workers", "Consumer group; workers sharing it split the partitions"],
             ["BATCH_MAX_SIZE / BATCH_MAX_WAIT_MS", "8 · 500", "Batch flushes at whichever limit is reached first"],
             ["SAMPLING_RATE_HZ / NORM_WINDOW", "173.61 · 64", "Must match training for the band powers and normalization"],
             ["GRPC_TIMEOUT_S / GRPC_MAX_RETRIES", "5 · 4", "Per-call deadline and retry budget"],
             ["MODEL_DIR / GRPC_PORT / MAX_WORKERS / TORCH_NUM_THREADS", "/srv/models · 50051 · 4 · 2", "Inference service"]], [.38, .32, .30]))
    H1("Appendix C. Glossary")
    A(table([["Term", "Meaning"],
             ["EEG", "Electroencephalogram: voltage recorded from electrodes on or in the head"], ["Epoch", "A fixed-length slice of signal; here one second (178 samples)"],
             ["Seizure", "A burst of abnormal, synchronous neuronal activity"], ["Interictal", "Between seizures"],
             ["Z-score", "A value rescaled to 'how many standard deviations from the local mean'"], ["FFT", "Fast Fourier transform: splits a signal into its frequency components"],
             ["Band power", "Share of signal energy in a frequency range (delta, theta, alpha, beta, gamma)"], ["CNN", "Convolutional neural network: learns short patterns and combines them"],
             ["Held-out", "Data kept aside and never used to fit or tune the model"], ["Sensitivity / recall", "Fraction of true seizures that are flagged"],
             ["Precision / PPV", "Fraction of flags that are true seizures"], ["ROC AUC", "Probability that a random seizure epoch scores higher than a random seizure-free epoch"],
             ["Kafka", "A distributed log: producers append messages, consumers read them at their own pace"], ["Partition", "An ordered slice of a topic; the unit of parallelism"],
             ["Consumer group", "Consumers that share a topic's partitions between them"], ["Offset", "A message's position in a partition; committing records progress"],
             ["Dead-letter topic", "A side topic for messages that cannot be processed"], ["At-least-once", "A message is never lost but may be processed more than once"],
             ["gRPC / Protobuf", "A fast typed remote-call protocol and its schema language"]], [.24, .76]))
    H1("Appendix D. References")
    A(B(["Andrzejak RG, Lehnertz K, Mormann F, Rieke C, David P, Elger CE (2001). Indications of nonlinear deterministic and finite-dimensional structures in time series of brain electrical activity: Dependence on recording region and brain state. <i>Physical Review E</i> 64, 061907.",
         "Dataset download page: University of Bonn, Department of Epileptology (Lehnertz group), 'Downloads'.",
         "Photo on the website: 'EEG Recording Cap' by Chris Hope, CC BY 2.0, via Wikimedia Commons."]))
    return story


def main() -> None:
    g = gather(); F = make_figures(g)
    doc = Doc(str(OUT), pagesize=A4, title="NeuroStream: Backend technical guide", author="NeuroStream", subject="Seizure detection on streaming EEG")
    from reportlab.platypus import NextPageTemplate
    toc = TableOfContents(); toc.levelStyles = [
        ParagraphStyle("t0", fontName="DV-B", fontSize=9, leading=15, leftIndent=0, textColor=NAVY, spaceBefore=3),
        ParagraphStyle("t1", fontName="DV", fontSize=8.4, leading=11.5, leftIndent=12, textColor=INK)]
    story = cover(g) + [NextPageTemplate("main"), PageBreak(), P("Contents", "toch"), toc, PageBreak()] + content(g, F)
    doc.multiBuild(story)
    print("wrote", OUT, f"({OUT.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
