"""The story of NeuroStream, written once and rendered to HTML (dashboard) and Markdown (README).

    .venv/Scripts/python demo/story.py --write-readme     # refreshes the block between the STORY markers in README.md
"""
from __future__ import annotations

import argparse
import base64
import html
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
META = json.loads((ROOT / "models" / "meta.json").read_text(encoding="utf-8"))
T = META["metrics"]["test"]

IMAGE = ROOT / "assets" / "eeg-recording-cap.jpg"
IMAGE_ALT = "A person wearing an EEG cap with electrodes attached across the scalp"
IMAGE_CREDIT = ('Photo: "EEG Recording Cap" by Chris Hope, '
                '<a href="https://commons.wikimedia.org/wiki/File:EEG_Recording_Cap.jpg">CC BY 2.0</a>, via Wikimedia Commons')
IMAGE_CREDIT_MD = ('Photo: "EEG Recording Cap" by Chris Hope, '
                   '[CC BY 2.0](https://commons.wikimedia.org/wiki/File:EEG_Recording_Cap.jpg), via Wikimedia Commons')

EYEBROW = "Clinical neurophysiology · Machine learning research prototype"
TITLE = "Can a system recognize a seizure as it happens?"
LEAD = ("Continuous EEG is how clinicians monitor patients at risk of seizures. Interpreting it is still largely manual, "
        "and some seizures show no sign at the bedside. We set out to test whether a model trained on labeled recordings "
        "can flag seizure activity within one second of signal, and whether it can run inside a real-time system "
        "that keeps working when parts of it fail.")

PROBLEM_TITLE = "The problem"
PROBLEM = [
    ("clock", "Review is retrospective",
     "Trained specialists read EEG, usually after the recording. Hours of signal per patient and a limited number of readers "
     "mean findings can arrive late."),
    ("eye", "Some seizures are silent",
     "Non-convulsive seizures can occur with no visible signs and are recognized only on the EEG."),
    ("timer", "Time matters",
     "Prolonged seizure activity is harmful and treatable. The sooner it is recognized, the sooner a clinician can act."),
]

QUESTION_TITLE = "What we wanted to find out"
QUESTIONS = [
    ("Detection", "How accurately can a model flag seizure activity in one-second epochs from recordings it has never seen?"),
    ("Delivery", "Can that model run in a streaming architecture that is validated, observable, and does not lose data when a component fails?"),
]
HELP_TITLE = "How it could help"
HELP = [
    "Continuous screening between specialist reviews.",
    "Triage: direct reading time to the segments the model flags first.",
    "Earlier notification of the care team.",
]
HELP_NOTE = "Decision support, not a diagnosis. The system flags; a clinician decides."

WORK_TITLE = "What we did"
WORK = [
    ("database", "Data",
     "500 single-channel EEG recordings from the University of Bonn: 100 recorded during seizures, 400 seizure-free "
     "(healthy volunteers and interictal intracranial recordings)."),
    ("shield", "Validation design",
     "75 recordings, 15 of them with seizures, were held out entirely and scored once at the end. Splits are by recording, "
     "so no epoch from a test recording appears in training."),
    ("network", "Model",
     "A 1D convolutional neural network reads the normalized waveform together with relative band power (delta to gamma). "
     "It trained on about 30,000 one-second epochs with class weighting. The alert threshold was tuned on validation data only."),
    ("server", "System",
     "Around the model: an ingestion gateway, a message queue (Kafka), a preprocessing worker with batching and retries, "
     "a gRPC inference service and an alerting step. An automated pipeline rebuilds the stack from scratch and replays "
     "held-out recordings end to end on every change."),
]

RESULTS_TITLE = f"Results on {T['confusion']['tp'] + T['confusion']['fn'] + T['confusion']['fp'] + T['confusion']['tn']:,} held-out epochs"
RESULTS = [
    (f"{T['accuracy']:.1%}", "Accuracy"),
    (f"{T['recall']:.1%}", "Sensitivity"),
    (f"{T['precision']:.1%}", "Precision (PPV)"),
    (f"{T['auc']:.3f}", "ROC AUC"),
]

LIMITS_TITLE = "Limits and next steps"
LIMITS = ("The Bonn recordings are clean, single-channel and short. Clinical EEG is multichannel, noisy and seizure-sparse, "
          "which raises the false-alarm burden, so these figures are not evidence of clinical performance. "
          "Next: multichannel recordings (CHB-MIT), patient-independent validation, and false alarms per hour. "
          "Research prototype, not a medical device.")
DATA_CREDIT = ("Data: Andrzejak et al. (2001), Phys. Rev. E 64, 061907, University of Bonn. "
               "Only the 75 held-out test recordings are included in this repository.")

# Thin-line icons (24px grid, stroke only), in the style of common clinical-UI icon sets.
ICONS = {
    "pulse": '<path d="M2 12h4l2.5-6 4 12 3-8 1.5 2H22"/>',
    "clock": '<circle cx="12" cy="12" r="9"/><path d="M12 7v5l3 2"/>',
    "eye": '<path d="M2 12s3.6-7 10-7 10 7 10 7-3.6 7-10 7S2 12 2 12z"/><circle cx="12" cy="12" r="3"/>',
    "timer": '<circle cx="12" cy="13" r="8"/><path d="M12 9v4l2.5 1.5M9 2h6"/>',
    "database": '<ellipse cx="12" cy="5" rx="8" ry="3"/><path d="M4 5v6c0 1.7 3.6 3 8 3s8-1.3 8-3V5M4 11v6c0 1.7 3.6 3 8 3s8-1.3 8-3v-6"/>',
    "shield": '<path d="M12 3l8 3v6c0 4.5-3.2 8.2-8 9-4.8-.8-8-4.5-8-9V6l8-3z"/><path d="M8.5 12l2.5 2.5 4.5-5"/>',
    "network": '<circle cx="5" cy="6" r="2"/><circle cx="5" cy="18" r="2"/><circle cx="12" cy="12" r="2"/><circle cx="19" cy="12" r="2"/><path d="M7 6.7l3.2 4M7 17.3l3.2-4M14 12h3"/>',
    "server": '<rect x="3" y="4" width="18" height="6" rx="1.5"/><rect x="3" y="14" width="18" height="6" rx="1.5"/><path d="M7 7h.01M7 17h.01"/>',
    "bell": '<path d="M6 8a6 6 0 0112 0c0 7 3 8 3 8H3s3-1 3-8M10 20a2 2 0 004 0"/>',
    "chevron": '<path d="M9 6l6 6-6 6"/>',
}


def icon(name: str, size: int = 24) -> str:
    return (f'<svg class="ic" width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="currentColor" '
            f'stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">{ICONS[name]}</svg>')


CSS = """
:root{--navy:#0b2545;--teal:#13808b;--ink:#1c2733;--mut:#5b6b7b;--line:#d9e0e7;--bg:#f5f7f9;--card:#fff;--alert:#a8322d;color-scheme:light}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font:16px/1.55 "Segoe UI",system-ui,-apple-system,Roboto,sans-serif}
a{color:var(--teal)}
.wrap{max-width:1080px;margin:0 auto;padding:0 20px}
header.top{background:var(--navy);color:#fff}
header.top .wrap{display:flex;align-items:center;justify-content:space-between;height:56px}
.brand{display:flex;align-items:center;gap:10px;font-weight:600;letter-spacing:.2px}.brand .ic{color:#6fd0d8}
.tag{font-size:12.5px;letter-spacing:.8px;text-transform:uppercase;color:#b9c7d8}
.hero{background:#fff;border-bottom:1px solid var(--line)}
.hero .wrap{display:grid;grid-template-columns:1.1fr .9fr;gap:40px;align-items:center;padding-top:44px;padding-bottom:44px}
.eyebrow{font-size:12.5px;letter-spacing:.9px;text-transform:uppercase;color:var(--teal);font-weight:600;margin:0 0 12px}
h1,h2,h3{font-family:Georgia,"Iowan Old Style","Times New Roman",serif;color:var(--navy);font-weight:600}
h1{font-size:38px;line-height:1.15;margin:0 0 16px}.lead{font-size:18px;color:#34475a;margin:0;max-width:60ch}
figure{margin:0}figure img{display:block;width:100%;height:340px;object-fit:cover;object-position:35% 30%;border-radius:6px;border:1px solid var(--line)}
figcaption,.credit{font-size:12px;color:var(--mut);margin-top:8px}
section.block{padding:40px 0 0}
section.block h2{font-size:26px;margin:0 0 16px}
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(240px,1fr));gap:16px}
.c{background:var(--card);border:1px solid var(--line);border-radius:6px;padding:20px}
.c .ic{color:var(--teal);margin-bottom:10px}.c h3{font-size:18px;margin:0 0 6px}.c p{margin:0;color:#3d4f61;font-size:15px}
.q{background:#fff;border:1px solid var(--line);border-left:4px solid var(--teal);border-radius:6px;padding:20px 24px}
.q dl{margin:0;display:grid;grid-template-columns:max-content 1fr;gap:10px 22px}.q dt{font-weight:600;color:var(--navy)}.q dd{margin:0}
.help{margin-top:16px}.help .note{margin:10px 0 0}
.help ul{margin:0;padding-left:20px}.help .c{display:block}.help li{margin-bottom:4px}.note{color:var(--mut);font-size:14px;margin:10px 0 0}
.work{display:grid;grid-template-columns:repeat(auto-fit,minmax(240px,1fr));gap:16px}
.stats{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:12px;margin-top:16px}
.stat{background:var(--navy);color:#fff;border-radius:6px;padding:16px 18px}.stat b{display:block;font:600 30px Georgia,serif}.stat span{font-size:13px;color:#b9c7d8}
.limits{background:#fff;border:1px solid var(--line);border-radius:6px;padding:20px 24px;color:#3d4f61}
h2.live{margin:0 0 6px}.sub{color:var(--mut);margin:0 0 16px;max-width:80ch}
.flow{display:flex;flex-wrap:wrap;align-items:center;gap:4px;margin:0 0 18px}
.stage{background:#fff;border:1px solid var(--line);border-radius:6px;padding:8px 12px;font-weight:600;font-size:14px;color:var(--navy)}
.stage small{display:block;font-weight:400;color:var(--mut);font-size:12px}.flow .ic{color:var(--mut)}
.sumrow{display:flex;flex-wrap:wrap;gap:12px;margin-bottom:18px}
.sum{background:#fff;border:1px solid var(--line);border-radius:6px;padding:10px 16px;font-size:13px;color:var(--mut)}.sum b{display:block;font:600 22px Georgia,serif;color:var(--navy)}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(300px,1fr));gap:12px}
.ep{background:#fff;border:1px solid var(--line);border-radius:6px;padding:12px 14px}
.ep.alert{border-color:var(--alert)}.ep svg.w{width:100%;display:block;margin:6px 0}
.row{display:flex;justify-content:space-between;align-items:center;gap:8px}
.truth{font-size:12px;color:var(--mut)}.truth.s{color:var(--alert);font-weight:600}
.verdict{text-align:right;font-weight:600}.verdict small{display:block}.verdict small{font-weight:400;color:var(--mut)}
.verdict.a{color:var(--alert)}.verdict.n{color:var(--teal)}.verdict .ic{vertical-align:-5px;margin-right:4px}
.flag{background:#f4e4c1;color:#7a5200;border-radius:3px;padding:0 6px;font-size:12px;font-weight:600;margin-top:2px}
footer.page{margin-top:40px;padding:24px 0 40px;border-top:1px solid var(--line);color:var(--mut);font-size:13px}footer.page p{margin:0 0 6px}
@media(max-width:820px){.hero .wrap{grid-template-columns:1fr;gap:24px;padding-top:28px;padding-bottom:28px}h1{font-size:30px}
 figure img{height:260px}.q dl{grid-template-columns:1fr;gap:2px}.tag{display:none}}
"""


def _image_data_uri() -> str:
    return "data:image/jpeg;base64," + base64.b64encode(IMAGE.read_bytes()).decode()


def render_top_html() -> str:
    problem = "".join(f'<div class="c">{icon(i)}<h3>{html.escape(t)}</h3><p>{html.escape(b)}</p></div>' for i, t, b in PROBLEM)
    questions = "".join(f"<dt>{html.escape(k)}</dt><dd>{html.escape(v)}</dd>" for k, v in QUESTIONS)
    helps = "".join(f"<li>{html.escape(x)}</li>" for x in HELP)
    work = "".join(f'<div class="c">{icon(i)}<h3>{html.escape(t)}</h3><p>{html.escape(b)}</p></div>' for i, t, b in WORK)
    stats = "".join(f'<div class="stat"><b>{v}</b><span>{html.escape(k)}</span></div>' for v, k in RESULTS)
    return f'''<header class="top"><div class="wrap"><div class="brand">{icon("pulse", 26)}<span>NeuroStream</span></div>
<span class="tag">Research prototype</span></div></header>
<div class="hero"><div class="wrap"><div><p class="eyebrow">{html.escape(EYEBROW)}</p><h1>{html.escape(TITLE)}</h1><p class="lead">{html.escape(LEAD)}</p></div>
<figure><img src="{_image_data_uri()}" alt="{html.escape(IMAGE_ALT)}"><figcaption>{IMAGE_CREDIT}</figcaption></figure></div></div>
<div class="wrap">
<section class="block"><h2>{html.escape(PROBLEM_TITLE)}</h2><div class="cards">{problem}</div></section>
<section class="block"><h2>{html.escape(QUESTION_TITLE)}</h2><div class="q"><dl>{questions}</dl></div>
<div class="help"><div class="c"><h3>{html.escape(HELP_TITLE)}</h3><ul>{helps}</ul><p class="note">{html.escape(HELP_NOTE)}</p></div></div></section>
<section class="block"><h2>{html.escape(WORK_TITLE)}</h2><div class="work">{work}</div>
<h3 style="margin:24px 0 0;font-size:18px">{html.escape(RESULTS_TITLE)}</h3><div class="stats">{stats}</div></section>
<section class="block"><h2>{html.escape(LIMITS_TITLE)}</h2><div class="limits">{html.escape(LIMITS)}</div></section>
</div>'''


def render_footer_html() -> str:
    return f'<footer class="page"><div class="wrap"><p>{IMAGE_CREDIT}</p><p>{html.escape(DATA_CREDIT)}</p></div></footer>'


def render_markdown() -> str:
    out = [f'<p align="center"><img src="assets/eeg-recording-cap.jpg" width="640" alt="{IMAGE_ALT}"><br><sub>{IMAGE_CREDIT}</sub></p>', "",
           f"*{EYEBROW}*", "", f"## {TITLE}", "", LEAD, "", f"### {PROBLEM_TITLE}", ""]
    out += [f"- **{t}.** {b}" for _, t, b in PROBLEM]
    out += ["", f"### {QUESTION_TITLE}", ""] + [f"- **{k}:** {v}" for k, v in QUESTIONS]
    out += ["", f"**{HELP_TITLE}.** " + " ".join(HELP), "", f"*{HELP_NOTE}*", "", f"### {WORK_TITLE}", ""]
    out += [f"{n}. **{t}.** {b}" for n, (_, t, b) in enumerate(WORK, 1)]
    out += ["", f"**{RESULTS_TITLE}:** " + " · ".join(f"{k} {v}" for v, k in RESULTS), "", f"### {LIMITS_TITLE}", "", LIMITS, "", f"<sub>{DATA_CREDIT}</sub>"]
    return "\n".join(out) + "\n"


START, END = "<!-- STORY:START (generated by demo/story.py, edit it there) -->", "<!-- STORY:END -->"


def write_readme(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    block = f"{START}\n{render_markdown()}{END}"
    text = re.sub(re.escape(START) + r".*?" + re.escape(END), lambda _: block, text, flags=re.S)
    path.write_text(text, encoding="utf-8", newline="\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--write-readme", action="store_true")
    if ap.parse_args().write_readme:
        write_readme(ROOT / "README.md")
        print("README.md updated")
