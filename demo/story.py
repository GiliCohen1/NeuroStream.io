"""The plain-language story of NeuroStream, rendered to Markdown for the README (the website has its own hand-authored copy in site/index.html).

    .venv/Scripts/python demo/story.py --write-readme     # refreshes the block between the STORY markers in README.md
"""
from __future__ import annotations

import argparse
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

def render_markdown() -> str:
    out = [f'<p align="center"><img src="site/assets/eeg-recording-cap.jpg" width="640" alt="{IMAGE_ALT}"><br><sub>{IMAGE_CREDIT}</sub></p>', "",
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
