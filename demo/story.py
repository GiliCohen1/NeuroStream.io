"""The plain-language story of NeuroStream, written once and rendered to HTML (dashboard) and Markdown (README).

    .venv/Scripts/python demo/story.py --write-readme     # refreshes the block between the STORY markers in README.md
"""
from __future__ import annotations

import argparse
import html
import re
from pathlib import Path

GOAL_TITLE = "What is this, and why does it exist?"
GOAL = [
    "Picture a patient in a hospital wearing a soft cap covered in tiny sensors. The sensors do not touch the brain. "
    "They simply listen to it, the way a microphone listens to a room. Every thought and every movement makes the brain "
    "produce faint electrical ripples, and the sensors draw them as a wavy line. That line is called an <b>EEG</b>.",

    "Sometimes, in some people, those ripples suddenly turn into a storm. That storm is a <b>seizure</b>. Seizures can be "
    "dangerous, and they can start without warning. But no nurse can stare at a wavy line for 24 hours, for every patient, "
    "without blinking.",

    "<b>So the goal of this project is simple to say: build a tireless digital watcher.</b> It looks at the brain signal "
    "every single second, decides whether the last second looks like a seizure, and the moment it does, it rings an alarm "
    "so a human doctor can step in.",

    "Two things make that hard. First, the signal never stops: it is a river, not a bucket, and nothing may be lost if one "
    "part of the system breaks. Second, nobody can write down a clear rule for what a seizure looks like. It is a pattern "
    "that is easy to recognize after seeing many examples and almost impossible to describe in words. So instead of "
    "giving the computer rules, we <b>show it examples and let it learn</b>.",
]
GOAL_NOTE = ("This is a learning project built on public research recordings. It shows how such a system can be built. "
             "It is not a medical device and must never be used to make decisions about real patients.")

STEPS_TITLE = "How we built it, step by step"
STEPS = [
    ("Find real examples",
     "We downloaded 500 short brain recordings that scientists at the University of Bonn made and shared for research. "
     "100 of them were recorded during a seizure. 400 were not. These are the 'examples' the computer will learn from."),
    ("Hide the exam paper",
     "Before teaching anything, we locked 75 recordings away, 15 with seizures. The computer never sees them while "
     "learning. It is like a teacher keeping the exam questions secret so a student cannot just memorize the answers. "
     "We locked away <i>whole recordings</i>, not slices of them, so the computer cannot cheat by recognizing a recording "
     "it has already seen."),
    ("Make the signals easy to compare",
     "Raw signals are messy: some are louder, some are shifted up or down. We rescale them so they can be compared fairly. "
     "Then we measure how much slow, medium and fast rhythm each second contains, much like the bass and treble bars on a "
     "stereo. Those five bars are extra clues for the computer."),
    ("Teach a computer by example",
     "We built a small <b>neural network</b>: a pile of simple pattern detectors loosely inspired by the brain. We showed it "
     "about 30,000 one-second snippets, each labeled 'seizure' or 'normal'. For every snippet it guessed, we told it how "
     "wrong it was, and it adjusted itself a tiny bit. We repeated this 30 times over the whole pile. Nobody told it what "
     "a seizure looks like. It worked that out."),
    ("Grade it on the secret exam",
     "Then we opened the locked recordings. On 1,725 one-second snippets it had never seen, it was right 97.6% of the time. "
     "It caught about 95 of every 100 seizure seconds. When it rang the alarm, it was right about 93 times out of 100. "
     "About 2 of every 100 calm seconds set off a false alarm. These are good results for a classroom-style test, "
     "and we are careful not to oversell them (see the honest limits below)."),
    ("Build the assembly line",
     "A smart model is not a system. We built the whole conveyor belt around it: a <b>simulated hospital machine</b> sends "
     "each second of signal, a <b>front desk</b> (the gateway) checks it is a proper form and hands back a receipt, a "
     "<b>mail room</b> (Kafka) holds every message safely until the next worker is ready, an <b>analyst</b> cleans and "
     "measures the signal, the <b>trained model</b> gives its verdict, and an <b>alarm bell</b> rings when it says seizure."),
    ("Make it hard to break",
     "Real systems fail, so we planned for it. If the model is briefly unreachable, the analyst waits and tries again "
     "instead of giving up. A message is only marked 'done' after it has really been handled, so a crash cannot silently "
     "lose it. A message that is garbage goes into a separate 'problem pile' for a human to look at, instead of "
     "blocking everything behind it."),
    ("Let robots check our work",
     "We wrote automatic tests, then set up a robot on GitHub that builds the entire system from scratch in the cloud, "
     "feeds the secret-exam recordings through the real conveyor belt, and only shows a green check if the alarms ring "
     "for the seizures and stay quiet for the normal brain activity. On the way it caught three real mistakes of ours "
     "that would have stopped the system from starting, and we fixed them."),
    ("Make it visible",
     "Finally, we turned the invisible into something you can look at: this page. Each card below is one real second "
     "of brain signal, what the computer decided, and whether the alarm rang."),
]

LIMITS_TITLE = "Honest limits"
LIMITS = [
    "The recordings are clean, short and carefully chosen. Real hospital signals are noisier, and real seizures are far rarer, "
    "which makes false alarms much more costly.",
    "Only one kind of signal from one research source was used. We have not shown it works on other patients, other machines "
    "or other hospitals.",
    "Missing even a few seizures would matter in real life. About 5 in 100 seizure seconds were missed here.",
]

GLOSSARY_TITLE = "Words you may have seen"
GLOSSARY = [
    ("EEG", "A recording of the brain's electrical ripples, drawn as a wavy line."),
    ("Seizure", "A sudden storm of electrical activity in the brain."),
    ("Neural network", "A computer program that learns patterns from examples instead of following written rules."),
    ("Kafka", "A very reliable mail room. Messages wait there safely until someone is ready to read them."),
    ("Gateway", "The front desk. It checks every incoming message is filled in properly."),
    ("gRPC", "A very fast way for two programs to talk to each other."),
    ("FFT", "A math trick that splits a wavy line into slow, medium and fast rhythms, like a stereo equalizer."),
]

CSS = """
.story{background:var(--card);border:1px solid var(--bd);border-radius:12px;padding:20px 22px;margin-bottom:20px}
.story h2{margin:0 0 10px;font-size:20px}.story p{margin:0 0 12px;max-width:75ch}
.story .note{border-left:3px solid #f5a623;padding:6px 12px;color:var(--mut);background:#f5a62311;border-radius:0 6px 6px 0}
ol.steps{list-style:none;counter-reset:s;margin:0;padding:0;display:grid;gap:10px}
ol.steps li{counter-increment:s;position:relative;padding:10px 12px 10px 52px;background:var(--bg);border-radius:8px}
ol.steps li::before{content:counter(s);position:absolute;left:12px;top:10px;width:28px;height:28px;border-radius:50%;
 background:#2f6fdb;color:#fff;font-weight:700;display:flex;align-items:center;justify-content:center}
ol.steps b.t{display:block;margin-bottom:2px}.story ul{margin:0;padding-left:20px}.story li{margin-bottom:4px}
dl.gl{display:grid;grid-template-columns:max-content 1fr;gap:4px 14px;margin:0}dl.gl dt{font-weight:700}dl.gl dd{margin:0;color:var(--mut)}
h2.live{margin:26px 0 10px}
@media(max-width:560px){dl.gl{grid-template-columns:1fr}}
"""


def render_html() -> str:
    steps = "".join(f'<li><b class="t">{html.escape(t)}</b>{b}</li>' for t, b in STEPS)
    limits = "".join(f"<li>{html.escape(x)}</li>" for x in LIMITS)
    gloss = "".join(f"<dt>{html.escape(k)}</dt><dd>{html.escape(v)}</dd>" for k, v in GLOSSARY)
    goal = "".join(f"<p>{p}</p>" for p in GOAL)
    return (f'<section class="story"><h2>{html.escape(GOAL_TITLE)}</h2>{goal}<p class="note">{html.escape(GOAL_NOTE)}</p></section>'
            f'<section class="story"><h2>{html.escape(STEPS_TITLE)}</h2><ol class="steps">{steps}</ol></section>'
            f'<section class="story"><h2>{html.escape(LIMITS_TITLE)}</h2><ul>{limits}</ul></section>'
            f'<section class="story"><h2>{html.escape(GLOSSARY_TITLE)}</h2><dl class="gl">{gloss}</dl></section>')


def _md(s: str) -> str:
    return re.sub(r"</?i>", "*", re.sub(r"</?b>", "**", s))


def render_markdown() -> str:
    out = [f"## {GOAL_TITLE}", ""]
    out += [f"{_md(p)}\n" for p in GOAL]
    out += [f"> {GOAL_NOTE}", "", f"## {STEPS_TITLE}", ""]
    out += [f"{i}. **{t}.** {_md(b)}" for i, (t, b) in enumerate(STEPS, 1)]
    out += ["", f"## {LIMITS_TITLE}", ""] + [f"- {x}" for x in LIMITS]
    out += ["", f"## {GLOSSARY_TITLE}", ""] + [f"- **{k}**: {v}" for k, v in GLOSSARY]
    return "\n".join(out) + "\n"


START, END = "<!-- STORY:START (generated by demo/story.py, edit it there) -->", "<!-- STORY:END -->"


def write_readme(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    block = f"{START}\n{render_markdown()}{END}"
    if START in text:
        text = re.sub(re.escape(START) + r".*?" + re.escape(END), lambda _: block, text, flags=re.S)
    else:  # first time: right after the title line + one-sentence summary
        head, _, rest = text.partition("\n\n")
        head2, _, rest2 = rest.partition("\n\n")
        text = f"{head}\n\n{head2}\n\n{block}\n\n---\n\n## For engineers\n\n{rest2}"
    path.write_text(text, encoding="utf-8", newline="\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--write-readme", action="store_true")
    if ap.parse_args().write_readme:
        write_readme(Path(__file__).resolve().parents[1] / "README.md")
        print("README.md updated")
