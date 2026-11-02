"""Animated explainer for the dashboard (web only): brain schematic + step-by-step decision flow + threshold explorer.

Everything shown is computed from the real held-out data and the trained model:
  * 5 consecutive seconds of a real held-out recording per class (seizure-free / seizure)
  * the epochs' normalized waveforms, band powers and model scores
  * score distribution over all 1,725 held-out epochs (drives the threshold explorer)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

FS = 173.61
WIN = 178
N_EPOCHS = 5


def build_data(root: Path) -> dict:
    sys.path[:0] = [str(root / "services" / "ml-server"), str(root / "services" / "worker")]
    from app.model import SpikeDetector
    from worker.preprocessing import preprocess

    det = SpikeDetector()
    thr = det.threshold
    files = json.loads((root / "models" / "heldout_test_files.json").read_text())
    scores, bands_by_class = [], {0: [], 1: []}
    per_file: dict[str, tuple[bool, np.ndarray, np.ndarray, list, list]] = {}
    for f in sorted(files, key=lambda f: f["file"]):
        sig = np.loadtxt(root / "data" / "heldout" / f["file"])
        eps = [preprocess(sig[s:s + WIN].tolist(), FS, 64) for s in range(0, len(sig) - WIN + 1, WIN)]
        p = det.score([e[0].tolist() for e in eps], [e[1].tolist() for e in eps])
        label = int(f["seizure"])
        for pp, (_, b) in zip(p, eps):
            scores.append([round(float(pp), 3), label])
            bands_by_class[label].append(b)
        per_file[f["file"]] = (f["seizure"], sig, np.array(p), [e[0] for e in eps], [e[1] for e in eps])

    def pick(seizure: bool, prefix: str | None):
        for name, (s, sig, p, norms, bands) in per_file.items():
            if s != seizure or (prefix and not name.lower().startswith(prefix)):
                continue
            ok = p[:N_EPOCHS] >= thr if seizure else p[:N_EPOCHS] < thr
            if ok.all():
                n = N_EPOCHS * WIN
                return {
                    "file": name,
                    "raw": [int(round(v)) for v in sig[:n]],
                    "norm": [round(float(v), 2) for e in norms[:N_EPOCHS] for v in e],
                    "bands": [[round(float(v), 4) for v in b] for b in bands[:N_EPOCHS]],
                    "p": [round(float(v), 4) for v in p[:N_EPOCHS]],
                    "amp": float(np.percentile(np.abs(sig[:n]), 99)),
                }
        raise RuntimeError("no suitable example recording")

    data = {
        "fs": FS, "thr": thr, "win": WIN,
        "ex": {"normal": pick(False, "n/"), "seizure": pick(True, "s/")},
        "bandMean": {"normal": [round(float(v), 4) for v in np.mean(bands_by_class[0], axis=0)],
                     "seizure": [round(float(v), 4) for v in np.mean(bands_by_class[1], axis=0)]},
        "scores": scores,
    }
    # Sanity: the explorer must reproduce the training-time metrics at the deployed threshold.
    meta = json.loads((root / "models" / "meta.json").read_text())["metrics"]["test"]["confusion"]
    tp = sum(1 for p, y in scores if y == 1 and p >= thr); fp = sum(1 for p, y in scores if y == 0 and p >= thr)
    assert (tp, fp) == (meta["tp"], meta["fp"]), f"explorer mismatch {(tp, fp)} vs {(meta['tp'], meta['fp'])}"
    return data


CSS = """
.mo{background:linear-gradient(180deg,#0b2545,#0a1d37);color:#cfe0ee;border-radius:8px;padding:22px;border:1px solid #14375f}
.mo h3{color:#fff;font-size:18px;margin:0 0 4px}.mo p{margin:0 0 10px}.mo small,.mo .cap{color:#8fa6bd;font-size:12.5px}
.mo-grid{display:grid;grid-template-columns:1.15fr .85fr;gap:24px;align-items:center}
.seg{display:inline-flex;border:1px solid #2b5282;border-radius:6px;overflow:hidden;margin-bottom:12px}
.seg button{background:transparent;color:#cfe0ee;border:0;padding:8px 16px;font:600 13px "Segoe UI",system-ui,sans-serif;cursor:pointer}
.seg button[aria-pressed=true]{background:#13808b;color:#fff}
.seg button:focus-visible,.stages button:focus-visible,.ctl:focus-visible{outline:2px solid #6fd0d8;outline-offset:2px}
#brain{width:100%;height:auto;display:block}
#live,#stagecv,#histcv{width:100%;height:auto;display:block;border-radius:6px}
.legend{list-style:none;margin:10px 0 0;padding:0;display:grid;gap:6px;font-size:14px}
.legend li{display:flex;gap:8px;align-items:flex-start}.dot{width:10px;height:10px;border-radius:50%;margin-top:6px;flex:none}
.stages{display:grid;grid-template-columns:repeat(6,1fr);gap:6px;list-style:none;margin:0 0 14px;padding:0}
.stages button{width:100%;text-align:left;background:#0f2f55;border:1px solid #1d426f;color:#9fb4c8;border-radius:6px;padding:8px 10px;cursor:pointer;font:600 13px "Segoe UI",system-ui,sans-serif;position:relative;overflow:hidden}
.stages button .n{display:inline-block;width:20px;height:20px;line-height:20px;text-align:center;border-radius:50%;background:#1d426f;color:#cfe0ee;margin-right:6px;font-size:12px}
.stages button[aria-current=step]{color:#fff;border-color:#13808b;background:#123e66}.stages button[aria-current=step] .n{background:#13808b}
.stages button i{position:absolute;left:0;bottom:0;height:2px;background:#6fd0d8;width:0}
.stage-text{min-height:3.6em;margin:12px 0 0}.stage-text b{color:#fff}
.ctlrow{display:flex;flex-wrap:wrap;gap:12px;align-items:center;justify-content:space-between;margin-bottom:12px}
.ctl{background:#0f2f55;border:1px solid #2b5282;color:#cfe0ee;border-radius:6px;padding:7px 14px;cursor:pointer;font:600 13px "Segoe UI",system-ui,sans-serif}
.readout{display:grid;grid-template-columns:repeat(auto-fit,minmax(140px,1fr));gap:10px;margin-top:12px}
.readout div{background:#0f2f55;border:1px solid #1d426f;border-radius:6px;padding:10px 12px}.readout b{display:block;color:#fff;font:600 22px Georgia,serif}.readout span{font-size:12.5px;color:#8fa6bd}
.slider{display:flex;gap:12px;align-items:center;margin:8px 0}.slider input{flex:1;accent-color:#13808b}.slider output{min-width:3.2em;color:#fff;font:600 16px Georgia,serif}
.why{display:grid;grid-template-columns:repeat(auto-fit,minmax(230px,1fr));gap:12px;margin-top:16px}
.why div{background:#0f2f55;border:1px solid #1d426f;border-radius:6px;padding:14px}.why b{color:#fff;display:block;margin-bottom:4px}.why span{font-size:14px;color:#b9c7d8}
@media(max-width:820px){.mo-grid{grid-template-columns:1fr}.stages{grid-template-columns:repeat(3,1fr)}}
"""

BODY = """
<section class="block" id="mechanism">
 <h2>Inside the brain: what a seizure looks like to an electrode</h2>
 <div class="mo"><div class="mo-grid">
  <div>
   <svg id="brain" viewBox="0 0 640 430" role="img" aria-label="Schematic side view of the brain with a network of neurons. In the seizure state, neurons at the hippocampal formation fire in synchrony and the activity spreads outward.">
    <defs>
     <radialGradient id="hipglow"><stop offset="0" stop-color="#6fd0d8" stop-opacity=".9"/><stop offset="1" stop-color="#6fd0d8" stop-opacity="0"/></radialGradient>
    </defs>
    <g id="rip"></g>
    <path id="cerebrum" d="M70 215 C55 150 105 85 190 62 C270 40 372 42 446 72 C520 102 566 160 566 222 C566 262 540 292 506 304 C480 313 455 308 436 316 C412 330 380 338 350 330 C326 324 306 330 282 336 C240 346 196 336 168 312 C130 292 82 268 70 215 Z" fill="rgba(111,208,216,.05)" stroke="#6fd0d8" stroke-opacity=".7" stroke-width="2"/>
    <g fill="none" stroke="#6fd0d8" stroke-opacity=".22" stroke-width="1.5" stroke-linecap="round">
     <path d="M300 56 C312 120 288 160 312 208"/><path d="M148 262 C220 236 292 236 362 254"/><path d="M470 92 C455 140 470 176 450 216"/>
     <path d="M110 160 C150 138 190 156 226 128"/><path d="M96 220 C140 200 170 214 214 190"/><path d="M360 70 C380 108 372 130 404 150"/>
     <path d="M500 150 C520 180 512 214 534 236"/><path d="M200 300 C240 288 270 296 300 284"/>
    </g>
    <path d="M420 322 C450 312 505 318 528 342 C536 366 500 392 455 392 C420 392 402 362 420 322 Z" fill="rgba(111,208,216,.04)" stroke="#6fd0d8" stroke-opacity=".5" stroke-width="1.6"/>
    <g fill="none" stroke="#6fd0d8" stroke-opacity=".2" stroke-width="1.2"><path d="M430 338 C455 330 500 336 520 350"/><path d="M426 352 C455 346 495 352 514 364"/><path d="M430 366 C455 362 485 366 500 376"/></g>
    <path d="M372 336 C376 372 372 396 380 414 L412 414 C410 392 412 360 416 330 Z" fill="rgba(111,208,216,.04)" stroke="#6fd0d8" stroke-opacity=".5" stroke-width="1.6"/>
    <g id="edges"></g><g id="nodes"></g>
    <circle cx="308" cy="290" r="46" fill="url(#hipglow)" opacity=".55"/>
    <path id="hippo" d="M250 292 C275 268 320 262 352 276 C372 285 376 300 362 304 C340 298 312 292 286 304 C268 312 246 306 250 292 Z" fill="rgba(19,128,139,.55)" stroke="#6fd0d8" stroke-width="2"/>
    <g id="elec"></g>
    <g font-size="12" fill="#cfe0ee" font-family="Segoe UI,system-ui,sans-serif">
     <path d="M300 300 L250 372" stroke="#6fd0d8" stroke-opacity=".7" fill="none"/><circle cx="300" cy="300" r="3" fill="#6fd0d8"/>
     <text x="30" y="384">Hippocampal formation</text><text x="30" y="400" fill="#8fa6bd">intracranial electrodes (patients)</text>
     <text x="30" y="24">Scalp electrodes, 10–20 system</text><text x="30" y="40" fill="#8fa6bd">healthy volunteers</text>
    </g>
   </svg>
   <p class="cap">Illustrative schematic, not anatomy to scale. The model analyzes one recording channel at a time and does not localize a seizure.</p>
  </div>
  <div>
   <div class="seg" role="group" aria-label="Brain activity state">
    <button type="button" data-mode="normal" aria-pressed="false">Seizure-free</button>
    <button type="button" data-mode="seizure" aria-pressed="true">Seizure</button>
   </div>
   <p id="brain-text"></p>
   <canvas id="live" width="520" height="150" aria-label="Live EEG trace from a real held-out recording"></canvas>
   <p class="cap" id="live-cap"></p>
   <ul class="legend">
    <li><span class="dot" style="background:#6fd0d8"></span><span><b style="color:#fff">Seizure-free:</b> neurons fire out of step, so their contributions cancel into a low-amplitude, irregular trace.</span></li>
    <li><span class="dot" style="background:#ff7a6b"></span><span><b style="color:#fff">Seizure:</b> a population fires in synchrony and the activity spreads. Summed at the electrode, it becomes large, rhythmic and regular.</span></li>
   </ul>
  </div>
 </div></div>
</section>

<section class="block" id="decision">
 <h2>How the decision is made</h2>
 <div class="mo">
  <div class="ctlrow">
   <div class="seg" role="group" aria-label="Example epoch type">
    <button type="button" data-mode="normal" aria-pressed="false">Seizure-free example</button>
    <button type="button" data-mode="seizure" aria-pressed="true">Seizure example</button>
   </div>
   <button type="button" class="ctl" id="playbtn" aria-pressed="true">Pause</button>
  </div>
  <ol class="stages" id="stagenav"></ol>
  <canvas id="stagecv" width="1000" height="300" role="img" aria-label="Animation of the current processing stage, driven by real held-out data"></canvas>
  <p class="stage-text" id="stagetext" aria-live="polite"></p>
  <p class="cap">Animated from a real held-out recording the model never trained on (<span id="exfile"></span>). Five consecutive one-second epochs.</p>
 </div>
</section>

<section class="block" id="threshold">
 <h2>Where the decision line sits, and why it matters to a clinician</h2>
 <div class="mo">
  <p>Every epoch gets a score between 0 and 1. The alert fires when the score reaches the decision line. Each setting trades missed seizures against false alarms, so the line is a clinical choice, not just a technical one.</p>
  <canvas id="histcv" width="1000" height="230" role="img" aria-label="Distribution of model scores for seizure and seizure-free epochs with the decision threshold"></canvas>
  <div class="slider"><label for="thr">Decision threshold</label><input id="thr" type="range" min="0.02" max="0.98" step="0.01"><output id="throut"></output><button type="button" class="ctl" id="thrreset">Deployed setting</button></div>
  <div class="readout">
   <div><b id="r-sens"></b><span>Sensitivity: seizure epochs flagged</span></div>
   <div><b id="r-prec"></b><span>Precision: alerts that were real</span></div>
   <div><b id="r-miss"></b><span>Seizure epochs missed</span></div>
   <div><b id="r-fa"></b><span>False alarms on seizure-free epochs</span></div>
  </div>
  <p class="cap" style="margin-top:10px">Illustration on the held-out test epochs. The deployed threshold was chosen on separate validation data; tuning a threshold on test data would overstate performance.</p>
  <div class="why">
   <div><b>Prioritize review</b><span>Flagged epochs go to the top of the reading queue, so specialist time goes where it is most likely to matter.</span></div>
   <div><b>Earlier recognition</b><span>The score updates every second, rather than after the recording is read.</span></div>
   <div><b>Transparent by design</b><span>A single score and a visible threshold. Every alert points to the exact one-second epoch a clinician can confirm.</span></div>
   <div><b>A second reader</b><span>The system flags and the clinician decides. It supports the decision; it does not replace it.</span></div>
  </div>
 </div>
</section>
"""

JS = r"""
(function(){
'use strict';
const D=JSON.parse(document.getElementById('ns-data').textContent);
const TAU=Math.PI*2, reduce=window.matchMedia('(prefers-reduced-motion: reduce)').matches;
const $=(s,r=document)=>r.querySelector(s), $$=(s,r=document)=>Array.from(r.querySelectorAll(s));
const C={bg:'#0b2545',grid:'rgba(255,255,255,.06)',txt:'#cfe0ee',mut:'#8fa6bd',ok:'#6fd0d8',bad:'#ff7a6b',line:'rgba(255,255,255,.2)'};
const BANDS=[['δ','delta · 0.5–4 Hz'],['θ','theta · 4–8 Hz'],['α','alpha · 8–13 Hz'],['β','beta · 13–30 Hz'],['γ','gamma · 30–100 Hz']];
const pct=v=>Math.round(v*100)+'%';
const bm=D.bandMean;
const STAGES=[
 ['Acquire','Electrodes record the voltage produced by large populations of neurons. Here: one channel, sampled '+D.fs+' times per second.'],
 ['Segment','The continuous stream is cut into one-second epochs of '+D.win+' samples. <b>An epoch is the unit of every decision.</b>'],
 ['Normalize','A sliding 64-sample window removes the local baseline and scale (z-score), so recordings from different patients and amplifiers become comparable.'],
 ['Characterize','A Fourier transform measures how power is distributed across the clinical frequency bands. On the held-out set, seizure epochs average <b>'+pct(bm.seizure[1])+' theta</b> versus '+pct(bm.normal[1])+' for seizure-free epochs, and <b>'+pct(bm.seizure[0])+' delta</b> versus '+pct(bm.normal[0])+'.'],
 ['Classify','A 1D convolutional network reads the normalized waveform together with the five band powers and outputs <b>P(seizure)</b>, a score between 0 and 1.'],
 ['Decide','If P(seizure) reaches <b>'+D.thr.toFixed(2)+'</b> the epoch is flagged and an alert is raised; below it, no alert. The threshold was tuned on validation data, not on the test set.']
];
const FROZEN=/freeze/.test(location.hash);
let mode='seizure', stage=0, playing=!reduce, stageT0=0, T=0, sync=1;
const STAGE_SECONDS=6.5;
const ease=x=>x<=0?0:x>=1?1:x*x*(3-2*x);
const lerp=(a,b,t)=>a+(b-a)*t;
const col=()=>mode==='seizure'?C.bad:C.ok;
const E=()=>D.ex[mode];

function setup(cv){const W=cv.width,H=cv.height,dpr=Math.min(2,window.devicePixelRatio||1);cv.width=W*dpr;cv.height=H*dpr;const g=cv.getContext('2d');g.setTransform(dpr,0,0,dpr,0,0);g.W=W;g.H=H;return g}
function frame(g){g.fillStyle=C.bg;g.fillRect(0,0,g.W,g.H);g.strokeStyle=C.grid;g.lineWidth=1;g.beginPath();for(let x=0;x<=g.W;x+=50){g.moveTo(x,0);g.lineTo(x,g.H)}for(let y=0;y<=g.H;y+=50){g.moveTo(0,y);g.lineTo(g.W,y)}g.stroke()}
function txt(g,s,x,y,size,color,align){g.font=(size||13)+'px "Segoe UI",system-ui,sans-serif';g.fillStyle=color||C.txt;g.textAlign=align||'left';g.fillText(s,x,y)}
function trace(g,arr,i0,n,x,y,w,h,amp,color,lw,total){total=total||n;g.beginPath();for(let k=0;k<n;k++){const v=arr[(i0+k)%arr.length];const px=x+k/(total-1)*w,py=y+h/2-(v/amp)*h/2*.92;k?g.lineTo(px,py):g.moveTo(px,py)}g.strokeStyle=color;g.lineWidth=lw||1.6;g.lineJoin='round';g.stroke()}
function segCopy(arr,k){return arr.slice(k*D.win,(k+1)*D.win)}

/* ---------- brain ---------- */
const brain=$('#brain'), cerebrum=$('#cerebrum');
const NS='http://www.w3.org/2000/svg';
function prng(seed){let a=seed>>>0;return function(){a=(a+0x6D2B79F5)>>>0;let t=a;t=Math.imul(t^t>>>15,t|1);t^=t+Math.imul(t^t>>>7,t|61);return((t^t>>>14)>>>0)/4294967296}}
const FOCUS={x:308,y:290}; const nodes=[], edges=[];
(function buildBrain(){
 const r=prng(7), gN=$('#nodes'), gE=$('#edges'); let tries=0;
 while(nodes.length<74&&tries++<6000){const x=84+r()*470,y=56+r()*270; if(!cerebrum.isPointInFill(new DOMPoint(x,y)))continue; if(nodes.some(n=>Math.hypot(n.x-x,n.y-y)<30))continue; nodes.push({x,y,ph:r()*TAU,f:.5+r()*.9,d:Math.hypot(x-FOCUS.x,y-FOCUS.y)})}
 const seen=new Set();
 nodes.forEach((a,i)=>{nodes.map((b,j)=>[j,Math.hypot(a.x-b.x,a.y-b.y)]).filter(p=>p[0]!==i).sort((p,q)=>p[1]-q[1]).slice(0,3).forEach(([j,d])=>{const k=i<j?i+'-'+j:j+'-'+i;if(seen.has(k)||d>95)return;seen.add(k);const l=document.createElementNS(NS,'line');l.setAttribute('x1',a.x);l.setAttribute('y1',a.y);l.setAttribute('x2',nodes[j].x);l.setAttribute('y2',nodes[j].y);l.setAttribute('stroke-width','1');gE.appendChild(l);edges.push({i,j,l})})});
 nodes.forEach(n=>{const h=document.createElementNS(NS,'circle'),c=document.createElementNS(NS,'circle');h.setAttribute('cx',n.x);h.setAttribute('cy',n.y);c.setAttribute('cx',n.x);c.setAttribute('cy',n.y);gN.appendChild(h);gN.appendChild(c);n.h=h;n.c=c});
 const rg=$('#rip');for(let i=0;i<3;i++){const c=document.createElementNS(NS,'circle');c.setAttribute('cx',FOCUS.x);c.setAttribute('cy',FOCUS.y);c.setAttribute('fill','none');c.setAttribute('stroke','#ff7a6b');rg.appendChild(c)}
 const L=cerebrum.getTotalLength(), ge=$('#elec');
 [0.07,0.13,0.19,0.25,0.31,0.37,0.43].forEach(f=>{const p=cerebrum.getPointAtLength(L*f),q=cerebrum.getPointAtLength(L*f+2);let tx=q.x-p.x,ty=q.y-p.y;const m=Math.hypot(tx,ty);tx/=m;ty/=m;const nx=ty,ny=-tx;const c=document.createElementNS(NS,'circle');c.setAttribute('cx',p.x+nx*11);c.setAttribute('cy',p.y+ny*11);c.setAttribute('r',5);c.setAttribute('fill','#0b2545');c.setAttribute('stroke','#fff');c.setAttribute('stroke-width','1.6');ge.appendChild(c)});
})();
function brainFrame(t,dt){
 sync+=((mode==='seizure'?1:0)-sync)*Math.min(1,dt*2.2);
 const mix=(a,b,k)=>Math.round(a+(b-a)*k);
 const act=nodes.map(n=>{const nA=.28+.22*Math.sin(TAU*n.f*t*.8+n.ph);const s=Math.sin(TAU*1.7*t-n.d*.032);const sA=.22+.78*Math.pow(Math.max(0,s),2)*(1-Math.min(1,n.d/560)*.35);return lerp(nA,sA,sync)});
 nodes.forEach((n,i)=>{const a=act[i];const r=mix(111,255,sync),g=mix(208,122,sync),b=mix(216,107,sync);const c='rgb('+r+','+g+','+b+')';n.c.setAttribute('r',(2.4+4.2*a).toFixed(2));n.c.setAttribute('fill',c);n.c.setAttribute('opacity',(.45+.55*a).toFixed(2));n.h.setAttribute('r',(6+11*a).toFixed(2));n.h.setAttribute('fill',c);n.h.setAttribute('opacity',(.04+.22*a).toFixed(2))});
 edges.forEach(e=>{const a=(act[e.i]+act[e.j])/2;const r=mix(111,255,sync),g=mix(208,122,sync),b=mix(216,107,sync);e.l.setAttribute('stroke','rgb('+r+','+g+','+b+')');e.l.setAttribute('stroke-opacity',(.08+.55*a*a+.1*sync*a).toFixed(2))});
 $$('#rip circle').forEach((c,k)=>{const p=((t*.55+k/3)%1);c.setAttribute('r',(10+p*300).toFixed(1));c.setAttribute('stroke-opacity',(sync*(1-p)*.38).toFixed(3));c.setAttribute('stroke-width',(2.2*(1-p)+.4).toFixed(2))});
}
const liveG=setup($('#live'));
function liveFrame(t){const g=liveG;frame(g);const e=E(),n=330,i0=Math.floor(t*D.fs)%e.raw.length,amp=Math.max(D.ex.normal.amp,D.ex.seizure.amp);trace(g,e.raw,i0,n,12,14,g.W-24,g.H-34,amp,col(),1.7);txt(g,'same scale in both states: ±'+Math.round(amp)+' (recorded units)',12,g.H-8,12,C.mut);txt(g,'1 s',g.W-12,g.H-8,12,C.mut,'right');g.strokeStyle=C.line;g.beginPath();g.moveTo(g.W-12-(D.fs/330)*(g.W-24),g.H-22);g.lineTo(g.W-12,g.H-22);g.stroke()}

/* ---------- stage drawers ---------- */
const stG=setup($('#stagecv'));
function drawers(g,t){
 const e=E(), W=g.W, H=g.H;
 switch(stage){
 case 0:{frame(g);const n=420,i0=Math.floor(t*D.fs)%e.raw.length;trace(g,e.raw,i0,n,40,50,W-80,170,e.amp,col(),1.8);
  for(let j=n-46;j<n;j++){const v=e.raw[(i0+j)%e.raw.length];g.fillStyle='#fff';g.beginPath();g.arc(40+j/(n-1)*(W-80),50+85-(v/e.amp)*85*.92,2.4,0,TAU);g.fill()}
  g.strokeStyle=C.line;g.setLineDash([4,4]);g.beginPath();g.moveTo(W-40,40);g.lineTo(W-40,230);g.stroke();g.setLineDash([]);
  txt(g,'Channel 1 · '+D.fs+' Hz · each dot is one sample',40,32,13,C.mut);txt(g,'now',W-40,252,12,C.mut,'center');txt(g,'Voltage over time (full scale ±'+Math.round(e.amp)+')',40,284,12,C.mut);break}
 case 1:{frame(g);const x=40,w=W-80;trace(g,e.raw,0,e.raw.length,x,50,w,170,e.amp,col(),1.3);const cur=Math.floor(t/1.3)%5;
  for(let j=0;j<=5;j++){g.strokeStyle=C.line;g.beginPath();g.moveTo(x+j*w/5,40);g.lineTo(x+j*w/5,232);g.stroke()}
  g.fillStyle=col()==C.bad?'rgba(255,122,107,.16)':'rgba(111,208,216,.16)';g.fillRect(x+cur*w/5,40,w/5,192);
  for(let j=0;j<5;j++)txt(g,String(j+1),x+(j+.5)*w/5,250,13,j===cur?'#fff':C.mut,'center');
  txt(g,'Epoch '+(cur+1)+' · 1 second · '+D.win+' samples',x,30,13,C.txt);txt(g,'5 consecutive epochs of one recording',x,284,12,C.mut);break}
 case 2:{frame(g);const ep=Math.floor(t/3.4)%5,u=(t%3.4)/3.4;const s=Math.floor(ease(u/.85)*D.win);const raw=segCopy(e.raw,ep),nm=e.norm.slice(ep*D.win,(ep+1)*D.win);const x=40,w=W-80;let mx=1;raw.forEach(v=>{mx=Math.max(mx,Math.abs(v))});
  trace(g,raw,0,D.win,x,34,w,100,mx,col(),1.5);
  const hw=64,lo=Math.max(0,s-hw/2),hi=Math.min(D.win-1,s+hw/2);g.fillStyle='rgba(255,255,255,.07)';g.fillRect(x+lo/(D.win-1)*w,30,Math.max(0,hi-lo)/(D.win-1)*w,110);
  g.beginPath();for(let j=0;j<=Math.min(s,D.win-1);j++){let a=0,c=0;for(let q=Math.max(0,j-32);q<=Math.min(D.win-1,j+32);q++){a+=raw[q];c++}const m=a/c;const px=x+j/(D.win-1)*w,py=34+50-(m/mx)*50*.92;j?g.lineTo(px,py):g.moveTo(px,py)}g.strokeStyle='#fff';g.setLineDash([5,4]);g.lineWidth=1.3;g.stroke();g.setLineDash([]);
  g.strokeStyle=C.line;g.beginPath();g.moveTo(x,196);g.lineTo(x+w,196);g.stroke();
  trace(g,nm,0,Math.max(2,s),x,150,w,92,3.6,col(),1.6,D.win);
  txt(g,'Raw epoch '+(ep+1)+'  ·  dashed line = local baseline (64-sample window)',x,26,12,C.mut);txt(g,'Normalized: local z-score (±3.6 σ)',x,262,12,C.mut);txt(g,'amplitude and baseline no longer depend on the patient or the amplifier',x,284,12,C.mut);break}
 case 3:{frame(g);const ep=Math.floor(t/3.2)%5,b=e.bands[ep],g0=ease(((t%3.2)/3.2)/.4);const x0=70,bw=110,gap=56,base=228,hmax=160;
  txt(g,'Relative power per band, epoch '+(ep+1)+' (dots = held-out averages: seizure-free ○, seizure ●)',40,28,13,C.txt);
  for(let j=0;j<5;j++){const bx=x0+j*(bw+gap),h=b[j]*hmax*1.55*g0;g.fillStyle=col()==C.bad?'rgba(255,122,107,.8)':'rgba(111,208,216,.8)';g.fillRect(bx,base-h,bw,h);g.strokeStyle=C.line;g.strokeRect(bx,base-b[j]*hmax*1.55,bw,b[j]*hmax*1.55);
   const yN=base-bm.normal[j]*hmax*1.55,yS=base-bm.seizure[j]*hmax*1.55;g.lineWidth=1.5;g.strokeStyle='#fff';g.beginPath();g.arc(bx+bw*.32,yN,5,0,TAU);g.stroke();g.fillStyle='#fff';g.beginPath();g.arc(bx+bw*.68,yS,5,0,TAU);g.fill();
   txt(g,BANDS[j][0]+'  '+pct(b[j]*g0),bx+bw/2,base+22,15,'#fff','center');txt(g,BANDS[j][1],bx+bw/2,base+40,11.5,C.mut,'center')}
  g.strokeStyle=C.line;g.beginPath();g.moveTo(50,base);g.lineTo(W-30,base);g.stroke();break}
 case 4:{frame(g);const per=2.6,k=Math.floor(t/per)%5,p=e.p[k],q=(t%per)/per*1.3;
  const nm=e.norm.slice(k*D.win,(k+1)*D.win);const lit=x=>Math.max(0,Math.min(1,(q-x)*4));
  g.strokeStyle=C.line;g.strokeRect(30,70,150,110);trace(g,nm,0,D.win,36,76,138,70,3.6,col(),1.2);
  for(let j=0;j<5;j++){g.fillStyle=col()==C.bad?'rgba(255,122,107,.8)':'rgba(111,208,216,.8)';const h=e.bands[k][j]*38*1.5;g.fillRect(40+j*26,176-h,18,h)}
  txt(g,'Normalized epoch',105,58,12,C.mut,'center');txt(g,'+ 5 band powers',105,198,12,C.mut,'center');
  const cx=[290,430,570];['Conv 1 · local shapes','Conv 2 · motifs','Conv 3 · patterns'].forEach((nme,i)=>{const a=lit(.12+i*.17);for(let r=0;r<3;r++){g.globalAlpha=.25+.75*a;g.fillStyle=col()==C.bad?'#ff7a6b':'#6fd0d8';g.fillRect(cx[i]-30+r*10,90+r*10,60-i*8,70-i*8);g.globalAlpha=1;g.strokeStyle='#0b2545';g.strokeRect(cx[i]-30+r*10,90+r*10,60-i*8,70-i*8)}txt(g,nme,cx[i]+6,58,12,C.mut,'center')});
  const dA=lit(.62);for(let r=0;r<6;r++){g.globalAlpha=.25+.75*dA;g.fillStyle='#fff';g.beginPath();g.arc(700,80+r*24,7,0,TAU);g.fill();g.globalAlpha=1}txt(g,'Dense',700,58,12,C.mut,'center');
  [[225,125],[355,125],[495,125],[630,125]].forEach((a,i)=>{const nx=[285,425,565,695][i];g.strokeStyle='rgba(255,255,255,'+(.15+.5*lit(.05+i*.17))+')';g.beginPath();g.moveTo(a[0]-8,a[1]);g.lineTo(nx-34,a[1]);g.stroke()});
  const oA=lit(.85),fill=ease((q-.85)*4);g.globalAlpha=.4+.6*oA;
  txt(g,'P(seizure-free)',790,92,12,C.mut);g.strokeStyle=C.line;g.strokeRect(790,98,170,16);g.fillStyle=C.ok;g.fillRect(790,98,170*(1-p)*fill,16);
  txt(g,'P(seizure)',790,150,12,C.mut);g.strokeRect(790,156,170,16);g.fillStyle=C.bad;g.fillRect(790,156,170*p*fill,16);g.globalAlpha=1;
  g.strokeStyle='rgba(255,255,255,'+(.15+.5*lit(.8))+')';g.beginPath();g.moveTo(710,125);g.lineTo(785,125);g.stroke();
  txt(g,'P(seizure) = '+(p*fill).toFixed(3),790,206,22,'#fff');txt(g,'Epoch '+(k+1)+' of 5 · every parameter was learned from labeled examples',40,284,12,C.mut);break}
 default:{frame(g);const ep=Math.floor(t/2.6)%5,p=e.p[ep],u=ease((t%2.6)/1.1),x=60,w=W-120,y=118;
  const tx=x+D.thr*w;g.fillStyle='rgba(111,208,216,.22)';g.fillRect(x,y-22,tx-x,44);g.fillStyle='rgba(255,122,107,.26)';g.fillRect(tx,y-22,x+w-tx,44);
  g.strokeStyle='#fff';g.lineWidth=2;g.beginPath();g.moveTo(tx,y-40);g.lineTo(tx,y+40);g.stroke();g.lineWidth=1;
  txt(g,'threshold '+D.thr.toFixed(2),tx,y-48,13,'#fff','center');txt(g,'no alert',x+(tx-x)/2,y+6,15,C.ok,'center');txt(g,'ALERT',tx+(x+w-tx)/2,y+6,15,C.bad,'center');
  for(let j=0;j<=10;j++){txt(g,(j/10).toFixed(1),x+j/10*w,y+58,11,C.mut,'center')}
  const mx=x+lerp(.5,p,u)*w;g.fillStyle='#fff';g.beginPath();g.moveTo(mx,y-26);g.lineTo(mx-8,y-40);g.lineTo(mx+8,y-40);g.closePath();g.fill();
  const flagged=p>=D.thr;g.globalAlpha=ease((t%2.6-1.0)/.4);txt(g,(flagged?'ALERT':'No alert')+'  ·  P(seizure) = '+p.toFixed(3)+(flagged?'  ≥  ':'  <  ')+D.thr.toFixed(2),W/2,214,22,flagged?C.bad:C.ok,'center');g.globalAlpha=1;
  txt(g,'Epoch '+(ep+1)+' of 5',40,284,12,C.mut);txt(g,'a clinician reviews every flagged epoch',W-40,284,12,C.mut,'right')}
 }
}

/* ---------- stage navigation ---------- */
const nav=$('#stagenav');
STAGES.forEach((s,i)=>{const li=document.createElement('li');li.innerHTML='<button type="button" data-stage="'+i+'"><span class="n">'+(i+1)+'</span>'+s[0]+'<i></i></button>';nav.appendChild(li)});
function setStage(i){stage=i;stageT0=T;$('#stagetext').innerHTML='<b>'+STAGES[i][0]+'.</b> '+STAGES[i][1];$$('#stagenav button').forEach((b,j)=>{if(j===i)b.setAttribute('aria-current','step');else b.removeAttribute('aria-current');b.querySelector('i').style.width='0'})}
nav.addEventListener('click',e=>{const b=e.target.closest('button');if(b)setStage(+b.dataset.stage)});
function setMode(m){mode=m;$$('.seg button').forEach(b=>b.setAttribute('aria-pressed',String(b.dataset.mode===m)));$('#brain-text').innerHTML=m==='seizure'?'<b style="color:#fff">Seizure.</b> Neurons around the hippocampal formation fire together, and the wave of synchronous activity spreads through connected tissue.':'<b style="color:#fff">Seizure-free.</b> Neurons fire independently, each at its own rhythm.';$('#live-cap').textContent='Real held-out recording ('+D.ex[m].file.split('/').pop()+'), not a simulation.';$('#exfile').textContent=D.ex[m].file.split('/').pop()}
$$('.seg button').forEach(b=>b.addEventListener('click',()=>setMode(b.dataset.mode)));
const pb=$('#playbtn');pb.addEventListener('click',()=>{playing=!playing;pb.textContent=playing?'Pause':'Play';pb.setAttribute('aria-pressed',String(playing));stageT0=T-(T-stageT0)});
pb.textContent=playing?'Pause':'Play';pb.setAttribute('aria-pressed',String(playing));

/* ---------- threshold explorer ---------- */
const hG=setup($('#histcv')),sl=$('#thr'),so=$('#throut');let thr=D.thr;sl.value=thr;
const BINS=20,cnt=[new Array(BINS).fill(0),new Array(BINS).fill(0)];D.scores.forEach(([p,y])=>{cnt[y][Math.min(BINS-1,Math.floor(p*BINS))]++});
const tot=[cnt[0].reduce((a,b)=>a+b),cnt[1].reduce((a,b)=>a+b)];
function hist(){const g=hG,x=50,w=g.W-100,y0=g.H-44,h=g.H-84;frame(g);const mx=Math.sqrt(Math.max(...cnt[0],...cnt[1]));
 for(let b=0;b<BINS;b++){const bw=w/BINS;[[0,C.ok,.18],[1,C.bad,.52]].forEach(([c,colr,off])=>{const v=Math.sqrt(cnt[c][b])/mx*h;g.fillStyle=colr;g.globalAlpha=.9;g.fillRect(x+b*bw+bw*off*.5+ (c?bw*.06:0),y0-v,bw*.42,v);g.globalAlpha=1})}
 g.strokeStyle=C.line;g.beginPath();g.moveTo(x,y0);g.lineTo(x+w,y0);g.stroke();for(let j=0;j<=10;j++)txt(g,(j/10).toFixed(1),x+j/10*w,y0+18,11,C.mut,'center');
 const tx=x+thr*w;g.fillStyle='rgba(255,122,107,.10)';g.fillRect(tx,24,x+w-tx,y0-24);g.strokeStyle='#fff';g.lineWidth=2;g.beginPath();g.moveTo(tx,18);g.lineTo(tx,y0);g.stroke();g.lineWidth=1;
 txt(g,'threshold '+thr.toFixed(2),Math.min(Math.max(tx,70),g.W-70),14,12,'#fff','center');txt(g,'epochs (square-root scale)',x,g.H-8,11.5,C.mut);
 g.fillStyle=C.ok;g.fillRect(g.W-250,g.H-18,10,10);txt(g,'seizure-free',g.W-236,g.H-9,11.5,C.txt);g.fillStyle=C.bad;g.fillRect(g.W-140,g.H-18,10,10);txt(g,'seizure',g.W-126,g.H-9,11.5,C.txt)}
function metrics(){let tp=0,fp=0,fn=0,tn=0;D.scores.forEach(([p,y])=>{const f=p>=thr;if(y&&f)tp++;else if(y)fn++;else if(f)fp++;else tn++});
 so.textContent=thr.toFixed(2);$('#r-sens').textContent=(tp/(tp+fn)*100).toFixed(1)+'%';$('#r-prec').textContent=(tp+fp?tp/(tp+fp)*100:100).toFixed(1)+'%';$('#r-miss').textContent=fn+' of '+(tp+fn);$('#r-fa').textContent=fp+' of '+(fp+tn);hist()}
sl.addEventListener('input',()=>{thr=+sl.value;metrics()});$('#thrreset').addEventListener('click',()=>{thr=D.thr;sl.value=thr;metrics()});

/* ---------- main loop ---------- */
const vis={brain:true,decision:true,threshold:true};
if('IntersectionObserver' in window){const io=new IntersectionObserver(es=>es.forEach(en=>{vis[en.target.id]=en.isIntersecting}),{rootMargin:'80px'});['mechanism','decision','threshold'].forEach(id=>io.observe(document.getElementById(id)))}
let last=performance.now();
function tick(nowMs){const dt=Math.min(.1,(nowMs-last)/1000);last=nowMs;if(!reduce&&!FROZEN)T+=dt;
 if(vis.mechanism!==false){brainFrame(reduce?3.2:T,dt);liveFrame(reduce?0:T)}
 if(vis.decision!==false){if(playing&&!reduce&&T-stageT0>STAGE_SECONDS){setStage((stage+1)%STAGES.length)}
  const t=reduce?1.4:(T-stageT0);drawers(stG,t);
  const bar=$('#stagenav button[aria-current=step] i');if(bar&&playing&&!reduce)bar.style.width=Math.min(100,(T-stageT0)/STAGE_SECONDS*100)+'%'}
 requestAnimationFrame(tick)}
/* deep link for review/screenshots: #stage=3&mode=normal */
const m=/stage=(\d)/.exec(location.hash),mm=/mode=(normal|seizure)/.exec(location.hash);
setMode(mm?mm[1]:'seizure');sync=mode==='seizure'?1:0;setStage(m?Math.min(5,+m[1]):0);if(m){playing=false;pb.textContent='Play';pb.setAttribute('aria-pressed','false')}
const tm=/t=([\d.]+)/.exec(location.hash);if(tm){T=+tm[1]}
metrics();
if(reduce){brainFrame(3.2,1);liveFrame(0);drawers(stG,1.4)}
requestAnimationFrame(tick);
})();
"""


def render_html(data: dict) -> str:
    payload = json.dumps(data, separators=(",", ":")).replace("</", "<\\/")
    return f'{BODY}<script type="application/json" id="ns-data">{payload}</script><script>{JS}</script>'
