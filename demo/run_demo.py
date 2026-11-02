"""No-Docker demo of NeuroStream, driven by REAL held-out EEG.

Real code:  worker preprocessing/batcher/gRPC client/alerting, gRPC server, trained CNN.
Real data:  recordings from the Bonn dataset that the model never saw during training.
Stand-ins:  an in-memory queue replaces Kafka; a function replaces the NestJS gateway.

Usage (repo root):  .venv\Scripts\python demo\run_demo.py --chunks 40
"""
from __future__ import annotations

import argparse
import logging
import queue
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "services" / "ml-server"), str(ROOT / "services" / "worker"), str(ROOT / "simulator")]


def generate_stubs() -> None:
    for pkg, dest in (("app", ROOT / "services/ml-server/app"), ("worker", ROOT / "services/worker/worker")):
        if (dest / "inference_pb2.py").exists():
            continue
        subprocess.check_call([sys.executable, "-m", "grpc_tools.protoc", "-I", str(ROOT / "proto"),
                               f"--python_out={dest}", f"--grpc_python_out={dest}", str(ROOT / "proto/inference.proto")])
        f = dest / "inference_pb2_grpc.py"
        f.write_text(f.read_text().replace("import inference_pb2 as", f"from {pkg} import inference_pb2 as"))


def validate(p: dict) -> None:
    """Mirror of the NestJS TelemetryChunkDto rules."""
    assert isinstance(p.get("patientId"), str) and 0 < len(p["patientId"]) <= 64, "bad patientId"
    assert isinstance(p.get("timestamp"), (int, float)) and p["timestamp"] > 0, "bad timestamp"
    v = p.get("signalVector")
    assert isinstance(v, list) and 16 <= len(v) <= 4096, "bad signalVector length"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks", type=int, default=40)
    ap.add_argument("--seizure-prob", type=float, default=0.25)
    ap.add_argument("--seed", type=int, default=3)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-8s %(name)s | %(message)s")
    generate_stubs()

    import numpy as np
    from app.server import build_server
    from simulate_patient import Replay
    from worker.alerting import build_alert, dispatch_alert
    from worker.batcher import Batcher
    from worker.config import Config
    from worker.inference_client import InferenceClient
    from worker.preprocessing import preprocess

    cfg, log = Config(), logging.getLogger("demo")
    server = build_server("50551")
    server.start()
    client = InferenceClient("localhost:50551")
    kafka: queue.Queue = queue.Queue()
    batcher: Batcher = Batcher(max_size=4, max_wait_ms=300)
    rng = np.random.default_rng(args.seed)
    replay = Replay(ROOT, rng)
    truth: dict[int, bool] = {}
    tp = fp = fn = tn = 0

    def flush() -> None:
        nonlocal tp, fp, fn, tn
        resp = client.predict_batch(batcher.drain())
        for p in resp.predictions:
            actual = truth[p.timestamp]
            tp += actual and p.anomaly_detected; fp += (not actual) and p.anomaly_detected
            fn += actual and not p.anomaly_detected; tn += (not actual) and not p.anomaly_detected
            log.getChild("worker").info("anomaly=%-5s confidence=%.3f   (ground truth: %s)",
                                        p.anomaly_detected, p.confidence, "SEIZURE" if actual else "normal")
            if p.anomaly_detected:
                dispatch_alert(build_alert(p.patient_id, p.timestamp, p.confidence, resp.model_version))

    for i in range(args.chunks):
        seizure = bool(rng.random() < args.seizure_prob)
        ts = int(time.time() * 1000) + i
        payload = {"patientId": "patient-001", "timestamp": ts, "signalVector": replay.next_chunk(seizure)}
        truth[ts] = seizure
        validate(payload)                                              # gateway
        kafka.put(payload)                                             # kafka
        raw = kafka.get()                                              # worker consumes
        sig, bands = preprocess(raw["signalVector"], cfg.sampling_rate_hz, cfg.norm_window)
        batcher.add((raw["patientId"], raw["timestamp"], sig, bands))
        if batcher.should_flush() or i == args.chunks - 1:
            flush()
        time.sleep(0.02)

    log.info("DONE  chunks=%d  seizures caught=%d missed=%d  false alarms=%d  correct normals=%d",
             args.chunks, tp, fn, fp, tn)
    client.close(); server.stop(0)


if __name__ == "__main__":
    main()
