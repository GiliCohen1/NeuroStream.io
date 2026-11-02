"""End-to-end check against the running docker compose stack.

Sends held-out real EEG through the REAL path
    HTTP -> NestJS gateway -> Kafka -> Python worker -> gRPC -> trained CNN -> alert
and asserts on the worker's logs:
  * seizure chunks (patient "e2e-seizure") raise CRITICAL ALERTs for most chunks
  * normal chunks  (patient "e2e-normal")  raise (almost) none
Exit code 0 = pass.  Usage: python scripts/e2e_check.py [--compose-dir .]
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "simulator"))
from simulate_patient import Replay  # noqa: E402

URL = "http://localhost:3000"
N_PER_CLASS = 12
MIN_SEIZURE_RECALL = 0.75   # held-out test recall is ~0.95; leave slack for tiny-sample noise
MAX_NORMAL_ALERTS = 1


def wait_for_gateway(timeout: float = 120) -> None:
    end = time.time() + timeout
    while time.time() < end:
        try:
            if requests.get(f"{URL}/health", timeout=2).status_code == 200:
                return
        except requests.RequestException:
            pass
        time.sleep(2)
    sys.exit("FAIL: gateway never became healthy")


def post(patient: str, chunk: list[float], ts: int) -> None:
    for attempt in range(10):  # the producer may still be connecting right after startup
        r = requests.post(f"{URL}/api/v1/telemetry/stream", timeout=10,
                          json={"patientId": patient, "timestamp": ts, "signalVector": chunk})
        if r.status_code == 202:
            return
        print(f"  retry {attempt + 1}: HTTP {r.status_code} {r.text[:120]}")
        time.sleep(2)
    sys.exit(f"FAIL: gateway rejected valid chunk ({r.status_code})")


def worker_alerts(compose_dir: str) -> list[dict]:
    logs = subprocess.run(["docker", "compose", "logs", "worker", "--no-log-prefix"], cwd=compose_dir,
                          capture_output=True, text=True, check=True).stdout
    alerts = []
    for line in logs.splitlines():
        if "CRITICAL ALERT" in line:
            alerts.append(json.loads(line[line.index("{"):]))
    return alerts


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--compose-dir", default=str(ROOT))
    args = ap.parse_args()

    wait_for_gateway()
    replay = Replay(ROOT, np.random.default_rng(11))
    base = int(time.time() * 1000)
    print(f"sending {N_PER_CLASS} seizure + {N_PER_CLASS} normal held-out chunks through the gateway")
    for i in range(N_PER_CLASS):
        post("e2e-seizure", replay.next_chunk(True), base + i)
        post("e2e-normal", replay.next_chunk(False), base + 1000 + i)

    deadline, alerts = time.time() + 90, []
    while time.time() < deadline:
        alerts = [a for a in worker_alerts(args.compose_dir) if a["signalTimestamp"] >= base]
        if sum(a["patientId"] == "e2e-seizure" for a in alerts) >= int(N_PER_CLASS * MIN_SEIZURE_RECALL):
            time.sleep(5)  # let any trailing (false) alerts for the normal patient land too
            alerts = [a for a in worker_alerts(args.compose_dir) if a["signalTimestamp"] >= base]
            break
        time.sleep(3)

    caught = sum(a["patientId"] == "e2e-seizure" for a in alerts)
    false_alarms = sum(a["patientId"] == "e2e-normal" for a in alerts)
    print(f"seizure chunks alerted: {caught}/{N_PER_CLASS}   false alarms on normal chunks: {false_alarms}/{N_PER_CLASS}")
    print(f"model version in alerts: {sorted({a['modelVersion'] for a in alerts})}")

    problems = []
    if caught < int(N_PER_CLASS * MIN_SEIZURE_RECALL):
        problems.append(f"too few seizure alerts ({caught}/{N_PER_CLASS})")
    if false_alarms > MAX_NORMAL_ALERTS:
        problems.append(f"too many false alarms ({false_alarms}/{N_PER_CLASS})")
    if any(not a["modelVersion"] for a in alerts):
        problems.append("alert missing modelVersion")
    if problems:
        sys.exit("FAIL: " + "; ".join(problems))
    print("PASS: full pipeline verified end to end")


if __name__ == "__main__":
    main()
