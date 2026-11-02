"""Closed-loop alerting: emits a simulated critical-notification payload."""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone

log = logging.getLogger("alerting")


def build_alert(patient_id: str, timestamp: int, confidence: float, model_version: str = "") -> dict:
    return {
        "type": "CRITICAL_NEURO_ANOMALY",
        "severity": "CRITICAL",
        "patientId": patient_id,
        "signalTimestamp": timestamp,
        "confidence": round(confidence, 4),
        "modelVersion": model_version,
        "raisedAt": datetime.now(timezone.utc).isoformat(),
        "action": "NOTIFY_ON_CALL_NEUROLOGIST",
    }


def dispatch_alert(alert: dict) -> None:
    # In production this would call a paging/notification service.
    log.critical("🚨 CRITICAL ALERT %s", json.dumps(alert))
