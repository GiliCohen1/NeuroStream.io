from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    kafka_brokers: str = os.getenv("KAFKA_BROKERS", "localhost:29092")
    topic: str = os.getenv("KAFKA_TOPIC", "neuro-raw-signals")
    dlq_topic: str = os.getenv("KAFKA_DLQ_TOPIC", "neuro-dead-letter")
    group_id: str = os.getenv("KAFKA_GROUP_ID", "neuro-preprocessing-workers")
    inference_target: str = os.getenv("INFERENCE_TARGET", "localhost:50051")
    batch_max_size: int = int(os.getenv("BATCH_MAX_SIZE", "8"))
    batch_max_wait_ms: int = int(os.getenv("BATCH_MAX_WAIT_MS", "500"))
    sampling_rate_hz: float = float(os.getenv("SAMPLING_RATE_HZ", "173.61"))
    norm_window: int = int(os.getenv("NORM_WINDOW", "64"))
    grpc_timeout_s: float = float(os.getenv("GRPC_TIMEOUT_S", "5"))
    grpc_max_retries: int = int(os.getenv("GRPC_MAX_RETRIES", "4"))
