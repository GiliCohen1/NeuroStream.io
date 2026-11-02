"""gRPC client with exponential backoff on transient failures."""
from __future__ import annotations

import logging
import random
import time
from typing import Sequence

import grpc
import numpy as np

from worker import inference_pb2, inference_pb2_grpc

log = logging.getLogger("inference-client")
_RETRYABLE = {grpc.StatusCode.UNAVAILABLE, grpc.StatusCode.DEADLINE_EXCEEDED, grpc.StatusCode.RESOURCE_EXHAUSTED}


class InferenceUnavailable(Exception):
    """Raised when the inference server cannot be reached after all retries."""


class InferenceClient:
    def __init__(self, target: str, timeout_s: float = 5.0, max_retries: int = 4):
        self._channel = grpc.insecure_channel(target)
        self._stub = inference_pb2_grpc.InferenceServiceStub(self._channel)
        self._timeout = timeout_s
        self._max_retries = max_retries

    def predict_batch(self, items: Sequence[tuple[str, int, np.ndarray, np.ndarray]]):
        """items: (patient_id, timestamp, normalized_signal, band_powers)."""
        request = inference_pb2.PredictBatchRequest(
            samples=[
                inference_pb2.Sample(
                    patient_id=pid, timestamp=ts, signal=sig.tolist(), band_powers=bands.tolist()
                )
                for pid, ts, sig, bands in items
            ]
        )
        for attempt in range(self._max_retries + 1):
            try:
                return self._stub.PredictBatch(request, timeout=self._timeout)
            except grpc.RpcError as err:
                code = err.code()
                if code not in _RETRYABLE or attempt == self._max_retries:
                    if code in _RETRYABLE:
                        raise InferenceUnavailable(str(err)) from err
                    raise
                delay = min(8.0, 0.25 * 2**attempt) * (0.5 + random.random() / 2)
                log.warning("inference %s (attempt %d/%d), retrying in %.2fs",
                            code.name, attempt + 1, self._max_retries, delay)
                time.sleep(delay)

    def close(self) -> None:
        self._channel.close()
