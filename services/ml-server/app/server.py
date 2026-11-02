"""gRPC inference server."""
from __future__ import annotations

import logging
import os
import signal
import time
from concurrent import futures

import grpc

import torch

from app import inference_pb2, inference_pb2_grpc
from app.model import SpikeDetector

log = logging.getLogger("ml-server")
MAX_BATCH = 256


class InferenceService(inference_pb2_grpc.InferenceServiceServicer):
    def __init__(self, detector: SpikeDetector) -> None:
        self._detector = detector

    def PredictBatch(self, request, context):
        started = time.perf_counter()
        samples = request.samples
        if not samples:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, "empty batch")
        if len(samples) > MAX_BATCH:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, f"batch too large (max {MAX_BATCH})")
        if any(len(s.signal) < 16 for s in samples):
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, "signal too short")

        try:
            scores = self._detector.score(
                [list(s.signal) for s in samples],
                [list(s.band_powers) for s in samples],
            )
        except Exception:
            log.exception("inference failed")
            context.abort(grpc.StatusCode.INTERNAL, "inference failure")

        predictions = [
            inference_pb2.Prediction(
                patient_id=s.patient_id,
                timestamp=s.timestamp,
                anomaly_detected=sc >= self._detector.threshold,
                confidence=round(sc, 4),
            )
            for s, sc in zip(samples, scores)
        ]
        latency = (time.perf_counter() - started) * 1000
        log.info("batch=%d latency=%.1fms anomalies=%d", len(samples), latency,
                 sum(p.anomaly_detected for p in predictions))
        return inference_pb2.PredictBatchResponse(
            predictions=predictions, model_version=self._detector.version, server_latency_ms=latency
        )


def build_server(port: str, detector: SpikeDetector | None = None) -> grpc.Server:
    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=int(os.getenv("MAX_WORKERS", "4"))),
        options=[("grpc.max_receive_message_length", 32 * 1024 * 1024)],
    )
    inference_pb2_grpc.add_InferenceServiceServicer_to_server(
        InferenceService(detector or SpikeDetector()), server
    )
    server.add_insecure_port(f"[::]:{port}")
    return server


def serve() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    torch.set_num_threads(int(os.getenv("TORCH_NUM_THREADS", "2")))
    port = os.getenv("GRPC_PORT", "50051")

    detector = SpikeDetector()
    server = build_server(port, detector)
    server.start()
    log.info("ML server listening on :%s (model=%s)", port, detector.version)

    def _stop(*_):
        log.info("shutting down")
        server.stop(grace=5)

    signal.signal(signal.SIGTERM, _stop)
    signal.signal(signal.SIGINT, _stop)
    server.wait_for_termination()


if __name__ == "__main__":
    serve()
