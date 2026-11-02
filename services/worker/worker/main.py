"""Kafka -> preprocess -> batch -> gRPC inference -> alert.

Delivery semantics: at-least-once. Offsets are committed only after a batch has been
fully handled (inference done or messages routed to the dead-letter topic).
"""
from __future__ import annotations

import json
import logging
import signal

from confluent_kafka import Consumer, KafkaError, Message, Producer, TopicPartition

from worker.alerting import build_alert, dispatch_alert
from worker.batcher import Batcher
from worker.config import Config
from worker.inference_client import InferenceClient, InferenceUnavailable
from worker.preprocessing import preprocess

log = logging.getLogger("worker")
MIN_SIGNAL_LEN = 16


class Worker:
    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self.running = True
        self.consumer = Consumer({
            "bootstrap.servers": cfg.kafka_brokers,
            "group.id": cfg.group_id,
            "enable.auto.commit": False,
            "auto.offset.reset": "earliest",
        })
        self.producer = Producer({"bootstrap.servers": cfg.kafka_brokers})
        self.client = InferenceClient(cfg.inference_target, cfg.grpc_timeout_s, cfg.grpc_max_retries)
        self.batcher: Batcher[tuple[Message, tuple]] = Batcher(cfg.batch_max_size, cfg.batch_max_wait_ms)

    # --- lifecycle -------------------------------------------------------
    def stop(self, *_: object) -> None:
        log.info("shutdown requested")
        self.running = False

    def run(self) -> None:
        self.consumer.subscribe([self.cfg.topic])
        log.info("worker started: topic=%s group=%s", self.cfg.topic, self.cfg.group_id)
        try:
            while self.running:
                msg = self.consumer.poll(0.1)
                if msg is not None:
                    self._ingest(msg)
                if self.batcher.should_flush():
                    self._flush()
            if len(self.batcher):
                self._flush()  # drain in-flight work before exiting
        finally:
            self.producer.flush(5)
            self.consumer.close()
            self.client.close()
            log.info("worker stopped")

    # --- message handling ------------------------------------------------
    def _ingest(self, msg: Message) -> None:
        err = msg.error()
        if err:
            if err.code() != KafkaError._PARTITION_EOF:
                log.error("kafka error: %s", err)
            return
        try:
            payload = json.loads(msg.value())
            raw = payload["signalVector"]
            if not isinstance(raw, list) or len(raw) < MIN_SIGNAL_LEN:
                raise ValueError("signalVector too short")
            signal_n, bands = preprocess(raw, self.cfg.sampling_rate_hz, self.cfg.norm_window)
            item = (str(payload["patientId"]), int(payload["timestamp"]), signal_n, bands)
        except (ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
            log.warning("poison message %s[%d]@%d: %s", msg.topic(), msg.partition(), msg.offset(), exc)
            self._dead_letter(msg, str(exc))
            self._commit([msg])
            return
        self.batcher.add((msg, item))

    def _flush(self) -> None:
        batch = self.batcher.drain()
        msgs = [m for m, _ in batch]
        items = [i for _, i in batch]
        try:
            response = self.client.predict_batch(items)
        except InferenceUnavailable as exc:
            # Do not commit: restart/rebalance will redeliver these messages.
            log.error("inference unavailable, %d messages left uncommitted: %s", len(msgs), exc)
            self._rewind(msgs)
            return
        except Exception as exc:  # non-retryable (e.g. INVALID_ARGUMENT): isolate the batch
            log.error("inference rejected batch: %s", exc)
            for m in msgs:
                self._dead_letter(m, f"inference error: {exc}")
            self._commit(msgs)
            return

        model_version = response.model_version
        for p in response.predictions:
            log.info("patient=%s ts=%d anomaly=%s confidence=%.3f",
                     p.patient_id, p.timestamp, p.anomaly_detected, p.confidence)
            if p.anomaly_detected:
                dispatch_alert(build_alert(p.patient_id, p.timestamp, p.confidence, model_version))
        self._commit(msgs)

    # --- kafka helpers ---------------------------------------------------
    def _commit(self, msgs: list[Message]) -> None:
        latest: dict[tuple[str, int], int] = {}
        for m in msgs:
            key = (m.topic(), m.partition())
            latest[key] = max(latest.get(key, -1), m.offset())
        offsets = [TopicPartition(t, p, o + 1) for (t, p), o in latest.items()]
        self.consumer.commit(offsets=offsets, asynchronous=False)

    def _rewind(self, msgs: list[Message]) -> None:
        """Seek back to the earliest failed offset per partition so processing is retried."""
        earliest: dict[tuple[str, int], int] = {}
        for m in msgs:
            key = (m.topic(), m.partition())
            earliest[key] = min(earliest.get(key, m.offset()), m.offset())
        for (t, p), o in earliest.items():
            self.consumer.seek(TopicPartition(t, p, o))

    def _dead_letter(self, msg: Message, reason: str) -> None:
        self.producer.produce(
            self.cfg.dlq_topic,
            value=msg.value(),
            key=msg.key(),
            headers=[("error", reason.encode()), ("source", f"{msg.topic()}:{msg.partition()}:{msg.offset()}".encode())],
        )
        self.producer.poll(0)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    worker = Worker(Config())
    signal.signal(signal.SIGTERM, worker.stop)
    signal.signal(signal.SIGINT, worker.stop)
    worker.run()


if __name__ == "__main__":
    main()
