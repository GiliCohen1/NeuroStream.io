"""Offset-commit safety of the worker, tested with fake Kafka and inference objects (no broker needed).

Kafka offsets are cumulative: committing offset N+1 for a partition declares *every* earlier offset in that
partition done. The worker must therefore never commit past a message that is still waiting in its batch buffer
or whose inference failed. The fake consumer below enforces exactly that rule on every commit call.
"""
from __future__ import annotations

import importlib
import json
import sys
import types
from types import SimpleNamespace

import numpy as np
import pytest

TOPIC = "neuro-raw-signals"
DLQ = "neuro-dead-letter"


class FakeMsg:
    def __init__(self, partition: int, offset: int, value: bytes):
        self._p, self._o, self._v = partition, offset, value

    def topic(self): return TOPIC
    def partition(self): return self._p
    def offset(self): return self._o
    def value(self): return self._v
    def key(self): return b"patient"
    def error(self): return None


class World:
    """Shared ledger: which offsets were consumed, and which were actually handled."""

    def __init__(self):
        self.consumed: dict[int, set[int]] = {}
        self.handled: set[tuple[int, int]] = set()     # (partition, offset) predicted or dead-lettered
        self.commits: list[tuple[int, int]] = []       # (partition, committed offset)
        self.seeks: list[tuple[int, int]] = []         # (partition, offset)
        self.dlq: list[dict] = []
        self.predict_calls: list[list[tuple[int, int]]] = []


class FakeConsumer:
    def __init__(self, world: World): self.w = world

    def commit(self, offsets, asynchronous=False):
        for tp in offsets:
            for off in self.w.consumed.get(tp.partition, ()):
                if off < tp.offset and (tp.partition, off) not in self.w.handled:
                    raise AssertionError(
                        f"commit of offset {tp.offset} on partition {tp.partition} would skip "
                        f"unprocessed offset {off} (it is still buffered or its inference failed)")
            self.w.commits.append((tp.partition, tp.offset))

    def seek(self, tp): self.w.seeks.append((tp.partition, tp.offset))
    def subscribe(self, *_): pass
    def poll(self, *_): return None
    def close(self): pass


class FakeProducer:
    def __init__(self, world: World): self.w = world

    def produce(self, topic, value=None, key=None, headers=None):
        h = {k: v.decode() for k, v in headers}
        _, part, off = h["source"].split(":")
        self.w.handled.add((int(part), int(off)))
        self.w.dlq.append({"topic": topic, "error": h["error"], "source": h["source"]})

    def poll(self, *_): pass
    def flush(self, *_): pass


class FakeClient:
    def __init__(self, world: World, unavailable_exc, fail: Exception | None = None):
        self.w, self.fail, self._unavailable = world, fail, unavailable_exc

    def predict_batch(self, items):
        decoded = []
        for _pid, ts, _sig, _bands in items:
            part, off = divmod(ts - 1_000_000, 1000)
            decoded.append((part, off))
        self.w.predict_calls.append(decoded)
        if self.fail is not None:
            raise self.fail
        self.w.handled.update(decoded)
        preds = [SimpleNamespace(patient_id="p", timestamp=ts, anomaly_detected=False, confidence=0.0)
                 for _pid, ts, _s, _b in items]
        return SimpleNamespace(predictions=preds, model_version="test")

    def close(self): pass


@pytest.fixture
def env(monkeypatch):
    """Import worker.main with Kafka and the gRPC client stubbed out, so the test needs only numpy and scipy."""
    class _Unused:
        def __init__(self, *a, **k): raise AssertionError("real Kafka client must not be built in unit tests")

    class TopicPartition:
        def __init__(self, topic, partition, offset=-1001): self.topic, self.partition, self.offset = topic, partition, offset

    class KafkaError:
        _PARTITION_EOF = -191

    ck = types.ModuleType("confluent_kafka")
    ck.Consumer = ck.Producer = _Unused
    ck.Message, ck.KafkaError, ck.TopicPartition = object, KafkaError, TopicPartition

    class InferenceUnavailable(Exception): pass

    ic = types.ModuleType("worker.inference_client")
    ic.InferenceClient, ic.InferenceUnavailable = _Unused, InferenceUnavailable

    monkeypatch.setitem(sys.modules, "confluent_kafka", ck)
    monkeypatch.setitem(sys.modules, "worker.inference_client", ic)
    monkeypatch.delitem(sys.modules, "worker.main", raising=False)
    mod = importlib.import_module("worker.main")

    def make(fail: Exception | None = None):
        world = World()
        worker = mod.Worker(mod.Config(), consumer=FakeConsumer(world), producer=FakeProducer(world),
                            client=FakeClient(world, InferenceUnavailable, fail))
        return world, worker

    yield SimpleNamespace(make=make, Unavailable=InferenceUnavailable)
    sys.modules.pop("worker.main", None)


def feed(world: World, worker, partition: int, offset: int, good: bool = True) -> None:
    if good:
        value = json.dumps({"patientId": "p", "timestamp": 1_000_000 + partition * 1000 + offset,
                            "signalVector": np.sin(np.arange(178) / 7).tolist()}).encode()
    else:
        value = b"this is not json {"
    world.consumed.setdefault(partition, set()).add(offset)
    worker._ingest(FakeMsg(partition, offset, value))


def test_poison_commit_never_skips_messages_still_in_the_batch(env):
    """THE BUG: offsets 0 and 1 are buffered, offset 2 is malformed. Committing 3 immediately would mark 0 and 1 done."""
    world, worker = env.make()
    feed(world, worker, 0, 0)
    feed(world, worker, 0, 1)
    feed(world, worker, 0, 2, good=False)        # the fake consumer raises if this commit overtakes 0 or 1

    assert world.predict_calls == [[(0, 0), (0, 1)]], "buffered messages must be processed before the poison is committed"
    assert [d["source"] for d in world.dlq] == [f"{TOPIC}:0:2"]
    assert max(o for p, o in world.commits if p == 0) == 3
    assert len(worker.batcher) == 0


def test_poison_alone_is_dead_lettered_and_committed(env):
    world, worker = env.make()
    feed(world, worker, 0, 0, good=False)

    assert world.predict_calls == []
    assert world.dlq == [{"topic": DLQ, "error": world.dlq[0]["error"], "source": f"{TOPIC}:0:0"}]
    assert world.dlq[0]["error"]
    assert world.commits == [(0, 1)]


def test_failed_flush_on_poison_commits_nothing_and_rewinds_every_partition(env):
    """If the buffered messages cannot be processed, the poison must not be committed either, and it must be redelivered."""
    world, worker = env.make(fail=env.Unavailable("down"))
    feed(world, worker, 0, 0)
    feed(world, worker, 0, 1)
    feed(world, worker, 1, 5, good=False)        # poison on a *different* partition

    assert world.commits == []
    assert world.dlq == [], "not dead-lettered yet: it will be handled again when redelivered"
    assert sorted(world.seeks) == [(0, 0), (1, 5)], "both the failed batch and the poison message must be redelivered"


def test_successful_flush_commits_highest_offset_plus_one_per_partition(env):
    world, worker = env.make()
    for part, off in [(0, 3), (0, 4), (1, 9)]:
        feed(world, worker, part, off)
    worker._flush()

    assert sorted(world.commits) == [(0, 5), (1, 10)]


def test_inference_unavailable_rewinds_without_committing(env):
    world, worker = env.make(fail=env.Unavailable("down"))
    for part, off in [(0, 4), (0, 5), (1, 2)]:
        feed(world, worker, part, off)
    worker._flush()

    assert world.commits == []
    assert sorted(world.seeks) == [(0, 4), (1, 2)]


def test_non_retryable_inference_error_dead_letters_the_whole_batch(env):
    world, worker = env.make(fail=ValueError("bad batch"))
    feed(world, worker, 0, 0)
    feed(world, worker, 0, 1)
    worker._flush()

    assert [d["source"] for d in world.dlq] == [f"{TOPIC}:0:0", f"{TOPIC}:0:1"]
    assert world.commits == [(0, 2)]
