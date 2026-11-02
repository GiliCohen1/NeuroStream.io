"""Dynamic micro-batcher: flush on size OR age, whichever comes first."""
from __future__ import annotations

import time
from typing import Callable, Generic, TypeVar

T = TypeVar("T")


class Batcher(Generic[T]):
    def __init__(self, max_size: int, max_wait_ms: int, clock: Callable[[], float] = time.monotonic):
        if max_size < 1:
            raise ValueError("max_size must be >= 1")
        self._max_size = max_size
        self._max_wait = max_wait_ms / 1000.0
        self._clock = clock
        self._items: list[T] = []
        self._oldest: float | None = None

    def add(self, item: T) -> None:
        if not self._items:
            self._oldest = self._clock()
        self._items.append(item)

    def __len__(self) -> int:
        return len(self._items)

    def should_flush(self) -> bool:
        if not self._items:
            return False
        if len(self._items) >= self._max_size:
            return True
        return (self._clock() - self._oldest) >= self._max_wait  # type: ignore[operator]

    def drain(self) -> list[T]:
        items, self._items, self._oldest = self._items, [], None
        return items
