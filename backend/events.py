"""Bounded in-memory event log plus the fan-out used by the SSE endpoint.

Every interesting thing the backend does is appended here with a monotonically
increasing ``seq``.  The browser's EventSource reconnects with ``Last-Event-ID``
and the SSE endpoint replays from this buffer, so a dropped connection never
duplicates or restarts inference (a reconnect is a plain GET).

``emit`` is called from the MLX worker thread as well as from the asyncio
thread, so subscribers are only touched through ``loop.call_soon_threadsafe``.
"""

from __future__ import annotations

import asyncio
import threading
import time
from collections import deque
from typing import Any

MAX_EVENTS = 3000
SUBSCRIBER_QUEUE = 4096


class EventLog:
    def __init__(self, max_events: int = MAX_EVENTS):
        self._events: deque[dict[str, Any]] = deque(maxlen=max_events)
        self._lock = threading.Lock()
        self._seq = 0
        self._subscribers: set[asyncio.Queue] = set()
        self._loop: asyncio.AbstractEventLoop | None = None

    def bind_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop

    def emit(self, type: str, data: dict[str, Any] | None = None) -> dict[str, Any]:
        """Append one event and push it to every connected subscriber."""
        with self._lock:
            self._seq += 1
            event = {
                "seq": self._seq,
                "time": time.time(),
                "type": type,
                "data": data or {},
            }
            self._events.append(event)
            subscribers = list(self._subscribers)

        if self._loop is not None:
            for queue in subscribers:
                self._loop.call_soon_threadsafe(self._offer, queue, event)
        return event

    @staticmethod
    def _offer(queue: asyncio.Queue, event: dict[str, Any]) -> None:
        """Put one event into a subscriber queue, replacing backlog on overflow.

        If a client cannot keep up we do not want to drip stale token events at
        it forever.  Drop the backlog and tell it to resync instead; it will
        refetch the generation record and /api/state.
        """
        try:
            queue.put_nowait(event)
            return
        except asyncio.QueueFull:
            pass
        while True:
            try:
                queue.get_nowait()
            except asyncio.QueueEmpty:
                break
        try:
            queue.put_nowait(
                {
                    "seq": event["seq"],
                    "time": event["time"],
                    "type": "resync",
                    "data": {"reason": "subscriber queue overflow; refetch state"},
                }
            )
        except asyncio.QueueFull:
            pass

    def subscribe(
        self, last_event_id: int | None
    ) -> tuple[asyncio.Queue, list[dict[str, Any]]]:
        """Register a subscriber and return its backlog (possibly one resync)."""
        queue: asyncio.Queue = asyncio.Queue(maxsize=SUBSCRIBER_QUEUE)
        with self._lock:
            self._subscribers.add(queue)
            oldest = self._events[0]["seq"] if self._events else self._seq + 1
            latest = self._seq
            if last_event_id is None:
                backlog: list[dict[str, Any]] = []
            elif last_event_id < oldest - 1:
                backlog = [
                    {
                        "seq": self._seq,
                        "time": time.time(),
                        "type": "resync",
                        "data": {
                            "reason": "Last-Event-ID fell out of the buffer",
                            "oldest_seq": oldest,
                        },
                    }
                ]
            elif last_event_id > latest:
                backlog = [
                    {
                        "seq": self._seq,
                        "time": time.time(),
                        "type": "resync",
                        "data": {"reason": "Last-Event-ID is ahead of this server"},
                    }
                ]
            else:
                backlog = [e for e in self._events if e["seq"] > last_event_id]
        return queue, backlog

    def unsubscribe(self, queue: asyncio.Queue) -> None:
        with self._lock:
            self._subscribers.discard(queue)

    def summary(self) -> dict[str, Any]:
        with self._lock:
            return {
                "seq": self._seq,
                "buffered": len(self._events),
                "oldest_seq": self._events[0]["seq"] if self._events else None,
                "subscribers": len(self._subscribers),
            }


def format_sse(event: dict[str, Any]) -> str:
    """One SSE frame.  The id is the seq, so Last-Event-ID is meaningful."""
    import json

    return (
        f"id: {event['seq']}\n"
        f"event: {event['type']}\n"
        f"data: {json.dumps(event['data'], separators=(',', ':'))}\n\n"
    )
