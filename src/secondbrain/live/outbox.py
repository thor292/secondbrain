"""After-commit change events, for apps that don't have them yet.

The rule: a message is published only after the database transaction that caused it has
committed. Otherwise subscribers could be told about a memory that was rolled back.

    bus = EventBus()
    bus.listeners.append(MqttPublisher(settings))

    with bus.collect():               # wrap the same block as your transaction
        with conn.transaction():
            ...insert the memory...
            bus.emit({"event": "created", "memory_id": mid, "brain_path": "root.acme.sales",
                      "sensitivity": 1, "user_id": "riya", "time": now.isoformat()})
    # leaving collect() normally delivers the events; an exception discards them

If your storage layer already has hooks (SQLAlchemy after_commit, Django on_commit, ...),
call `bus.notify(events)` from there instead of using `collect()`.
"""
import logging
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Callable, Iterator

log = logging.getLogger(__name__)
_box: ContextVar[list[dict] | None] = ContextVar("sb_mqtt_outbox", default=None)


class EventBus:
    def __init__(self) -> None:
        self.listeners: list[Callable[[list[dict]], None]] = []

    @contextmanager
    def collect(self) -> Iterator[None]:
        box: list[dict] = []
        token = _box.set(box)
        try:
            yield
        finally:
            _box.reset(token)
        self.notify(box)  # only reached when the block did not raise

    def emit(self, event: dict) -> None:
        """Queue an event inside `collect()`. Outside it, the event is ignored."""
        box = _box.get()
        if box is not None:
            box.append(event)

    def notify(self, events: list[dict]) -> None:
        if not events:
            return
        for listener in self.listeners:
            try:
                listener(events)
            except Exception:  # a listener failure must never undo a committed write
                log.exception("event listener failed")
