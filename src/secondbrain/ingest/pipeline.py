"""What happens after a company's data arrives: forecasts and signals are recalculated.

``refresh_company`` is the one hook. Recalculating is expensive (every
product's models are backtested), so arrivals don't call it directly: they
mark the company as changed, and a ``RefreshScheduler`` refreshes each
changed company at most once per ``SB_INGEST_REFRESH_MINUTES``. The last
batch of a burst is never left out: a company that changes again during the
quiet period is refreshed when the period ends.

With several processes (the API and the MQTT bridge), the quiet period is
shared through Redis, so together they still refresh a company at most once
per period.
"""

import logging
import threading
import time
from collections.abc import Callable

import psycopg
import redis
from pydantic_settings import BaseSettings, SettingsConfigDict

from secondbrain import evolution, signals
from secondbrain.config import Settings
from secondbrain.forecasting.__main__ import run as run_forecasts
from secondbrain.ingest.processor import fill_gaps
from secondbrain.models import LLM, Embedder

log = logging.getLogger(__name__)

FORECAST_HORIZON = 120


class IngestSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="SB_INGEST_", env_file=".env", extra="ignore")

    refresh_minutes: float = 10.0
    # Off: data is stored but nothing is recalculated until asked (POST /ingest/refresh).
    auto_refresh: bool = True


def refresh_company(settings: Settings, llm: LLM, embedder: Embedder, company: str) -> None:
    """Recalculate a company's forecasts and signals from the data it has now."""
    with psycopg.connect(settings.admin_database_url, autocommit=True) as conn:
        fill_gaps(conn, company)
    if hasattr(evolution, "refresh"):
        # Judges outcomes on the real sales, then reforecasts and recalculates signals.
        evolution.refresh(settings, llm, embedder, company)
        return
    try:
        run_forecasts(settings.admin_database_url, under=company, horizon=FORECAST_HORIZON)
    except ValueError as exc:
        # Typically too little history to backtest yet; signals still use the last forecasts.
        log.warning("Forecasts for %s not refreshed: %s", company, exc)
    signals.run(settings, llm, company)


class RefreshScheduler:
    """Debounced refreshes: at most one per company per ``min_interval`` seconds."""

    def __init__(self, refresh: Callable[[str], None], min_interval: float,
                 redis_url: str | None = None, tick: float = 5.0):
        self._refresh = refresh
        self._interval = min_interval
        self._tick = tick
        self._redis = redis.Redis.from_url(redis_url) if redis_url else None
        self._dirty: set[str] = set()
        self._last: dict[str, float] = {}
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None

    def mark(self, company: str) -> None:
        with self._lock:
            self._dirty.add(company)

    def _may_run(self, company: str) -> bool:
        if time.monotonic() - self._last.get(company, -1e12) < self._interval:
            return False
        if self._redis is None or self._interval <= 0:
            return True
        try:
            return bool(self._redis.set(f"sb:ingest:refreshed:{company}", 1, nx=True,
                                        px=max(int(self._interval * 1000), 1)))
        except redis.RedisError:
            return True  # without Redis, fall back to this process's own timing

    def run_due(self) -> list[str]:
        """Refresh every changed company whose quiet period is over; returns them."""
        with self._lock:
            due = [c for c in sorted(self._dirty) if self._may_run(c)]
            self._dirty.difference_update(due)
            for c in due:
                self._last[c] = time.monotonic()
        for company in due:
            self.refresh_now(company, mark_time=False)
        return due

    def refresh_now(self, company: str, mark_time: bool = True) -> None:
        if mark_time:
            with self._lock:
                self._last[company] = time.monotonic()
                self._dirty.discard(company)
        try:
            self._refresh(company)
            log.info("Refreshed forecasts and signals for %s", company)
        except Exception:  # a failed refresh must not stop the next ones
            log.exception("Refresh of %s failed", company)

    def start(self) -> "RefreshScheduler":
        if self._thread is None:
            self._thread = threading.Thread(target=self._loop, name="ingest-refresh", daemon=True)
            self._thread.start()
        return self

    def _loop(self) -> None:
        while True:
            time.sleep(self._tick)
            self.run_due()


def scheduler_for(settings: Settings, llm: LLM, embedder: Embedder,
                  ingest: IngestSettings | None = None) -> RefreshScheduler:
    ingest = ingest or IngestSettings()
    return RefreshScheduler(lambda company: refresh_company(settings, llm, embedder, company),
                            ingest.refresh_minutes * 60, settings.redis_url)
