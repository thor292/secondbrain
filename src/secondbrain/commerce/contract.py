"""Live Commerce: the message contract between the simulator, the engine and the wall.

Three parts that only talk over MQTT (and the engine's small HTTP API), so the
simulator can be swapped for real platform feeds without touching the rest:

    simulator  --(ingest broker, commerce/v1/...)-->  engine  --(live broker, sb/commerce/...)-->  wall (browser)
         ^                                               |
         +-------------- commerce/v1/_control -----------+  (apply offer, speed, pause)

Ingest broker: the development broker on localhost:51883 (anonymous). Live
broker: the permission-checked broker from ``secondbrain.live`` (publisher
login to write, the monitor login reads everything over WebSockets).

Money is Indian rupees (float, rupees). Times: ``sim_ts`` is the simulated
clock as ISO 8601 without timezone (IST implied), ``*_ms`` fields are wall
clock epoch milliseconds (``wall_ms()``), used for latency.

Ingest topics (simulator publishes, engine subscribes; QoS 0 for orders, 1 otherwise)
-------------------------------------------------------------------------------------

``commerce/v1/_world`` (retained) - who is connected:
    {"started_ms", "sim_start": "2026-10-20T00:00:00", "platforms": [
        {"id": "shopify", "name": "Shopify", "kind": "D2C online stores", "color": "#95BF47"}, ...],
     "merchants": [{"id": "shopify-0001", "platform": "shopify", "name": "Urban Threads",
                    "city": "Bengaluru", "category": "Fashion", "x": 0.0-1.0, "y": 0.0-1.0}, ...],
     "customers": 120000}
    (x, y: the merchant's position on the wall's map of India, longitude/latitude scaled to 0-1.)

``commerce/v1/{platform}/{merchant}/catalog`` (retained) - products:
    {"products": [{"sku", "name", "category", "price", "cost", "stock", "lead_days", "perishable": bool}]}

``commerce/v1/{platform}/{merchant}/history`` - backfill, sent once at start:
    {"days": ["2026-08-25", ...], "skus": [...], "units": [[per day, per sku]], "prices": [[...]]}
    ``units[d][s]`` is units of ``skus[s]`` sold on ``days[d]``; ``prices`` the average price paid.

``commerce/v1/{platform}/{merchant}/orders`` - live orders, one message per merchant per tick:
    {"sent_ms", "sim_ts", "orders": [{"id", "customer", "sku", "qty", "price", "offer": signal id | null}]}

``commerce/v1/{platform}/{merchant}/stock`` - stock after restocks and at each sim day close:
    {"sim_ts", "stock": {"sku": units}, "restocked": {"sku": units}}

``commerce/v1/_clock`` - every tick: {"sent_ms", "sim_ts", "speed", "running": bool, "orders_per_sec"}

``commerce/v1/_control`` - commands (the engine's HTTP API publishes these; both sides listen):
    {"cmd": "start"} | {"cmd": "pause"} | {"cmd": "speed", "speed": SPEEDS key}
    {"cmd": "apply_offer", "signal_id", "platform", "merchant", "sku", "discount": 0.08,
     "until": sim_ts, "by": "merchant" | "autopilot"}
    {"cmd": "reset"}

Live topics (engine publishes with the publisher login, the wall reads with the monitor login)
---------------------------------------------------------------------------------------------

``sb/commerce/stats`` - every 500 ms of wall time:
    {"sent_ms", "sim_ts", "speed", "running", "merchants", "customers", "series",
     "orders_total", "units_total", "revenue_total", "orders_per_sec", "revenue_per_sec", "msgs_per_sec",
     "active_customers_today", "signals_total", "signals_by_kind": {kind: n}, "offers_applied",
     "uplift_revenue", "platforms": {pid: {"orders_total", "revenue_total", "orders_per_sec"}},
     "latency_ms": {"broker_p50", "broker_p95", "process_p50", "signal_p50", "end_to_end_p50", "end_to_end_p95"},
     "accuracy": {"wape", "baseline_wape", "days_scored", "series_scored", "by_platform": {pid: wape}},
     "learning": {"outcomes", "hit_rate", "elasticity": {category: {"prior", "learned", "n"}}},
     "pipeline": {"ingested", "forecasts_updated", "signals_raised", "delivered"},
     "festival": {"name", "date", "days_to"} | null}
    ``latency_ms.broker``: order sent by the simulator -> received by the engine.
    ``process``: received -> state and forecasts updated. ``signal``: updated -> signal published.
    ``end_to_end``: order sent -> signal published (the wall adds its own delivery time).

``sb/commerce/ticker`` - a sample of live orders, at most ~20 a second, batched every 250 ms:
    {"sent_ms", "orders": [{"platform", "merchant", "merchant_name", "city", "product", "qty", "amount", "offer": bool}]}

``sb/commerce/signals/{platform}/{merchant}`` - one decision signal:
    {"id", "sent_ms", "sim_ts", "platform", "merchant", "merchant_name", "city", "sku", "product", "category",
     "kind": "discount" | "restock" | "winback" | "surge" | "festival" | "clearance",
     "urgency": "high" | "medium" | "low", "title", "detail",
     "action": {"type": "discount" | "restock" | "coupon" | "hold_price", "discount": 0.08, "hours": 48,
                "units": 140, "coupon": 50, "customers": 38},
     "expected": {"units", "revenue", "margin", "uplift_units"},
     "confidence": 0.0-1.0,
     "evidence": {"actual", "forecast", "deviation", "stock", "cover_days", "elasticity"},
     "latency_ms": order sent -> signal published, "autopilot": bool (true: applied automatically)}

``sb/commerce/outcomes/{platform}/{merchant}`` - an applied offer's window closed:
    {"signal_id", "sku", "product", "merchant_name", "discount", "expected_uplift_units", "actual_uplift_units",
     "uplift_pct", "revenue_gain", "worked": bool, "elasticity_before", "elasticity_after", "category"}

Engine HTTP API (ENGINE_PORT; the app proxies it at /commerce/api/...)
---------------------------------------------------------------------

    GET  /status                         same body as the last stats message
    GET  /merchants                      [{"id", "platform", "name", "city", "category", "x", "y",
                                           "orders_today", "revenue_today", "open_signals"}]
    GET  /merchants/{id}                 merchant, products with stock, cover days, today's actual vs
                                         forecast, open signals, recent outcomes
    GET  /merchants/{id}/series/{sku}    {"history": [{"day", "units"}] (last 42 days),
                                          "forecast": [{"day", "f", "lo", "hi"}] (next 14 days),
                                          "today": {"day", "actual", "expected_so_far", "forecast"},
                                          "offers": [{"from", "until", "discount"}], "model", "wape"}
    GET  /signals?limit=50               most recent signals, newest first
    POST /control                        body: a _control command (not apply_offer's "by"); publishes it
    POST /offers/{signal_id}/apply       the merchant accepts a signal's offer (by "merchant")
"""

import os
import time
from datetime import date, datetime
from zoneinfo import ZoneInfo

INGEST_HOST = "localhost"
INGEST_PORT = 51883
ENGINE_PORT = int(os.environ.get("SB_COMMERCE_ENGINE_PORT", "8766"))

# Overridable so tests and parallel runs on one broker never see each other's messages.
PREFIX = os.environ.get("SB_COMMERCE_PREFIX", "commerce/v1")
WORLD = f"{PREFIX}/_world"
CLOCK = f"{PREFIX}/_clock"
CONTROL = f"{PREFIX}/_control"

LIVE_PREFIX = os.environ.get("SB_COMMERCE_LIVE_PREFIX", "sb/commerce")  # must stay under sb/ (live broker ACL)
STATS = f"{LIVE_PREFIX}/stats"
TICKER = f"{LIVE_PREFIX}/ticker"

# Simulated seconds per wall second. The wall shows the labels.
SPEEDS = {
    "1x": 1,            # real time
    "1min": 60,         # a minute a second
    "1h": 3600,         # an hour a second
    "1day": 14400,      # a day in 6 seconds
}
DEFAULT_SPEED = "1h"

IST = ZoneInfo("Asia/Kolkata")


def ist_now() -> datetime:
    """The real time in India, without timezone (sim times are IST implied)."""
    return datetime.now(IST).replace(tzinfo=None)


# Live means today: the simulated day starts at today's midnight (IST), and on Start the
# simulator replays today's orders up to the real time (CATCH_UP_SPEED), then runs at the
# chosen speed. Pinned in tests with SB_COMMERCE_SIM_START.
SIM_START = os.environ.get("SB_COMMERCE_SIM_START") or ist_now().strftime("%Y-%m-%dT00:00:00")
CATCH_UP_SPEED = 14400
HISTORY_DAYS = 56

# Festivals both sides know in advance, like a merchant's sale calendar.
# The simulator decides how strongly each one moves demand; the engine has to learn it.
FESTIVALS = {
    date(2026, 10, 20): "Dussehra",
    date(2026, 11, 6): "Dhanteras",
    date(2026, 11, 8): "Diwali",
    date(2026, 11, 11): "Bhai Dooj",
    date(2026, 11, 24): "Wedding season",
    date(2026, 12, 25): "Christmas",
}


def wall_ms() -> int:
    return int(time.time() * 1000)


def merchant_topic(platform: str, merchant: str, kind: str) -> str:
    """kind: catalog, history, orders, stock."""
    return f"{PREFIX}/{platform}/{merchant}/{kind}"


def parse_merchant_topic(topic: str) -> tuple[str, str, str] | None:
    """(platform, merchant, kind) for a merchant topic, None for _world, _clock, _control."""
    if not topic.startswith(PREFIX + "/"):
        return None
    parts = topic[len(PREFIX) + 1:].split("/")
    if len(parts) != 3 or parts[0].startswith("_"):
        return None
    return parts[0], parts[1], parts[2]


def signal_topic(platform: str, merchant: str) -> str:
    return f"{LIVE_PREFIX}/signals/{platform}/{merchant}"


def outcome_topic(platform: str, merchant: str) -> str:
    return f"{LIVE_PREFIX}/outcomes/{platform}/{merchant}"


def next_festival(day: date) -> tuple[date, str] | None:
    upcoming = [(d, n) for d, n in sorted(FESTIVALS.items()) if d >= day]
    return upcoming[0] if upcoming else None
