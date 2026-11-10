"""Live Commerce on TimescaleDB: every order and the whole sales history in hypertables, and SQL
signals from them over MQTT.

The engine (``engine.py``) keeps its forecasts in memory for speed. This process is the data
platform next to it, a second consumer of the same ingest stream, the way a company's warehouse
sits next to its operational systems:

- **Writer.** Subscribes to the platforms' order and history messages on the ingest broker and
  COPYs them into hypertables in batches: ``cm_order_lines`` (one row per order line, chunked by
  simulated day), ``cm_products`` (the catalogues) and ``cm_product_daily`` (every product, every
  day: the 14-month backfill, then each live day as it closes, zeros and sold-out days included).
- **Rollups.** Continuous aggregates keep hourly and daily totals per merchant (``cm_sales_1h``,
  ``cm_sales_1d``, hierarchical on the hourly one) and daily units per product (``cm_sku_1d``). They
  are refreshed for the window the simulated clock just passed (simulated time runs ahead of the
  wall clock, so time-based refresh policies would not fit) and answer in real time beyond it.
- **Columnstore.** Simulated days older than two days are converted to TimescaleDB's columnstore
  (compressed, segmented by merchant), and the compression ratio is measured.
- **Watcher.** Each simulated day, SQL over the stored days reports facts worth acting on: rupees
  each store lost to empty shelves, and products out of stock three days running. Each simulated
  hour it compares every store with its own platform's other stores at that hour (festivals and
  weekdays move them all alike, so they cancel out) and reports a store far off its usual. Each
  finding goes out as a signal on the live broker, with the query and how long it took, and is
  kept in ``cm_findings``. Product-level spikes are left to the engine's forecasts: at one product
  in one store, a day's random swing is as large as most real changes, and a rule on raw history
  can't tell them apart (measured against the simulator's injected events).
- **Stats.** Once a second: rows written and per second, batch latency, chunks, columnstore ratio,
  rollup sizes, query times.

Run it with the rest of Live Commerce (``python -m secondbrain.commerce --tsdb``) or alone:
``uv run python -m secondbrain.commerce.tsdb``. Needs PostgreSQL with the TimescaleDB extension
(``docker-compose.timescale.yml``). Topics are in ``contract.py`` ("TimescaleDB").
"""

import argparse
import json
import logging
import math
import os
import queue
import threading
import time
from collections import deque
from datetime import datetime, timedelta

import psycopg
from psycopg.types.json import Jsonb

from secondbrain.commerce import contract as C

log = logging.getLogger("secondbrain.commerce.tsdb")

BATCH_SECONDS = 0.25
COMPRESS_AFTER_DAYS = 2
STATS_EVERY = 1.0
SIZES_EVERY = 5.0
HOUR_LOOKBACK_DAYS = 7          # a store-hour against the same hour on up to the last 7 days...
HOUR_MIN_DAYS = 2               # ...once at least two of those days exist
HOUR_Z = 5.0                    # standard deviations from usual (Poisson plus 22% day-to-day swing)
HOUR_MIN_USUAL = 8              # orders: quieter store-hours are too small to judge
HOUR_RATIO = (0.5, 1.8)         # ...and this far from usual, both ways
STREAK_DAYS = 3                 # a product out of stock at the close of this many days running
STOCKOUT_MIN_LOST = 25_000      # rupees lost to empty shelves in a day before a store is reported
PER_ROUND = 6                   # findings published per check, strongest first
COOLDOWN_HOURS = 12             # one finding per merchant (or product) per 12 simulated hours

SCHEMA = """
CREATE TABLE IF NOT EXISTS cm_order_lines (
    ts        timestamptz NOT NULL,
    platform  text        NOT NULL,
    merchant  text        NOT NULL,
    sku       text        NOT NULL,
    customer  text,
    qty       integer     NOT NULL,
    price     real        NOT NULL,
    revenue   real        NOT NULL,
    offer     boolean     NOT NULL DEFAULT false
);
SELECT create_hypertable('cm_order_lines', by_range('ts', INTERVAL '1 day'), if_not_exists => true);
CREATE INDEX IF NOT EXISTS cm_order_lines_merchant_ts ON cm_order_lines (merchant, ts DESC);

CREATE TABLE IF NOT EXISTS cm_products (
    platform  text NOT NULL,
    merchant  text NOT NULL,
    sku       text NOT NULL,
    name      text,
    category  text,
    price     real,
    PRIMARY KEY (merchant, sku)
);

-- every product, every day: the history backfill, then each live day written at its close
CREATE TABLE IF NOT EXISTS cm_product_daily (
    day       date    NOT NULL,
    platform  text    NOT NULL,
    merchant  text    NOT NULL,
    sku       text    NOT NULL,
    units     integer NOT NULL,
    revenue   real    NOT NULL,
    sold_out  boolean NOT NULL DEFAULT false,
    live      boolean NOT NULL DEFAULT false
);
SELECT create_hypertable('cm_product_daily', by_range('day', INTERVAL '1 month'), if_not_exists => true);
CREATE INDEX IF NOT EXISTS cm_product_daily_merchant_day ON cm_product_daily (merchant, day DESC);

-- products that ended a live day sold out (from the day-close stock messages)
CREATE TABLE IF NOT EXISTS cm_stockouts (
    day       date NOT NULL,
    merchant  text NOT NULL,
    sku       text NOT NULL,
    PRIMARY KEY (day, merchant, sku)
);

-- what the watcher reported, as it went out over MQTT
CREATE TABLE IF NOT EXISTS cm_findings (
    sim_ts    timestamptz NOT NULL,
    kind      text        NOT NULL,
    platform  text        NOT NULL,
    merchant  text        NOT NULL,
    body      jsonb       NOT NULL
);
SELECT create_hypertable('cm_findings', by_range('sim_ts', INTERVAL '7 days'), if_not_exists => true);

CREATE MATERIALIZED VIEW IF NOT EXISTS cm_sales_1h WITH (timescaledb.continuous, timescaledb.materialized_only = false) AS
    SELECT time_bucket(INTERVAL '1 hour', ts, 'Asia/Kolkata') AS bucket, platform, merchant,
           count(*) AS orders, sum(qty) AS units, sum(revenue) AS revenue,
           sum(revenue) FILTER (WHERE offer) AS offer_revenue
    FROM cm_order_lines GROUP BY 1, 2, 3 WITH NO DATA;

CREATE MATERIALIZED VIEW IF NOT EXISTS cm_sales_1d WITH (timescaledb.continuous, timescaledb.materialized_only = false) AS
    SELECT time_bucket(INTERVAL '1 day', bucket, 'Asia/Kolkata') AS bucket, platform, merchant,
           sum(orders) AS orders, sum(units) AS units, sum(revenue) AS revenue, sum(offer_revenue) AS offer_revenue
    FROM cm_sales_1h GROUP BY 1, 2, 3 WITH NO DATA;

CREATE MATERIALIZED VIEW IF NOT EXISTS cm_sku_1d WITH (timescaledb.continuous, timescaledb.materialized_only = false) AS
    SELECT time_bucket(INTERVAL '1 day', ts, 'Asia/Kolkata') AS bucket, platform, merchant, sku,
           sum(qty) AS units, sum(revenue) AS revenue
    FROM cm_order_lines GROUP BY 1, 2, 3, 4 WITH NO DATA;

"""

DROP = """
DROP VIEW IF EXISTS cm_sku_daily;                -- earlier layouts
DROP TABLE IF EXISTS cm_history_daily CASCADE;
DROP MATERIALIZED VIEW IF EXISTS cm_sales_1d;
DROP MATERIALIZED VIEW IF EXISTS cm_sales_1h;
DROP MATERIALIZED VIEW IF EXISTS cm_sku_1d;
DROP TABLE IF EXISTS cm_order_lines;
DROP TABLE IF EXISTS cm_product_daily;
DROP TABLE IF EXISTS cm_products;
DROP TABLE IF EXISTS cm_stockouts;
DROP TABLE IF EXISTS cm_findings;
"""

# The watcher's questions, shown with every finding. Each compares a store with its own platform's
# other stores at the same moment, so festivals, weekdays and seasons, which move every store alike,
# cancel out: what is left is local (a competitor's price cut, a viral product, empty shelves).
HOUR_SQL = """WITH h AS MATERIALIZED (
  SELECT platform, merchant, orders, revenue FROM cm_sales_1h WHERE bucket = %(hour)s
), past AS MATERIALIZED (   -- the same hour on the last days
  SELECT merchant, sum(orders)::float AS orders, count(*) AS hours FROM cm_sales_1h
  WHERE bucket = ANY(%(same_hours)s) GROUP BY merchant
), j AS MATERIALIZED (
  SELECT h.*, p.orders AS past, p.hours FROM h JOIN past p USING (merchant) WHERE p.hours >= %(min_days)s
), platform AS MATERIALIZED (
  SELECT platform, sum(orders)::float AS orders, sum(past) AS past, count(*) AS stores FROM j GROUP BY platform
), e AS MATERIALIZED (   -- usual: the store's past hours, moved the way its platform's other stores moved
  SELECT j.*, j.past / j.hours * (p.orders - j.orders) / nullif((p.past - j.past) / j.hours, 0) AS usual
  FROM j JOIN platform p USING (platform) WHERE p.stores >= 10
)
SELECT platform, merchant, orders, revenue, usual, hours, (orders - usual) / sqrt(usual + 0.05 * usual * usual) AS z
FROM e
WHERE usual >= %(min_usual)s AND abs(orders - usual) / sqrt(usual + 0.05 * usual * usual) >= %(z)s
  AND (orders >= %(hi)s * usual OR orders <= %(lo)s * usual)
ORDER BY abs(orders - usual) / sqrt(usual + 0.05 * usual * usual) DESC
LIMIT 40"""

STREAK_SQL = """WITH s AS MATERIALIZED (   -- out of stock at the close of each of the last N days, in stock the day before
  SELECT platform, merchant, sku FROM cm_product_daily
  WHERE day BETWEEN %(day)s - %(streak)s + 1 AND %(day)s AND sold_out
  GROUP BY platform, merchant, sku HAVING count(*) = %(streak)s
), fresh AS MATERIALIZED (
  SELECT s.* FROM s JOIN cm_product_daily d USING (merchant, sku)
  WHERE d.day = %(day)s - %(streak)s AND NOT d.sold_out
), b AS MATERIALIZED (   -- what it sells on an in-stock day (the three weeks before)
  SELECT d.merchant, d.sku, avg(d.units) AS units, sum(d.revenue) / nullif(sum(d.units), 0) AS price
  FROM cm_product_daily d JOIN fresh USING (merchant, sku)
  WHERE d.day BETWEEN %(day)s - 21 AND %(day)s - %(streak)s AND NOT d.sold_out GROUP BY d.merchant, d.sku
), t AS MATERIALIZED (   -- what it sold in the streak, before each day's shelf emptied
  SELECT d.merchant, d.sku, sum(d.units) AS sold FROM cm_product_daily d JOIN fresh USING (merchant, sku)
  WHERE d.day BETWEEN %(day)s - %(streak)s + 1 AND %(day)s GROUP BY d.merchant, d.sku
)
SELECT f.platform, f.merchant, f.sku, b.units AS usual, greatest(b.units * %(streak)s - t.sold, 0) * b.price AS lost
FROM fresh f JOIN b USING (merchant, sku) JOIN t USING (merchant, sku)
WHERE b.units >= 1
ORDER BY lost DESC
LIMIT 40"""

STOCKOUT_SQL = """WITH t AS MATERIALIZED (   -- products that sold out that day, and what they sold before the shelf emptied
  SELECT platform, merchant, sku, units, revenue FROM cm_product_daily WHERE day = %(day)s AND sold_out
), b AS MATERIALIZED (   -- what they sell on an in-stock day (last 7 days)
  SELECT d.merchant, d.sku, avg(d.units) AS units, sum(d.revenue) / nullif(sum(d.units), 0) AS price
  FROM cm_product_daily d JOIN t USING (merchant, sku)
  WHERE d.day BETWEEN %(day)s - 7 AND %(day)s - 1 AND NOT d.sold_out
  GROUP BY d.merchant, d.sku
)
SELECT t.platform, t.merchant, count(*) AS products, sum(greatest(b.units - t.units, 0) * b.price) AS lost,
       sum(t.revenue) AS sold, (array_agg(t.sku ORDER BY greatest(b.units - t.units, 0) * b.price DESC))[1:3] AS worst
FROM t JOIN b USING (merchant, sku)
GROUP BY t.platform, t.merchant
HAVING sum(greatest(b.units - t.units, 0) * b.price) >= %(min_lost)s
ORDER BY lost DESC
LIMIT 40"""

CLOSE_DAY_SQL = """INSERT INTO cm_product_daily (day, platform, merchant, sku, units, revenue, sold_out, live)
SELECT %(day)s, p.platform, p.merchant, p.sku, coalesce(s.units, 0), coalesce(s.revenue, 0), so.sku IS NOT NULL, true
FROM cm_products p
LEFT JOIN cm_sku_1d s ON s.bucket = %(start)s AND s.merchant = p.merchant AND s.sku = p.sku
LEFT JOIN cm_stockouts so ON so.day = %(day)s AND so.merchant = p.merchant AND so.sku = p.sku"""

STATS_SQL = {
    "chunks": "SELECT count(*), count(*) FILTER (WHERE is_compressed) FROM timescaledb_information.chunks "
              "WHERE hypertable_name = 'cm_order_lines'",
    "size": "SELECT hypertable_size('cm_order_lines'), hypertable_size('cm_product_daily')",
    "columnstore": "SELECT coalesce(sum(before_compression_total_bytes), 0), coalesce(sum(after_compression_total_bytes), 0) "
                   "FROM (SELECT * FROM chunk_columnstore_stats('cm_order_lines') UNION ALL "
                   "SELECT * FROM chunk_columnstore_stats('cm_product_daily')) s WHERE compression_status = 'Compressed'",
    "rollups": "SELECT (SELECT count(*) FROM cm_sales_1h), (SELECT count(*) FROM cm_sales_1d), (SELECT count(*) FROM cm_sku_1d)",
}


def ist(ts: str) -> datetime:
    return datetime.fromisoformat(ts).replace(tzinfo=C.IST)


def setup(conn: psycopg.Connection, reset: bool = False) -> None:
    """Create the hypertables, rollups and view (``reset``: drop everything first)."""
    if conn.execute("SELECT 1 FROM pg_extension WHERE extname = 'timescaledb'").fetchone() is None:
        conn.execute("CREATE EXTENSION IF NOT EXISTS timescaledb")
    if reset:
        conn.execute(DROP)
    conn.execute(SCHEMA)
    for table, order in (("cm_order_lines", "ts"), ("cm_product_daily", "day")):
        on = conn.execute("SELECT compression_enabled FROM timescaledb_information.hypertables WHERE hypertable_name = %s",
                          (table,)).fetchone()
        if on and not on[0]:
            conn.execute(f"ALTER TABLE {table} SET (timescaledb.enable_columnstore = true, "
                         f"timescaledb.segmentby = 'platform, merchant', timescaledb.orderby = '{order}')")


class Writer:
    """Turns ingest messages into rows and COPYs them in batches (one thread, its own connection)."""

    def __init__(self, dsn: str):
        self.dsn = dsn
        self.inbox: queue.Queue = queue.Queue(maxsize=20000)
        self.rows_total = 0
        self.history_rows = 0
        self.window: deque = deque(maxlen=40)       # (wall, rows) per batch, for rows per second
        self.batch_ms: deque = deque(maxlen=40)
        self.sim_ts: str | None = None
        self.running = False
        self.resets = 0                             # bumped after each reset, so the watcher starts over
        self.history_at = 0.0                       # when the last history message landed
        self.reset_requested = threading.Event()
        self.ready = threading.Event()
        self.lock = threading.Lock()

    def put(self, topic: str, payload: bytes) -> None:
        try:
            self.inbox.put_nowait((topic, payload))
        except queue.Full:   # never block the network thread; the stats show the backlog instead
            log.warning("writer backlog full, dropped a message on %s", topic)

    def rate(self) -> float:
        w = list(self.window)
        if len(w) < 2 or time.time() - w[-1][0] > 2:   # nothing written for 2 s: idle
            return 0.0
        span = w[-1][0] - w[0][0]
        return sum(r for _, r in w[1:]) / span if span > 0 else 0.0

    def run(self) -> None:
        with psycopg.connect(self.dsn, autocommit=True) as conn:
            setup(conn, reset=True)   # the simulator sends its history again at every start: one run per table
            self.ready.set()
            pending: list[tuple] = []
            last = time.monotonic()
            while True:
                try:
                    topic, payload = self.inbox.get(timeout=BATCH_SECONDS)
                except queue.Empty:
                    topic = None
                if self.reset_requested.is_set():
                    pending.clear()
                    setup(conn, reset=True)
                    with self.lock:
                        self.rows_total = self.history_rows = 0
                        self.window.clear()
                    self.sim_ts = None   # the old run's clock would hold the watcher at its last day
                    self.reset_requested.clear()
                    self.resets += 1
                if topic is not None:
                    self._route(conn, topic, payload, pending)
                if pending and (time.monotonic() - last >= BATCH_SECONDS or len(pending) >= 50000):
                    self._copy_lines(conn, pending)
                    pending = []
                    last = time.monotonic()

    def _route(self, conn, topic: str, payload: bytes, pending: list) -> None:
        if topic == C.CLOCK:
            m = json.loads(payload)
            self.sim_ts, self.running = m.get("sim_ts"), bool(m.get("running"))
            return
        if topic == C.CONTROL:
            if json.loads(payload).get("cmd") == "reset":
                self.reset_requested.set()
            return
        where = C.parse_merchant_topic(topic)
        if where is None:
            return
        platform, merchant, kind = where
        if kind == "orders":
            m = json.loads(payload)
            ts = ist(m["sim_ts"])
            for o in m.get("orders", []):
                q = int(o.get("qty", 1))
                p = float(o.get("price", 0))
                pending.append((ts, platform, merchant, o["sku"], o.get("customer"), q, p, q * p, o.get("offer") is not None))
        elif kind == "catalog":
            rows = [(platform, merchant, p["sku"], p.get("name"), p.get("category"), p.get("price"))
                    for p in json.loads(payload).get("products", [])]
            with conn.cursor() as cur:
                cur.executemany("INSERT INTO cm_products VALUES (%s, %s, %s, %s, %s, %s) ON CONFLICT (merchant, sku) "
                                "DO UPDATE SET name = excluded.name, category = excluded.category, price = excluded.price",
                                rows)
        elif kind == "history":
            self._copy_history(conn, platform, merchant, json.loads(payload))
            self.history_at = time.monotonic()
        elif kind == "stock":
            m = json.loads(payload)
            out = [(ist(m["sim_ts"]).date(), merchant, sku) for sku, units in (m.get("stock") or {}).items() if units <= 0]
            if out:
                with conn.cursor() as cur:
                    cur.executemany("INSERT INTO cm_stockouts VALUES (%s, %s, %s) ON CONFLICT DO NOTHING", out)

    def _copy_lines(self, conn, rows: list[tuple]) -> None:
        t0 = time.perf_counter()
        with conn.cursor() as cur, cur.copy(
                "COPY cm_order_lines (ts, platform, merchant, sku, customer, qty, price, revenue, offer) FROM STDIN") as cp:
            for r in rows:
                cp.write_row(r)
        self._count(len(rows), t0)

    def _copy_history(self, conn, platform: str, merchant: str, m: dict) -> None:
        t0 = time.perf_counter()
        days, skus = m.get("days", []), m.get("skus", [])
        units, prices, out = m.get("units", []), m.get("prices", []), m.get("sold_out") or []
        n = 0
        with conn.cursor() as cur, cur.copy(
                "COPY cm_product_daily (day, platform, merchant, sku, units, revenue, sold_out) FROM STDIN") as cp:
            for d, day in enumerate(days):
                row_u = units[d]
                row_p = prices[d] if d < len(prices) else [0.0] * len(skus)
                row_o = out[d] if d < len(out) else [0] * len(skus)
                for s, sku in enumerate(skus):
                    u = int(row_u[s])
                    cp.write_row((day, platform, merchant, sku, u, u * float(row_p[s]), bool(row_o[s])))
                    n += 1
        with self.lock:
            self.history_rows += n
        self._count(n, t0)

    def _count(self, n: int, t0: float) -> None:
        with self.lock:
            self.rows_total += n
            self.window.append((time.time(), n))
            self.batch_ms.append((time.perf_counter() - t0) * 1000)


class Watcher:
    """Refreshes rollups, moves old days to the columnstore, runs the SQL checks, publishes stats."""

    def __init__(self, dsn: str, writer: Writer, publish, names: dict[str, dict]):
        self.dsn, self.w, self.publish, self.names = dsn, writer, publish, names
        self.done_hour: datetime | None = None      # last simulated hour checked
        self.done_day: datetime | None = None
        self.compressed_before: datetime | None = None
        self.cool: dict[tuple, datetime] = {}
        self.queries: dict[str, float] = {}
        self.signals_total = 0
        self.lost_today = 0.0
        self.by_kind: dict[str, int] = {}
        self.sizes: dict = {}
        self.last_sizes = 0.0
        self.resets = writer.resets
        self.history_done = False

    def run(self) -> None:
        self.w.ready.wait()
        with psycopg.connect(self.dsn, autocommit=True) as conn:
            # The checks join a few thousand rows each way; fresh chunks have no statistics yet, and a
            # nested loop picked on a bad estimate turns milliseconds into minutes.
            conn.execute("SET enable_nestloop = off")
            self.conn = conn
            last_stats = 0.0
            while True:
                try:
                    if self.w.resets != self.resets:   # the simulation was reset: start over
                        self.__init__(self.dsn, self.w, self.publish, self.names)
                    self._tick(conn)
                    if time.monotonic() - last_stats >= STATS_EVERY:
                        self._stats(conn)
                        last_stats = time.monotonic()
                except psycopg.Error as exc:   # e.g. a reset dropped the tables mid-query
                    log.warning("watcher: %s", exc)
                    time.sleep(1)
                time.sleep(0.2)

    def _timed(self, conn, name: str, sql: str, params=None) -> list:
        t0 = time.perf_counter()
        rows = conn.execute(sql, params).fetchall()
        self.queries[name] = round((time.perf_counter() - t0) * 1000, 1)
        return rows

    def _history_to_columnstore(self, conn) -> None:
        """Once the backfill has landed (no history message for 5 s): all of it to the columnstore, then
        fresh planner statistics."""
        t0 = time.perf_counter()
        for (c,) in conn.execute("SELECT c::text FROM show_chunks('cm_product_daily') c").fetchall():
            conn.execute("CALL convert_to_columnstore(%s, if_not_columnstore => true)", (c,))
        conn.execute("ANALYZE cm_product_daily")
        self.queries["history to columnstore"] = round((time.perf_counter() - t0) * 1000, 1)
        self.history_done = True
        self.last_sizes = 0.0

    def _tick(self, conn) -> None:
        if (not self.history_done and self.w.history_rows and self.w.inbox.qsize() < 100
                and time.monotonic() - self.w.history_at > 5):
            self._history_to_columnstore(conn)
        if not self.w.sim_ts:
            return
        now = ist(self.w.sim_ts)
        hour = now.replace(minute=0, second=0, microsecond=0) - timedelta(hours=1)   # the last closed hour
        day = now.replace(hour=0, minute=0, second=0, microsecond=0) - timedelta(days=1)
        if self.done_hour is None or hour < self.done_hour:   # first tick, or the clock went back
            self.done_hour, self.done_day = hour, day
            return
        if hour > self.done_hour:
            self._refresh(conn, "cm_sales_1h", self.done_hour, hour + timedelta(hours=1))
            self._hour_check(conn, hour)
            self.done_hour = hour
        if day > self.done_day:
            self._refresh(conn, "cm_sales_1d", self.done_day, day + timedelta(days=1))
            self._refresh(conn, "cm_sku_1d", self.done_day, day + timedelta(days=1))
            self._close_days(conn, self.done_day, day)
            self._stockout_check(conn, day)
            self._streak_check(conn, day)
            self._compress(conn, now)
            self.done_day = day

    def _close_days(self, conn, after: datetime, through: datetime) -> None:
        """Closed live days into cm_product_daily: every catalogue product, zeros and sold-out days included,
        so history and live days read the same."""
        t0 = time.perf_counter()
        d = after + timedelta(days=1)
        while d <= through:
            conn.execute(CLOSE_DAY_SQL, {"day": d.date(), "start": d})
            d += timedelta(days=1)
        self.queries["close day"] = round((time.perf_counter() - t0) * 1000, 1)

    def _refresh(self, conn, view: str, start: datetime, end: datetime) -> None:
        t0 = time.perf_counter()
        conn.execute(f"CALL refresh_continuous_aggregate('{view}', %s, %s)", (start, end))
        self.queries[f"refresh {view}"] = round((time.perf_counter() - t0) * 1000, 1)

    def _compress(self, conn, now: datetime) -> None:
        before = now.replace(hour=0, minute=0, second=0, microsecond=0) - timedelta(days=COMPRESS_AFTER_DAYS)
        t0 = time.perf_counter()
        chunks = conn.execute("SELECT c::text FROM show_chunks('cm_order_lines', older_than => %s) c", (before,)).fetchall()
        for (c,) in chunks:
            conn.execute("CALL convert_to_columnstore(%s, if_not_columnstore => true)", (c,))
        conn.execute("ANALYZE cm_order_lines")
        if chunks:
            self.queries["convert_to_columnstore"] = round((time.perf_counter() - t0) * 1000, 1)
        self.compressed_before = before

    def _cool(self, key: tuple, at: datetime) -> bool:
        last = self.cool.get(key)
        if last is not None and at - last < timedelta(hours=COOLDOWN_HOURS):
            return False
        self.cool[key] = at
        return True

    def _hour_check(self, conn, hour: datetime) -> None:
        same = [hour - timedelta(days=k) for k in range(1, HOUR_LOOKBACK_DAYS + 1)]
        params = {"hour": hour, "same_hours": same, "min_days": HOUR_MIN_DAYS, "z": HOUR_Z, "min_usual": HOUR_MIN_USUAL,
                  "lo": HOUR_RATIO[0], "hi": HOUR_RATIO[1]}
        rows = self._timed(conn, "hour check", HOUR_SQL, params)
        sent = 0
        for platform, merchant, orders, revenue, usual, hours, z in rows:
            if sent >= PER_ROUND or not self._cool((merchant, "hour"), hour):
                continue
            usual = float(usual)
            ratio = orders / max(usual, 0.1)
            up = ratio > 1
            name = self.names.get(merchant, {}).get("name", merchant)
            when = f"{hour:%a} {hour:%H}:00-{(hour + timedelta(hours=1)):%H}:00"
            title = (f"{name}: {orders} orders {when}, {ratio:.1f}× its usual" if up else
                     f"{name}: only {orders} orders {when}, {ratio:.0%} of its usual")
            detail = (f"Usual: {usual:.0f} orders, from this hour on its last {hours} days, moved the way the platform's "
                      f"other stores moved this hour, so the festival and the day of the week are already in it. "
                      f"{_inr(float(revenue))} this hour.")
            self._emit({"kind": "hour_spike" if up else "hour_drop", "window": "hour", "platform": platform,
                        "merchant": merchant, "sku": None, "product": None, "title": title, "detail": detail,
                        "value": int(orders), "usual": round(usual, 1), "ratio": round(ratio, 2), "z": round(float(z), 1),
                        "period_start": hour.isoformat(), "sql": _show(HOUR_SQL, params),
                        "query_ms": self.queries.get("hour check")})
            sent += 1

    def _streak_check(self, conn, day: datetime) -> None:
        d = day.date()
        params = {"day": d, "streak": STREAK_DAYS}
        rows = self._timed(conn, "out-of-stock streak", STREAK_SQL, params)
        for platform, merchant, sku, usual, lost in rows[:PER_ROUND]:
            info = self.names.get(merchant, {})
            name, product = info.get("name", merchant), info.get("products", {}).get(sku, sku)
            title = f"{product} at {name}: out of stock {STREAK_DAYS} days running, about {_inr(float(lost))} lost"
            detail = (f"Sold out at the close of every day since {d - timedelta(days=STREAK_DAYS - 1):%a %d %b}. "
                      f"It sells about {float(usual):.0f} a day when in stock; the restock hasn't arrived or wasn't ordered.")
            self._emit({"kind": "stockout_streak", "window": "day", "platform": platform, "merchant": merchant,
                        "sku": sku, "product": product, "title": title, "detail": detail, "value": round(float(lost)),
                        "usual": round(float(usual), 1), "ratio": None, "z": None, "days": STREAK_DAYS,
                        "period_start": d.isoformat(), "sql": _show(STREAK_SQL, params),
                        "query_ms": self.queries.get("out-of-stock streak")})

    def _stockout_check(self, conn, day: datetime) -> None:
        d = day.date()
        params = {"day": d, "min_lost": STOCKOUT_MIN_LOST}
        rows = self._timed(conn, "stock-out check", STOCKOUT_SQL, params)
        self.lost_today = sum(float(r[3]) for r in rows)
        for platform, merchant, products, lost, sold, worst in rows[:PER_ROUND]:
            info = self.names.get(merchant, {})
            name = info.get("name", merchant)
            names = ", ".join(info.get("products", {}).get(s, s) for s in (worst or []))
            title = f"{name}: about {_inr(float(lost))} of sales lost to empty shelves on {d:%a %d %b}"
            detail = (f"{products} product{'s' if products != 1 else ''} sold out that day ({names}); estimate from "
                      f"what each sells on an in-stock day. They sold {_inr(float(sold or 0))} before running out.")
            self._emit({"kind": "stockout_loss", "window": "day", "platform": platform, "merchant": merchant,
                        "sku": None, "product": None, "title": title, "detail": detail, "value": round(float(lost)),
                        "usual": None, "ratio": None, "z": None, "products": int(products),
                        "period_start": d.isoformat(), "sql": _show(STOCKOUT_SQL, params),
                        "query_ms": self.queries.get("stock-out check")})

    def _emit(self, s: dict) -> None:
        self.signals_total += 1
        self.by_kind[s["kind"]] = self.by_kind.get(s["kind"], 0) + 1
        info = self.names.get(s["merchant"], {})
        s.update({"id": f"tsdb-{s['merchant']}-{s['kind']}-{s['period_start']}", "source": "timescaledb",
                  "sent_ms": C.wall_ms(), "sim_ts": self.w.sim_ts, "merchant_name": info.get("name", s["merchant"]),
                  "city": info.get("city"), "category": info.get("category")})
        self.publish(C.tsdb_signal_topic(s["platform"], s["merchant"]), s, 1)
        self.conn.execute("INSERT INTO cm_findings VALUES (%s, %s, %s, %s, %s)",
                          (ist(s["sim_ts"]), s["kind"], s["platform"], s["merchant"], Jsonb(s)))

    def _stats(self, conn) -> None:
        if time.monotonic() - self.last_sizes >= SIZES_EVERY:
            try:
                (chunks, compressed), = self._timed(conn, "chunks", STATS_SQL["chunks"])
                (lines_b, hist_b), = self._timed(conn, "size", STATS_SQL["size"])
                (before, after), = self._timed(conn, "columnstore", STATS_SQL["columnstore"])
                (h1, d1, s1), = self._timed(conn, "rollups", STATS_SQL["rollups"])
                self.sizes = {"chunks": chunks, "columnstore_chunks": compressed, "lines_bytes": lines_b or 0,
                              "history_bytes": hist_b or 0, "columnstore_before_bytes": int(before),
                              "columnstore_after_bytes": int(after),
                              # a sliver of a chunk compresses badly; the ratio means something from a few MB
                              "columnstore_ratio": round(float(before) / float(after), 1) if after and before >= 5_000_000 else None,
                              "rollup_rows": {"cm_sales_1h": h1, "cm_sales_1d": d1, "cm_sku_1d": s1}}
            except psycopg.Error as exc:
                log.debug("stats: %s", exc)
            self.last_sizes = time.monotonic()
        with self.w.lock:
            rows, hist, ms = self.w.rows_total, self.w.history_rows, list(self.w.batch_ms)
        self.publish(C.TSDB_STATS, {
            "sent_ms": C.wall_ms(), "sim_ts": self.w.sim_ts, "running": self.w.running,
            "rows_total": rows, "order_lines": rows - hist, "history_rows": hist,
            "rows_per_sec": round(self.w.rate()), "batch_ms_p50": round(_pct(ms, 50), 1),
            "backlog": self.w.inbox.qsize(), **self.sizes,
            "checked_hour": self.done_hour.isoformat() if self.done_hour else None,
            "checked_day": self.done_day.date().isoformat() if self.done_day else None,
            "query_ms": self.queries, "signals_total": self.signals_total, "signals_by_kind": self.by_kind,
            "stockout_lost_last_day": round(self.lost_today),
        }, 0)


def _pct(xs: list[float], p: float) -> float:
    if not xs:
        return 0.0
    s = sorted(xs)
    return s[min(len(s) - 1, int(math.ceil(p / 100 * len(s))) - 1)]


def _inr(x: float) -> str:
    from secondbrain.money import inr_short
    return inr_short(x)


def _show(sql: str, params: dict) -> str:
    """The query as it ran, parameters written in, for the page to display."""
    out = sql
    for k, v in params.items():
        if isinstance(v, list):
            v = "ARRAY[" + ", ".join(f"'{x.isoformat()}'" for x in v[:3]) + (", …]" if len(v) > 3 else "]")
        elif hasattr(v, "isoformat"):
            v = f"'{v.isoformat()}'"
        out = out.replace(f"%({k})s", str(v))
    return out


def _fatal(target):
    """A thread that dies takes the process with it, so the launcher sees it instead of a silent half."""
    def run():
        try:
            target()
        except BaseException:
            log.exception("%s stopped", threading.current_thread().name)
            os._exit(1)
    return run


def _names() -> dict[str, dict]:
    """Merchant and product names from the engine (empty when it isn't up yet; ids are shown instead)."""
    import httpx
    try:
        ms = httpx.get(f"http://127.0.0.1:{C.ENGINE_PORT}/merchants", timeout=3).json()
        return {m["id"]: {"name": m["name"], "city": m.get("city"), "category": m.get("category"), "products": {}}
                for m in ms}
    except Exception:
        return {}


def run() -> None:
    import paho.mqtt.client as mqtt

    from secondbrain.config import get_settings
    from secondbrain.live import MqttSettings
    from secondbrain.live.realtime import PUBLISHER

    settings = get_settings()
    dsn = settings.admin_database_url
    writer = Writer(dsn)
    names: dict[str, dict] = {}

    live_s = MqttSettings.from_app(settings)
    live = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id=f"sb-commerce-tsdb-{time.time_ns() % 10**8}")
    live.username_pw_set(PUBLISHER, live_s.publisher_password)
    live.connect_async(live_s.host, live_s.port)
    live.loop_start()

    def publish(topic: str, payload: dict, qos: int = 0) -> None:
        live.publish(topic, json.dumps(payload, separators=(",", ":"), default=str), qos=qos)

    ingest = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id=f"sb-commerce-tsdb-in-{time.time_ns() % 10**8}")
    ingest.max_inflight_messages_set(1000)

    def on_message(client, userdata, msg):
        where = C.parse_merchant_topic(msg.topic)
        if where and where[2] == "catalog":   # product names for the findings
            info = names.setdefault(where[1], {"name": where[1], "products": {}})
            for p in json.loads(msg.payload).get("products", []):
                info.setdefault("products", {})[p["sku"]] = p.get("name", p["sku"])
        elif msg.topic == C.WORLD:
            for m in json.loads(msg.payload).get("merchants", []):
                info = names.setdefault(m["id"], {"products": {}})
                info.update({"name": m["name"], "city": m.get("city"), "category": m.get("category")})
        writer.put(msg.topic, msg.payload)

    ingest.on_connect = lambda c, *a: c.subscribe(f"{C.PREFIX}/#", qos=1)
    ingest.on_message = on_message
    ingest.connect_async(C.INGEST_HOST, C.INGEST_PORT)
    ingest.loop_start()

    for k, v in _names().items():
        names.setdefault(k, {"products": {}}).update({x: v[x] for x in ("name", "city", "category")})
    watcher = Watcher(dsn, writer, publish, names)
    for name, target in (("tsdb-writer", writer.run), ("tsdb-watcher", watcher.run)):
        threading.Thread(target=_fatal(target), name=name, daemon=True).start()
    log.info("TimescaleDB writer and watcher running on %s", dsn.rsplit("@", 1)[-1])
    try:
        while True:
            time.sleep(10)
            log.info("rows %s (%s/s), backlog %s, signals %s", writer.rows_total, round(writer.rate()),
                     writer.inbox.qsize(), watcher.signals_total)
    except KeyboardInterrupt:
        pass


def main() -> None:
    parser = argparse.ArgumentParser(description="Live Commerce on TimescaleDB: writer, rollups, SQL signals")
    parser.add_argument("--reset", action="store_true", help="drop and recreate the TimescaleDB tables, then exit")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    if args.reset:
        from secondbrain.config import get_settings
        with psycopg.connect(get_settings().admin_database_url, autocommit=True) as conn:
            setup(conn, reset=True)
        print("TimescaleDB tables recreated.")
        return
    run()


if __name__ == "__main__":
    main()
