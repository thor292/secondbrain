"""Live Commerce on TimescaleDB (commerce/tsdb.py, tsdb_api.py), against the TimescaleDB container.

Start it first: ``docker compose -f docker-compose.timescale.yml -p secondbrain-ts up -d --wait``.
Skipped when it isn't running (the main suite's PostgreSQL has no TimescaleDB).
"""

import json
import os
from datetime import date, datetime, timedelta

import psycopg
import pytest

from secondbrain.commerce import contract as C
from secondbrain.commerce import tsdb, tsdb_api
from secondbrain.config import Settings

SERVER = "postgresql://secondbrain_admin:dev_admin_password@localhost:55433"
DB = f"secondbrain_tsdb_test{os.environ.get('SB_TEST_SUFFIX', '')}"
DAY = date(2026, 11, 12)


@pytest.fixture(scope="module")
def dsn() -> str:
    try:
        with psycopg.connect(f"{SERVER}/postgres", autocommit=True, connect_timeout=2) as conn:
            if conn.execute("SELECT 1 FROM pg_available_extensions WHERE name = 'timescaledb'").fetchone() is None:
                pytest.skip("no TimescaleDB on :55433")
            if not conn.execute("SELECT 1 FROM pg_database WHERE datname = %s", (DB,)).fetchone():
                conn.execute(f'CREATE DATABASE "{DB}"')
    except psycopg.OperationalError:
        pytest.skip("TimescaleDB container not running (docker-compose.timescale.yml)")
    return f"{SERVER}/{DB}"


@pytest.fixture
def conn(dsn):
    with psycopg.connect(dsn, autocommit=True) as c:
        tsdb.setup(c, reset=True)
        c.execute("SET enable_nestloop = off")
        yield c


def at(d: date, hour: int = 12) -> datetime:
    return datetime(d.year, d.month, d.day, hour, tzinfo=C.IST)


def products(conn, rows) -> None:
    conn.cursor().executemany("INSERT INTO cm_products VALUES (%s, %s, %s, %s, %s, %s)", rows)


def daily(conn, rows) -> None:
    """(day, merchant, sku, units, sold_out) on platform dmart, price 100."""
    conn.cursor().executemany(
        "INSERT INTO cm_product_daily (day, platform, merchant, sku, units, revenue, sold_out, live) "
        "VALUES (%s, 'dmart', %s, %s, %s, %s * 100, %s, true)", [(d, m, s, u, u, so) for d, m, s, u, so in rows])


def lines(conn, rows) -> None:
    """(ts, merchant, orders) on platform dmart, one unit at 100 each."""
    w = tsdb.Writer("")
    w._copy_lines(conn, [(ts, "dmart", m, "SKU", None, 1, 100.0, 100.0, False) for ts, m, n in rows for _ in range(n)])


class Recorder:
    def __init__(self):
        self.sent = []

    def __call__(self, topic, payload, qos=0):
        self.sent.append((topic, payload))


def watcher(conn, names=None) -> tuple[tsdb.Watcher, Recorder]:
    w = tsdb.Writer("")
    w.sim_ts = at(DAY + timedelta(days=1), 0).replace(tzinfo=None).isoformat()
    rec = Recorder()
    wa = tsdb.Watcher("", w, rec, names or {})
    wa.conn = conn
    return wa, rec


def test_writer_routes_every_kind_of_message(conn):
    w = tsdb.Writer("")
    pending: list = []
    w._route(conn, C.merchant_topic("dmart", "dmart-0001", "catalog"),
             json.dumps({"products": [{"sku": "A", "name": "Atta 10kg", "category": "Staples", "price": 450}]}), pending)
    w._route(conn, C.merchant_topic("dmart", "dmart-0001", "history"),
             json.dumps({"days": ["2026-11-01", "2026-11-02"], "skus": ["A"], "units": [[3], [0]],
                         "prices": [[450.0], [0.0]], "sold_out": [[0], [1]]}), pending)
    w._route(conn, C.merchant_topic("dmart", "dmart-0001", "orders"),
             json.dumps({"sim_ts": "2026-11-03T10:15:00", "orders": [
                 {"id": "o1", "customer": "c1", "sku": "A", "qty": 2, "price": 450, "offer": None},
                 {"id": "o2", "customer": "c2", "sku": "A", "qty": 1, "price": 405, "offer": "sig-1"}]}), pending)
    w._route(conn, C.merchant_topic("dmart", "dmart-0001", "stock"),
             json.dumps({"sim_ts": "2026-11-03T23:59:59", "stock": {"A": 0}, "restocked": {}}), pending)
    w._route(conn, C.CLOCK, json.dumps({"sim_ts": "2026-11-03T23:59:59", "running": True}), pending)
    w._copy_lines(conn, pending)

    assert conn.execute("SELECT name, price FROM cm_products").fetchall() == [("Atta 10kg", 450.0)]
    assert conn.execute("SELECT day, units, revenue, sold_out FROM cm_product_daily ORDER BY day").fetchall() == [
        (date(2026, 11, 1), 3, 1350.0, False), (date(2026, 11, 2), 0, 0.0, True)]
    assert conn.execute("SELECT qty, revenue, offer FROM cm_order_lines ORDER BY qty").fetchall() == [
        (1, 405.0, True), (2, 900.0, False)]
    assert conn.execute("SELECT day, sku FROM cm_stockouts").fetchall() == [(date(2026, 11, 3), "A")]
    assert (w.sim_ts, w.running, w.rows_total, w.history_rows) == ("2026-11-03T23:59:59", True, 4, 2)


def test_closing_a_day_writes_every_product_with_zeros_and_sold_out(conn):
    products(conn, [("dmart", "m1", "A", "Atta", "Staples", 100), ("dmart", "m1", "B", "Bread", "Staples", 100),
                    ("dmart", "m2", "A", "Atta", "Staples", 100)])
    lines(conn, [(at(DAY, 11), "m1", 5)])
    conn.execute("UPDATE cm_order_lines SET sku = 'A'")
    conn.execute("INSERT INTO cm_stockouts VALUES (%s, 'm2', 'A')", (DAY,))
    start = at(DAY, 0)
    conn.execute("CALL refresh_continuous_aggregate('cm_sku_1d', %s, %s)", (start, start + timedelta(days=1)))
    conn.execute(tsdb.CLOSE_DAY_SQL, {"day": DAY, "start": start})

    got = conn.execute("SELECT merchant, sku, units, sold_out, live FROM cm_product_daily ORDER BY 1, 2").fetchall()
    assert got == [("m1", "A", 5, False, True), ("m1", "B", 0, False, True), ("m2", "A", 0, True, True)]


def test_stockout_loss_prices_what_sold_out_products_normally_sell(conn):
    rows = []
    for k in range(1, 8):   # a normal week: A sells 20 a day at m1, B 10 a day
        rows += [(DAY - timedelta(days=k), "m1", "A", 20, False), (DAY - timedelta(days=k), "m1", "B", 10, False)]
    rows += [(DAY, "m1", "A", 5, True), (DAY, "m1", "B", 10, False)]   # A ran out after 5
    daily(conn, rows)
    wa, rec = watcher(conn, {"m1": {"name": "DMart Test", "products": {"A": "Atta 10kg"}}})
    wa.queries = {}
    import secondbrain.commerce.tsdb as t
    old, t.STOCKOUT_MIN_LOST = t.STOCKOUT_MIN_LOST, 1000
    try:
        wa._stockout_check(conn, at(DAY, 0))
    finally:
        t.STOCKOUT_MIN_LOST = old

    (topic, s), = rec.sent
    assert topic == C.tsdb_signal_topic("dmart", "m1")
    assert s["kind"] == "stockout_loss" and s["value"] == 1500 and s["products"] == 1   # (20 - 5) x ₹100
    assert "Atta 10kg" in s["detail"] and "DMart Test" in s["title"]
    stored = conn.execute("SELECT kind, body->>'id' FROM cm_findings").fetchall()
    assert stored == [("stockout_loss", s["id"])]


def test_out_of_stock_streak_is_reported_once_when_it_reaches_three_days(conn):
    rows = []
    for k in range(4, 15):   # in stock, 10 a day, until DAY - 3
        for sku in ("A", "B", "C"):
            rows.append((DAY - timedelta(days=k), "m1", sku, 10, False))
    for k in range(0, 3):
        rows.append((DAY - timedelta(days=k), "m1", "A", 0, True))       # out three days running
        rows.append((DAY - timedelta(days=k), "m1", "C", 0, False))      # nobody bought, but on the shelf
    rows.append((DAY - timedelta(days=3), "m1", "A", 10, False))
    rows += [(DAY - timedelta(days=k), "m1", "B", 0, True) for k in range(0, 4)]   # out four days: reported yesterday
    rows += [(DAY - timedelta(days=3), "m1", "C", 10, False)]
    daily(conn, rows)
    wa, rec = watcher(conn)
    wa._streak_check(conn, at(DAY, 0))

    assert [(s["sku"], s["value"], s["days"]) for _, s in rec.sent] == [("A", 3000, 3)]   # 3 days x 10 x ₹100


def test_hour_check_flags_one_store_but_not_a_festival_hour_for_all(conn):
    stores = [f"m{i:02d}" for i in range(12)]
    rows = []
    for k in range(1, 4):   # the same hour on the three days before: 10 orders each
        rows += [(at(DAY - timedelta(days=k), 19), m, 10) for m in stores]
    # this hour: a festival doubles every store; m00 also has something of its own going on
    rows += [(at(DAY, 19), m, 20) for m in stores[1:]] + [(at(DAY, 19), "m00", 70)]
    lines(conn, rows)
    for v in ("cm_sales_1h", "cm_sales_1d"):
        conn.execute(f"CALL refresh_continuous_aggregate('{v}', %s, %s)", (at(DAY - timedelta(days=4), 0), at(DAY, 23)))
    wa, rec = watcher(conn)
    wa._hour_check(conn, at(DAY, 19))

    flagged = [(s["merchant"], s["value"], round(s["usual"])) for _, s in rec.sent]
    assert flagged == [("m00", 70, 20)]   # usual = its 10, moved like its peers (x2); the others are just the festival


def test_findings_overview_and_merchant_series_from_the_api(dsn, conn):
    settings = Settings(_env_file=None, admin_database_url=dsn)
    daily(conn, [(DAY - timedelta(days=k), "m1", "A", 10 + k, False) for k in range(5)])
    conn.execute("INSERT INTO cm_findings VALUES (%s, 'stockout_loss', 'dmart', 'm1', %s)",
                 (at(DAY), json.dumps({"id": "f1", "kind": "stockout_loss"})))
    assert tsdb_api.available(settings)
    assert tsdb_api.findings(settings)["findings"] == [{"id": "f1", "kind": "stockout_loss"}]
    ov = tsdb_api.overview(settings)
    assert {h["name"] for h in ov["hypertables"]} >= {"cm_order_lines", "cm_product_daily", "cm_findings"}
    assert {r["name"] for r in ov["rollups"]} == {"cm_sales_1h", "cm_sales_1d", "cm_sku_1d"}
    assert all(q["ms"] >= 0 and q["sql"] for q in ov["queries"].values())
    mx = tsdb_api.merchant(settings, "m1", days=7)
    assert [d["units"] for d in mx["days"]] == [14, 13, 12, 11, 10] and mx["to"] == DAY


def test_api_answers_503_without_the_tables():
    settings = Settings(_env_file=None, admin_database_url="postgresql://nobody:x@localhost:1/none")
    assert not tsdb_api.available(settings)
    with pytest.raises(tsdb_api.Unavailable):
        tsdb_api.overview(settings)


def test_show_writes_the_parameters_into_the_query():
    sql = tsdb._show("SELECT * FROM t WHERE day = %(day)s AND b = ANY(%(hours)s) AND z >= %(z)s",
                     {"day": DAY, "hours": [at(DAY, h) for h in range(5)], "z": 4.0})
    assert "day = '2026-11-12'" in sql and "z >= 4.0" in sql and ", …]" in sql


def test_watcher_follows_the_clock_back_after_a_reset(conn):
    wa, _ = watcher(conn)
    wa.history_done = True
    wa.w.sim_ts = at(DAY + timedelta(days=5), 6).replace(tzinfo=None).isoformat()   # the old run, days ahead
    wa._tick(conn)
    wa.w.sim_ts = at(DAY, 6).replace(tzinfo=None).isoformat()                       # a new run from today
    wa._tick(conn)
    assert wa.done_day == at(DAY - timedelta(days=1), 0) and wa.done_hour == at(DAY, 5)
