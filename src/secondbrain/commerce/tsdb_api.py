"""The app's read-only view of the Live Commerce TimescaleDB tables (``tsdb.py`` writes them).

Every answer carries the SQL it ran and how long it took, so the page can show the database
working, not just its results. Without the tables (a copy of the demo on plain PostgreSQL) the
endpoints answer 503 and ``available`` is false.
"""

import time
from datetime import timedelta

import psycopg
from psycopg.rows import dict_row

from secondbrain.config import Settings

OVERVIEW_SQL = """SELECT
  (SELECT extversion FROM pg_extension WHERE extname = 'timescaledb') AS version,
  (SELECT count(*) FROM timescaledb_information.chunks WHERE hypertable_name = 'cm_order_lines') AS chunks,
  (SELECT count(*) FROM timescaledb_information.chunks WHERE hypertable_name = 'cm_order_lines' AND is_compressed) AS columnstore_chunks,
  (SELECT count(*) FROM timescaledb_information.chunks WHERE hypertable_name = 'cm_product_daily') AS history_chunks,
  hypertable_size('cm_order_lines') AS lines_bytes,
  hypertable_size('cm_product_daily') AS history_bytes,
  (SELECT approximate_row_count('cm_order_lines')) AS order_lines,
  (SELECT approximate_row_count('cm_product_daily')) AS history_rows,
  (SELECT min(day) FROM cm_product_daily) AS history_from,
  (SELECT max(ts) FROM cm_order_lines) AS latest"""

HYPERTABLES_SQL = """SELECT hypertable_name AS name, num_chunks AS chunks, compression_enabled AS columnstore,
  (SELECT string_agg(column_name || ' (' || time_interval || ')', ', ') FROM timescaledb_information.dimensions d
    WHERE d.hypertable_name = h.hypertable_name) AS partitioned_by
FROM timescaledb_information.hypertables h ORDER BY 1"""

ROLLUPS_SQL = """SELECT view_name AS name, materialized_only,
  regexp_replace(view_definition, '\\s+', ' ', 'g') AS definition
FROM timescaledb_information.continuous_aggregates ORDER BY 1"""

MERCHANT_SQL = """SELECT day, sum(units) AS units, round(sum(revenue)::numeric) AS revenue,
       bool_or(live) AS live, count(*) FILTER (WHERE sold_out) AS sold_out_products
FROM cm_product_daily
WHERE merchant = %(merchant)s AND day >= %(since)s
GROUP BY day ORDER BY day"""

HOURS_SQL = """SELECT bucket, orders, units, round(revenue::numeric) AS revenue
FROM cm_sales_1h
WHERE merchant = %(merchant)s AND bucket >= %(since)s
ORDER BY bucket"""

LAST_YEAR_SQL = """SELECT sum(units) AS units, round(sum(revenue)::numeric) AS revenue
FROM cm_product_daily
WHERE merchant = %(merchant)s AND day BETWEEN %(from)s AND %(to)s"""

TOP_SQL = """SELECT platform, merchant, sum(orders) AS orders, round(sum(revenue)::numeric) AS revenue,
       round(sum(offer_revenue)::numeric) AS offer_revenue
FROM cm_sales_1h
WHERE bucket > (SELECT max(bucket) FROM cm_sales_1h) - %(span)s::interval
GROUP BY platform, merchant ORDER BY sum(revenue) DESC LIMIT %(limit)s"""

PLATFORM_SQL = """SELECT platform, time_bucket(INTERVAL '1 day', bucket, 'Asia/Kolkata') AS day,
       sum(orders) AS orders, round(sum(revenue)::numeric) AS revenue
FROM cm_sales_1d
GROUP BY 1, 2 ORDER BY 2, 1"""


FINDINGS_SQL = """SELECT body FROM cm_findings ORDER BY sim_ts DESC LIMIT %(limit)s"""


class Unavailable(Exception):
    """TimescaleDB or its Live Commerce tables are not there."""


def _connect(settings: Settings) -> psycopg.Connection:
    return psycopg.connect(settings.admin_database_url, row_factory=dict_row, connect_timeout=3)


def available(settings: Settings) -> bool:
    try:
        with _connect(settings) as conn:
            return conn.execute("SELECT to_regclass('cm_order_lines') IS NOT NULL AS ok").fetchone()["ok"]
    except psycopg.Error:
        return False


def _run(conn, sql: str, params: dict | None = None) -> tuple[list[dict], dict]:
    t0 = time.perf_counter()
    rows = conn.execute(sql, params).fetchall()
    return rows, {"sql": sql, "params": {k: str(v) for k, v in (params or {}).items()},
                  "ms": round((time.perf_counter() - t0) * 1000, 1), "rows": len(rows)}


def _guard(fn):
    def wrapper(settings: Settings, *args, **kwargs):
        try:
            with _connect(settings) as conn:
                if not conn.execute("SELECT to_regclass('cm_order_lines') IS NOT NULL AS ok").fetchone()["ok"]:
                    raise Unavailable("The Live Commerce TimescaleDB tables are not set up "
                                      "(uv run python -m secondbrain.commerce --tsdb)")
                return fn(conn, *args, **kwargs)
        except psycopg.Error as exc:
            raise Unavailable(f"TimescaleDB is not reachable: {exc.__class__.__name__}") from exc
    return wrapper


@_guard
def overview(conn) -> dict:
    (o,), q1 = _run(conn, OVERVIEW_SQL)
    hyper, q2 = _run(conn, HYPERTABLES_SQL)
    rollups, q3 = _run(conn, ROLLUPS_SQL)
    plat, q4 = _run(conn, PLATFORM_SQL)
    return {"database": o, "hypertables": hyper, "rollups": rollups, "platform_days": plat,
            "queries": {"overview": q1, "hypertables": q2, "rollups": q3, "platform_days": q4}}


@_guard
def merchant(conn, merchant_id: str, days: int = 60) -> dict:
    days = max(7, min(days, 430))
    latest = conn.execute("SELECT max(day) AS d FROM cm_product_daily WHERE merchant = %s", (merchant_id,)).fetchone()["d"]
    if latest is None:
        return {"merchant": merchant_id, "days": [], "hours": [], "queries": {}}
    since = latest - timedelta(days=days - 1)
    daily, q1 = _run(conn, MERCHANT_SQL, {"merchant": merchant_id, "since": since})
    hours, q2 = _run(conn, HOURS_SQL, {"merchant": merchant_id, "since": latest - timedelta(days=2)})
    window = (since - timedelta(days=364), latest - timedelta(days=364))
    (ly,), q3 = _run(conn, LAST_YEAR_SQL, {"merchant": merchant_id, "from": window[0], "to": window[1]})
    return {"merchant": merchant_id, "from": since, "to": latest, "days": daily, "hours": hours,
            "last_year": {**ly, "from": window[0], "to": window[1]},
            "queries": {"daily": q1, "hours": q2, "last_year": q3}}


@_guard
def top(conn, hours: int = 24, limit: int = 10) -> dict:
    rows, q = _run(conn, TOP_SQL, {"span": f"{max(1, min(hours, 24 * 30))} hours", "limit": max(1, min(limit, 50))})
    return {"hours": hours, "merchants": rows, "queries": {"top": q}}


@_guard
def findings(conn, limit: int = 60) -> dict:
    rows, q = _run(conn, FINDINGS_SQL, {"limit": max(1, min(limit, 500))})
    return {"findings": [r["body"] for r in rows], "queries": {"findings": q}}


__all__ = ["Unavailable", "available", "findings", "merchant", "overview", "top"]
