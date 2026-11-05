"""Time passing: what happened after the brain's signals, how good its forecasts
were, and what it learns from that.

``refresh`` is the loop that runs whenever new sales arrive, real or simulated:

1. **Grade past forecasts** against what really sold (``forecast_accuracy``):
   error (WAPE), bias and how often real sales fell inside the 80% and 95%
   ranges. Days a product was out of stock are left out: zero sales there
   show an empty shelf, not zero demand.
2. **Judge run-out warnings** whose window has passed, from the recorded
   sales (``in_stock = false`` days) and stock:

   - prevented: someone acted and it didn't run out;
   - happened_anyway: someone acted, but it still ran out;
   - happened: nobody acted and it ran out;
   - false_alarm: nobody acted and it didn't run out.

   Each outcome is saved as a memory in the product's brain, so the brain
   remembers what happened, maintenance and root learning can use it, and the
   next signals can cite it. Signals nobody answered before their deadline
   become "expired".
3. **Recalculate** forecasts and signals from the new "today".

The brain's "today" is the latest day of sales data, so nothing is judged
before the data covers it. ``simulate`` is a demo-only source of that data:
it writes the same tables real ingestion does (sales with ``in_stock``
flags, stock positions), so simulated and real sales go through one path.
``advance`` is simulate + refresh.

    uv run python -m secondbrain.evolution --days 28            # demo: simulate, then refresh
    uv run python -m secondbrain.evolution --refresh-only       # real data: just refresh
"""

import argparse
import hashlib
from dataclasses import dataclass, field
from datetime import date, timedelta

import numpy as np
import psycopg
from psycopg.rows import dict_row, tuple_row

from secondbrain import schema as db_schema
from secondbrain.config import Settings, get_settings
from secondbrain.forecasting import store
from secondbrain.forecasting.__main__ import run as run_forecasts
from secondbrain.memory.store import vector_literal
from secondbrain.models import LLM, Embedder

FORECAST_HORIZON = 120
# A run-out counts as the one warned about from two weeks before the predicted
# date to a week after it; a warning is judged once the data covers that window.
EARLY_DAYS = 14
LATE_DAYS = 7
# Forecasts are graded on their first four weeks: the horizon ordering
# decisions rest on, the same for every forecast however often they are rerun.
ACCURACY_DAYS = 28

SCHEMA_SQL = """
-- How each forecast did against what really sold, over its first weeks.
-- Totals are kept next to the ratios so any set of forecasts can be summed.
CREATE TABLE forecast_accuracy (
    forecast_id     uuid PRIMARY KEY REFERENCES forecasts(id) ON DELETE CASCADE,
    brain_path      ltree NOT NULL,
    sku             text NOT NULL,
    cutoff          date NOT NULL,
    model           text NOT NULL,
    scored_through  date NOT NULL,     -- last day of actuals graded so far
    complete        boolean NOT NULL,  -- every day it is graded on has happened
    days            integer NOT NULL,  -- in-stock days graded
    censored_days   integer NOT NULL,  -- out-of-stock days left out (sales there aren't demand)
    actual          double precision NOT NULL,
    forecast        double precision NOT NULL,
    abs_error       double precision NOT NULL,
    in80            integer NOT NULL,  -- days real sales fell inside the 80% range
    in95            integer NOT NULL,
    wape            real,  -- NULL when nothing sold
    bias            real,  -- positive: forecast too high
    coverage80      real,
    coverage95      real,
    scored_at       timestamptz NOT NULL DEFAULT now(),
    FOREIGN KEY (brain_path, sku) REFERENCES products
);
CREATE INDEX forecast_accuracy_product_idx ON forecast_accuracy (brain_path, sku, scored_through);

-- What each company's last refresh saw, so a refresh on unchanged data
-- changes nothing. System table: the app role can't read it.
CREATE TABLE evolution_state (
    company       ltree PRIMARY KEY REFERENCES brains(path),
    data_through  date NOT NULL,
    digest        text NOT NULL,
    refreshed_at  timestamptz NOT NULL DEFAULT now()
);

ALTER TABLE forecast_accuracy ENABLE ROW LEVEL SECURITY;
ALTER TABLE forecast_accuracy FORCE ROW LEVEL SECURITY;
ALTER TABLE evolution_state ENABLE ROW LEVEL SECURITY;
ALTER TABLE evolution_state FORCE ROW LEVEL SECURITY;
-- Visible exactly when the product is (the subquery runs under products' own policy).
CREATE POLICY forecast_accuracy_read ON forecast_accuracy FOR SELECT
    USING (EXISTS (SELECT 1 FROM products p
                   WHERE p.brain_path = forecast_accuracy.brain_path AND p.sku = forecast_accuracy.sku));
GRANT SELECT ON forecast_accuracy TO secondbrain_app;
"""

DROP_SQL = "DROP TABLE IF EXISTS forecast_accuracy, evolution_state CASCADE;"


def ensure_schema(conn: psycopg.Connection) -> None:
    """Add the tables to a database created before they existed (purely additive)."""
    with conn.cursor(row_factory=tuple_row) as cur:
        if cur.execute("SELECT to_regclass('forecast_accuracy') IS NULL").fetchone()[0]:
            cur.execute(SCHEMA_SQL)


@dataclass
class ProductDays:
    sku: str
    name: str
    sold: int = 0
    lost: int = 0
    out_of_stock_days: list[date] = field(default_factory=list)
    arrivals: list[str] = field(default_factory=list)
    price: float = 0.0
    # Day by day, for the demo's charts: stock at the end of the day and demand.
    days: list[date] = field(default_factory=list)
    stock: list[int] = field(default_factory=list)
    demand: list[int] = field(default_factory=list)
    arrived_on: list[date] = field(default_factory=list)


def _fmt(d: date) -> str:
    return f"{d:%b} {d.day}"


def data_through(conn: psycopg.Connection, under: str) -> dict[str, date]:
    """Latest day of sales data per brain: each brain's "today"."""
    with conn.cursor(row_factory=dict_row) as cur:
        return {r["brain"]: r["day"] for r in cur.execute(
            """SELECT brain_path::text AS brain, max(day) AS day FROM sales_daily
               WHERE brain_path <@ %s::ltree GROUP BY 1""", (under,))}


# ---------- demo-only data source ----------

def simulate(conn: psycopg.Connection, brain: str, days: int, rng: np.random.Generator) -> tuple[date, date, dict[str, ProductDays]]:
    """Advance one brain's sales and stock by ``days``. Returns (first day, last day, per product).

    Demo only. Writes what real ingestion would: one sales row per product and
    day (``in_stock = false`` when demand went unmet) and the stock position
    the morning after the last day.
    """
    # Every product continues from the day after the brain's last sales day, so
    # histories stay contiguous.
    last_sale = data_through(conn, brain).get(brain)
    # Acting on a run-out signal places the recommended order on the brain's
    # date (not the wall clock): the day it was raised, or today if someone
    # acted later. It goes into the stock record once, as stock on order, and
    # arrives after the lead time.
    for o in conn.execute(
        """SELECT s.id, s.details FROM signals s
           WHERE s.kind = 'stockout' AND s.status = 'acted' AND s.details->>'brain' = %s
             AND NOT coalesce((s.details->>'ordered')::boolean, false)""",
        (brain,),
    ).fetchall():
        d = o["details"]
        placed = date.fromisoformat(d["as_of"])
        if last_sale:
            placed = max(placed, last_sale + timedelta(days=1))
        eta = placed + timedelta(days=d["lead_time_days"])
        conn.execute(
            """UPDATE inventory SET on_order = on_order + %s,
                      order_eta = CASE WHEN order_eta IS NULL OR on_order = 0 THEN %s ELSE least(order_eta, %s) END
               WHERE brain_path = %s::ltree AND sku = %s""",
            (d["quantity"], eta, eta, brain, d["sku"]),
        )
        conn.execute("""UPDATE signals SET details = details || jsonb_build_object('ordered', true, 'eta', %s::text)
                        WHERE id = %s""", (eta.isoformat(), o["id"]))
    rows = conn.execute(
        """SELECT p.sku, p.name, p.base_price::float AS price, i.as_of, i.on_hand, i.on_order, i.order_eta,
                  i.lead_time_days, i.sku IS NOT NULL AS tracked
           FROM products p LEFT JOIN inventory i USING (brain_path, sku) WHERE p.brain_path = %s::ltree""",
        (brain,),
    ).fetchall()
    # A product without stock data never runs out.
    first = last_sale + timedelta(days=1) if last_sale else min(r["as_of"] for r in rows if r["as_of"])
    last = first + timedelta(days=days - 1)
    promos = store.load_promotions(conn, brain)
    result, sales_rows, updates = {}, [], []
    for r in rows:
        forecast = store.latest_forecast(conn, brain, r["sku"])
        if forecast is None:
            continue  # nothing to draw sales from: this product's data stops
        points = {date.fromisoformat(p["day"]): p for p in forecast["points"]}
        tail = forecast["points"][-1]["forecast"]
        stock = r["on_hand"] if r["tracked"] else 10**9
        on_order, eta = r["on_order"] or 0, r["order_eta"]
        days_out = ProductDays(r["sku"], r["name"], price=r["price"])
        for i in range(days):
            day = first + timedelta(days=i)
            if on_order and eta and day >= eta:
                stock += on_order
                days_out.arrivals.append(f"{on_order:,} units on order arrived {_fmt(day)}")
                days_out.arrived_on.append(day)
                on_order, eta = 0, None
            demand = int(rng.poisson(max(points.get(day, {"forecast": tail})["forecast"], 0.1)))
            sold = min(demand, stock)
            stock -= sold
            if sold < demand:
                days_out.out_of_stock_days.append(day)
            days_out.sold += sold
            days_out.lost += demand - sold
            days_out.days.append(day)
            days_out.stock.append(stock)
            days_out.demand.append(demand)
            discount = max((p["discount"] for p in promos
                            if p["starts"] <= day <= p["ends"] and p["sku"] in (None, r["sku"])), default=0.0)
            price = round(r["price"] * (1 - discount), 2)
            sales_rows.append({"sku": r["sku"], "day": day, "units": sold, "revenue": round(sold * price, 2),
                               "price": price, "discount": discount, "in_stock": sold == demand})
        if r["tracked"]:
            updates.append((r["sku"], stock, on_order, eta, last + timedelta(days=1)))
            result[r["sku"]] = days_out

    # Looked up before the COPY: a connection can't run other queries while copying.
    sensitivity = conn.execute("SELECT sensitivity FROM products WHERE brain_path = %s::ltree LIMIT 1", (brain,)).fetchone()["sensitivity"]
    with conn.cursor().copy(
        "COPY sales_daily (brain_path, sku, day, units, revenue, price, discount, in_stock, sensitivity) FROM STDIN"
    ) as copy:
        for s in sales_rows:
            copy.write_row((brain, s["sku"], s["day"], s["units"], s["revenue"], s["price"], s["discount"], s["in_stock"], sensitivity))
    for sku, on_hand, on_order, eta, as_of in updates:
        conn.execute(
            """UPDATE inventory SET on_hand = %s, on_order = %s, order_eta = %s, as_of = %s
               WHERE brain_path = %s::ltree AND sku = %s""",
            (on_hand, on_order, eta, as_of, brain, sku),
        )
    return first, last, result


# ---------- grading forecasts ----------

def grade(days: list[dict]) -> dict:
    """Grade a forecast on actual days: ``[{units, in_stock, forecast, lo80, hi80, lo95, hi95}]``.

    Out-of-stock days are censored: what sold there is a floor on demand, not
    demand, so counting them would make an accurate forecast look too high.
    """
    graded = [d for d in days if d["in_stock"]]
    totals = {
        "days": len(graded),
        "censored_days": len(days) - len(graded),
        "actual": float(sum(d["units"] for d in graded)),
        "forecast": float(sum(d["forecast"] for d in graded)),
        "abs_error": float(sum(abs(d["units"] - d["forecast"]) for d in graded)),
        "in80": sum(d["lo80"] <= d["units"] <= d["hi80"] for d in graded),
        "in95": sum(d["lo95"] <= d["units"] <= d["hi95"] for d in graded),
    }
    return {**totals, **_ratios(totals)}


def _ratios(t: dict) -> dict:
    """WAPE and bias as in forecasting.metrics, and interval coverage, from totals."""
    actual, days = t["actual"], t["days"]
    return {"wape": t["abs_error"] / actual if actual else None,
            "bias": (t["forecast"] - actual) / actual if actual else None,
            "coverage80": t["in80"] / days if days else None,
            "coverage95": t["in95"] / days if days else None}


def score_forecasts(conn: psycopg.Connection, under: str) -> int:
    """Grade every forecast under a brain whose first weeks have (partly) happened.

    Each is graded on its first ACCURACY_DAYS days after the cutoff. Complete
    grades are left alone, and a row is only rewritten when new actuals
    changed it. Returns how many rows were written.
    """
    through = data_through(conn, under)
    with conn.cursor(row_factory=dict_row) as cur:
        rows = cur.execute(
            """SELECT f.id, f.brain_path::text AS brain, f.sku, f.cutoff, f.model,
                      (f.points->-1->>'day')::date AS last_point,
                      p.forecast, p.lo80, p.hi80, p.lo95, p.hi95, s.day, s.units, s.in_stock
               FROM forecasts f
               LEFT JOIN forecast_accuracy a ON a.forecast_id = f.id
               CROSS JOIN LATERAL jsonb_to_recordset(f.points)
                    AS p(day date, forecast float8, lo80 float8, hi80 float8, lo95 float8, hi95 float8)
               JOIN sales_daily s ON s.brain_path = f.brain_path AND s.sku = f.sku AND s.day = p.day
               WHERE f.brain_path <@ %s::ltree AND NOT coalesce(a.complete, false)
                 AND p.day > f.cutoff AND p.day <= f.cutoff + %s
               ORDER BY f.id, s.day""",
            (under, ACCURACY_DAYS),
        ).fetchall()
    by_forecast: dict = {}
    for r in rows:
        by_forecast.setdefault(r["id"], []).append(r)
    written = 0
    for forecast_id, days in by_forecast.items():
        f, g = days[0], grade(days)
        end = min(f["cutoff"] + timedelta(days=ACCURACY_DAYS), f["last_point"])
        written += conn.execute(
            """INSERT INTO forecast_accuracy (forecast_id, brain_path, sku, cutoff, model, scored_through, complete,
                   days, censored_days, actual, forecast, abs_error, in80, in95, wape, bias, coverage80, coverage95)
               VALUES (%s, %s::ltree, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
               ON CONFLICT (forecast_id) DO UPDATE SET
                   scored_through = EXCLUDED.scored_through, complete = EXCLUDED.complete, days = EXCLUDED.days,
                   censored_days = EXCLUDED.censored_days, actual = EXCLUDED.actual, forecast = EXCLUDED.forecast,
                   abs_error = EXCLUDED.abs_error, in80 = EXCLUDED.in80, in95 = EXCLUDED.in95, wape = EXCLUDED.wape,
                   bias = EXCLUDED.bias, coverage80 = EXCLUDED.coverage80, coverage95 = EXCLUDED.coverage95,
                   scored_at = now()
               WHERE (forecast_accuracy.scored_through, forecast_accuracy.days, forecast_accuracy.censored_days,
                      forecast_accuracy.complete)
                     IS DISTINCT FROM (EXCLUDED.scored_through, EXCLUDED.days, EXCLUDED.censored_days, EXCLUDED.complete)""",
            (forecast_id, f["brain"], f["sku"], f["cutoff"], f["model"], days[-1]["day"],
             through.get(f["brain"], date.min) >= end, g["days"], g["censored_days"], g["actual"], g["forecast"],
             g["abs_error"], g["in80"], g["in95"], g["wape"], g["bias"], g["coverage80"], g["coverage95"]),
        ).rowcount
    return written


_SUMS = """count(*) AS forecasts, coalesce(sum(days), 0) AS days, coalesce(sum(censored_days), 0) AS censored_days,
           coalesce(sum(actual), 0) AS actual, coalesce(sum(forecast), 0) AS forecast,
           coalesce(sum(abs_error), 0) AS abs_error, coalesce(sum(in80), 0) AS in80, coalesce(sum(in95), 0) AS in95,
           max(scored_through) AS through"""


def _summary(row: dict | None) -> dict | None:
    if not row or not row["days"]:
        return None
    out = {k: (float(v) if k in ("actual", "forecast", "abs_error") else v) for k, v in row.items()}
    return {**out, **_ratios(out)}


def realized_accuracy(conn: psycopg.Connection, brain: str, sku: str, days: int = 28) -> dict | None:
    """How this product's forecasts did against real sales recently, or None before any were graded.

    Sums the graded forecasts that include actuals from the last ``days`` days
    of this product's data (each graded on its first ACCURACY_DAYS days, out of
    stock days left out). Returns ``{wape, bias, coverage80, coverage95, days,
    censored_days, forecasts, through, ...totals}``. Works on the admin
    connection or as a user (row-level security applies).
    """
    with conn.cursor(row_factory=dict_row) as cur:
        row = cur.execute(
            f"""SELECT {_SUMS} FROM forecast_accuracy
                WHERE brain_path = %(brain)s::ltree AND sku = %(sku)s
                  AND scored_through > (SELECT max(day) FROM sales_daily
                                        WHERE brain_path = %(brain)s::ltree AND sku = %(sku)s) - %(days)s""",
            {"brain": brain, "sku": sku, "days": days},
        ).fetchone()
    return _summary(row)


def describe_accuracy(a: dict | None, days: int = 28) -> str | None:
    """Plain-language grade, e.g. "Forecasts were off by 12% from real sales over 28 graded days; ..."."""
    if not a or a["wape"] is None:
        return None
    text = f"Forecasts were off by {a['wape']:.0%} from real sales over {a['days']:,} graded product-days"
    if abs(a["bias"]) >= 0.03:
        text += f" (running {abs(a['bias']):.0%} {'high' if a['bias'] > 0 else 'low'})"
    text += (f"; {a['coverage80']:.0%} of days fell inside the 80% range"
             f" and {a['coverage95']:.0%} inside the 95% range.")
    if a["censored_days"]:
        text += f" {a['censored_days']} out-of-stock day(s) were left out: sales there show the empty shelf, not demand."
    return text


# ---------- judging warnings ----------

def _forecast_in_force(conn: psycopg.Connection, brain: str, sku: str, start: date, end: date) -> dict[date, float]:
    """For each day, what the latest forecast made before that day expected."""
    with conn.cursor(row_factory=dict_row) as cur:
        return {r["day"]: r["forecast"] for r in cur.execute(
            """SELECT DISTINCT ON (p.day) p.day, p.forecast
               FROM forecasts f CROSS JOIN LATERAL jsonb_to_recordset(f.points) AS p(day date, forecast float8)
               WHERE f.brain_path = %s::ltree AND f.sku = %s AND p.day BETWEEN %s AND %s AND f.cutoff < p.day
               ORDER BY p.day, f.created_at DESC""",
            (brain, sku, start, end))}


def judge(conn: psycopg.Connection, embedder: Embedder, settings: Settings, company: str) -> dict:
    """Expire unanswered signals whose deadline passed, and judge run-out warnings
    whose window the recorded data now covers. Needs a dict_row connection.

    Returns ``{"outcomes": [...], "expired": n, "waiting": [...]}``; ``waiting``
    lists warnings that can't be judged because the product's data stopped.
    """
    result = {"outcomes": [], "expired": 0, "waiting": []}
    through = data_through(conn, company)
    if not through:
        return result
    today = max(through.values())
    # Only when new data arrived since the last refresh: a signal raised on
    # stale stock data may already be due before "today", and must get its day.
    last = conn.execute("SELECT data_through FROM evolution_state WHERE company = %s::ltree", (company,)).fetchone()
    if last is None or last["data_through"] < today:
        result["expired"] = conn.execute(
            "UPDATE signals SET status = 'expired' WHERE company = %s::ltree AND status = 'open' AND due < %s",
            (company, today),
        ).rowcount
    signals = conn.execute(
        """SELECT s.id, s.status, s.details, s.title, s.impact_value::float AS impact_value FROM signals s
           LEFT JOIN signal_outcomes o ON o.signal_id = s.id
           WHERE s.company = %(company)s::ltree AND s.kind = 'stockout' AND o.signal_id IS NULL
             AND s.status IN ('acted', 'dismissed', 'expired')
             -- Already empty when raised: there was no run-out left to prevent.
             AND NOT coalesce((s.details->>'already_out')::boolean, false)
             -- An unanswered warning raised again for the same run-out is one
             -- story: only the latest warning is judged, so it counts once.
             AND NOT (s.status = 'expired' AND EXISTS (
                 SELECT 1 FROM signals n
                 WHERE n.company = s.company AND n.kind = 'stockout' AND n.created_at > s.created_at
                   AND n.details->>'brain' = s.details->>'brain' AND n.details->>'sku' = s.details->>'sku'
                   AND NOT coalesce((n.details->>'already_out')::boolean, false)
                   AND (n.details->>'as_of')::date <= (s.details->>'runs_out')::date + %(late)s))
           ORDER BY s.created_at""",
        {"company": company, "late": LATE_DAYS},
    ).fetchall()
    cast = db_schema.vector_type(settings.embedding_dim)[0].split("(")[0]
    for s in signals:
        d = s["details"]
        predicted = date.fromisoformat(d["runs_out"])
        start = max(date.fromisoformat(d["as_of"]), predicted - timedelta(days=EARLY_DAYS))
        end = predicted + timedelta(days=LATE_DAYS)
        if through.get(d["brain"], date.min) < end:
            continue  # the data doesn't cover the window yet
        days = conn.execute(
            """SELECT day, units, in_stock FROM sales_daily
               WHERE brain_path = %s::ltree AND sku = %s AND day BETWEEN %s AND %s ORDER BY day""",
            (d["brain"], d["sku"], start, end),
        ).fetchall()
        out_days = [r["day"] for r in days if not r["in_stock"]]
        # A stock count of zero inside the window is a run-out too, even on a day without demand.
        stock = conn.execute("SELECT as_of, on_hand FROM inventory WHERE brain_path = %s::ltree AND sku = %s",
                             (d["brain"], d["sku"])).fetchone()
        if stock and stock["on_hand"] == 0 and start <= stock["as_of"] <= end:
            out_days = sorted(set(out_days) | {stock["as_of"]})
        if not out_days and (not days or days[-1]["day"] < end):
            # The product's data stopped: no run-out seen, but no proof it didn't happen.
            seen = days[-1]["day"] if days else None
            result["waiting"].append(f"{d['name']}: no sales data " + (f"after {_fmt(seen)}" if seen else "in the window")
                                     + f", so the warning for {_fmt(predicted)} can't be judged yet")
            continue
        # Sales lost: what the forecast in force expected on out-of-stock days, minus what still sold.
        expected = _forecast_in_force(conn, d["brain"], d["sku"], start, end)
        sold = {r["day"]: r["units"] for r in days}
        lost = round(sum(max(expected.get(day, 0.0) - sold.get(day, 0), 0.0) for day in out_days))
        lost_value = lost * d["price"]
        acted = s["status"] == "acted"
        if acted and not out_days:
            outcome = "prevented"
            detail = (f"{d['name']}: acted on the warning (order of {d['quantity']:,} units from {d['supplier']}) "
                      f"and it did not run out through {_fmt(end)}.")
            value = s["impact_value"]
        elif acted:
            outcome = "happened_anyway"
            detail = (f"{d['name']}: acted on the warning, but it still ran out on {_fmt(out_days[0])} "
                      f"before the order arrived; about {lost:,} sales lost (${lost_value:,.0f}).")
            value = -lost_value
        elif out_days:
            outcome = "happened"
            detail = (f"{d['name']}: the warning was not acted on and it ran out on {_fmt(out_days[0])}, as predicted "
                      f"for {_fmt(predicted)}; about {lost:,} sales lost (${lost_value:,.0f}).")
            value = -lost_value
        else:
            outcome = "false_alarm"
            detail = (f"{d['name']}: the warning was not acted on and it did not run out through {_fmt(end)}; "
                      f"demand was lower than forecast.")
            value = 0
        # The brain remembers what happened, in the product's brain, so later
        # signals, maintenance and root learning can use it.
        [embedding] = embedder.embed([detail])
        memory_id = conn.execute(
            f"""INSERT INTO memories (brain_path, memory_type, tier, sensitivity, content, embedding, embedding_model, source)
                SELECT %s::ltree, 'episodic', 'long_term', sensitivity, %s, %s::{cast}, %s, 'signal outcome'
                FROM products WHERE brain_path = %s::ltree AND sku = %s RETURNING id""",
            (d["brain"], "Outcome: " + detail, vector_literal(embedding), embedder.name, d["brain"], d["sku"]),
        ).fetchone()["id"]
        conn.execute(
            """INSERT INTO signal_outcomes (signal_id, result, detail, value, evaluated_on, memory_id)
               VALUES (%s, %s, %s, %s, %s, %s)""",
            (s["id"], outcome, detail, value, through[d["brain"]], memory_id),
        )
        result["outcomes"].append({"title": s["title"], "result": outcome, "detail": detail})
    return result


# ---------- the loop ----------

def _inputs_digest(conn: psycopg.Connection, company: str) -> str:
    """Fingerprint of everything stock signals are calculated from: stock,
    latest forecasts, feedback and outcomes. Unchanged means nothing to recalculate."""
    rows = conn.execute(
        """SELECT concat_ws('|', 'stock', brain_path, sku, as_of, on_hand, on_order, order_eta) AS k
           FROM inventory WHERE brain_path <@ %(c)s::ltree
           UNION ALL
           SELECT concat_ws('|', 'forecast', brain_path, sku, max(created_at)) FROM forecasts
           WHERE brain_path <@ %(c)s::ltree GROUP BY brain_path, sku
           UNION ALL
           SELECT concat_ws('|', 'feedback', s.id, s.status, o.result) FROM signals s
           LEFT JOIN signal_outcomes o ON o.signal_id = s.id
           WHERE s.company = %(c)s::ltree AND s.status <> 'open'
           ORDER BY 1""",
        {"c": company},
    ).fetchall()
    return hashlib.sha256("\n".join(r["k"] for r in rows).encode()).hexdigest()


def _stale_brains(conn: psycopg.Connection, company: str) -> list[str]:
    """Brains with sales newer than a product's latest forecast."""
    return [r["brain"] for r in conn.execute(
        """SELECT DISTINCT s.brain_path::text AS brain
           FROM (SELECT brain_path, sku, max(day) AS day FROM sales_daily
                 WHERE brain_path <@ %s::ltree GROUP BY 1, 2) s
           LEFT JOIN LATERAL (SELECT max(cutoff) AS cutoff FROM forecasts f
                              WHERE f.brain_path = s.brain_path AND f.sku = s.sku) f ON true
           WHERE f.cutoff IS NULL OR f.cutoff < s.day ORDER BY 1""",
        (company,),
    ).fetchall()]


def company_accuracy(conn: psycopg.Connection, company: str, days: int = 90) -> dict | None:
    """``realized_accuracy`` summed over every product under a brain (admin connection).

    Forecasts are graded on their first 4 weeks, so a quarter-long window
    still has grades after a long jump forward in the demo."""
    with conn.cursor(row_factory=dict_row) as cur:
        return _summary(cur.execute(
            f"""SELECT {_SUMS} FROM forecast_accuracy
                WHERE brain_path <@ %(c)s::ltree
                  AND scored_through > (SELECT max(day) FROM sales_daily WHERE brain_path <@ %(c)s::ltree) - %(days)s""",
            {"c": company, "days": days},
        ).fetchone())


def refresh(settings: Settings, llm: LLM, embedder: Embedder, company: str) -> dict:
    """Run when new sales arrive: grade past forecasts, judge warnings up to the
    latest data day, re-forecast and recalculate signals.

    Idempotent: a second run on the same data changes nothing.
    """
    from secondbrain import signals as signals_module

    summary = {"company": company, "data_through": None, "scored": 0, "outcomes": [], "expired": 0,
               "waiting": [], "forecasted": [], "signals_recalculated": False, "signals": 0,
               "accuracy": None, "accuracy_text": None}
    with psycopg.connect(settings.admin_database_url, row_factory=dict_row) as conn:
        ensure_schema(conn)
        through = data_through(conn, company)
        if not through:
            return summary
        today = max(through.values())
        summary["data_through"] = today.isoformat()
        summary["scored"] = score_forecasts(conn, company)
        summary.update(judge(conn, embedder, settings, company))
        summary["forecasted"] = _stale_brains(conn, company)
        conn.commit()
    for brain in summary["forecasted"]:
        run_forecasts(settings.admin_database_url, under=brain, horizon=FORECAST_HORIZON)
    with psycopg.connect(settings.admin_database_url, row_factory=dict_row) as conn:
        digest = _inputs_digest(conn, company)
        last = conn.execute("SELECT data_through, digest FROM evolution_state WHERE company = %s::ltree",
                            (company,)).fetchone()
        if last is None or (last["data_through"], last["digest"]) != (today, digest):
            signals_module.run(settings, llm, company)
            summary["signals_recalculated"] = True
            conn.execute(
                """INSERT INTO evolution_state (company, data_through, digest) VALUES (%s::ltree, %s, %s)
                   ON CONFLICT (company) DO UPDATE SET data_through = EXCLUDED.data_through,
                       digest = EXCLUDED.digest, refreshed_at = now()""",
                (company, today, digest),
            )
        summary["signals"] = conn.execute(
            "SELECT count(*) AS n FROM signals WHERE company = %s::ltree AND status = 'open'", (company,)).fetchone()["n"]
        summary["accuracy"] = company_accuracy(conn, company)
        summary["accuracy_text"] = describe_accuracy(summary["accuracy"])
    return summary


POLICIES = ("as_is", "follow", "ignore")


def follow_advice(conn: psycopg.Connection, company: str) -> int:
    """Demo: act on every open run-out warning, as a business following the brain would."""
    return conn.execute(
        """UPDATE signals SET status = 'acted', feedback_by = NULL, feedback_at = now(),
                              feedback_note = 'Followed the brain''s advice (simulation)'
           WHERE company = %s::ltree AND kind = 'stockout' AND status = 'open'""",
        (company,),
    ).rowcount


def snapshot(conn: psycopg.Connection, company: str) -> dict:
    """The few numbers that show a company's brain moving: its date, what is
    at stake now, what acting saved and ignoring lost, and forecast error."""
    row = conn.execute(
        """SELECT (SELECT max(as_of) FROM inventory WHERE brain_path <@ %(c)s::ltree) AS today,
                  (SELECT count(*) FROM signals WHERE company = %(c)s::ltree AND status = 'open') AS open_signals,
                  (SELECT coalesce(sum(impact_value), 0)::float FROM signals
                    WHERE company = %(c)s::ltree AND status = 'open') AS at_stake,
                  (SELECT count(*) FROM memories WHERE brain_path <@ %(c)s::ltree
                    AND source = 'signal outcome') AS memories,
                  (SELECT count(*) FROM forecast_accuracy WHERE brain_path <@ %(c)s::ltree) AS forecasts_graded""",
        {"c": company},
    ).fetchone()
    results = {r["result"]: r for r in conn.execute(
        """SELECT o.result, count(*) AS n, coalesce(sum(o.value), 0)::float AS value FROM signal_outcomes o
           JOIN signals s ON s.id = o.signal_id WHERE s.company = %s::ltree GROUP BY o.result""", (company,))}
    get = lambda k, f: results[k][f] if k in results else 0
    accuracy = company_accuracy(conn, company)
    return {
        "today": row["today"].isoformat() if row["today"] else None,
        "open_signals": row["open_signals"],
        "at_stake": row["at_stake"],
        "saved": get("prevented", "value"),
        "lost": -(get("happened", "value") + get("happened_anyway", "value")),
        "prevented": get("prevented", "n"),
        "came_true": get("happened", "n") + get("happened_anyway", "n"),
        "false_alarms": get("false_alarm", "n"),
        "forecast_error": accuracy["wape"] if accuracy else None,
        # What the brain has learned from: outcomes it remembers, forecasts it graded.
        "memories": row["memories"],
        "forecasts_graded": row["forecasts_graded"],
    }


def advance(settings: Settings, llm: LLM, embedder: Embedder, company: str, days: int, seed: int = 7,
            policy: str = "as_is", step: int = 7) -> dict:
    """Demo: move a company forward by ``days`` of simulated sales, then refresh.

    ``policy`` decides what the business does with run-out warnings: "as_is"
    keeps what people chose, "follow" acts on every warning as it appears
    (checked every ``step`` days, so warnings raised along the way are acted on
    too), "ignore" leaves them unanswered. Returns before/after numbers and
    each product's day-by-day stock for the charts.
    """
    from secondbrain import signals as signals_module

    if policy not in POLICIES:
        raise ValueError(f"policy must be one of {POLICIES}")
    rng = np.random.default_rng(seed)
    summary = {"company": company, "days": days, "policy": policy, "products": {}, "outcomes": [], "acted": 0}
    series: dict[str, ProductDays] = {}
    with psycopg.connect(settings.admin_database_url, row_factory=dict_row) as conn:
        ensure_schema(conn)
        summary["before"] = snapshot(conn, company)
        brains = [r["brain"] for r in conn.execute(
            "SELECT DISTINCT brain_path::text AS brain FROM inventory WHERE brain_path <@ %s::ltree ORDER BY 1", (company,))]
    if not brains:
        return summary
    remaining = days
    while remaining > 0:
        chunk = min(step, remaining) if policy == "follow" else remaining
        with psycopg.connect(settings.admin_database_url, row_factory=dict_row) as conn:
            if policy == "follow":
                summary["acted"] += follow_advice(conn, company)
            for brain in brains:
                first, last, simulated = simulate(conn, brain, chunk, rng)
                summary.setdefault("from", first.isoformat())
                summary["to"] = last.isoformat()
                for sku, part in simulated.items():
                    _extend(series, sku, part)
            conn.commit()
            remaining -= chunk
            if remaining > 0:
                # Between steps: judge what came due and raise new run-out
                # warnings from the current forecasts; no re-forecast or AI calls.
                summary["outcomes"] += judge(conn, embedder, settings, company)["outcomes"]
                conn.commit()
                signals_module.generate(conn, llm, company, business=False, settings=settings)
    result = refresh(settings, llm, embedder, company)
    summary.update({k: result[k] for k in ("expired", "waiting", "signals", "accuracy", "accuracy_text")})
    summary["outcomes"] += result["outcomes"]
    with psycopg.connect(settings.admin_database_url, row_factory=dict_row) as conn:
        summary["after"] = snapshot(conn, company)
        summary["top_signals"] = [dict(r) for r in conn.execute(
            """SELECT kind, title, action, urgency, impact_value::float AS impact_value, due FROM signals
               WHERE company = %s::ltree AND status = 'open'
               ORDER BY CASE urgency WHEN 'high' THEN 0 WHEN 'medium' THEN 1 ELSE 2 END, impact_value DESC NULLS LAST
               LIMIT 5""", (company,))]
    summary["products"] = {sku: {"name": p.name, "sold": p.sold, "lost": p.lost, "lost_value": round(p.lost * p.price),
                                 "out_of_stock_days": len(p.out_of_stock_days), "arrivals": p.arrivals,
                                 "days": [d.isoformat() for d in p.days], "stock": p.stock, "demand": p.demand,
                                 "arrived_on": [d.isoformat() for d in p.arrived_on]}
                           for sku, p in series.items()}
    return summary


def _extend(series: dict[str, ProductDays], sku: str, part: ProductDays) -> None:
    if sku not in series:
        series[sku] = part
        return
    whole = series[sku]
    whole.sold += part.sold
    whole.lost += part.lost
    for name in ("out_of_stock_days", "arrivals", "days", "stock", "demand", "arrived_on"):
        getattr(whole, name).extend(getattr(part, name))


def main() -> None:
    from secondbrain.models import build_models

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--company", default="root.lumen")
    parser.add_argument("--days", type=int, default=28)
    parser.add_argument("--refresh-only", action="store_true", help="no simulated sales: grade, judge and recalculate")
    args = parser.parse_args()
    settings = get_settings()
    llm, embedder = build_models(settings)
    if args.refresh_only:
        result = refresh(settings, llm, embedder, args.company)
        print(f"{result['company']}: data through {result['data_through']}, {result['scored']} forecast grade(s) updated")
    else:
        result = advance(settings, llm, embedder, args.company, args.days)
        print(f"{result['company']}: {result.get('from')} to {result.get('to')}")
        for sku, p in result["products"].items():
            print(f"  {sku:<16} sold {p['sold']:>6,}  lost {p['lost']:>5,}  out of stock {p['out_of_stock_days']} day(s)  {'; '.join(p['arrivals'])}")
    for o in result["outcomes"]:
        print(f"  [{o['result']}] {o['detail']}")
    for w in result.get("waiting", []):
        print(f"  [waiting] {w}")
    if result.get("accuracy_text"):
        print(f"  {result['accuracy_text']}")
    print(f"  {result.get('signals', 0)} open signal(s) after recalculating")


if __name__ == "__main__":
    main()


# Reads below run on the app connection as a user, under row-level security.

def _accuracy_weeks(cur) -> list[dict]:
    """Per week: how the forecasts made that week did, and how the warnings judged that week turned out."""
    weeks: dict = {}
    for r in cur.execute(
        f"SELECT date_trunc('week', cutoff)::date AS week, {_SUMS} FROM forecast_accuracy GROUP BY 1"
    ).fetchall():
        week = r.pop("week")
        weeks[week] = {"week": week, "forecasts": _summary(r), "warnings": {}}
    for r in cur.execute(
        """SELECT date_trunc('week', evaluated_on)::date AS week, result, count(*) AS n
           FROM signal_outcomes GROUP BY 1, 2"""
    ).fetchall():
        weeks.setdefault(r["week"], {"week": r["week"], "forecasts": None, "warnings": {}})["warnings"][r["result"]] = r["n"]
    for w in weeks.values():
        judged = sum(w["warnings"].values())
        # A warning was right when the run-out came, or was prevented by acting on it.
        w["warning_precision"] = (judged - w["warnings"].get("false_alarm", 0)) / judged if judged else None
    return [weeks[k] for k in sorted(weeks)]


def history(conn: psycopg.Connection) -> dict:
    """How this person's brain has evolved: track record, forecast grades,
    feedback, outcomes and facts it updated, limited to what they may see."""
    with conn.cursor(row_factory=dict_row) as cur:
        outcomes = cur.execute(
            """SELECT o.result, count(*) AS n, coalesce(sum(o.value), 0)::float AS value
               FROM signal_outcomes o GROUP BY o.result"""
        ).fetchall()
        feedback = cur.execute("SELECT status, count(*) AS n FROM signals GROUP BY status").fetchall()
        clock = cur.execute("SELECT max(as_of) AS today FROM inventory").fetchone()["today"]
        events = cur.execute(
            """SELECT 'outcome' AS kind, o.evaluated_on::timestamptz AS at, o.result AS label, o.detail AS text, NULL AS who
               FROM signal_outcomes o
               UNION ALL
               SELECT 'feedback', s.feedback_at, s.status, s.title || coalesce(' (' || s.feedback_note || ')', ''), s.feedback_by
               FROM signals s WHERE s.status IN ('acted', 'dismissed')
               UNION ALL
               SELECT 'expired', s.due::timestamptz, 'expired', s.title, NULL FROM signals s WHERE s.status = 'expired'
               UNION ALL
               SELECT 'updated', m.created_at, CASE WHEN m.superseded_by IS NOT NULL THEN m.status ELSE 'partly updated' END,
                      split_part(m.content, E'\n', 1) || ' — ' || coalesce(m.status_reason, ''), NULL
               FROM memories m WHERE m.superseded_by IS NOT NULL OR m.amended_by IS NOT NULL
               ORDER BY at DESC NULLS LAST LIMIT 40"""
        ).fetchall()
        accuracy = None
        # A database created before forecast grading existed has no grades yet.
        if cur.execute("SELECT to_regclass('forecast_accuracy') IS NOT NULL AS ok").fetchone()["ok"]:
            recent = _summary(cur.execute(
                f"""SELECT {_SUMS} FROM forecast_accuracy
                    WHERE scored_through > (SELECT max(day) FROM sales_daily) - 28"""
            ).fetchone())
            accuracy = {"recent": recent, "summary": describe_accuracy(recent), "weeks": _accuracy_weeks(cur)}
    results = {r["result"]: {"count": r["n"], "value": r["value"]} for r in outcomes}
    came_true = results.get("happened", {}).get("count", 0)
    false_alarms = results.get("false_alarm", {}).get("count", 0)
    return {
        "today": clock,
        "outcomes": results,
        "feedback": {r["status"]: r["n"] for r in feedback},
        "saved": results.get("prevented", {}).get("value", 0),
        "lost": -sum(results.get(k, {}).get("value", 0) for k in ("happened", "happened_anyway")),
        "warning_accuracy": (came_true / (came_true + false_alarms)) if came_true + false_alarms else None,
        "accuracy": accuracy,
        "events": events,
    }


def hq(conn: psycopg.Connection) -> dict:
    """The root operator's overview of every connected company (admin connection).

    Aggregates only: counts, money, outcomes and forecast grades, never a
    company's notes, so it is consistent with companies sharing only patterns
    with the root.
    """
    ensure_schema(conn)
    companies = conn.execute(
        """SELECT b.path::text AS path, b.name, b.sharing,
                  (SELECT count(*) FROM brains d WHERE d.path <@ b.path AND d.kind = 'department') AS departments,
                  (SELECT count(*) FROM users u WHERE u.home_brain <@ b.path) AS people,
                  (SELECT count(*) FROM memories m WHERE m.brain_path <@ b.path AND m.source IS NOT NULL
                      AND m.source NOT IN ('signal outcome')) AS document_sections,
                  (SELECT count(*) FROM memories m WHERE m.brain_path <@ b.path AND m.source IS NULL) AS notes,
                  (SELECT count(*) FROM memories m WHERE m.brain_path <@ b.path
                      AND (m.superseded_by IS NOT NULL OR m.amended_by IS NOT NULL)) AS facts_updated,
                  (SELECT count(*) FROM entities e WHERE e.company = b.path) AS entities,
                  (SELECT max(as_of) FROM inventory i WHERE i.brain_path <@ b.path) AS today
           FROM brains b WHERE b.kind = 'company' ORDER BY b.name"""
    ).fetchall()
    for c in companies:
        c["signals"] = {r["status"]: r["n"] for r in conn.execute(
            "SELECT status, count(*) AS n FROM signals WHERE company = %s::ltree GROUP BY status", (c["path"],))}
        c["at_stake"] = conn.execute(
            "SELECT coalesce(sum(impact_value), 0)::float AS v FROM signals WHERE company = %s::ltree AND status = 'open'",
            (c["path"],)).fetchone()["v"]
        c["outcomes"] = {r["result"]: {"count": r["n"], "value": r["v"]} for r in conn.execute(
            """SELECT o.result, count(*) AS n, coalesce(sum(o.value), 0)::float AS v FROM signal_outcomes o
               JOIN signals s ON s.id = o.signal_id WHERE s.company = %s::ltree GROUP BY o.result""", (c["path"],))}
        c["accuracy"] = company_accuracy(conn, c["path"])
        c["accuracy_text"] = describe_accuracy(c["accuracy"])
    lessons = [r["content"] for r in conn.execute(
        "SELECT content FROM memories WHERE brain_path = 'root' AND status = 'active' ORDER BY created_at")]
    return {"companies": companies, "lessons": lessons}
