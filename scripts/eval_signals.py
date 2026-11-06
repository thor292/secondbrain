"""How good are the brain's stock signals? Measured, not assumed.

Runs many simulated futures for the Acme test fixture (the same sales history
the tests use) through the public ``evolution.advance`` and scores what the
signals told people to do against what then happened:

- warnings: when nobody acts, did the products the brain warned about run out
  (precision), and did any product run out without a week's warning (recall)?
- lead time: when people order on the "order by" date, does the order arrive
  before the shelf is empty?
- quantity: once it arrives, does it last (no run-out in the following 30
  days) without leaving months of stock behind?
- confidence: is the stated confidence a fair probability that an ignored
  warning comes true (Brier score; lower is better)?
- money: lost margin plus the cost of holding stock, against doing nothing
  and against a naive reorder-point rule a spreadsheet could run.

A scenario is a seed (the randomness of daily sales), a supplier lead time,
starting stock for each product (days of cover) and a demand shock: reality
is the first forecast times 0.8, 1.0 or 1.25 (with daily randomness), so the
brain is tested on demand it did not see coming and has to catch up as its
weekly forecasts learn from the sales that happen. Each scenario runs under three policies: ignore every
signal, act on stock signals when they are due, and the reorder-point rule.

Everything runs on separate databases (secondbrain_test_signals_eval_*),
never on the demo database. A template is built once (schema, Acme, sales,
first forecast) and every scenario starts from a fresh copy of it.

    uv run python scripts/eval_signals.py --workers 5 --out eval.json
    uv run python scripts/eval_signals.py --quick        # a handful of scenarios
    uv run python scripts/eval_signals.py --compare before.json after.json
"""

import os

# One thread per worker: several workers share the machine, and numerical
# libraries otherwise each start a thread per core.
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse  # noqa: E402
import itertools
import json
import math
import statistics
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import date, timedelta
from pathlib import Path

import psycopg
from psycopg.rows import dict_row

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

from scripts.generate_sales import END, PRODUCTS, generate  # noqa: E402
from secondbrain import admin, evolution  # noqa: E402
from secondbrain import signals as signals_module  # noqa: E402
from secondbrain.config import Settings  # noqa: E402
from secondbrain.forecasting import store  # noqa: E402
from secondbrain.forecasting.__main__ import run as run_forecasts  # noqa: E402
from secondbrain.models import build_models  # noqa: E402
from secondbrain.money import inr  # noqa: E402
from secondbrain.schema import create_schema  # noqa: E402
from secondbrain.sensitivity import Sensitivity  # noqa: E402

# Gradient boosting is the slowest model by far (most of a weekly refit) and
# the weekly refits dominate the run time, so the evaluation forecasts with
# the other three models unless SB_EVAL_ALL_MODELS=1. Signals see the same
# kind of forecast (points, ranges, backtest error) either way.
if not os.environ.get("SB_EVAL_ALL_MODELS"):
    from secondbrain.forecasting import backtest, models as forecast_models

    backtest.default_models = lambda: [m for m in forecast_models.default_models() if m.name != "gradient_boosting"]

SERVER = "postgresql://secondbrain_admin:dev_admin_password@localhost:55432"
APP_SERVER = "postgresql://secondbrain_app:dev_app_password@localhost:55432"
PREFIX = "secondbrain_test_signals_eval"
TEMPLATE = f"{PREFIX}_base"
COMPANY, BRAIN = "root.acme", "root.acme.sales"
SKUS = ["brass-floor-lamp", "kantha-throw", "sheesham-sofa"]  # a lamp, a throw and a sofa
START = END + timedelta(days=1)  # the first day after the sales history

STEP_DAYS = 7  # people look at their signals once a week
WINDOW_DAYS = 84  # long enough for a 28-day lead time plus a month after arrival
NOTICE_DAYS = 7  # a run-out needs at least a week's warning to count as warned
SLACK_DAYS = 14  # a warning came true if it ran out by its predicted date plus this
COVER_AFTER_ARRIVAL = 30
OVERSTOCK_DAYS = 60  # more than this many days of stock left 30 days after arrival is overstock
HOLDING_RATE = 0.25  # yearly cost of holding stock, as a share of its cost
COST_SHARE = 0.5  # unit cost as a share of the selling price

SEEDS = [1, 2]
LEAD_TIMES = [7, 14, 28]
SHOCKS = [0.8, 1.0, 1.25]
COVERS = [12, 25, 40, 60, 90]  # starting days of stock, rotated over the products
POLICIES = ["ignore", "signals", "reorder_point"]


def settings_for(db: str) -> Settings:
    return Settings(_env_file=None, admin_database_url=f"{SERVER}/{db}", app_database_url=f"{APP_SERVER}/{db}",
                    model_provider="stub", embedding_dim=256, graph_prefix="sbeval")


def build_template() -> None:
    """Schema, the Acme company, its sales history and a first forecast, once."""
    with psycopg.connect(f"{SERVER}/secondbrain", autocommit=True) as conn:
        conn.execute(f'DROP DATABASE IF EXISTS "{TEMPLATE}"')
        conn.execute(f'CREATE DATABASE "{TEMPLATE}"')
    with psycopg.connect(f"{SERVER}/{TEMPLATE}", autocommit=True) as conn:
        conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        conn.execute("CREATE EXTENSION IF NOT EXISTS ltree")
        conn.execute("GRANT USAGE ON SCHEMA public TO secondbrain_app")
    settings = settings_for(TEMPLATE)
    create_schema(settings)
    rows, promos = generate()
    with psycopg.connect(settings.admin_database_url) as conn:
        admin.onboard(conn, json.loads((ROOT / "tests/fixtures/acme/company.json").read_text()))
        store.save_history(conn, BRAIN, Sensitivity.INTERNAL,
                           [(p.sku, p.name, p.category, p.price) for p in PRODUCTS if p.sku in SKUS],
                           [r for r in rows if r["sku"] in SKUS], [vars(p) for p in promos])
    run_forecasts(settings.admin_database_url, under=COMPANY, horizon=evolution.FORECAST_HORIZON)


def fresh_database(db: str) -> Settings:
    with psycopg.connect(f"{SERVER}/secondbrain", autocommit=True) as conn:
        conn.execute(f'DROP DATABASE IF EXISTS "{db}"')
        for attempt in range(20):
            try:
                conn.execute(f'CREATE DATABASE "{db}" TEMPLATE "{TEMPLATE}"')
                break
            except psycopg.errors.ObjectInUse:  # another worker is copying the template
                time.sleep(0.5 + attempt * 0.2)
        conn.execute(f'GRANT CONNECT ON DATABASE "{db}" TO secondbrain_app')
    return settings_for(db)


def scenarios(quick: bool = False) -> list[dict]:
    out = []
    for i, (seed, lead, shock) in enumerate(itertools.product(SEEDS, LEAD_TIMES, SHOCKS)):
        covers = {sku: COVERS[(i + k * 2) % len(COVERS)] for k, sku in enumerate(SKUS)}
        out.append({"seed": seed, "lead": lead, "shock": shock, "covers": covers})
    return out[::6] if quick else out


# ---------- one run ----------

def _reality(conn: psycopg.Connection, truth: dict, shock: float) -> None:
    """Store what demand really is as the newest forecast, which the
    simulation draws sales from: the first forecast (from the history alone)
    times the shock. It stays fixed for the whole run; the brain's own
    forecasts, refitted each week from the sales that happen, never feed back
    into it. The brain's next forecast replaces it before signals are made."""
    for sku, f in truth.items():
        points = [{**p, **{k: p[k] * shock for k in ("forecast", "lo80", "hi80", "lo95", "hi95") if k in p}}
                  for p in f["points"]]
        store.save_forecast(conn, BRAIN, sku, f["cutoff"], f["model"], f["reason"], points, f["backtest"])
    conn.commit()


def _today(conn) -> date:
    return conn.execute("SELECT max(as_of) AS d FROM inventory WHERE brain_path = %s::ltree", (BRAIN,)).fetchone()["d"]


def _open_signals(conn) -> list[dict]:
    return conn.execute(
        """SELECT id, kind, urgency, confidence, due, details, evidence, title FROM signals
           WHERE company = %s::ltree AND status = 'open' AND kind IN ('stockout', 'backorder')""",
        (COMPANY,)).fetchall()


def _reorder_point(conn, today: date, lead: int) -> list[dict]:
    """The naive rule: reorder when stock plus orders fall below the last 28
    days' average sales times (lead time + a week); order enough for the lead
    time plus a month."""
    placed = []
    for inv in conn.execute("SELECT sku, on_hand, on_order, order_eta FROM inventory WHERE brain_path = %s::ltree",
                            (BRAIN,)).fetchall():
        avg = conn.execute(
            """SELECT coalesce(sum(units), 0)::float / 28 AS a FROM sales_daily
               WHERE brain_path = %s::ltree AND sku = %s AND day >= %s AND day < %s""",
            (BRAIN, inv["sku"], today - timedelta(days=28), today)).fetchone()["a"]
        position = inv["on_hand"] + inv["on_order"]
        if inv["on_order"] or position > avg * (lead + 7):
            continue
        quantity = int(math.ceil(max(avg * (lead + COVER_AFTER_ARRIVAL) - position, 0) / 10) * 10)
        if quantity:
            eta = today + timedelta(days=lead)
            conn.execute("UPDATE inventory SET on_order = %s, order_eta = %s WHERE brain_path = %s::ltree AND sku = %s",
                         (quantity, eta, BRAIN, inv["sku"]))
            placed.append({"sku": inv["sku"], "day": today, "quantity": quantity, "eta": eta})
    conn.commit()
    return placed


def run_scenario(args: tuple[int, dict, str]) -> dict:
    worker, scenario, policy = args
    t0 = time.time()
    settings = fresh_database(f"{PREFIX}_w{worker}")
    llm, embedder = build_models(settings)
    lead, shock = scenario["lead"], scenario["shock"]
    with psycopg.connect(settings.admin_database_url, row_factory=dict_row) as conn:
        products = {r["sku"]: r for r in conn.execute(
            "SELECT sku, name, base_price::float AS price FROM products WHERE brain_path = %s::ltree", (BRAIN,))}
        truth = {sku: store.latest_forecast(conn, BRAIN, sku) for sku in SKUS}
        rows = []
        for sku in SKUS:
            points = store.latest_forecast(conn, BRAIN, sku)["points"][:28]
            daily = sum(p["forecast"] for p in points) / len(points)
            rows.append({"sku": sku, "as_of": START, "on_hand": int(daily * scenario["covers"][sku]), "on_order": 0,
                         "order_eta": None, "supplier": "Northwind", "lead_time_days": lead,
                         "unit_cost": round(products[sku]["price"] * COST_SHARE, 2)})
        store.save_inventory(conn, BRAIN, rows)
        conn.commit()
    signals_module.run(settings, llm, COMPANY)

    stock = {r["sku"]: r["on_hand"] for r in rows}
    pending = {sku: (0, None) for sku in SKUS}  # (on order, arrives) as the simulation sees it
    daily_stock = {sku: {} for sku in SKUS}
    daily_sold = {sku: {} for sku in SKUS}
    warnings, orders, lost = [], [], {sku: 0 for sku in SKUS}
    for step in range(WINDOW_DAYS // STEP_DAYS):
        with psycopg.connect(settings.admin_database_url, row_factory=dict_row) as conn:
            today = _today(conn)
            open_now = _open_signals(conn)
            for s in open_now:
                d = s["details"]
                sku = d.get("sku") or s["evidence"]["products"][0]["sku"]
                warnings.append({"step": step, "day": today.isoformat(), "kind": s["kind"], "sku": sku,
                                 "runs_out": d.get("runs_out"), "order_by": d.get("order_by"),
                                 "quantity": d.get("quantity"), "urgency": s["urgency"],
                                 "confidence": float(s["confidence"])})
            if policy == "signals":
                # Ordered on the last check before the deadline: the honest test of "order by".
                for s in open_now:
                    if s["kind"] == "stockout" and s["due"] < today + timedelta(days=STEP_DAYS):
                        conn.execute("UPDATE signals SET status = 'acted', feedback_at = now() WHERE id = %s", (s["id"],))
                        d = s["details"]
                        orders.append({"sku": d["sku"], "day": today.isoformat(), "quantity": d["quantity"],
                                       "eta": (today + timedelta(days=lead)).isoformat(), "due": s["due"].isoformat(),
                                       "runs_out": d["runs_out"]})
                        # The simulation adds it to what's on order, keeping the earliest arrival.
                        q, eta = pending[d["sku"]]
                        new_eta = today + timedelta(days=lead)
                        pending[d["sku"]] = (q + d["quantity"], new_eta if not q or eta is None else min(eta, new_eta))
                conn.commit()
            elif policy == "reorder_point":
                for o in _reorder_point(conn, today, lead):
                    orders.append({**o, "day": o["day"].isoformat(), "eta": o["eta"].isoformat()})
                    pending[o["sku"]] = (o["quantity"], o["eta"])
            _reality(conn, truth, shock)
        result = evolution.advance(settings, llm, embedder, COMPANY, STEP_DAYS, seed=scenario["seed"] * 1000 + step)
        with psycopg.connect(settings.admin_database_url, row_factory=dict_row) as conn:
            sold = {(r["sku"], r["day"]): r["units"] for r in conn.execute(
                "SELECT sku, day, units FROM sales_daily WHERE brain_path = %s::ltree AND day >= %s AND day < %s",
                (BRAIN, today, today + timedelta(days=STEP_DAYS)))}
            after = {r["sku"]: r for r in conn.execute(
                "SELECT sku, on_hand, on_order, order_eta FROM inventory WHERE brain_path = %s::ltree", (BRAIN,))}
        for sku in SKUS:
            q, eta = pending[sku]
            for i in range(STEP_DAYS):
                day = today + timedelta(days=i)
                if q and eta and day >= eta:
                    stock[sku] += q
                    q, eta = 0, None
                stock[sku] -= sold.get((sku, day), 0)
                daily_stock[sku][day.isoformat()] = stock[sku]
                daily_sold[sku][day.isoformat()] = sold.get((sku, day), 0)
            stock[sku] = after[sku]["on_hand"]  # the simulation's word is final
            pending[sku] = (after[sku]["on_order"], after[sku]["order_eta"])
            lost[sku] += result["products"].get(sku, {}).get("lost", 0)
    with psycopg.connect(settings.admin_database_url, row_factory=dict_row) as conn:
        out_days = {}
        for r in conn.execute(
                """SELECT sku, array_agg(day ORDER BY day) AS days FROM sales_daily
                   WHERE brain_path = %s::ltree AND day >= %s AND NOT in_stock GROUP BY sku""", (BRAIN, START)):
            out_days[r["sku"]] = [d.isoformat() for d in r["days"]]
        demand = {}
        for sku in SKUS:
            points = store.latest_forecast(conn, BRAIN, sku)["points"][:COVER_AFTER_ARRIVAL]
            demand[sku] = sum(p["forecast"] for p in points) / len(points)
    return {"scenario": scenario, "policy": policy, "db": settings.admin_database_url.rsplit("/", 1)[1], "seconds": round(time.time() - t0, 1),
            "products": {sku: {"price": products[sku]["price"], "unit_cost": round(products[sku]["price"] * COST_SHARE, 2),
                               "lost": lost[sku], "out_days": out_days.get(sku, []), "stock": daily_stock[sku], "sold": daily_sold[sku],
                               "end_daily_demand": demand[sku]} for sku in SKUS},
            "warnings": warnings, "orders": orders}


# ---------- scoring ----------

def _d(s: str) -> date:
    return date.fromisoformat(s)


def money(run: dict) -> dict:
    lost_margin = holding = excess = 0.0
    for p in run["products"].values():
        margin = p["price"] - p["unit_cost"]
        lost_margin += p["lost"] * margin
        holding += sum(max(s, 0) for s in p["stock"].values()) * p["unit_cost"] * HOLDING_RATE / 365
        last_day = max(p["stock"])
        rate = _rate(p, _d(last_day) - timedelta(days=27), 28)
        excess += max(p["stock"][last_day] - rate * OVERSTOCK_DAYS, 0) * p["unit_cost"]
    return {"lost_margin": lost_margin, "holding": holding, "cost": lost_margin + holding, "excess_tied_up": excess}


def score_warnings(run: dict) -> dict:
    """Precision, recall, timing and calibration of run-out warnings when nobody acts."""
    end = START + timedelta(days=WINDOW_DAYS - 1)
    tp = fp = judged = misses = actual = 0
    timing, brier, calibration = [], [], []
    for sku, p in run["products"].items():
        first_out = _d(p["out_days"][0]) if p["out_days"] else None
        mine = [w for w in run["warnings"] if w["sku"] == sku]
        first = next((w for w in mine if w["runs_out"] or w["kind"] == "backorder"), None)
        if first and first["kind"] == "stockout":
            predicted = _d(first["runs_out"])
            if predicted + timedelta(days=SLACK_DAYS) <= end:
                judged += 1
                came_true = first_out is not None and first_out <= predicted + timedelta(days=SLACK_DAYS)
                tp += came_true
                fp += not came_true
                brier.append((first["confidence"] - came_true) ** 2)
                calibration.append((first["confidence"], came_true))
                if came_true:
                    timing.append((first_out - predicted).days)
        if first_out is not None and first_out <= end - timedelta(days=0):
            actual += 1
            warned = any(_d(w["day"]) <= first_out - timedelta(days=NOTICE_DAYS) for w in mine)
            misses += not warned
    return {"judged": judged, "true": tp, "false_alarms": fp, "actual_runouts": actual, "missed": misses,
            "timing_errors": timing, "brier": brier, "calibration": calibration}


def _rate(p: dict, start: date, days: int) -> float:
    """Units sold a day on in-stock days from ``start`` (out-of-stock days understate demand)."""
    out = set(p["out_days"])
    sold = [p["sold"][d] for d in sorted(p["sold"]) if start <= _d(d) < start + timedelta(days=days) and d not in out]
    return max(statistics.mean(sold) if sold else p["end_daily_demand"], 0.1)


def _cover(p: dict, units: float, start: date, days: int = COVER_AFTER_ARRIVAL) -> float:
    """Days of sales ``units`` would cover, at the rate sold in the following days."""
    return units / _rate(p, start, days)


def score_orders(run: dict) -> dict:
    """For orders placed from signals or the rule: in time? enough? too much?"""
    on_time = late = short = over = judged_qty = in_time_judged = in_time_ok = 0
    buffers, cover_left = [], []
    for o in run["orders"]:
        p = run["products"][o["sku"]]
        eta = _d(o["eta"])
        before = [d for d in p["out_days"] if _d(o["day"]) <= _d(d) < eta]
        if before:
            late += 1
        else:
            on_time += 1
        # Lead-time adequacy proper: orders the brain expected to arrive before
        # the run-out (it wasn't already too late when the warning came).
        if o.get("runs_out") and _d(o["runs_out"]) >= eta:
            in_time_judged += 1
            in_time_ok += not before
        stock_before = p["stock"].get((eta - timedelta(days=1)).isoformat())
        if stock_before is not None:
            buffers.append(_cover(p, stock_before, eta))
        check = eta + timedelta(days=COVER_AFTER_ARRIVAL)
        if check.isoformat() in p["stock"]:
            judged_qty += 1
            if any(eta <= _d(d) < check for d in p["out_days"]):
                short += 1
            days_left = _cover(p, p["stock"][check.isoformat()], check - timedelta(days=COVER_AFTER_ARRIVAL))
            cover_left.append(days_left)
            over += days_left > OVERSTOCK_DAYS
    return {"orders": len(run["orders"]), "on_time": on_time, "late": late,
            "in_time_judged": in_time_judged, "in_time_ok": in_time_ok, "qty_judged": judged_qty,
            "short": short, "over": over, "buffer_days": buffers, "cover_left_days": cover_left}


def summarize(runs: list[dict]) -> dict:
    by = {}
    for r in runs:
        by.setdefault(r["policy"], []).append(r)
    out = {}
    if "ignore" in by:
        w = [score_warnings(r) for r in by["ignore"]]
        s = lambda k: sum(x[k] for x in w)
        flat = lambda k: [v for x in w for v in x[k]]
        cal = flat("calibration")
        out["warnings"] = {
            "judged": s("judged"), "precision": s("true") / s("judged") if s("judged") else None,
            "false_alarm_rate": s("false_alarms") / s("judged") if s("judged") else None,
            "actual_runouts": s("actual_runouts"),
            "recall": 1 - s("missed") / s("actual_runouts") if s("actual_runouts") else None,
            "missed": s("missed"),
            "timing_mae_days": statistics.mean(abs(t) for t in flat("timing_errors")) if flat("timing_errors") else None,
            "timing_bias_days": statistics.mean(flat("timing_errors")) if flat("timing_errors") else None,
            "brier": statistics.mean(flat("brier")) if flat("brier") else None,
            "mean_confidence": statistics.mean(c for c, _ in cal) if cal else None,
        }
    for policy in ("signals", "reorder_point"):
        if policy not in by:
            continue
        o = [score_orders(r) for r in by[policy]]
        s = lambda k: sum(x[k] for x in o)
        flat = lambda k: [v for x in o for v in x[k]]
        out[policy] = {
            "orders": s("orders"), "on_time": s("on_time") / s("orders") if s("orders") else None,
            "in_time_judged": s("in_time_judged"),
            "lead_time_adequate": s("in_time_ok") / s("in_time_judged") if s("in_time_judged") else None,
            "qty_judged": s("qty_judged"),
            "short_after_arrival": s("short") / s("qty_judged") if s("qty_judged") else None,
            "overstock": s("over") / s("qty_judged") if s("qty_judged") else None,
            "median_buffer_days_at_arrival": statistics.median(flat("buffer_days")) if flat("buffer_days") else None,
            "median_cover_left_days": statistics.median(flat("cover_left_days")) if flat("cover_left_days") else None,
        }
    totals = {policy: {k: sum(money(r)[k] for r in rs) for k in ("lost_margin", "holding", "cost", "excess_tied_up")}
              for policy, rs in by.items()}
    out["money"] = totals
    if "ignore" in totals:
        for policy in ("signals", "reorder_point"):
            if policy in totals:
                out["money"][policy]["saved_vs_nothing"] = totals["ignore"]["cost"] - totals[policy]["cost"]
        if "signals" in totals and "reorder_point" in totals:
            out["money"]["signals"]["saved_vs_reorder_point"] = totals["reorder_point"]["cost"] - totals["signals"]["cost"]
    out["runs"] = {p: len(rs) for p, rs in by.items()}
    return out


def report(summary: dict, title: str = "") -> str:
    pct = lambda v: "—" if v is None else f"{v:.0%}"
    num = lambda v, f="{:.1f}": "—" if v is None else f.format(v)
    lines = [title] if title else []
    w = summary.get("warnings")
    if w:
        lines += [f"Run-out warnings (ignored): precision {pct(w['precision'])} of {w['judged']} judged, "
                  f"false alarms {pct(w['false_alarm_rate'])}, recall {pct(w['recall'])} "
                  f"({w['missed']} of {w['actual_runouts']} run-outs without a week's warning), "
                  f"timing error {num(w['timing_mae_days'])} days (bias {num(w['timing_bias_days'], '{:+.1f}')}), "
                  f"Brier {num(w['brier'], '{:.3f}')} at mean confidence {pct(w['mean_confidence'])}"]
    for policy, label in (("signals", "Acting on signals when due"), ("reorder_point", "Naive reorder point")):
        o = summary.get(policy)
        if o:
            adequate = (f" ({pct(o['lead_time_adequate'])} of the {o['in_time_judged']} that weren't already too late)"
                        if o.get("in_time_judged") else "")
            lines += [f"{label}: {o['orders']} orders, arrived before run-out {pct(o['on_time'])}{adequate}, "
                      f"ran short within 30 days of arrival {pct(o['short_after_arrival'])}, "
                      f"overstock (>{OVERSTOCK_DAYS} days left) {pct(o['overstock'])}, "
                      f"median {num(o['median_buffer_days_at_arrival'])} days of stock left at arrival, "
                      f"{num(o['median_cover_left_days'])} days 30 days later"]
    for policy, m in summary["money"].items():
        extra = "".join(f", {k.replace('_', ' ')} {inr(m[k])}" for k in ("saved_vs_nothing", "saved_vs_reorder_point") if k in m)
        lines += [f"Money, {policy}: lost margin {inr(m['lost_margin'])} + holding {inr(m['holding'])} = "
                  f"{inr(m['cost'])}; overstock tied up at the end {inr(m['excess_tied_up'])}{extra}"]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--quick", action="store_true", help="a few scenarios, for a smoke test")
    parser.add_argument("--limit", type=int, help="only the first N scenarios")
    parser.add_argument("--policies", default=",".join(POLICIES))
    parser.add_argument("--reuse", help="earlier results JSON whose runs of other policies to reuse (e.g. reorder_point)")
    parser.add_argument("--out", default="signal_eval.json")
    parser.add_argument("--keep-template", action="store_true", help="reuse the template database from an earlier run")
    parser.add_argument("--compare", nargs=2, metavar=("BEFORE", "AFTER"))
    args = parser.parse_args()

    if args.compare:
        for label, path in zip(("Before", "After"), args.compare):
            print(report(summarize(json.loads(Path(path).read_text())["runs"]), f"\n== {label} ({path}) =="))
        return

    if not args.keep_template:
        t = time.time()
        build_template()
        print(f"Template ready in {time.time() - t:.0f}s", flush=True)
    policies = args.policies.split(",")
    jobs = [(scenario, policy) for scenario in scenarios(args.quick)[:args.limit] for policy in policies]
    runs = []
    if args.reuse:
        runs = [r for r in json.loads(Path(args.reuse).read_text())["runs"] if r["policy"] not in policies]
    print(f"{len(jobs)} runs on {args.workers} workers", flush=True)
    with ProcessPoolExecutor(args.workers) as pool:
        futures = [pool.submit(_run_in_slot, scenario, policy) for scenario, policy in jobs]
        for n, f in enumerate(futures, 1):
            r = f.result()
            runs.append(r)
            print(f"  {n}/{len(jobs)} {r['policy']:<13} lead {r['scenario']['lead']:>2} shock {r['scenario']['shock']} "
                  f"seed {r['scenario']['seed']} ({r['seconds']}s)", flush=True)
    drop_worker_databases({r["db"] for r in runs if "db" in r})
    Path(args.out).write_text(json.dumps({"runs": runs}, default=str))
    print(report(summarize(runs), f"\n== {args.out} =="))


def drop_worker_databases(names: set[str]) -> None:
    with psycopg.connect(f"{SERVER}/secondbrain", autocommit=True) as conn:
        for name in names:
            if name.startswith(f"{PREFIX}_w"):
                conn.execute(f'DROP DATABASE IF EXISTS "{name}"')


def _run_in_slot(scenario: dict, policy: str) -> dict:
    """Each worker process keeps one database of its own, named by its pid."""
    return run_scenario((os.getpid(), scenario, policy))


if __name__ == "__main__":
    main()
