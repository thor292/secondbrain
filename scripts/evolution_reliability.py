"""Reliability check for the evolving brain: does judging, grading and
learning stay consistent over many simulated weeks?

On this checkout's *test* database (never the demo one), for each seed and
step length: load Acme's fixture sales as the tests do, set random stock
levels and lead times, then repeatedly answer the open run-out warnings the
way a busy team would (act on some, dismiss some, ignore the rest) and
``advance`` the company. At the end it reports:

- run-out warning precision: judged warnings that came true or were
  prevented, against false alarms (and the strict version: ignored warnings
  only, since a prevented run-out can't prove the warning right);
- forecast interval coverage against the nominal 80% and 95%, WAPE and bias;
- outcome counts, and warnings still waiting to be judged;
- inconsistencies: stock that doesn't add up (orders counted twice or lost),
  gaps in the sales history, outcomes judged before the data covered their
  window, the same run-out judged twice, outcomes without a memory, and
  whether a second ``refresh`` on the same data changes anything.

    SB_TEST_SUFFIX=_evolve uv run python -m scripts.evolution_reliability
    SB_TEST_SUFFIX=_evolve uv run python -m scripts.evolution_reliability --seeds 1 2 3 --steps 14 28 --days 112

Signals someone answered stay quiet for 14 wall-clock days; between steps the
script moves feedback times back by the simulated days, as time would pass in
production.
"""

import argparse
import json
from collections import Counter
from datetime import date, timedelta

import numpy as np
import psycopg
from psycopg.rows import dict_row

from scripts.generate_lumen_sales import PRODUCTS, generate
from secondbrain import admin, evolution
from secondbrain.forecasting import store
from secondbrain.forecasting.__main__ import run as run_forecasts
from secondbrain.models import build_models
from secondbrain.schema import create_schema
from secondbrain.sensitivity import Sensitivity
from tests.conftest import EXAMPLES, TEST_DB, make_test_settings

COMPANY, BRAIN = "root.acme", "root.acme.sales"
SKUS = ["halo-lamp", "cloud-throw", "aria-sofa"]
SUPPLIERS = {"halo-lamp": "Lampworks", "cloud-throw": "Weavers", "aria-sofa": "Sofa Co"}


def setup(settings, rng: np.random.Generator) -> dict:
    """Fresh schema, Acme onboarded, fixture sales, forecasts and random stock. Returns the starting stock."""
    create_schema(settings, reset=True)
    rows, promos = generate()
    with psycopg.connect(settings.admin_database_url) as conn:
        admin.onboard(conn, json.loads((EXAMPLES / "acme" / "company.json").read_text()))
        store.save_history(conn, BRAIN, Sensitivity.INTERNAL,
                           [(p.sku, p.name, p.category, p.price) for p in PRODUCTS if p.sku in SKUS],
                           [r for r in rows if r["sku"] in SKUS], [vars(p) for p in promos])
    run_forecasts(settings.admin_database_url, under=COMPANY, horizon=evolution.FORECAST_HORIZON)
    stock = []
    with psycopg.connect(settings.admin_database_url, row_factory=dict_row) as conn:
        for sku in SKUS:
            points = store.latest_forecast(conn, BRAIN, sku)["points"][:56]
            daily = sum(p["forecast"] for p in points) / len(points)
            stock.append({"sku": sku, "as_of": date(2026, 11, 4), "on_hand": int(daily * rng.uniform(8, 70)),
                          "on_order": 0, "order_eta": None, "supplier": SUPPLIERS[sku],
                          "lead_time_days": int(rng.choice([14, 21, 28, 42])), "unit_cost": 50})
        store.save_inventory(conn, BRAIN, stock)
    return {s["sku"]: s["on_hand"] for s in stock}


def answer_warnings(settings, rng: np.random.Generator) -> Counter:
    """Act on half the open run-out warnings, dismiss a quarter, ignore the rest."""
    answered = Counter()
    with psycopg.connect(settings.admin_database_url, row_factory=dict_row) as conn:
        for s in conn.execute("SELECT id FROM signals WHERE company = %s::ltree AND kind = 'stockout' AND status = 'open'",
                              (COMPANY,)).fetchall():
            choice = str(rng.choice(["acted", "dismissed", "ignored"], p=[0.5, 0.25, 0.25]))
            answered[choice] += 1
            if choice != "ignored":
                conn.execute("""UPDATE signals SET status = %s, feedback_by = 'vikram', feedback_at = now()
                                WHERE id = %s""", (choice, s["id"]))
    return answered


def snapshot(conn) -> dict:
    """Everything a refresh could change, to check that a second one changes nothing."""
    q = lambda sql: conn.execute(sql).fetchall()  # noqa: E731
    return {
        "forecasts": q("SELECT id FROM forecasts ORDER BY id"),
        "signals": q("SELECT id, status, due FROM signals ORDER BY id"),
        "outcomes": q("SELECT signal_id, result, value FROM signal_outcomes ORDER BY signal_id"),
        "grades": q("SELECT forecast_id, scored_through, days, complete, scored_at FROM forecast_accuracy ORDER BY 1"),
        "memories": q("SELECT count(*) FROM memories"),
        "state": q("SELECT company, data_through, digest, refreshed_at FROM evolution_state"),
    }


def check(conn, start_stock: dict, start: date) -> dict:
    """Invariants after a run; each list holds the problems found."""
    problems = {k: [] for k in ("stock_does_not_add_up", "sales_gaps", "judged_too_early",
                                "same_runout_judged_twice", "outcome_without_memory")}
    for sku in SKUS:
        sold = conn.execute("SELECT coalesce(sum(units), 0) AS n FROM sales_daily WHERE brain_path = %s::ltree "
                            "AND sku = %s AND day >= %s", (BRAIN, sku, start)).fetchone()["n"]
        ordered = conn.execute(
            """SELECT coalesce(sum((details->>'quantity')::int), 0) AS n FROM signals
               WHERE details->>'sku' = %s AND coalesce((details->>'ordered')::boolean, false)""", (sku,)).fetchone()["n"]
        end = conn.execute("SELECT on_hand, on_order FROM inventory WHERE brain_path = %s::ltree AND sku = %s",
                           (BRAIN, sku)).fetchone()
        if start_stock[sku] + ordered - sold != end["on_hand"] + end["on_order"]:
            problems["stock_does_not_add_up"].append(
                f"{sku}: {start_stock[sku]} + ordered {ordered} - sold {sold} != {end['on_hand']} on hand + {end['on_order']} on order")
        span = conn.execute("SELECT count(*) AS n, max(day) - min(day) + 1 AS span FROM sales_daily "
                            "WHERE brain_path = %s::ltree AND sku = %s AND day >= %s", (BRAIN, sku, start)).fetchone()
        if span["n"] != span["span"]:
            problems["sales_gaps"].append(f"{sku}: {span['n']} rows over {span['span']} days")
    judged = conn.execute(
        """SELECT s.id, s.status, s.details, o.result, o.evaluated_on, o.memory_id FROM signal_outcomes o
           JOIN signals s ON s.id = o.signal_id ORDER BY s.created_at""").fetchall()
    for o in judged:
        runs_out = date.fromisoformat(o["details"]["runs_out"])
        if o["evaluated_on"] < runs_out + timedelta(days=evolution.LATE_DAYS):
            problems["judged_too_early"].append(f"{o['details']['sku']} {runs_out} judged {o['evaluated_on']}")
        if o["memory_id"] is None:
            problems["outcome_without_memory"].append(str(o["id"]))
    # Two unanswered warnings for the same product whose windows overlap are one run-out.
    ignored = [o for o in judged if o["status"] == "expired"]
    for a, b in zip(ignored, ignored[1:]):
        if a["details"]["sku"] == b["details"]["sku"]:
            end_a = date.fromisoformat(a["details"]["runs_out"]) + timedelta(days=evolution.LATE_DAYS)
            if date.fromisoformat(b["details"]["as_of"]) <= end_a:
                problems["same_runout_judged_twice"].append(f"{a['details']['sku']}: {a['details']['runs_out']} and {b['details']['runs_out']}")
    return problems


def scenario(settings, llm, embedder, seed: int, step: int, days: int) -> dict:
    rng = np.random.default_rng(seed)
    start_stock = setup(settings, rng)
    evolution.refresh(settings, llm, embedder, COMPANY)
    answered, waiting = Counter(), set()
    for i in range(days // step):
        answered += answer_warnings(settings, rng)
        result = evolution.advance(settings, llm, embedder, COMPANY, days=step, seed=seed * 1000 + i)
        waiting.update(result["waiting"])
        with psycopg.connect(settings.admin_database_url) as conn:
            conn.execute("UPDATE signals SET feedback_at = feedback_at - make_interval(days => %s) "
                         "WHERE feedback_at IS NOT NULL", (step,))
    with psycopg.connect(settings.admin_database_url, row_factory=dict_row) as conn:
        before = snapshot(conn)
    again = evolution.refresh(settings, llm, embedder, COMPANY)
    with psycopg.connect(settings.admin_database_url, row_factory=dict_row) as conn:
        after = snapshot(conn)
        problems = check(conn, start_stock, date(2026, 11, 4))
        outcomes = Counter({r["result"]: r["n"] for r in conn.execute(
            "SELECT result, count(*) AS n FROM signal_outcomes GROUP BY 1")})
        grades = conn.execute(f"SELECT {evolution._SUMS} FROM forecast_accuracy WHERE complete").fetchone()
        unjudged = conn.execute(
            """SELECT count(*) AS n FROM signals s LEFT JOIN signal_outcomes o ON o.signal_id = s.id
               WHERE s.kind = 'stockout' AND s.status <> 'open' AND o.signal_id IS NULL
                 AND NOT coalesce((s.details->>'already_out')::boolean, false)""").fetchone()["n"]
    problems["second_refresh_changed_something"] = [k for k in before if before[k] != after[k]] + (
        ["reported changes"] if again["scored"] or again["outcomes"] or again["expired"] or again["forecasted"]
        or again["signals_recalculated"] else [])
    return {"seed": seed, "step": step, "days": days, "answered": answered, "outcomes": outcomes,
            "grades": evolution._summary(grades), "unjudged": unjudged, "waiting": sorted(waiting), "problems": problems}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--steps", type=int, nargs="+", default=[14, 28], help="days per advance")
    parser.add_argument("--days", type=int, default=112, help="simulated days per scenario")
    args = parser.parse_args()
    settings = make_test_settings()
    assert settings.admin_database_url.endswith("/" + TEST_DB) and TEST_DB.startswith("secondbrain_test")
    llm, embedder = build_models(settings)
    print(f"Database {TEST_DB}; {args.days} simulated days per scenario\n")
    totals, grade_totals, problems = Counter(), Counter(), Counter()
    for seed in args.seeds:
        for step in args.steps:
            r = scenario(settings, llm, embedder, seed, step, args.days)
            o, g = r["outcomes"], r["grades"] or {}
            totals += o
            for k in ("days", "censored_days", "actual", "forecast", "abs_error", "in80", "in95"):
                grade_totals[k] += g.get(k, 0)
            found = {k: v for k, v in r["problems"].items() if v}
            problems.update({k: len(v) for k, v in found.items()})
            print(f"seed {seed}, {step}-day steps: answered {dict(r['answered'])}; outcomes {dict(o)}; "
                  f"{r['unjudged']} closed warning(s) unjudged")
            if g:
                print(f"    forecasts: WAPE {g['wape']:.1%}, bias {g['bias']:+.1%}, 80% range {g['coverage80']:.1%}, "
                      f"95% range {g['coverage95']:.1%} over {g['days']} days ({g['censored_days']} out-of-stock left out)")
            for w in r["waiting"]:
                print(f"    waiting: {w}")
            for k, v in found.items():
                print(f"    PROBLEM {k}: {v}")
    judged = sum(totals.values())
    right = totals["happened"] + totals["happened_anyway"] + totals["prevented"]
    strict = totals["happened"] + totals["false_alarm"]
    overall = evolution._ratios(dict(grade_totals)) if grade_totals["days"] else None
    print("\nAll scenarios")
    print(f"  outcomes: {dict(totals)}")
    if judged:
        print(f"  run-out warning precision: {right}/{judged} = {right / judged:.0%} came true or were prevented")
    if strict:
        print(f"  ignored warnings that came true: {totals['happened']}/{strict} = {totals['happened'] / strict:.0%}")
    if overall:
        print(f"  forecasts: WAPE {overall['wape']:.1%}, bias {overall['bias']:+.1%}; "
              f"80% range held {overall['coverage80']:.1%} of days (nominal 80%), "
              f"95% range {overall['coverage95']:.1%} (nominal 95%) over {grade_totals['days']} product-days")
    print(f"  problems: {dict(problems) or 'none'}")


if __name__ == "__main__":
    main()
