"""Time passing: what happened after the brain's signals, and what it learns from that.

For the demo, ``advance`` moves a company's clock forward by some days:
sales happen (drawn from the forecast with realistic day-to-day randomness),
stock runs down, stock already on order arrives, and so do orders from run-out
signals someone acted on, after the supplier's lead time. Then every run-out
signal whose date has come is judged:

- prevented: someone acted and it didn't run out;
- happened_anyway: someone acted, but it still ran out;
- happened: nobody acted and it ran out;
- false_alarm: nobody acted and it didn't run out.

Each outcome is saved as a memory in the product's brain, so the brain
remembers what happened, maintenance and root learning can use it, and the
next signals can cite it. Signals nobody answered before their deadline
become "expired". Forecasts and signals are then recalculated from the new
"today".

In production the same evaluation runs on real sales as they arrive; only
the simulated sales are demo-only.

    uv run python -m secondbrain.evolution --days 28
"""

import argparse
from dataclasses import dataclass, field
from datetime import date, timedelta

import numpy as np
import psycopg
from psycopg.rows import dict_row

from secondbrain.config import Settings, get_settings
from secondbrain.forecasting import store
from secondbrain.forecasting.__main__ import run as run_forecasts
from secondbrain.memory.store import vector_literal
from secondbrain.models import LLM, Embedder
from secondbrain.schema import vector_type

FORECAST_HORIZON = 120


@dataclass
class ProductDays:
    sku: str
    name: str
    sold: int = 0
    lost: int = 0
    out_of_stock_days: list[date] = field(default_factory=list)
    arrivals: list[str] = field(default_factory=list)


def _fmt(d: date) -> str:
    return f"{d:%b} {d.day}"


def simulate(conn: psycopg.Connection, brain: str, days: int, rng: np.random.Generator) -> tuple[date, date, dict[str, ProductDays]]:
    """Advance one brain's sales and stock by ``days``. Returns (first day, last day, per product)."""
    # Acting on a run-out signal places the recommended order on the day the
    # signal was raised (the brain's date, not the wall clock). It goes into the
    # stock record once, as stock on order, and arrives after the lead time.
    for o in conn.execute(
        """SELECT s.id, s.details FROM signals s
           WHERE s.kind = 'stockout' AND s.status = 'acted' AND s.details->>'brain' = %s
             AND NOT coalesce((s.details->>'ordered')::boolean, false)""",
        (brain,),
    ).fetchall():
        d = o["details"]
        eta = date.fromisoformat(d["as_of"]) + timedelta(days=d["lead_time_days"])
        conn.execute(
            """UPDATE inventory SET on_order = on_order + %s,
                      order_eta = CASE WHEN order_eta IS NULL OR on_order = 0 THEN %s ELSE least(order_eta, %s) END
               WHERE brain_path = %s::ltree AND sku = %s""",
            (d["quantity"], eta, eta, brain, d["sku"]),
        )
        conn.execute("""UPDATE signals SET details = details || jsonb_build_object('ordered', true, 'eta', %s::text)
                        WHERE id = %s""", (eta.isoformat(), o["id"]))
    # Every product moves forward, so all sales histories end on the same day;
    # one without stock data is treated as never running out.
    rows = conn.execute(
        """SELECT p.sku, p.name, p.base_price::float AS price, i.as_of, i.on_hand, i.on_order, i.order_eta,
                  i.lead_time_days, i.sku IS NOT NULL AS tracked
           FROM products p LEFT JOIN inventory i USING (brain_path, sku) WHERE p.brain_path = %s::ltree""",
        (brain,),
    ).fetchall()
    start = min(r["as_of"] for r in rows if r["as_of"])
    rows = [{**r, "as_of": r["as_of"] or start, "on_hand": r["on_hand"] if r["tracked"] else 10**9,
             "on_order": r["on_order"] or 0} for r in rows]
    promos = store.load_promotions(conn, brain)
    first = min(r["as_of"] for r in rows)
    last = first + timedelta(days=days - 1)
    result, sales_rows, updates = {}, [], []
    for r in rows:
        forecast = store.latest_forecast(conn, brain, r["sku"])
        points = {date.fromisoformat(p["day"]): p for p in forecast["points"]}
        tail = forecast["points"][-1]["forecast"]
        stock, on_order, eta = r["on_hand"], r["on_order"], r["order_eta"]
        days_out = ProductDays(r["sku"], r["name"])
        for i in range(days):
            day = r["as_of"] + timedelta(days=i)
            if on_order and eta and day >= eta:
                stock += on_order
                days_out.arrivals.append(f"{on_order:,} units on order arrived {_fmt(day)}")
                on_order, eta = 0, None
            demand = int(rng.poisson(max(points.get(day, {"forecast": tail})["forecast"], 0.1)))
            sold = min(demand, stock)
            stock -= sold
            if sold < demand:
                days_out.out_of_stock_days.append(day)
            days_out.sold += sold
            days_out.lost += demand - sold
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


def judge(conn: psycopg.Connection, embedder: Embedder, settings: Settings, company: str,
          until: date, simulated: dict[str, ProductDays]) -> list[dict]:
    """Judge run-out signals whose date has come, and expire unanswered ones whose deadline passed."""
    conn.execute(
        "UPDATE signals SET status = 'expired' WHERE company = %s::ltree AND status = 'open' AND due < %s",
        (company, until),
    )
    signals = conn.execute(
        """SELECT s.id, s.status, s.details, s.title, s.impact_value::float AS impact_value FROM signals s
           LEFT JOIN signal_outcomes o ON o.signal_id = s.id
           WHERE s.company = %s::ltree AND s.kind = 'stockout' AND o.signal_id IS NULL
             AND s.status IN ('acted', 'dismissed', 'expired')
             AND (s.details->>'runs_out')::date <= %s
             -- Already empty when raised: there was no run-out left to prevent.
             AND NOT coalesce((s.details->>'already_out')::boolean, false)""",
        (company, until),
    ).fetchall()
    cast = vector_type(settings.embedding_dim)[0].split("(")[0]
    outcomes = []
    for s in signals:
        d = s["details"]
        product = simulated.get(d["sku"])
        if product is None:
            continue
        predicted = date.fromisoformat(d["runs_out"])
        # Run-outs around the predicted date count; a week's slack either side.
        ran_out = [day for day in product.out_of_stock_days if day >= predicted - timedelta(days=14)]
        lost_value = product.lost * d["price"]
        acted = s["status"] == "acted"
        if acted and not ran_out:
            result = "prevented"
            detail = (f"{d['name']}: acted on the warning; the order of {d['quantity']:,} units from {d['supplier']} "
                      f"arrived in time and it did not run out through {_fmt(until)}.")
            value = s["impact_value"]
        elif acted:
            result = "happened_anyway"
            detail = (f"{d['name']}: acted on the warning, but it still ran out on {_fmt(ran_out[0])} "
                      f"before the order arrived; about {product.lost:,} sales lost (${lost_value:,.0f}).")
            value = -lost_value
        elif ran_out:
            result = "happened"
            detail = (f"{d['name']}: the warning was not acted on and it ran out on {_fmt(ran_out[0])}, as predicted "
                      f"for {_fmt(predicted)}; about {product.lost:,} sales lost (${lost_value:,.0f}).")
            value = -lost_value
        else:
            result = "false_alarm"
            detail = (f"{d['name']}: the warning was not acted on and it did not run out through {_fmt(until)}; "
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
            (s["id"], result, detail, value, until, memory_id),
        )
        outcomes.append({"title": s["title"], "result": result, "detail": detail})
    return outcomes


def advance(settings: Settings, llm: LLM, embedder: Embedder, company: str, days: int, seed: int = 7) -> dict:
    """Move a company forward by ``days``: sales, stock, outcomes, new forecasts and signals."""
    from secondbrain import signals as signals_module

    rng = np.random.default_rng(seed)
    summary = {"company": company, "days": days, "products": {}, "outcomes": []}
    with psycopg.connect(settings.admin_database_url, row_factory=dict_row) as conn:
        brains = [r["brain"] for r in conn.execute(
            "SELECT DISTINCT brain_path::text AS brain FROM inventory WHERE brain_path <@ %s::ltree", (company,))]
        until = None
        for brain in brains:
            first, last, simulated = simulate(conn, brain, days, rng)
            until = last
            summary["from"], summary["to"] = first.isoformat(), last.isoformat()
            summary["products"].update({k: {"sold": v.sold, "lost": v.lost, "out_of_stock_days": len(v.out_of_stock_days),
                                            "arrivals": v.arrivals} for k, v in simulated.items()})
            summary["outcomes"] += judge(conn, embedder, settings, company, last, simulated)
        conn.commit()
    if until is None:
        return summary
    run_forecasts(settings.admin_database_url, under=company, horizon=FORECAST_HORIZON)
    summary["signals"] = len(signals_module.run(settings, llm, company)[company])
    return summary


def main() -> None:
    from secondbrain.models import build_models

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--company", default="root.lumen")
    parser.add_argument("--days", type=int, default=28)
    args = parser.parse_args()
    settings = get_settings()
    llm, embedder = build_models(settings)
    result = advance(settings, llm, embedder, args.company, args.days)
    print(f"{result['company']}: {result.get('from')} to {result.get('to')}")
    for sku, p in result["products"].items():
        print(f"  {sku:<16} sold {p['sold']:>6,}  lost {p['lost']:>5,}  out of stock {p['out_of_stock_days']} day(s)  {'; '.join(p['arrivals'])}")
    for o in result["outcomes"]:
        print(f"  [{o['result']}] {o['detail']}")
    print(f"  {result.get('signals', 0)} open signal(s) after recalculating")


if __name__ == "__main__":
    main()


# Reads below run on the app connection as a user, under row-level security.

def history(conn: psycopg.Connection) -> dict:
    """How this person's brain has evolved: track record, feedback, outcomes and
    facts it updated, limited to what they may see."""
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
        "events": events,
    }


def hq(conn: psycopg.Connection) -> dict:
    """The root operator's overview of every connected company (admin connection).

    Aggregates only: counts, money and outcomes, never a company's notes, so it
    is consistent with companies sharing only patterns with the root.
    """
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
    lessons = [r["content"] for r in conn.execute(
        "SELECT content FROM memories WHERE brain_path = 'root' AND status = 'active' ORDER BY created_at")]
    return {"companies": companies, "lessons": lessons}
