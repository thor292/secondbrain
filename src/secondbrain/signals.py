"""Decision signals: what each business should do next, why, and by when.

Two sources, both run as a system job per company:

- **Stock signals** are calculated. The forecast, stock on hand, stock on
  order and the supplier's lead time give a run-out date, the last day to
  order, how much to order, and the revenue at risk. Products with far more
  stock than demand get an excess-stock signal.
- **Business signals** come from the model. It reads the company's recent
  notes and the anonymous lessons learned at the root, and proposes actions.
  Every one must cite the notes or lessons it rests on; any that doesn't, or
  cites something that doesn't exist, is dropped.

A signal is only visible to someone who can see all of its evidence, enforced
by row-level security like everything else. People mark signals as acted on
or dismissed; that feedback is how the brain learns what is useful.

    uv run python -m secondbrain.signals
"""

import json
import math
import re
from dataclasses import dataclass, field
from datetime import date, timedelta

import psycopg
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

from secondbrain.config import Settings, get_settings
from secondbrain.models import LLM

SCHEMA_SQL = """
CREATE TABLE signals (
    id             uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    company        ltree NOT NULL REFERENCES brains(path),
    kind           text NOT NULL CHECK (kind IN ('stockout', 'backorder', 'excess_stock', 'business')),
    fingerprint    text NOT NULL,  -- same situation, same fingerprint, across runs
    title          text NOT NULL,
    action         text NOT NULL,
    reason         text NOT NULL,
    impact         text NOT NULL,
    impact_value   numeric(14, 2),  -- dollars at stake, when it can be estimated
    urgency        text NOT NULL CHECK (urgency IN ('high', 'medium', 'low')),
    confidence     real NOT NULL CHECK (confidence BETWEEN 0 AND 1),
    due            date,
    -- {"memories": [ids], "products": [{"brain", "sku"}], "lessons": [text],
    --  "facts": [text]}; lessons are copied in because company users can't read the root.
    evidence       jsonb NOT NULL,
    status         text NOT NULL DEFAULT 'open' CHECK (status IN ('open', 'acted', 'dismissed')),
    feedback_by    text REFERENCES users(id),
    feedback_note  text,
    feedback_at    timestamptz,
    created_at     timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX signals_company_idx ON signals USING gist (company);

-- Visible only within the viewer's company, and only if every memory and
-- product it cites is visible to them; the subqueries run under their own
-- row-level security.
CREATE FUNCTION sb_can_see_signal(company ltree, evidence jsonb) RETURNS boolean
LANGUAGE sql STABLE AS $$
    SELECT EXISTS (SELECT 1 FROM sb_my_grants() g WHERE g.brain_path <@ company OR company <@ g.brain_path)
       AND NOT EXISTS (
            SELECT 1 FROM jsonb_array_elements_text(coalesce(evidence->'memories', '[]')) AS e(id)
            WHERE NOT EXISTS (SELECT 1 FROM memories m WHERE m.id = e.id::uuid))
       AND NOT EXISTS (
            SELECT 1 FROM jsonb_array_elements(coalesce(evidence->'products', '[]')) AS e(p)
            WHERE NOT EXISTS (SELECT 1 FROM products pr
                              WHERE pr.brain_path = (e.p->>'brain')::ltree AND pr.sku = e.p->>'sku'))
$$;

ALTER TABLE signals ENABLE ROW LEVEL SECURITY;
ALTER TABLE signals FORCE ROW LEVEL SECURITY;
CREATE POLICY signals_read ON signals FOR SELECT USING (sb_can_see_signal(company, evidence));
CREATE POLICY signals_feedback ON signals FOR UPDATE
    USING (sb_can_see_signal(company, evidence))
    WITH CHECK (feedback_by = current_setting('app.user_id', true));

GRANT SELECT ON signals TO secondbrain_app;
GRANT UPDATE (status, feedback_by, feedback_note, feedback_at) ON signals TO secondbrain_app;
GRANT EXECUTE ON FUNCTION sb_can_see_signal(ltree, jsonb) TO secondbrain_app;
"""

DROP_SQL = """
DROP TABLE IF EXISTS signals CASCADE;
DROP FUNCTION IF EXISTS sb_can_see_signal(ltree, jsonb);
"""

# Stock rules
SAFETY_DAYS = 7  # order this many days before the lead time would strictly require
COVER_AFTER_ARRIVAL = 30  # an order should last this long after it arrives
EXCESS_COVER_DAYS = 180
Z80 = 1.2816  # the 80% range is forecast ± Z80 standard deviations
# Feedback: a signal someone acted on or dismissed isn't raised again for a while.
QUIET_DAYS = 14
MAX_BUSINESS_SIGNALS = 5

URGENCY_ORDER = {"high": 0, "medium": 1, "low": 2}


@dataclass
class Signal:
    kind: str
    fingerprint: str
    title: str
    action: str
    reason: str
    impact: str
    urgency: str
    confidence: float
    due: date | None = None
    impact_value: float | None = None
    evidence: dict = field(default_factory=dict)


def _day(d) -> date:
    return d if isinstance(d, date) else date.fromisoformat(d)


def _fmt(d: date) -> str:
    return f"{d:%b} {d.day}"


def _round_up(units: float, step: int = 10) -> int:
    return int(math.ceil(units / step) * step)


def project_stock(on_hand: int, on_order: int, eta: date | None, points: list[dict], key: str = "forecast") -> list[tuple[date, float]]:
    """Stock left at the end of each forecast day, with incoming stock added on arrival."""
    stock, out = float(on_hand), []
    for p in points:
        day = _day(p["day"])
        if on_order and eta and day >= eta and (not out or out[-1][0] < eta):
            stock += on_order
        stock -= p[key]
        out.append((day, stock))
    return out


def stock_signals(conn: psycopg.Connection, company: str) -> list[Signal]:
    """Run-out, backorder and excess-stock signals for every product with stock data."""
    rows = conn.execute(
        """SELECT i.brain_path::text AS brain, i.sku, p.name, p.base_price::float AS price, i.as_of,
                  i.on_hand, i.on_order, i.order_eta, i.supplier, i.lead_time_days, i.unit_cost::float AS unit_cost,
                  f.points, f.model, f.backtest
           FROM inventory i JOIN products p USING (brain_path, sku)
           JOIN LATERAL (SELECT points, model, backtest FROM forecasts f
                         WHERE f.brain_path = i.brain_path AND f.sku = i.sku
                         ORDER BY created_at DESC LIMIT 1) f ON true
           WHERE i.brain_path <@ %s::ltree ORDER BY i.sku""",
        (company,),
    ).fetchall()
    out = []
    for r in rows:
        points, today = r["points"], r["as_of"]
        wape = r["backtest"][r["model"]]["wape"]
        confidence = round(max(0.3, min(0.95, 1 - wape)), 2)
        evidence = {"products": [{"brain": r["brain"], "sku": r["sku"]}], "facts": [
            f"{r['on_hand']:,} on hand" + (f", {r['on_order']:,} on order arriving {_fmt(r['order_eta'])}" if r["on_order"] else ""),
            f"{r['supplier']} lead time: {r['lead_time_days']} days",
            f"Forecast model: {r['model']} (typical error {wape:.0%})",
        ]}
        daily = sum(p["forecast"] for p in points) / len(points)
        projection = project_stock(r["on_hand"], r["on_order"], r["order_eta"], points)
        runout = next((d for d, s in projection if s <= 0), None)

        if r["on_hand"] == 0 and r["on_order"] and r["order_eta"] and r["order_eta"] > today:
            gap = [p for p in points if _day(p["day"]) < r["order_eta"]]
            lost_units = sum(p["forecast"] for p in gap)
            lost = lost_units * r["price"]
            out.append(Signal(
                kind="backorder", fingerprint=f"backorder:{r['sku']}",
                title=f"{r['name']} is out of stock until {_fmt(r['order_eta'])}",
                action=f"Take pre-orders now and promise delivery after {_fmt(r['order_eta'])}, so the "
                       f"~{lost_units:,.0f} customers expected before then aren't lost.",
                reason=f"Nothing on hand; {r['on_order']:,} units from {r['supplier']} arrive {_fmt(r['order_eta'])}. "
                       f"The forecast expects about {lost_units:,.0f} orders before that.",
                impact=f"About ${lost:,.0f} in sales at risk", impact_value=round(lost, 2),
                urgency="high", confidence=confidence, due=today, evidence=evidence))
            continue

        if runout is not None:
            order_by = runout - timedelta(days=r["lead_time_days"] + SAFETY_DAYS)
            arrival = max(today, order_by) + timedelta(days=r["lead_time_days"])
            # Enough to last a month after arrival: expected demand plus one safety
            # margin for the whole period. (Adding up each day's upper range would
            # overstate it, since high and low days cancel out over a month.)
            needed_until = arrival + timedelta(days=COVER_AFTER_ARRIVAL)
            period = [p for p in points if _day(p["day"]) <= needed_until]
            expected = sum(p["forecast"] for p in period)
            spread = math.sqrt(sum(((p["hi80"] - p["forecast"]) / Z80) ** 2 for p in period))
            incoming = r["on_order"] if r["order_eta"] and r["order_eta"] <= needed_until else 0
            quantity = _round_up(max(expected + Z80 * spread - r["on_hand"] - incoming, daily * COVER_AFTER_ARRIVAL))
            lost_units = sum(p["forecast"] for p in points if runout <= _day(p["day"]) < arrival) if order_by < today else 0
            # If nobody acts until the shelf is empty, new stock takes the lead time to arrive:
            # that gap is what's at stake.
            restock = runout + timedelta(days=r["lead_time_days"])
            at_risk = sum(p["forecast"] for p in points if runout <= _day(p["day"]) < restock) * r["price"]
            peak = max(points, key=lambda p: p["forecast"])
            before_peak = runout <= _day(peak["day"])
            days_left = (order_by - today).days
            if order_by < today:
                action = (f"Order {quantity:,} units from {r['supplier']} today and ask them to expedite: at the normal "
                          f"{r['lead_time_days']}-day lead time they arrive {_fmt(arrival)}, after the run-out.")
            else:
                action = f"Order {quantity:,} units from {r['supplier']} by {_fmt(order_by)} ({r['lead_time_days']}-day lead time)."
            out.append(Signal(
                kind="stockout", fingerprint=f"stockout:{r['sku']}",
                title=f"{r['name']} runs out on {_fmt(runout)}"
                      + (f", before its peak on {_fmt(_day(peak['day']))}" if before_peak else ""),
                action=action,
                reason=f"{r['on_hand']:,} units on hand cover the forecast until {_fmt(runout)}. "
                       f"Demand peaks at about {peak['forecast']:.0f} a day on {_fmt(_day(peak['day']))}.",
                impact=f"About ${at_risk:,.0f} in lost sales if it runs out before new stock arrives" +
                       (f"; ~{lost_units:,.0f} units lost even if ordered today" if lost_units else ""),
                impact_value=round(at_risk, 2),
                urgency="high" if days_left <= 7 else "medium" if days_left <= 21 else "low",
                confidence=confidence, due=max(today, order_by), evidence=evidence))
            continue

        cover = r["on_hand"] / daily if daily else float("inf")
        if cover > EXCESS_COVER_DAYS:
            tied_up = r["on_hand"] * r["unit_cost"]
            excess_units = r["on_hand"] - daily * 90
            out.append(Signal(
                kind="excess_stock", fingerprint=f"excess:{r['sku']}",
                title=f"{r['name']} has {cover:,.0f} days of stock",
                action=f"Pause reorders from {r['supplier']} and consider a promotion to sell about "
                       f"{excess_units:,.0f} extra units over the next 3 months.",
                reason=f"{r['on_hand']:,} units on hand against about {daily:.1f} sold a day.",
                impact=f"About ${tied_up:,.0f} of cash tied up in stock", impact_value=round(tied_up, 2),
                urgency="low", confidence=confidence, due=None, evidence=evidence))
    return out


BUSINESS_SYSTEM = f"""You are the Second Brain of a business. From its recent notes and from lessons
learned across many businesses, propose at most {MAX_BUSINESS_SIGNALS} decision signals: concrete actions
that protect or grow the business in the coming weeks.

Rules, all mandatory:
- Each signal rests on evidence: cite note ids (N1, N2, ...) and/or lesson ids (L1, ...). Never invent facts.
- Combine notes when they explain each other (e.g. a carrier delay and a rise in support tickets).
- The action is specific: who should do what, by when if a date is known.
- Skip stock and reordering; those are handled separately.
- Prefer fewer, stronger signals over many weak ones.
- Never mention the ids (N1, L2, ...) in the title, action, reason or impact; put them only in "notes" and "lessons".

Reply with JSON only:
{{"signals": [{{"title": "...", "action": "...", "reason": "...", "impact": "...",
 "urgency": "high|medium|low", "confidence": 0.0-1.0, "notes": ["N1"], "lessons": ["L2"]}}]}}"""


def business_signals(conn: psycopg.Connection, llm: LLM, company: str) -> list[Signal]:
    notes = conn.execute(
        """SELECT id::text, brain_path::text AS brain, memory_type, content, created_at FROM memories
           WHERE brain_path <@ %s::ltree AND source IS NULL AND status = 'active'
             AND created_at > now() - interval '60 days'
           ORDER BY created_at DESC LIMIT 40""",
        (company,),
    ).fetchall()
    lessons = [r["content"] for r in conn.execute(
        "SELECT content FROM memories WHERE brain_path = 'root' AND status = 'active' ORDER BY created_at LIMIT 20"
    ).fetchall()]
    if not notes and not lessons:
        return []
    note_ids = {f"N{i}": n for i, n in enumerate(notes, 1)}
    lesson_ids = {f"L{i}": text for i, text in enumerate(lessons, 1)}
    prompt = "\n".join(
        ["Recent notes:"] + [f"{k} ({n['created_at']:%Y-%m-%d}, {n['brain'].split('.')[-1]}): {n['content']}"
                             for k, n in note_ids.items()]
        + ["", "Lessons learned across businesses:"] + [f"{k}: {t}" for k, t in lesson_ids.items()]
    )
    reply = llm.complete(BUSINESS_SYSTEM, prompt)
    match = re.search(r"\{.*\}", reply, re.DOTALL)
    try:
        items = json.loads(match.group(0)).get("signals", []) if match else []
    except json.JSONDecodeError:
        items = []
    out = []
    for item in items[:MAX_BUSINESS_SIGNALS]:
        cited_notes = [note_ids[k] for k in item.get("notes", []) if k in note_ids]
        cited_lessons = [lesson_ids[k] for k in item.get("lessons", []) if k in lesson_ids]
        # No evidence, no signal: it must rest on something the brain actually knows.
        if not (cited_notes or cited_lessons) or not item.get("title") or not item.get("action"):
            continue
        urgency = item.get("urgency") if item.get("urgency") in URGENCY_ORDER else "medium"
        try:
            confidence = max(0.0, min(1.0, float(item.get("confidence", 0.6))))
        except (TypeError, ValueError):
            confidence = 0.6
        # Readers don't know the N/L ids; drop any the model put in the text anyway.
        clean = lambda t: re.sub(r"\s*\(?\b[NL]\d+(?:\s*,\s*[NL]\d+)*\b\)?", "", str(t or "")).strip()
        item = {**item, **{k: clean(item.get(k)) for k in ("title", "action", "reason", "impact")}}
        out.append(Signal(
            kind="business", fingerprint="business:" + re.sub(r"[^a-z0-9]+", "-", item["title"].lower())[:60],
            title=item["title"].strip(), action=item["action"].strip(), reason=str(item.get("reason", "")).strip(),
            impact=str(item.get("impact", "")).strip(), urgency=urgency, confidence=round(confidence, 2),
            evidence={"memories": [n["id"] for n in cited_notes], "lessons": cited_lessons}))
    return out


def generate(conn: psycopg.Connection, llm: LLM, company: str) -> list[Signal]:
    """Replace a company's open signals with fresh ones. Signals someone acted on
    or dismissed recently are not raised again."""
    conn.row_factory = dict_row
    quiet = {r["fingerprint"] for r in conn.execute(
        """SELECT fingerprint FROM signals WHERE company = %s::ltree AND status <> 'open'
           AND feedback_at > now() - make_interval(days => %s)""",
        (company, QUIET_DAYS),
    ).fetchall()}
    fresh = [s for s in stock_signals(conn, company) + business_signals(conn, llm, company)
             if s.fingerprint not in quiet]
    conn.execute("DELETE FROM signals WHERE company = %s::ltree AND status = 'open'", (company,))
    for s in fresh:
        conn.execute(
            """INSERT INTO signals (company, kind, fingerprint, title, action, reason, impact, impact_value,
                                    urgency, confidence, due, evidence)
               VALUES (%s::ltree, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)""",
            (company, s.kind, s.fingerprint, s.title, s.action, s.reason, s.impact, s.impact_value,
             s.urgency, s.confidence, s.due, Jsonb(s.evidence)),
        )
    conn.commit()
    return fresh


def run(settings: Settings, llm: LLM, company: str | None = None) -> dict[str, list[Signal]]:
    with psycopg.connect(settings.admin_database_url, row_factory=dict_row) as conn:
        companies = [company] if company else [r["path"] for r in conn.execute(
            "SELECT path::text FROM brains WHERE kind = 'company' ORDER BY path").fetchall()]
        return {c: generate(conn, llm, c) for c in companies}


# Reads and feedback below run on the app connection as the user, under row-level security.

def visible(conn: psycopg.Connection, include_closed: bool = False) -> list[dict]:
    with conn.cursor(row_factory=dict_row) as cur:
        rows = cur.execute(
            f"""SELECT id::text, company::text, kind, title, action, reason, impact, impact_value::float AS impact_value,
                       urgency, confidence, due, evidence, status, feedback_by, feedback_note, feedback_at, created_at
                FROM signals {"" if include_closed else "WHERE status = 'open'"}"""
        ).fetchall()
    rows.sort(key=lambda r: (r["status"] != "open", URGENCY_ORDER[r["urgency"]], r["due"] or date.max, -(r["impact_value"] or 0)))
    return rows


def evidence_details(conn: psycopg.Connection, signal: dict) -> dict:
    """The cited notes as this user may read them (they can, or they wouldn't see the signal)."""
    ids = signal["evidence"].get("memories", [])
    with conn.cursor(row_factory=dict_row) as cur:
        notes = cur.execute(
            """SELECT id::text, brain_path::text AS brain, memory_type, content, created_by, created_at
               FROM memories WHERE id = ANY(%s::uuid[]) ORDER BY created_at""",
            (ids,),
        ).fetchall() if ids else []
    return {**signal["evidence"], "notes": notes}


def feedback(conn: psycopg.Connection, signal_id: str, user_id: str, status: str, note: str | None) -> bool:
    row = conn.execute(
        """UPDATE signals SET status = %s, feedback_by = %s, feedback_note = %s, feedback_at = now()
           WHERE id = %s::uuid RETURNING id""",
        (status, user_id, note, signal_id),
    ).fetchone()
    return row is not None


def main() -> None:
    from secondbrain.models import build_models

    settings = get_settings()
    llm, _ = build_models(settings)
    for company, signals in run(settings, llm).items():
        print(f"\n{company}: {len(signals)} open signal(s)")
        for s in sorted(signals, key=lambda s: URGENCY_ORDER[s.urgency]):
            print(f"  [{s.urgency:>6}] {s.title}\n           -> {s.action}")


if __name__ == "__main__":
    main()
