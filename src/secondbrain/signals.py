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
from statistics import NormalDist

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
    -- Structured facts behind a stock signal (sku, quantity, lead time, run-out
    -- date), so its outcome can be judged later.
    details        jsonb NOT NULL DEFAULT '{}',
    -- expired: nobody answered before the deadline passed.
    status         text NOT NULL DEFAULT 'open' CHECK (status IN ('open', 'acted', 'dismissed', 'expired')),
    feedback_by    text REFERENCES users(id),
    feedback_note  text,
    feedback_at    timestamptz,
    created_at     timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX signals_company_idx ON signals USING gist (company);

-- What actually happened after a signal: did the run-out it warned about happen?
CREATE TABLE signal_outcomes (
    signal_id    uuid PRIMARY KEY REFERENCES signals(id) ON DELETE CASCADE,
    result       text NOT NULL CHECK (result IN ('prevented', 'happened', 'happened_anyway', 'false_alarm')),
    detail       text NOT NULL,
    value        numeric(14, 2),  -- dollars saved or lost
    evaluated_on date NOT NULL,
    memory_id    uuid REFERENCES memories(id) ON DELETE SET NULL
);

ALTER TABLE signal_outcomes ENABLE ROW LEVEL SECURITY;
ALTER TABLE signal_outcomes FORCE ROW LEVEL SECURITY;
-- An outcome is visible exactly when its signal is.
CREATE POLICY outcomes_read ON signal_outcomes FOR SELECT
    USING (EXISTS (SELECT 1 FROM signals s WHERE s.id = signal_id));
GRANT SELECT ON signal_outcomes TO secondbrain_app;

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
DROP TABLE IF EXISTS signal_outcomes, signals CASCADE;
DROP FUNCTION IF EXISTS sb_can_see_signal(ltree, jsonb);
"""

# Stock rules
SAFETY_DAYS = 7  # order this many days before the lead time would strictly require
COVER_AFTER_ARRIVAL = 30  # an order should last this long after it arrives
EXCESS_COVER_DAYS = 180
Z80 = 1.2816  # the 80% range is forecast ± Z80 standard deviations
# A warning "runs out on X" came true if it ran out by X plus this many days
# (the slack the outcome judge allows); confidence is the chance of that.
CAME_TRUE_SLACK = 14
# Day-to-day errors partly cancel out, but a forecast that is off is usually
# off for weeks: this share of the backtest error (WAPE) is treated as a
# shift of the whole period when estimating how uncertain total demand is.
SYSTEMATIC_ERROR = 0.5
# The order-by date leaves at most this chance of running out before the
# order arrives (and never less than SAFETY_DAYS before the expected run-out).
ORDER_RISK = 0.1
# Order quantities cover expected demand plus this many standard deviations.
QUANTITY_Z = 0.84  # one-sided 80%: re-warnings catch the rest
# Forecast quality (when the forecasts carry a grade): a weak forecast's
# confidence is scaled down; a product with no backtest has an estimated error.
WEAK_FORECAST_TRUST = 0.8
ESTIMATED_WAPE = 0.6
NO_SALES_DAILY = 0.05  # below this many a day, the forecast expects no sales
# Feedback: a signal someone acted on or dismissed isn't raised again for a while.
QUIET_DAYS = 14
# Real-sales grades need this many graded days before they override the backtest.
MIN_GRADED_DAYS = 14
MAX_BUSINESS_SIGNALS = 5

# Learning from feedback and outcomes
LEARN_LOOKBACK_DAYS = 180
# How many past warnings the forecast's own probability is worth: with fewer
# outcomes than this the forecast dominates, with more the track record does.
PRIOR_WEIGHT = 4
# A run-out prevented by acting suggests the warning was real but can't prove
# it (it might not have run out anyway), so it counts as half a warning that came true.
PREVENTED_WEIGHT = 0.5
# Other products' outcomes say something about this one, but less than its own.
OTHER_PRODUCTS_WEIGHT = 0.5
# Dismissing the same signal this often, for the same reason, lowers its priority.
REPEATED_DISMISSALS = 2
# Dismissal notes grouped by what they mean, so "we have a second warehouse"
# and "there is stock in the other warehouse" count as the same reason.
DISMISS_REASONS = {
    "stock elsewhere": ("warehouse", "elsewhere", "other store", "other location", "transfer", "backup", "spare"),
    "already handled": ("already", "ordered", "on its way", "placed", "handled", "covered"),
    "not a priority": ("priority", "not important", "later", "not now", "doesn't matter", "low impact", "ignore"),
    "data is wrong": ("wrong", "incorrect", "error", "inaccurate", "not accurate", "miscount", "bad data"),
    "phasing out": ("discontinu", "phase out", "phasing out", "end of line", "delist", "stop selling"),
}

URGENCY_ORDER = {"high": 0, "medium": 1, "low": 2}
LOWER = {"high": "medium", "medium": "low", "low": "low"}
HIGHER = {"low": "medium", "medium": "high", "high": "high"}
_NORMAL = NormalDist()


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
    details: dict = field(default_factory=dict)


def _day(d) -> date:
    return d if isinstance(d, date) else date.fromisoformat(d)


def _fmt(d: date) -> str:
    return f"{d:%b} {d.day}"


def _round_up(units: float, step: int = 10) -> int:
    return int(math.ceil(units / step) * step)


def _times(n: int) -> str:
    return "once" if n == 1 else "twice" if n == 2 else f"{n} times"


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


def demand_outlook(points: list[dict], wape: float) -> list[tuple[date, float, float]]:
    """Expected total demand from the first forecast day to the end of each
    day, and its standard deviation.

    Day-to-day errors are independent (their ranges add up as variances), but
    a forecast that is off tends to stay off for weeks, so a share of the
    backtest error (WAPE) is added as a shift of the whole period.
    """
    out, mean, daily = [], 0.0, 0.0
    for p in points:
        mean += p["forecast"]
        daily += ((p["hi80"] - p["forecast"]) / Z80) ** 2
        out.append((_day(p["day"]), mean, math.sqrt(daily + (SYSTEMATIC_ERROR * wape * mean) ** 2)))
    return out


def runout_probability(on_hand: int, on_order: int, eta: date | None, points: list[dict], wape: float,
                       until: date) -> float:
    """The chance that demand through ``until`` is more than the stock
    available by then: what a run-out warning's confidence means (the chance
    it comes true if nobody acts)."""
    outlook = [o for o in demand_outlook(points, wape) if o[0] <= until]
    if not outlook:
        return 0.0
    _, mean, spread = outlook[-1]
    stock = on_hand + (on_order if on_order and eta and eta <= until else 0)
    return _NORMAL.cdf((mean - stock) / (spread or 1.0))


def risk_day(on_hand: int, on_order: int, eta: date | None, points: list[dict], wape: float) -> date | None:
    """The first day the chance of having run out reaches ORDER_RISK."""
    for day, mean, spread in demand_outlook(points, wape):
        stock = on_hand + (on_order if on_order and eta and eta <= day else 0)
        if _NORMAL.cdf((mean - stock) / (spread or 1.0)) >= ORDER_RISK:
            return day
    return None


def track_record(conn: psycopg.Connection, company: str) -> dict:
    """How the brain's run-out warnings have done at this company.

    Only warnings nobody acted on show whether the prediction was right (a
    prevented run-out can't tell us), so accuracy comes from those.
    """
    counts = {r["result"]: r["n"] for r in conn.execute(
        """SELECT o.result, count(*) AS n FROM signal_outcomes o JOIN signals s ON s.id = o.signal_id
           WHERE s.company = %s::ltree GROUP BY o.result""",
        (company,),
    ).fetchall()}
    came_true, false_alarms = counts.get("happened", 0), counts.get("false_alarm", 0)
    saved = conn.execute(
        """SELECT coalesce(sum(o.value), 0)::float AS v FROM signal_outcomes o JOIN signals s ON s.id = o.signal_id
           WHERE s.company = %s::ltree AND o.result = 'prevented'""",
        (company,),
    ).fetchone()["v"]
    return {**counts, "came_true": came_true, "false_alarms": false_alarms, "saved": saved,
            # Smoothed: with no history it is 0.5.
            "accuracy": (came_true + 1) / (came_true + false_alarms + 2)}


# ---------- learning from feedback and outcomes ----------

@dataclass
class Record:
    """What people did with earlier signals for one product (or one kind of
    signal), and what happened after."""
    acted: int = 0
    dismissed: int = 0
    expired: int = 0
    prevented: int = 0
    happened: int = 0
    happened_anyway: int = 0
    false_alarm: int = 0
    # Dismissals since anyone last acted on it: [(note, outcome or None)], oldest first.
    dismissals: list = field(default_factory=list)

    def add(self, status: str, note: str | None, result: str | None) -> None:
        setattr(self, status, getattr(self, status) + 1)
        if result:
            setattr(self, result, getattr(self, result) + 1)
        if status == "acted":
            self.dismissals = []
        elif status == "dismissed":
            self.dismissals.append((note, result))

    @property
    def came_true(self) -> float:
        return self.happened + self.happened_anyway + PREVENTED_WEIGHT * self.prevented


def learned_records(conn: psycopg.Connection, company: str) -> dict[str, Record]:
    """Records per signal fingerprint (one product and kind) and per kind ("kind:stockout")."""
    records: dict[str, Record] = {}
    for r in conn.execute(
        """SELECT s.kind, s.fingerprint, s.status, s.feedback_note, o.result FROM signals s
           LEFT JOIN signal_outcomes o ON o.signal_id = s.id
           WHERE s.company = %s::ltree AND s.status <> 'open'
             AND coalesce(s.feedback_at, s.created_at) > now() - make_interval(days => %s)
           ORDER BY coalesce(s.feedback_at, s.created_at)""",
        (company, LEARN_LOOKBACK_DAYS),
    ).fetchall():
        for key in (r["fingerprint"], f"kind:{r['kind']}"):
            records.setdefault(key, Record()).add(r["status"], r["feedback_note"], r["result"])
    return records


def dismissal_reason(note: str | None) -> str:
    """What a dismissal note means, in a few words; notes with no known
    reason keep their own words, so two identical notes still match."""
    text = (note or "").lower().strip()
    if not text:
        return "no reason given"
    for reason, words in DISMISS_REASONS.items():
        if any(w in text for w in words):
            return reason
    return " ".join(re.findall(r"[a-z]+", text))


def repeated_dismissal(record: Record | None) -> tuple[int, str | None, int]:
    """The largest group of dismissals for the same reason since anyone last
    acted: (how many, a note people wrote, how many of those still ran out)."""
    groups: dict[str, list] = {}
    for note, result in (record.dismissals if record else []):
        groups.setdefault(dismissal_reason(note), []).append((note, result))
    if not groups:
        return 0, None, 0
    same = max(groups.values(), key=len)
    note = next((n for n, _ in reversed(same) if n), None)
    return len(same), note, sum(1 for _, result in same if result == "happened")


def learn(signal: Signal, records: dict[str, Record], probability: float | None = None) -> Signal | None:
    """Adjust a stock signal by what happened to earlier ones like it, and say
    why. Returns None when people have made clear they don't need it this early.

    - Confidence: the forecast's probability (when there is one) blended with
      how earlier warnings for this product, and less so for other products,
      turned out.
    - Ran out although someone acted: raised one level more urgent.
    - Dismissed repeatedly for the same reason, and the dismissals proved right:
      one level less urgent, and a run-out warning is held back until its
      order-by date is within a week. If those warnings came true anyway, the
      priority stays and the signal says so.
    """
    own = records.get(signal.fingerprint) or Record()
    kind = records.get(f"kind:{signal.kind}") or Record()
    facts = signal.evidence.setdefault("facts", [])

    if probability is not None:
        others_true = max(kind.came_true - own.came_true, 0)
        others_false = max(kind.false_alarm - own.false_alarm, 0)
        came_true = own.came_true + OTHER_PRODUCTS_WEIGHT * others_true
        false_alarms = own.false_alarm + OTHER_PRODUCTS_WEIGHT * others_false
        signal.confidence = round(min(0.97, max(0.05, (PRIOR_WEIGHT * probability + came_true)
                                                / (PRIOR_WEIGHT + came_true + false_alarms))), 2)
        why = (f"Why {signal.confidence:.0%} confidence: the forecast gives a {probability:.0%} chance it runs out "
               f"by {_fmt(_day(signal.details['runs_out']) + timedelta(days=CAME_TRUE_SLACK))} if nothing is done")
        ignored = own.happened + own.false_alarm
        if ignored:
            why += f"; {own.happened} of {ignored} ignored warnings for this product came true"
        if own.prevented:
            why += f"; acting on it prevented a run-out {_times(own.prevented)}"
        if others_true + others_false and not ignored:
            why += f"; other products' warnings here came true {others_true:g} of {others_true + others_false:g} times"
        facts.append(why + ".")

    if own.happened_anyway:
        signal.urgency = HIGHER[signal.urgency]
        facts.append(f"Learned: it ran out {_times(own.happened_anyway)} even though someone acted on the warning, "
                     f"so it is raised with higher urgency.")

    count, note, came_true_anyway = repeated_dismissal(own)
    quoted = f' ("{note}")' if note else ""
    if count >= REPEATED_DISMISSALS and came_true_anyway * 2 >= count and came_true_anyway:
        facts.append(f"Dismissed {_times(count)} before{quoted}, but it ran out anyway {_times(came_true_anyway)}, "
                     f"so its priority stays.")
    elif count >= REPEATED_DISMISSALS:
        days_left = (signal.due - _day(signal.details["as_of"])).days if signal.due and signal.details.get("as_of") else 0
        if signal.kind == "stockout" and days_left > SAFETY_DAYS:
            return None
        signal.urgency = LOWER[signal.urgency]
        facts.append(f"Learned: dismissed {_times(count)} for the same reason{quoted}, so it now has lower priority"
                     + (" and is raised only close to the order-by date." if signal.kind == "stockout" else "."))
    elif count == 1 and note:
        facts.append(f'Dismissed before: "{note}".')
    return signal


def stock_signals(conn: psycopg.Connection, company: str, records: dict[str, Record] | None = None) -> list[Signal]:
    """Run-out, backorder and excess-stock signals for every product with stock data."""
    from secondbrain.evolution import realized_accuracy  # evolution imports this module
    record = track_record(conn, company)
    records = learned_records(conn, company) if records is None else records
    rows = conn.execute(
        """SELECT i.brain_path::text AS brain, i.sku, p.name, p.base_price::float AS price, i.as_of,
                  i.on_hand, i.on_order, i.order_eta, i.supplier, i.lead_time_days, i.unit_cost::float AS unit_cost,
                  f.points, f.model, f.backtest, f.quality
           FROM inventory i JOIN products p USING (brain_path, sku)
           -- to_jsonb: a forecast's quality grade may not exist on older databases.
           JOIN LATERAL (SELECT points, model, backtest, to_jsonb(f)->'quality' AS quality FROM forecasts f
                         WHERE f.brain_path = i.brain_path AND f.sku = i.sku
                         ORDER BY created_at DESC LIMIT 1) f ON true
           WHERE i.brain_path <@ %s::ltree ORDER BY i.sku""",
        (company,),
    ).fetchall()
    out = []
    for r in rows:
        points, today = r["points"], r["as_of"]
        quality = r["quality"] or {}
        # Products without a backtest get an estimated error.
        wape = quality.get("wape") or (r["backtest"] or {}).get(r["model"], {}).get("wape", ESTIMATED_WAPE)
        # Once real sales have graded this product's forecasts, a worse real
        # error than the backtest's wins: the brain trusts what it has seen.
        realized = realized_accuracy(conn, r["brain"], r["sku"])
        graded = bool(realized and realized["days"] >= MIN_GRADED_DAYS and realized["wape"] is not None)
        if graded:
            wape = max(wape, realized["wape"])
        # A forecast graded weak (few or irregular sales) is trusted less.
        trust = WEAK_FORECAST_TRUST if quality.get("grade") == "weak" else 1.0
        # Backorders and excess stock rest on the forecast alone; run-out
        # warnings get a probability of their own below.
        confidence = round(max(0.3, min(0.97, (1 - wape) * trust)), 2)
        evidence = {"products": [{"brain": r["brain"], "sku": r["sku"]}], "facts": [
            f"{r['on_hand']:,} on hand" + (f", {r['on_order']:,} on order arriving {_fmt(r['order_eta'])}" if r["on_order"] else ""),
            f"{r['supplier']} lead time: {r['lead_time_days']} days",
            f"Forecast model: {r['model']} (typical error {wape:.0%})",
        ] + ([f"Forecast quality is weak ({quality.get('pattern') or 'irregular'} sales), so confidence is lowered."]
             if trust < 1 else []) + ([f"Checked against real sales: forecasts were off by {realized['wape']:.0%} "
                                       f"over the last {realized['days']} days"] if graded else [])}
        base = {"brain": r["brain"], "sku": r["sku"], "name": r["name"], "as_of": today.isoformat()}
        daily = sum(p["forecast"] for p in points) / len(points)
        projection = project_stock(r["on_hand"], r["on_order"], r["order_eta"], points)
        # A product nobody is expected to buy can't run out.
        runout = next((d for d, s in projection if s <= 0), None) if daily >= NO_SALES_DAILY else None

        if r["on_hand"] == 0 and r["on_order"] and r["order_eta"] and r["order_eta"] > today:
            gap = [p for p in points if _day(p["day"]) < r["order_eta"]]
            lost_units = sum(p["forecast"] for p in gap)
            lost = lost_units * r["price"]
            evidence["facts"].append(f"Why {confidence:.0%} confidence: the forecast's typical error is {wape:.0%}.")
            signal = learn(Signal(
                kind="backorder", fingerprint=f"backorder:{r['sku']}",
                title=f"{r['name']} is out of stock until {_fmt(r['order_eta'])}",
                action=f"Take pre-orders now and promise delivery after {_fmt(r['order_eta'])}, so the "
                       f"~{lost_units:,.0f} customers expected before then aren't lost.",
                reason=f"Nothing on hand; {r['on_order']:,} units from {r['supplier']} arrive {_fmt(r['order_eta'])}. "
                       f"The forecast expects about {lost_units:,.0f} orders before that.",
                impact=f"About ${lost:,.0f} in sales at risk", impact_value=round(lost, 2),
                urgency="high", confidence=confidence, due=today, evidence=evidence, details=base), records)
            out += [signal] if signal else []
            continue

        if runout is not None:
            if record["came_true"] + record["false_alarms"]:
                evidence["facts"].append(f"Track record here: {record['came_true']} of {record['came_true'] + record['false_alarms']} "
                                         f"ignored run-out warnings came true")
            # Early enough that running out before the order arrives is unlikely:
            # the more uncertain the forecast, the earlier.
            risky = risk_day(r["on_hand"], r["on_order"], r["order_eta"], points, wape) or runout
            order_by = min(runout - timedelta(days=SAFETY_DAYS), risky) - timedelta(days=r["lead_time_days"])
            arrival = max(today, order_by) + timedelta(days=r["lead_time_days"])
            # Enough to last a month after arrival: expected demand plus a safety
            # margin for the whole period, from the same outlook as the order-by date.
            needed_until = arrival + timedelta(days=COVER_AFTER_ARRIVAL)
            outlook = [o for o in demand_outlook(points, wape) if o[0] <= needed_until]
            _, expected, spread = outlook[-1]
            incoming = r["on_order"] if r["order_eta"] and r["order_eta"] <= needed_until else 0
            quantity = _round_up(max(expected + QUANTITY_Z * spread - r["on_hand"] - incoming, daily * COVER_AFTER_ARRIVAL))
            lost_units = sum(p["forecast"] for p in points if runout <= _day(p["day"]) < arrival) if order_by < today else 0
            # If nobody acts until the shelf is empty, new stock takes the lead time to arrive:
            # that gap is what's at stake.
            restock = runout + timedelta(days=r["lead_time_days"])
            at_risk = sum(p["forecast"] for p in points if runout <= _day(p["day"]) < restock) * r["price"]
            peak = max(points, key=lambda p: p["forecast"])
            before_peak = runout <= _day(peak["day"])
            days_left = (order_by - today).days
            details = {**base, "price": r["price"], "quantity": quantity, "supplier": r["supplier"],
                       "lead_time_days": r["lead_time_days"]}
            if r["on_hand"] <= 0 and not r["on_order"]:
                evidence["facts"].append("Why high confidence: there is nothing on hand or on order.")
                signal = learn(Signal(
                    kind="stockout", fingerprint=f"stockout:{r['sku']}",
                    title=f"{r['name']} is out of stock now, with nothing on order",
                    action=f"Order {quantity:,} units from {r['supplier']} today; they arrive about "
                           f"{_fmt(today + timedelta(days=r['lead_time_days']))} ({r['lead_time_days']}-day lead time).",
                    reason=f"Nothing on hand or on order. The forecast expects about {daily:.0f} orders a day.",
                    impact=f"About ${at_risk:,.0f} in sales lost before new stock can arrive", impact_value=round(at_risk, 2),
                    urgency="high", confidence=0.97, due=today, evidence=evidence,
                    details={**details, "runs_out": today.isoformat(), "order_by": today.isoformat(), "already_out": True}),
                    records)
                out += [signal] if signal else []
                continue
            if order_by < today:
                action = (f"Order {quantity:,} units from {r['supplier']} today and ask them to expedite: at the normal "
                          f"{r['lead_time_days']}-day lead time they arrive {_fmt(arrival)}, after the run-out.")
            else:
                action = f"Order {quantity:,} units from {r['supplier']} by {_fmt(order_by)} ({r['lead_time_days']}-day lead time)."
            probability = trust * runout_probability(r["on_hand"], r["on_order"], r["order_eta"], points, wape,
                                                     runout + timedelta(days=CAME_TRUE_SLACK))
            signal = learn(Signal(
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
                confidence=probability, due=max(today, order_by), evidence=evidence,
                details={**details, "runs_out": runout.isoformat(), "order_by": order_by.isoformat()}),
                records, probability)
            out += [signal] if signal else []
            continue

        if r["on_hand"] <= 0:
            continue
        # A discontinued product's forecast is all zeros: no number of days of stock means anything.
        no_sales = daily < NO_SALES_DAILY
        cover = float("inf") if no_sales else r["on_hand"] / daily
        own = records.get(f"excess:{r['sku']}")
        # Dismissed repeatedly for the same reason: only flag it when it's twice as bad.
        threshold = EXCESS_COVER_DAYS * (2 if repeated_dismissal(own)[0] >= REPEATED_DISMISSALS else 1)
        if cover > threshold:
            tied_up = r["on_hand"] * r["unit_cost"]
            excess_units = r["on_hand"] - daily * 90
            evidence["facts"].append(f"Why {confidence:.0%} confidence: the forecast's typical error is {wape:.0%}.")
            if no_sales:
                title = f"{r['name']}: no sales expected, {r['on_hand']:,} units on hand"
                action = (f"Stop reorders from {r['supplier']} and clear the {r['on_hand']:,} units "
                          f"(return to the supplier, discount or bundle).")
                reason = f"The forecast expects no sales; {r['on_hand']:,} units on hand, ${tied_up:,.0f} tied up."
            else:
                title = f"{r['name']} has {cover:,.0f} days of stock"
                action = (f"Pause reorders from {r['supplier']} and consider a promotion to sell about "
                          f"{excess_units:,.0f} extra units over the next 3 months.")
                reason = f"{r['on_hand']:,} units on hand against about {daily:.1f} sold a day."
            signal = learn(Signal(
                kind="excess_stock", fingerprint=f"excess:{r['sku']}",
                title=title, action=action, reason=reason,
                impact=f"About ${tied_up:,.0f} of cash tied up in stock", impact_value=round(tied_up, 2),
                urgency="low", confidence=confidence, due=None, evidence=evidence, details=base), records)
            out += [signal] if signal else []
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
- Learn from the feedback section: don't propose again what people dismissed (respect their reasons) unless
  something new changes the case, and favour the kinds of action they acted on. Use outcomes as evidence.

Reply with JSON only:
{{"signals": [{{"title": "...", "action": "...", "reason": "...", "impact": "...",
 "urgency": "high|medium|low", "confidence": 0.0-1.0, "notes": ["N1"], "lessons": ["L2"]}}]}}"""


# Numbers in a business signal must come from its evidence. A number is
# supported if it appears in a cited note or lesson (or a cited note's date),
# or is a sum, difference or product of two numbers in them; a percentage may
# also be one number as a share of, or change from, another. Two kinds of
# number are advice rather than facts and are always fine: a deadline in the
# coming months ("by November 21") and a short duration ("within 48 hours").
NUMBER = re.compile(r"(?<![\w.])[$€£]?(\d[\d,]*(?:\.\d+)?)(?:\s*(k|m|thousand|million)\b)?(\s*(?:%|percent\b))?",
                    re.IGNORECASE)
DURATION = re.compile(r"^(?:st|nd|rd|th)?\s*(?:-\s*)?(?:business |working )?(?:day|week|month|hour|minute|quarter)s?\b",
                      re.IGNORECASE)
MAX_DURATION = 72  # "within 72 hours" is advice; "a 400-day backlog" is a claim
MONTHS = ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"]
_MONTH = r"(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?"
DATES = re.compile(rf"\b{_MONTH}\s+(\d{{1,2}})(?:st|nd|rd|th)?\b(?:,?\s+(\d{{4}}))?|\b(\d{{1,2}})(?:st|nd|rd|th)?\s+(?:of\s+)?{_MONTH}\b"
                   rf"(?:,?\s+(\d{{4}}))?|\b(\d{{4}})-(\d{{2}})-(\d{{2}})\b", re.IGNORECASE)
DEADLINE_DAYS = 120  # dates this far ahead (or a week back) read as deadlines
SCALE = {"k": 1e3, "thousand": 1e3, "m": 1e6, "million": 1e6}
TOLERANCE = 0.02  # rounding: "about 1,200" for 1,187


@dataclass
class Number:
    written: str
    value: float
    percent: bool
    advice: bool  # a deadline or a short duration, not a claim about the business


def _deadline(m: re.Match, today: date) -> bool:
    """A calendar date in the coming months: a deadline the advice sets."""
    g = m.groups()
    try:
        if g[6]:
            d = date(int(g[6]), int(g[7]), int(g[8]))
        else:
            month = MONTHS.index((g[0] or g[4]).lower()[:3]) + 1
            day, year = int(g[1] or g[3]), int(g[2] or g[5] or today.year)
            d = date(year, month, day)
            if not (g[2] or g[5]) and d < today - timedelta(days=7):
                d = date(year + 1, month, day)  # "January 5" said in November
    except ValueError:
        return False
    return today - timedelta(days=7) <= d <= today + timedelta(days=DEADLINE_DAYS)


def _numbers(text: str, today: date | None = None) -> list[Number]:
    text, today = text or "", today or date.today()
    deadlines = [m.span() for m in DATES.finditer(text) if _deadline(m, today)]
    out = []
    for m in NUMBER.finditer(text):
        try:
            value = float(m.group(1).replace(",", "")) * SCALE.get((m.group(2) or "").lower(), 1)
        except ValueError:
            continue
        in_deadline = any(a <= m.start() < b for a, b in deadlines)
        duration = value <= MAX_DURATION and bool(DURATION.match(text[m.end():]))
        out.append(Number(m.group(0).strip(), value, bool(m.group(3)), in_deadline or duration))
    return out


def evidence_numbers(texts: list[str], dates: list[date] = ()) -> dict[str, set[float]]:
    """Numbers a signal may use: those in its evidence, simple arithmetic on
    them, and the parts of the cited notes' dates."""
    values = sorted({n.value for t in texts for n in _numbers(t)})[:60]  # a few thousand pairs at most
    plain, percent = set(values), set(values)
    for a in values:
        for b in values:
            plain |= {a + b, abs(a - b), a * b}
            if b:
                percent |= {a / b * 100, abs(a - b) / b * 100}
    for d in dates:
        plain |= {d.day, d.month, d.year}
    return {"plain": plain, "percent": percent}


def unsupported_numbers(text: str, known: dict[str, set[float]]) -> list[str]:
    """Numbers in ``text`` that its evidence doesn't contain or directly imply."""
    close = lambda v, pool: any(abs(v - k) <= max(0.5, TOLERANCE * abs(k)) for k in pool)
    return [n.written for n in _numbers(text)
            if not n.advice and not close(n.value, known["percent" if n.percent else "plain"])]


_STOP = {"the", "and", "for", "with", "this", "that", "from", "into", "their", "they", "them", "our", "your", "before",
         "after", "about", "next", "more", "less", "than", "have", "has", "will", "should", "now", "all", "its", "both"}


def _words(text: str) -> set[str]:
    """Content words, cut to five letters so "notify" and "notifications" match."""
    return {w[:5] for w in re.findall(r"[a-z]+", text.lower()) if len(w) > 2 and w not in _STOP}


def _same_advice(a: dict, b: dict) -> bool:
    """Two business signals give the same advice. The model words its titles
    differently on every run, so: a fair share of the same title words, or
    mostly the same notes with some of the same title words. (Measured on the
    real model: repeats share 0.3 to 0.9 of their title words, different
    advice under 0.25.)"""
    wa, wb = _words(a["title"]), _words(b["title"])
    titles = len(wa & wb) / max(len(wa | wb), 1)
    na, nb = set(a.get("memories") or ()), set(b.get("memories") or ())
    notes = len(na & nb) / max(len(na | nb), 1)
    return titles >= 0.3 or (notes >= 0.5 and titles >= 0.15)


def business_signals(conn: psycopg.Connection, llm: LLM, company: str,
                     records: dict[str, Record] | None = None) -> list[Signal]:
    notes = conn.execute(
        """SELECT id::text, brain_path::text AS brain, memory_type, content, created_at FROM memories
           WHERE brain_path <@ %s::ltree AND (source IS NULL OR source = 'signal outcome') AND status = 'active'
             AND created_at > now() - interval '60 days'
           ORDER BY created_at DESC LIMIT 40""",
        (company,),
    ).fetchall()
    # Lessons measured from outcomes first: they rest on what actually happened.
    lessons = [r["content"] for r in conn.execute(
        """SELECT content FROM memories WHERE brain_path = 'root' AND status = 'active'
           ORDER BY coalesce(source LIKE 'outcome lesson%%', false) DESC, created_at LIMIT 20"""
    ).fetchall()]
    # Advice must be about this business: lessons from elsewhere alone aren't enough.
    if not notes:
        return []
    records = learned_records(conn, company) if records is None else records
    feedback = conn.execute(
        """SELECT s.title, s.status, s.feedback_note, o.detail AS outcome FROM signals s
           LEFT JOIN signal_outcomes o ON o.signal_id = s.id
           WHERE s.company = %s::ltree AND s.status <> 'open'
           ORDER BY coalesce(s.feedback_at, s.created_at) DESC LIMIT 15""",
        (company,),
    ).fetchall()
    # Earlier business advice, open or answered: new advice that repeats one
    # keeps its fingerprint, so feedback on it (and the quiet period) carries over.
    earlier = [{**r, "memories": r["evidence"].get("memories")} for r in conn.execute(
        """SELECT fingerprint, title, action, evidence FROM signals
           WHERE company = %s::ltree AND kind = 'business'
             AND coalesce(feedback_at, created_at) > now() - make_interval(days => %s)
           ORDER BY created_at DESC""",
        (company, LEARN_LOOKBACK_DAYS),
    ).fetchall()]
    note_ids = {f"N{i}": n for i, n in enumerate(notes, 1)}
    lesson_ids = {f"L{i}": text for i, text in enumerate(lessons, 1)}
    prompt = "\n".join(
        ["Recent notes:"] + [f"{k} ({n['created_at']:%Y-%m-%d}, {n['brain'].split('.')[-1]}): {n['content']}"
                             for k, n in note_ids.items()]
        + ["", "Lessons learned across businesses:"] + [f"{k}: {t}" for k, t in lesson_ids.items()]
        + ["", "Feedback on earlier signals at this business:"]
        + ([f"- {f['status'].upper()}: {f['title']}" + (f" (reason: {f['feedback_note']})" if f["feedback_note"] else "")
            + (f" -> outcome: {f['outcome']}" if f["outcome"] else "") for f in feedback] or ["(none yet)"])
    )
    reply = llm.complete(BUSINESS_SYSTEM, prompt)
    match = re.search(r"\{.*\}", reply, re.DOTALL)
    try:
        items = json.loads(match.group(0)).get("signals", []) if match else []
    except json.JSONDecodeError:
        items = []
    out = []
    for item in items[:MAX_BUSINESS_SIGNALS]:
        if not isinstance(item, dict):
            continue
        cited_notes = [note_ids[k] for k in item.get("notes") or [] if k in note_ids]
        cited_lessons = [lesson_ids[k] for k in item.get("lessons") or [] if k in lesson_ids]
        # No evidence, no signal: it must rest on something the brain actually knows about this business.
        if not cited_notes or not item.get("title") or not item.get("action"):
            continue
        urgency = item.get("urgency") if item.get("urgency") in URGENCY_ORDER else "medium"
        try:
            confidence = max(0.0, min(1.0, float(item.get("confidence", 0.6))))
        except (TypeError, ValueError):
            confidence = 0.6
        # Readers don't know the N/L ids; drop any the model put in the text anyway.
        clean = lambda t: re.sub(r"\s*\(?\b[NL]\d+(?:\s*,\s*[NL]\d+)*\b\)?", "", str(t or "")).strip()
        item = {**item, **{k: clean(item.get(k)) for k in ("title", "action", "reason", "impact")}}
        # Invented numbers: in the title or action (what people act on) the
        # signal is dropped; in the reason or impact it is kept but flagged.
        known = evidence_numbers([n["content"] for n in cited_notes] + cited_lessons,
                                 [n["created_at"].date() for n in cited_notes] + [date.today()])
        if unsupported_numbers(item["title"] + " " + item["action"], known):
            continue
        facts = []
        doubtful = unsupported_numbers(item["reason"] + " " + item["impact"], known)
        if doubtful:
            confidence = min(confidence, 0.4)
            facts.append(f"Check before relying on it: {', '.join(doubtful)} isn't in the cited notes.")
        candidate = {"title": item["title"], "action": item["action"], "memories": [n["id"] for n in cited_notes]}
        if any(_same_advice(candidate, s.evidence | {"title": s.title, "action": s.action}) for s in out):
            continue  # the same advice twice in one run
        repeat = next((e for e in earlier if _same_advice(candidate, e)), None)
        fingerprint = repeat["fingerprint"] if repeat else "business:" + re.sub(r"[^a-z0-9]+", "-", item["title"].lower())[:60]
        signal = learn(Signal(
            kind="business", fingerprint=fingerprint,
            title=item["title"], action=item["action"], reason=item["reason"],
            impact=item["impact"], urgency=urgency, confidence=round(confidence, 2),
            evidence={"memories": candidate["memories"], "lessons": cited_lessons, "facts": facts}), records)
        if signal:
            out.append(signal)
    return out


def generate(conn: psycopg.Connection, llm: LLM, company: str) -> list[Signal]:
    """Replace a company's open signals with fresh ones. Signals someone acted on
    or dismissed recently are not raised again.

    "Recently" is measured on the business's own clock for stock signals (the
    date of its stock data, which runs ahead of the wall clock in a simulation)
    and on the wall clock for advice from notes.
    """
    conn.row_factory = dict_row
    today = conn.execute("SELECT max(as_of) AS d FROM inventory WHERE brain_path <@ %s::ltree",
                         (company,)).fetchone()["d"] or date.today()
    quiet = {r["fingerprint"] for r in conn.execute(
        """SELECT fingerprint FROM signals WHERE company = %s::ltree AND status <> 'open'
           AND feedback_at > now() - make_interval(days => %s)
           AND (details->>'as_of' IS NULL OR (details->>'as_of')::date > %s::date - %s)""",
        (company, QUIET_DAYS, today, QUIET_DAYS),
    ).fetchall()}
    records = learned_records(conn, company)
    fresh = [s for s in stock_signals(conn, company, records) + business_signals(conn, llm, company, records)
             if s.fingerprint not in quiet]
    conn.execute("DELETE FROM signals WHERE company = %s::ltree AND status = 'open'", (company,))
    for s in fresh:
        conn.execute(
            """INSERT INTO signals (company, kind, fingerprint, title, action, reason, impact, impact_value,
                                    urgency, confidence, due, evidence, details)
               VALUES (%s::ltree, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)""",
            (company, s.kind, s.fingerprint, s.title, s.action, s.reason, s.impact, s.impact_value,
             s.urgency, s.confidence, s.due, Jsonb(s.evidence), Jsonb(s.details)),
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
            f"""SELECT s.id::text, s.company::text, s.kind, s.title, s.action, s.reason, s.impact,
                       s.impact_value::float AS impact_value, s.urgency, s.confidence, s.due, s.evidence,
                       s.status, s.feedback_by, s.feedback_note, s.feedback_at, s.created_at,
                       o.result AS outcome, o.detail AS outcome_detail, o.value::float AS outcome_value,
                       o.evaluated_on AS outcome_on
                FROM signals s LEFT JOIN signal_outcomes o ON o.signal_id = s.id
                {"" if include_closed else "WHERE s.status = 'open'"}"""
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


def quality(conn: psycopg.Connection) -> dict:
    """The brain's measured track record over the signals this person may see
    (row-level security decides which): how often ignored run-out warnings
    came true, how often acting prevented the run-out, money saved and lost,
    and what people dismiss signals for."""
    with conn.cursor(row_factory=dict_row) as cur:
        rows = cur.execute(
            """SELECT s.kind, s.status, s.feedback_note, o.result, o.value::float AS value
               FROM signals s LEFT JOIN signal_outcomes o ON o.signal_id = s.id"""
        ).fetchall()
    count = lambda **kw: sum(1 for r in rows if all(r[k] == v for k, v in kw.items()))
    came_true, false_alarms = count(result="happened"), count(result="false_alarm")
    prevented, anyway = count(result="prevented"), count(result="happened_anyway")
    reasons: dict[str, int] = {}
    for r in rows:
        if r["status"] == "dismissed":
            reasons[dismissal_reason(r["feedback_note"])] = reasons.get(dismissal_reason(r["feedback_note"]), 0) + 1
    out = {
        "signals": len(rows),
        "by_status": {k: count(status=k) for k in ("open", "acted", "dismissed", "expired")},
        "by_kind": {k: count(kind=k) for k in ("stockout", "backorder", "excess_stock", "business")},
        "warnings_judged": came_true + false_alarms,
        "warnings_came_true": came_true,
        "acted_judged": prevented + anyway,
        "prevented": prevented,
        "saved": sum(r["value"] or 0 for r in rows if r["result"] == "prevented"),
        "lost": -sum(r["value"] or 0 for r in rows if r["result"] in ("happened", "happened_anyway")),
        "dismissal_reasons": dict(sorted(reasons.items(), key=lambda kv: -kv[1])),
    }
    parts = []
    if out["warnings_judged"]:
        parts.append(f"{came_true} of {came_true + false_alarms} ignored run-out warnings came true")
    if out["acted_judged"]:
        parts.append(f"acting prevented the run-out {prevented} of {prevented + anyway} times (${out['saved']:,.0f} saved)")
    if reasons:
        top, n = next(iter(out["dismissal_reasons"].items()))
        parts.append(f"most dismissals: {top} ({n})")
    out["summary"] = ("Track record: " + "; ".join(parts) + ".") if parts else \
        "Track record: no outcomes judged yet. Act on or dismiss signals; outcomes are judged when their date comes."
    return out


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
