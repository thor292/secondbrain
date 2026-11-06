"""Learning from feedback and outcomes, business-signal checks, and the measured track record."""

import json
from datetime import date, timedelta

import psycopg
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

from secondbrain import learning, signals
from secondbrain.forecasting import store
from secondbrain.forecasting.__main__ import run as run_forecasts
from secondbrain.signals import Record, Signal
from tests.conftest import ADMIN_TOKEN
from tests.test_forecasting import history, load_acme_sales  # noqa: F401  (fixture)

TODAY = date(2026, 11, 6)


def stockout(urgency="medium", due_in=14) -> Signal:
    return Signal(kind="stockout", fingerprint="stockout:brass-floor-lamp", title="Halo runs out", action="Order",
                  reason="", impact="", urgency=urgency, confidence=0.5, due=TODAY + timedelta(days=due_in),
                  evidence={"facts": []},
                  details={"sku": "brass-floor-lamp", "as_of": TODAY.isoformat(),
                           "runs_out": (TODAY + timedelta(days=due_in + 21)).isoformat()})


def record(*events) -> dict[str, Record]:
    """Records from (status, note, outcome) events for the halo lamp's run-out warnings."""
    records = {"stockout:brass-floor-lamp": Record(), "kind:stockout": Record()}
    for status, note, result in events:
        for r in records.values():
            r.add(status, note, result)
    return records


def test_runout_probability_follows_stock_and_uncertainty():
    points = [{"day": (TODAY + timedelta(days=i)).isoformat(), "forecast": 10.0, "hi80": 14.0} for i in range(30)]
    until = TODAY + timedelta(days=29)
    low, even, high = (signals.runout_probability(stock, 0, None, points, 0.2, until) for stock in (200, 300, 400))
    assert low > 0.9 and abs(even - 0.5) < 0.01 and high < 0.1
    # Stock arriving in time counts; a less accurate forecast is less sure either way.
    assert signals.runout_probability(200, 200, TODAY + timedelta(days=5), points, 0.2, until) < 0.1
    assert signals.runout_probability(200, 0, None, points, 0.6, until) < low


def test_track_record_drives_confidence_and_explains_it():
    fresh = signals.learn(stockout(), record(), probability=0.6)
    assert fresh.confidence == 0.6
    assert fresh.evidence["facts"][0].startswith("Why 60% confidence: the forecast gives a 60% chance it runs out")

    proven = signals.learn(stockout(), record(*[("dismissed", None, "happened")] * 3 + [("acted", None, "prevented")]),
                           probability=0.6)
    assert proven.confidence > 0.75
    why = proven.evidence["facts"][0]
    assert "3 of 3 ignored warnings for this product came true" in why and "prevented a run-out once" in why

    wrong = signals.learn(stockout(), record(*[("expired", None, "false_alarm")] * 3), probability=0.6)
    assert wrong.confidence < 0.4 and "0 of 3 ignored warnings" in wrong.evidence["facts"][0]


def test_ran_out_anyway_raises_urgency():
    signal = signals.learn(stockout("medium"), record(("acted", "ordered", "happened_anyway")), probability=0.6)
    assert signal.urgency == "high"
    assert any("even though someone acted" in f for f in signal.evidence["facts"])


def test_repeated_dismissals_for_the_same_reason_lower_priority():
    assert signals.dismissal_reason("We have a second warehouse") == signals.dismissal_reason(
        "there's stock in the other warehouse") == "stock elsewhere"
    assert signals.dismissal_reason("Not a priority right now") == "not a priority"
    assert signals.dismissal_reason(None) == "no reason given"

    twice = record(("dismissed", "We have a second warehouse", "false_alarm"),
                   ("dismissed", "stock in the other warehouse", None))
    # Far from its deadline it is held back; close to it, raised one level lower, with the reason.
    assert signals.learn(stockout("low", due_in=20), twice) is None
    near = signals.learn(stockout("high", due_in=3), twice)
    assert near.urgency == "medium"
    assert any('dismissed twice for the same reason ("stock in the other warehouse")' in f for f in near.evidence["facts"])

    # Different reasons aren't a pattern; one dismissal is only shown.
    mixed = record(("dismissed", "We have a second warehouse", None), ("dismissed", "not a priority", None))
    assert signals.learn(stockout("high", due_in=3), mixed).urgency == "high"
    once = signals.learn(stockout("high", due_in=3), record(("dismissed", "not a priority", None)))
    assert once.urgency == "high" and 'Dismissed before: "not a priority".' in once.evidence["facts"]

    # Dismissals that proved wrong don't lower anything, and the signal says so.
    wrong = record(("dismissed", "second warehouse", "happened"), ("dismissed", "other warehouse", "happened"))
    kept = signals.learn(stockout("medium", due_in=10), wrong)
    assert kept.urgency == "medium" and any("ran out anyway twice" in f for f in kept.evidence["facts"])

    # Acting on it afterwards resets the pattern.
    assert signals.learn(stockout("low", due_in=20), record(*twice_events(), ("acted", None, "prevented"))) is not None


def twice_events():
    return [("dismissed", "We have a second warehouse", None), ("dismissed", "other warehouse", None)]


def test_number_validator():
    known = signals.evidence_numbers(["Bluebird wants 400 throws at 25 dollars each by December 12"], [date(2026, 11, 2)])
    assert signals.unsupported_numbers("Offer 400 throws at $25 by Dec 12", known) == []
    assert signals.unsupported_numbers("Quote $10,000 for 400 throws within 2 days", known) == []  # 400 x 25, a duration
    assert signals.unsupported_numbers("Expect 35% more tickets and 1,250 orders", known) == ["35%", "1,250"]
    assert signals.unsupported_numbers("Since the note on 2026-11-02", known) == []


class ScriptedLLM:
    """Replies to the decision-signals prompt with a fixed list, and records prompts."""
    name = "scripted"

    def __init__(self, items):
        self.items, self.prompts = items, []

    def complete(self, system, prompt):
        self.prompts.append(prompt)
        return json.dumps({"signals": self.items})


def business(title, action, reason="", impact="", notes=("N1",), lessons=()):
    return {"title": title, "action": action, "reason": reason, "impact": impact, "urgency": "high",
            "confidence": 0.8, "notes": list(notes), "lessons": list(lessons)}


def test_business_signals_are_checked(client, settings):
    client.post("/remember", json={"content": "Bluebird Stores wants 400 throws before December 12"},
                headers={"X-User-Id": "asha"})
    client.post("/consolidate", headers={"X-User-Id": "asha"})
    llm = ScriptedLLM([
        business("Confirm Bluebird's 400 throws", "Call Bluebird to confirm the order by Dec 12",
                 reason="A bulk order", impact="400 throws"),
        business("Confirm the Bluebird throws order", "Call Bluebird to confirm the order",
                 reason="duplicate wording"),  # the same advice again
        business("Win 7,777 new customers", "Run a campaign"),  # invented number in the title
        business("Price the bulk order", "Offer a bulk price to Bluebird", impact="About $18,000 in revenue"),
        business("Generic advice", "Do it", notes=(), lessons=("L1",)),  # lessons only: not about this business
    ])
    result = signals.run(settings, llm, "root.acme")["root.acme"]
    titles = [s.title for s in result if s.kind == "business"]
    assert titles == ["Confirm Bluebird's 400 throws", "Price the bulk order"]
    flagged = next(s for s in result if s.title == "Price the bulk order")
    assert flagged.confidence == 0.4 and "18,000" in flagged.evidence["facts"][0]

    # Dismissed advice isn't raised again in other words, and keeps what people said.
    first = next(s for s in client.get("/signals", headers={"X-User-Id": "asha"}).json()
                 if s["title"] == "Confirm Bluebird's 400 throws")
    client.post(f"/signals/{first['id']}/feedback", json={"status": "dismissed", "note": "already handled"},
                headers={"X-User-Id": "asha"})
    llm.items = [business("Confirm the 400 Bluebird throws", "Call Bluebird to confirm the order by Dec 12")]
    assert [s for s in signals.run(settings, llm, "root.acme")["root.acme"] if s.kind == "business"] == []
    assert "DISMISSED: Confirm Bluebird's 400 throws (reason: already handled)" in llm.prompts[-1]


def test_no_notes_no_business_signals(client, settings):
    llm = ScriptedLLM([business("Anything", "Do it")])
    with psycopg.connect(settings.admin_database_url, row_factory=dict_row) as conn:
        assert signals.business_signals(conn, llm, "root.globex") == []
    assert llm.prompts == []  # nothing to read, so the model isn't asked


def add_history(conn, company, sku, events, lead=14):
    """Closed run-out signals with outcomes, as if weeks had passed: (status, note, result, minutes to act)."""
    for status, note, result, minutes in events:
        signal_id = conn.execute(
            """INSERT INTO signals (company, kind, fingerprint, title, action, reason, impact, urgency, confidence,
                                    due, evidence, details, status, feedback_note, feedback_at, created_at)
               VALUES (%s::ltree, 'stockout', %s, 'Runs out', 'Order', '', '', 'medium', 0.6, %s, %s, %s, %s, %s,
                       now() - interval '30 days' + make_interval(mins => %s), now() - interval '30 days')
               RETURNING id""",
            (company, f"stockout:{sku}", TODAY, Jsonb({"products": [{"brain": f"{company}.sales", "sku": sku}]}),
             Jsonb({"sku": sku, "lead_time_days": lead, "as_of": "2026-10-01", "runs_out": "2026-10-20"}),
             status, note, minutes)).fetchone()[0]
        conn.execute("INSERT INTO signal_outcomes (signal_id, result, detail, value, evaluated_on) "
                     "VALUES (%s, %s, 'judged', %s, %s)",
                     (signal_id, result, 1000 if result == "prevented" else -500, TODAY))
    conn.commit()


def test_learning_changes_live_signals_and_track_record(client, settings, history):  # noqa: F811
    load_acme_sales(settings, history)
    run_forecasts(settings.admin_database_url, under="root.acme", horizon=56)
    with psycopg.connect(settings.admin_database_url) as conn:
        daily = sum(p["forecast"] for p in store.latest_forecast(conn, "root.acme.sales", "brass-floor-lamp")["points"]) / 56
        store.save_inventory(conn, "root.acme.sales", [
            # Order-by date three weeks out: low urgency, an early warning.
            {"sku": "brass-floor-lamp", "as_of": TODAY, "on_hand": int(daily * 50), "on_order": 0, "order_eta": None,
             "supplier": "Lampworks", "lead_time_days": 21, "unit_cost": 150},
        ])
    llm = ScriptedLLM([])
    by_kind = lambda: {s.kind: s for s in signals.run(settings, llm, "root.acme")["root.acme"]}
    # Without any history: an early warning.
    with psycopg.connect(settings.admin_database_url) as conn:
        conn.execute("DELETE FROM signals")
    base = by_kind()["stockout"]
    assert base.urgency in ("low", "medium") and (base.due - TODAY).days > signals.SAFETY_DAYS
    with psycopg.connect(settings.admin_database_url) as conn:
        add_history(conn, "root.acme", "brass-floor-lamp", [
            ("dismissed", "We have a second warehouse", "false_alarm", 60),
            ("dismissed", "stock is in the other warehouse", "false_alarm", 60),
        ])
    # Twice dismissed for stock elsewhere, rightly: the early warning is held back.
    assert "stockout" not in by_kind()

    # People acted later and still ran out: the pattern resets and urgency goes up.
    with psycopg.connect(settings.admin_database_url) as conn:
        add_history(conn, "root.acme", "brass-floor-lamp", [("acted", None, "happened_anyway", 120)])
    warning = by_kind()["stockout"]
    facts = " ".join(warning.evidence["facts"])
    assert warning.urgency == signals.HIGHER[base.urgency] and "even though someone acted" in facts and "Why " in facts

    # The measured track record, for those who may see the product's signals.
    quality = client.get("/signals/quality", headers={"X-User-Id": "asha"}).json()
    assert quality["warnings_judged"] == 2 and quality["warnings_came_true"] == 0
    assert quality["acted_judged"] == 1 and quality["prevented"] == 0
    assert quality["dismissal_reasons"] == {"stock elsewhere": 2}
    assert quality["summary"].startswith("Track record: 0 of 2 ignored run-out warnings came true")
    hidden = client.get("/signals/quality", headers={"X-User-Id": "meera"}).json()
    assert hidden["signals"] == 0 and hidden["summary"].startswith("Track record: no outcomes")
    assert client.get("/signals/quality", headers={"X-User-Id": "nobody"}).status_code == 401

    # A discontinued product (an all-zero forecast) is excess stock with no sales expected, not "inf days".
    with psycopg.connect(settings.admin_database_url) as conn:
        f = store.latest_forecast(conn, "root.acme.sales", "brass-floor-lamp")
        zero = [{**p, **{k: 0.0 for k in ("forecast", "lo80", "hi80", "lo95", "hi95") if k in p}} for p in f["points"]]
        store.save_forecast(conn, "root.acme.sales", "brass-floor-lamp", f["cutoff"], f["model"], f["reason"], zero, f["backtest"])
        conn.commit()
    gone = by_kind()
    assert "stockout" not in gone
    assert gone["excess_stock"].title.startswith("Moradabad brass floor lamp: no sales expected")
    assert "inf" not in gone["excess_stock"].title + gone["excess_stock"].reason

    # Product and supplier names count as identifying for root learning; their plain words don't.
    with psycopg.connect(settings.admin_database_url) as conn:
        forbidden = learning._forbidden_terms(conn, "root.acme", [])
    assert {"moradabad", "lampworks"} <= forbidden and "lamp" not in forbidden


def test_outcome_lessons_reach_the_root_anonymously(client, settings):
    with psycopg.connect(settings.admin_database_url) as conn:
        # Acme shares patterns; Globex doesn't, so its outcomes never count.
        add_history(conn, "root.acme", "brass-floor-lamp",
                    [("acted", None, "prevented", 30)] * 9 + [("acted", None, "happened_anyway", 30)]
                    + [("dismissed", "not a priority", "happened", 60)] * 4 + [("expired", None, "happened", 0)] * 3
                    + [("dismissed", "we have a second warehouse", "false_alarm", 60)], lead=21)
        add_history(conn, "root.globex", "widget", [("dismissed", None, "false_alarm", 60)] * 20)

    learned = client.post("/admin/learn", headers={"X-Admin-Token": ADMIN_TOKEN}).json()["outcome_lessons"]
    assert "Run-out warnings acted on within two days were prevented about nine times in ten." in learned
    assert "Run-out warnings nobody acted on came true about nine times in ten." in learned
    assert ("When the supplier's lead time is long, stock ordered on a run-out warning still arrived too late "
            "about one time in ten.") in learned
    # Too few dismissals "for stock elsewhere" to say anything yet.
    assert not any("elsewhere" in text for text in learned)
    with psycopg.connect(settings.admin_database_url) as conn:
        forbidden = learning._forbidden_terms(conn, "root.acme", [])
        assert {"acme", "retail", "asha", "vikram"} <= forbidden
    assert all(learning.is_anonymous(text, forbidden) for text in learned)

    # Nothing new, nothing stored twice; new outcomes replace a lesson rather than add one.
    assert client.post("/admin/learn", headers={"X-Admin-Token": ADMIN_TOKEN}).json()["outcome_lessons"] == []
    with psycopg.connect(settings.admin_database_url) as conn:
        add_history(conn, "root.acme", "brass-floor-lamp", [("acted", None, "happened_anyway", 30)] * 5)
    again = client.post("/admin/learn", headers={"X-Admin-Token": ADMIN_TOKEN}).json()["outcome_lessons"]
    assert "Run-out warnings acted on within two days were prevented about six times in ten." in again
    with psycopg.connect(settings.admin_database_url) as conn:
        active = [r[0] for r in conn.execute(
            "SELECT content FROM memories WHERE brain_path = 'root' AND source = 'outcome lesson: acted promptly' "
            "AND status = 'active'")]
    assert active == ["Run-out warnings acted on within two days were prevented about six times in ten."]

    # Business signals read them first, as lessons from across businesses.
    client.post("/remember", json={"content": "Bluebird Stores wants a bulk order of throws"}, headers={"X-User-Id": "asha"})
    client.post("/consolidate", headers={"X-User-Id": "asha"})
    llm = ScriptedLLM([])
    signals.run(settings, llm, "root.acme")
    assert "L1: Run-out warnings" in llm.prompts[-1]
