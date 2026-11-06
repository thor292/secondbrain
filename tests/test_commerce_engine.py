"""Live Commerce engine: forecaster, decisions and payloads on synthetic data, no broker needed."""
import json
from datetime import date, datetime, timedelta

import numpy as np
import pytest

from secondbrain.commerce import contract as C
from secondbrain.commerce.engine import Engine, create_app
from secondbrain.commerce.forecast import _PRIOR, Forecaster, LiveAccuracy
from secondbrain.commerce.offers import (Cooldowns, Elasticity, Offer, OfferBook, choose_defence, choose_discount,
                                         format_inr, margin_floor, restock_plan)
from secondbrain.commerce.state import Registry
from secondbrain.forecasting.metrics import wape

WEEKDAY = np.array([0.8, 0.9, 1.0, 1.0, 1.1, 1.4, 0.8])
SIM_DAY = date.fromisoformat(C.SIM_START[:10])

SIGNAL_KEYS = {"id", "sent_ms", "sim_ts", "platform", "merchant", "merchant_name", "city", "sku", "product", "category",
               "kind", "urgency", "title", "detail", "action", "expected", "confidence", "evidence", "latency_ms",
               "autopilot"}
OUTCOME_KEYS = {"signal_id", "sku", "product", "merchant_name", "discount", "expected_uplift_units",
                "actual_uplift_units", "uplift_pct", "revenue_gain", "worked", "elasticity_before", "elasticity_after",
                "category"}
STATS_KEYS = {"sent_ms", "sim_ts", "speed", "running", "merchants", "customers", "series", "orders_total", "units_total",
              "revenue_total", "orders_per_sec", "revenue_per_sec", "msgs_per_sec", "active_customers_today",
              "signals_total", "signals_by_kind", "offers_applied", "uplift_revenue", "platforms", "latency_ms",
              "accuracy", "learning", "pipeline", "festival"}


# ------------------------------------------------------------------ forecaster

def synthetic(n=300, level=30.0, start=date(2026, 8, 25), end=date(2026, 10, 24), lift=1.6, seed=1):
    rng = np.random.default_rng(seed)
    days = [start + timedelta(d) for d in range((end - start).days + 1)]
    rate = np.array([[level * WEEKDAY[d.weekday()] * (lift if d in C.FESTIVALS else 1.0) for d in days]] * n)
    return days, rng.poisson(rate).astype(float)


def fitted(days, Y, category="Fashion"):
    fc = Forecaster(days[0])
    rows = np.array([fc.add_series(category, -2.5) for _ in range(len(Y))])
    fc.set_history(rows, np.arange(len(days)), Y.T, np.ones_like(Y.T))
    fc.fit(rows, len(days) - 1)
    return fc, rows


def test_forecaster_recovers_weekday_pattern_and_festival_lift():
    days, Y = synthetic()
    fc, rows = fitted(days, Y)
    season = fc.st.season[rows].mean(axis=0)
    assert np.allclose(season, WEEKDAY / WEEKDAY.mean(), atol=0.06)
    lift, seen = fc.festival_lift("Dussehra")
    assert lift[fc.cat[rows[0]]] == pytest.approx(1.6, abs=0.12)
    assert seen[fc.cat[rows[0]]] == 1
    nxt = date(2026, 10, 25)
    assert fc.f_day[rows[0]] == fc.day_index(nxt)
    expected = 30 * WEEKDAY[nxt.weekday()]
    assert fc.f_next[rows].mean() == pytest.approx(expected, rel=0.08)
    lo, hi = fc.lo_next[rows], fc.hi_next[rows]
    assert np.all(lo < fc.f_next[rows]) and np.all(hi > fc.f_next[rows])


def test_wape_is_computed_right_and_beats_seasonal_naive():
    acc = LiveAccuracy()
    y, f = np.array([10.0, 0.0, 5.0, 5.0]), np.array([8.0, 1.0, 5.0, 7.0])
    s = {"rows": np.arange(4), "y": y, "f": f, "lo": f - 2, "hi": f + 2, "naive": np.array([5.0, 5, 5, 5]),
         "naive_ok": np.ones(4, bool)}
    acc.score(s, np.array([0, 0, 1, 1]), np.array([0, 0, 1, 1]))
    out = acc.summary(["a", "b"], {"abs": 0, "y": 0, "naive_abs": 0, "naive_y": 0, "days": 0})
    assert out["wape"] == pytest.approx(5 / 20) == pytest.approx(wape(y, f))
    assert out["baseline_wape"] == pytest.approx(10 / 20)
    assert out["by_platform"] == {"a": pytest.approx(3 / 10), "b": pytest.approx(2 / 10)}

    days, Y = synthetic(n=200, level=12.0, end=date(2026, 10, 18), lift=1.0)
    fc, rows = fitted(days, Y)
    bt = fc.backtest
    assert bt["abs"] / bt["y"] < bt["naive_abs"] / bt["naive_y"]
    live = LiveAccuracy()
    rng = np.random.default_rng(5)
    for k in range(1, 15):  # live days, scored against the previous close's forecast
        d = days[-1] + timedelta(k)
        y = rng.poisson(12 * WEEKDAY[d.weekday()], len(rows)).astype(float)
        live.score(fc.close_day(fc.day_index(d), y, np.ones(len(rows)), np.zeros(len(rows), bool)),
                   np.zeros(len(rows), int), np.zeros(len(rows), int))
    out = live.summary(["p"], fc.backtest)
    assert out["days_scored"] == 14 and out["series_scored"] == len(rows)
    assert out["wape"] < out["baseline_wape"]
    assert 0.7 <= out["coverage_80"] <= 0.97


# ------------------------------------------------------------------ decisions

def test_discount_choice_is_sensible_and_keeps_the_margin_floor():
    # elastic, high margin, overstocked: a real discount pays
    best = choose_discount([1000], [500], [400], [20], [10], [60], [-2.5], [0.15], margin_floor(np.array([1000.0]), np.array([500.0])))
    assert 0.05 <= best["discount"][0] <= 0.2 and best["gain"][0] > 0 and best["uplift"][0] > 0
    # thin-margin staple, not overstocked: discounting only gives margin away
    best = choose_discount([100], [90], [50], [20], [10], [60], [-1.0], [0.15], margin_floor(np.array([100.0]), np.array([90.0])))
    assert best["discount"][0] == 0
    rng = np.random.default_rng(0)
    n = 2000
    price = rng.uniform(20, 5000, n)
    cost = price * rng.uniform(0.4, 0.95, n)
    floor = margin_floor(price, cost)
    best = choose_discount(price, cost, rng.uniform(0, 500, n), rng.uniform(0, 50, n), rng.uniform(0, 50, n),
                           rng.uniform(0, 300, n), -rng.uniform(0.5, 4, n), rng.uniform(0, 0.8, n), floor)
    assert np.all(best["margin"] >= floor - 1e-9)
    assert np.all(best["gain"] >= 0)


def test_restock_fires_when_cover_is_below_lead_time():
    plan = restock_plan([10, 200], [5, 5], [2, 2], [4, 4])
    assert plan["need"].tolist() == [True, False]
    assert plan["cover"][0] == pytest.approx(2.0)
    assert plan["units"][0] >= 5 * (4 + 7) - 10


def test_cooldown_holds_a_signal_back_for_its_sim_hours():
    cd = Cooldowns()
    key = ("m1", "sku1", "discount")
    assert cd.ready(key, 100.0)
    cd.hit(key, 100.0)
    assert not cd.ready(key, 147.0) and cd.ready(key, 148.0)
    assert cd.ready(("m1", "sku1", "restock"), 101.0)


def test_elasticity_moves_toward_the_observed_uplift_and_outcome_matches_contract():
    book = OfferBook(Elasticity())
    start = datetime(2026, 10, 21, 10)
    o = Offer("sig-1", 0, "dmart", "m1", "Sharma Kirana", "BAS5", "Basmati 5kg", "Staples", 0.10, start,
              start + timedelta(hours=48), "merchant", 500.0, 100.0, 11.1, -1.0)
    book.apply(o)
    book.record(0, 130, 130 * 450.0)
    assert book.due(start + timedelta(hours=47)) == []
    (done,) = book.due(start + timedelta(hours=48))
    out = book.evaluate(done)
    observed = np.log(1.3) / np.log(0.9)  # -2.49
    assert observed < out["elasticity_after"] < out["elasticity_before"] == -1.0
    assert out["worked"] and out["actual_uplift_units"] == 30
    assert OUTCOME_KEYS <= set(out)
    json.dumps(out)
    assert book.hit_rate() == 1.0


def test_rupees_are_formatted_the_indian_way():
    assert format_inr(45600) == "₹45,600"
    assert format_inr(340000) == "₹3.40 lakh"
    assert format_inr(21500000) == "₹2.15 crore"


# ------------------------------------------------------------------ engine on messages

def feed_setup(eng, rate=60.0, category="Fashion", n_skus=3, seed=2):
    rng = np.random.default_rng(seed)
    world = {"started_ms": 1, "sim_start": C.SIM_START,
             "platforms": [{"id": "shopify", "name": "Shopify", "kind": "D2C", "color": "#95BF47"}],
             "merchants": [{"id": "shopify-0001", "platform": "shopify", "name": "Urban Threads", "city": "Bengaluru",
                            "category": category, "x": 0.4, "y": 0.6}], "customers": 1000}
    skus = [f"SKU{k}" for k in range(n_skus)]
    catalog = {"products": [{"sku": s, "name": f"Kurta {k}", "category": category, "price": 1000.0, "cost": 500.0,
                             "stock": 5000, "lead_days": 7, "perishable": False} for k, s in enumerate(skus)]}
    days = [SIM_DAY - timedelta(C.HISTORY_DAYS - d) for d in range(C.HISTORY_DAYS)]
    units = [[int(rng.poisson(rate * WEEKDAY[d.weekday()])) for _ in skus] for d in days]
    history = {"days": [d.isoformat() for d in days], "skus": skus, "units": units,
               "prices": [[1000.0] * n_skus for _ in days]}
    t = lambda kind: C.merchant_topic("shopify", "shopify-0001", kind)
    eng.process([(C.WORLD, world, 0), (t("catalog"), catalog, 0), (t("history"), history, 0)])
    return skus


def live_hours(eng, skus, hours, factor, rate=60.0, seed=3, day=SIM_DAY):
    """Orders hour by hour; ``factor`` per sku scales its demand."""
    rng = np.random.default_rng(seed)
    share = _PRIOR / _PRIOR.sum()
    for h in hours:
        ts = datetime.combine(day, datetime.min.time()) + timedelta(hours=h, minutes=59)
        now = C.wall_ms()
        orders = []
        for s in skus:
            for _ in range(rng.poisson(rate * WEEKDAY[day.weekday()] * share[h] * factor.get(s, 1.0))):
                orders.append({"id": f"o{len(orders)}", "customer": f"c{rng.integers(500)}", "sku": s, "qty": 1,
                               "price": 1000.0, "offer": None})
        batch = [(C.merchant_topic("shopify", "shopify-0001", "orders"),
                  {"sent_ms": now - 2, "sim_ts": ts.isoformat(), "orders": orders}, now),
                 (C.CLOCK, {"sent_ms": now, "sim_ts": ts.isoformat(), "speed": 3600, "running": True,
                            "orders_per_sec": 10}, now)]
        eng.process(batch)


@pytest.fixture
def engine():
    live, control = [], []
    eng = Engine(publish_live=lambda topic, payload, qos=0: live.append((topic, payload)),
                 publish_control=control.append, autopilot=0.0)
    eng.live, eng.control = live, control
    return eng


def signals(eng):
    return [p for t, p in eng.live if "/signals/" in t]


def test_nowcast_flags_a_midday_drop_with_a_profitable_discount(engine):
    skus = feed_setup(engine)
    assert not engine.fc.fitted[:3].any()  # waits for the history burst to end, or the clock
    live_hours(engine, skus, range(0, 15), {"SKU0": 0.3})
    assert engine.fc.fitted[:3].all()
    sigs = signals(engine)
    drops = [s for s in sigs if s["kind"] == "discount"]
    assert [s["sku"] for s in drops] == ["SKU0"]
    s = drops[0]
    assert s["action"]["type"] == "discount" and 0 < s["action"]["discount"] <= 0.2
    assert s["expected"]["margin"] >= 0.12 and s["expected"]["uplift_units"] > 0
    assert s["evidence"]["deviation"] > 0.4 and "below forecast" in s["title"]
    assert set(s) == SIGNAL_KEYS and not s["autopilot"]
    json.dumps(s)
    # the next hourly checks don't repeat it (cooldown)
    live_hours(engine, skus, range(15, 18), {"SKU0": 0.3}, seed=4)
    assert len([x for x in signals(engine) if x["kind"] == "discount"]) == 1


def test_applied_offer_closes_with_an_outcome_and_payloads_match_contract(engine):
    skus = feed_setup(engine)
    live_hours(engine, skus, range(0, 15), {"SKU0": 0.3})
    sig = next(s for s in signals(engine) if s["kind"] == "discount")
    client = __import__("fastapi.testclient", fromlist=["TestClient"]).TestClient(create_app(engine))
    assert client.post(f"/offers/{sig['id']}/apply").status_code == 200
    cmd = engine.control[-1]
    assert cmd["cmd"] == "apply_offer" and cmd["by"] == "merchant" and cmd["signal_id"] == sig["id"]
    engine.process([(C.CONTROL, cmd, 0)])
    assert engine.book.applied == 1
    assert client.post(f"/offers/{sig['id']}/apply").status_code == 409
    # demand answers the discount (x1.5), through the window and past it
    live_hours(engine, skus, range(15, 24), {"SKU0": 0.45}, seed=5)
    for k, day in enumerate([SIM_DAY + timedelta(1), SIM_DAY + timedelta(2)]):
        live_hours(engine, skus, range(24), {"SKU0": 1.5}, seed=6 + k, day=day)
    live_hours(engine, skus, range(0, 16), {}, seed=9, day=SIM_DAY + timedelta(3))
    outs = [p for t, p in engine.live if "/outcomes/" in t]
    assert len(outs) == 1 and OUTCOME_KEYS <= set(outs[0]) and outs[0]["signal_id"] == sig["id"]
    assert outs[0]["actual_uplift_units"] > 0
    engine.last_stats_at = engine.last_ticker_at = 0
    engine.tick()
    stats = next(p for t, p in reversed(engine.live) if t == C.STATS)
    assert set(stats) >= STATS_KEYS
    assert set(stats["latency_ms"]) >= {"broker_p50", "broker_p95", "process_p50", "signal_p50", "end_to_end_p50",
                                         "end_to_end_p95"}
    assert stats["accuracy"]["days_scored"] == 3 and stats["accuracy"]["wape"] is not None
    assert set(stats["accuracy"]) >= {"wape", "baseline_wape", "days_scored", "series_scored", "by_platform"}
    assert stats["learning"]["outcomes"] == 1 and set(stats["pipeline"]) >= {"ingested", "forecasts_updated",
                                                                             "signals_raised", "delivered"}
    assert stats["festival"]["name"] == "Dhanteras"
    json.dumps(stats)
    ticker = [p for t, p in engine.live if t == C.TICKER]
    assert ticker and set(ticker[-1]["orders"][0]) == {"platform", "merchant", "merchant_name", "city", "product",
                                                       "qty", "amount", "offer"}
    # API views
    m = client.get("/merchants/shopify-0001").json()
    assert {"id", "platform", "name", "city", "category", "orders_today", "revenue_today", "products",
            "open_signals", "outcomes"} <= set(m)
    assert {"sku", "name", "category", "price", "stock", "lead_days", "cover_days", "today"} <= set(m["products"][0])
    series = client.get("/merchants/shopify-0001/series/SKU0").json()
    assert len(series["history"]) == 42 and len(series["forecast"]) == 14 and series["offers"]
    assert series["product"] == "Kurta 0" and series["today"]["forecast"] > 0
    assert client.get("/merchants").json()[0]["id"] == "shopify-0001"
    assert client.get("/signals?limit=1").json()[0]["id"]
    assert client.post("/control", json={"cmd": "speed", "speed": "warp"}).status_code == 400
    assert client.post("/control", json={"cmd": "pause"}).status_code == 200 and engine.control[-1] == {"cmd": "pause"}


def test_autopilot_applies_offers_itself(engine):
    engine.autopilot = 1.0
    skus = feed_setup(engine)
    live_hours(engine, skus, range(0, 15), {"SKU0": 0.3})
    sig = next(s for s in signals(engine) if s["kind"] == "discount")
    assert sig["autopilot"] and engine.control[-1]["by"] == "autopilot"


def test_reset_clears_the_run_and_asks_for_the_retained_setup_again(engine):
    calls = []
    engine.on_reset = lambda: calls.append(1)
    skus = feed_setup(engine)
    live_hours(engine, skus, range(8, 11), {})
    assert engine.totals["orders"] > 0
    engine.process([(C.CONTROL, {"cmd": "reset"}, 0)])
    assert calls == [1] and engine.totals["orders"] == 0 and engine.reg.n == 0 and engine.today is None


def test_volume_defence_picks_the_smallest_discount_that_wins_back_the_drop():
    # snacks-like: 19% margin, elasticity -1.4; 30 units/day lost of 100, customers worth ₹60 a unit later
    price, cost = np.array([100.0]), np.array([81.0])
    best = choose_defence(price, cost, [1000], [70], [30], [-1.4], margin_floor(price, cost), [60.0])
    d = best["discount"][0]
    assert d > 0 and best["margin"][0] >= margin_floor(price, cost)[0] - 1e-9
    smaller = [g for g in (0.05, 0.08, 0.10, 0.12, 0.15, 0.20) if g < d]
    assert all(70 * ((1 - g) ** -1.4 - 1) < 0.5 * 30 for g in smaller) or best["recovered_share"][0] < 0.5
    # customers worth nothing later: giving margin away on units that sell anyway doesn't pay
    assert choose_defence(price, cost, [1000], [70], [30], [-1.4], margin_floor(price, cost), [0.0])["discount"][0] == 0


def test_winback_counts_regulars_away_twice_their_usual_interval():
    reg = Registry()
    m = reg.merchant("dmart-1", "dmart")
    for k in range(40):              # regulars every ~4 days, last seen on day 10
        for day in (2, 6, 10):
            reg.visit(m, f"r{k}", day)
    for k in range(40):              # still coming
        for day in (12, 16, 20):
            reg.visit(m, f"a{k}", day)
    count, interval = reg.lapsed(m, 21)
    assert count == 40 and interval == pytest.approx(4.0)
    assert reg.m_visits[m] == 240


def test_drop_persistence_is_learned_from_flagged_products_without_offers(engine):
    assert engine.persistence() == pytest.approx(0.5)
    engine.drop_obs = [(0.0, 1.0, False)] * 45   # dips that fully reverted
    assert engine.persistence() == pytest.approx(0.05)
    assert engine.persistence(True) == pytest.approx(0.5)


# ------------------------------------------------------------------ command centre: HQ, Graph and Chat

MERCHANTS = [  # (id, platform, name, category, product category, price, cost)
    ("shopify-0001", "shopify", "Desi Living", "Home Decor", "Fashion", 1000.0, 500.0),
    ("shopify-0002", "shopify", "Kurta Co", "Fashion", "Fashion", 1200.0, 600.0),
    ("shopify-0003", "shopify", "Silk Route", "Fashion", "Fashion", 900.0, 450.0),
    ("dmart-0001", "dmart", "DMart Powai", "Hypermarket", "Staples", 100.0, 90.0),
    ("dmart-0002", "dmart", "DMart Thane", "Hypermarket", "Staples", 120.0, 107.0),
]


def feed_network(eng, rate=30.0, seed=2):
    rng = np.random.default_rng(seed)
    world = {"started_ms": 1, "sim_start": C.SIM_START,
             "platforms": [{"id": "shopify", "name": "Shopify", "kind": "D2C", "color": "#95BF47"},
                           {"id": "dmart", "name": "DMart", "kind": "Value", "color": "#00833E"}],
             "merchants": [{"id": m, "platform": p, "name": n, "city": "Mumbai", "category": c, "x": 0.5, "y": 0.5}
                           for m, p, n, c, *_ in MERCHANTS], "customers": 5000}
    batch = [(C.WORLD, world, 0)]
    days = [SIM_DAY - timedelta(C.HISTORY_DAYS - d) for d in range(C.HISTORY_DAYS)]
    for m, p, n, c, pc, price, cost in MERCHANTS:
        skus = [f"{m}-S{k}" for k in range(2)]
        t = lambda kind: C.merchant_topic(p, m, kind)
        batch.append((t("catalog"), {"products": [{"sku": s, "name": f"{n} item {k}", "category": pc, "price": price,
                                                    "cost": cost, "stock": 400, "lead_days": 5, "perishable": False}
                                                   for k, s in enumerate(skus)]}, 0))
        batch.append((t("history"), {"days": [d.isoformat() for d in days], "skus": skus,
                                     "units": [[int(rng.poisson(rate * WEEKDAY[d.weekday()])) for _ in skus] for d in days],
                                     "prices": [[price] * 2 for _ in days]}, 0))
    eng.process(batch)


def network_hours(eng, hours, day=SIM_DAY, seed=3, rate=30.0):
    rng = np.random.default_rng(seed)
    share = _PRIOR / _PRIOR.sum()
    for h in hours:
        ts = (datetime.combine(day, datetime.min.time()) + timedelta(hours=h, minutes=59)).isoformat()
        now = C.wall_ms()
        batch = []
        for m, p, n, c, pc, price, cost in MERCHANTS:
            orders = [{"id": f"{m}-{h}-{k}", "customer": f"{m}-c{rng.integers(60)}", "sku": f"{m}-S{k % 2}", "qty": 1,
                       "price": price, "offer": None}
                      for k in range(rng.poisson(2 * rate * WEEKDAY[day.weekday()] * share[h]))]
            batch.append((C.merchant_topic(p, m, "orders"), {"sent_ms": now, "sim_ts": ts, "orders": orders}, now))
        batch.append((C.CLOCK, {"sent_ms": now, "sim_ts": ts, "speed": 3600, "running": True}, now))
        eng.process(batch)


def fake_signal(eng, merchant, urgency="medium", revenue=1000.0, kind="restock"):
    m = eng.reg.merchant_index[merchant]
    i = eng.reg.series_of[m][0]
    eng._queue({"kind": kind, "i": i, "score": revenue, "urgency": urgency, "title": f"{kind} at {merchant}",
                "detail": "test", "confidence": 0.5, "action": {"type": "restock", "units": 10, "hours": 72},
                "expected": {"units": 10, "revenue": revenue, "margin": 0.2, "uplift_units": 1},
                "evidence": eng._evidence(i, 1, 1, 0, 1, -1)}, m)
    eng._flush(C.wall_ms())


def closed_offer(eng, merchant, discount=0.2, base=50.0, units=70.0, stockout=False):
    m = eng.reg.merchant_index[merchant]
    i = eng.reg.series_of[m][0]
    start = eng.now
    o = Offer(f"sig-x{len(eng.outcome_log)}", i, eng.reg.merchants[m]["platform"], merchant,
              eng.reg.merchants[m]["name"], eng.reg.sku[i], eng.reg.name[i], eng.reg.category[i], discount, start,
              start + timedelta(hours=48), "merchant", float(eng.reg.price[i]), base, 5.0, -2.5, assured_uplift=2.0)
    o.units, o.stockout = units, stockout
    eng._log_outcome(o, eng.book.evaluate(o))


@pytest.fixture
def network(engine):
    feed_network(engine)
    for d in range(6):    # six closed days (six daily price checks), then half of the next
        network_hours(engine, range(24), day=SIM_DAY + timedelta(d), seed=3 + d)
    network_hours(engine, range(0, 12), day=SIM_DAY + timedelta(6), seed=10)
    return engine


def test_lessons_come_only_from_real_evidence(network):
    eng = network
    client = __import__("fastapi.testclient", fromlist=["TestClient"]).TestClient(create_app(eng))
    kinds = lambda **q: [l["kind"] for l in client.get("/lessons", params=q).json()]
    # flat history prices (no promotions), no closed offers, festival days that sold like any other day:
    # the only lesson is the one the price checks taught (DMart staples at ~10% margin)
    lessons = client.get("/lessons").json()
    assert [(l["kind"], l["category"]) for l in lessons] == [("no_discount", "Staples")]
    nd = lessons[0]
    checked = sum(v[0] for (m, c), v in eng.price_checks.items() if c == "Staples")
    assert nd["evidence_n"] == checked >= 2 and nd["learned_from"] == 2 and nd["applies_to"] == 2
    assert "Discounts don't pay on Staples" in nd["title"] and nd["platform"] is None
    assert kinds(platform="shopify") == []          # Shopify's fashion margins are 50%: no such lesson there
    # two closed offers are not yet a lesson; the third is
    closed_offer(eng, "shopify-0002", base=50, units=70)
    closed_offer(eng, "shopify-0003", base=40, units=60)
    assert "price_response" not in kinds()
    closed_offer(eng, "shopify-0002", base=30, units=33)
    closed_offer(eng, "shopify-0003", base=20, units=5, stockout=True)   # censored: not evidence of response
    pr = next(l for l in client.get("/lessons").json() if l["kind"] == "price_response")
    mean_uplift = np.mean([70 / 50 - 1, 60 / 40 - 1, 33 / 30 - 1])
    assert pr["evidence_n"] == 3 and pr["learned_from"] == 2 and pr["applies_to"] == 3
    assert f"sold {mean_uplift:+.0%} on average across 3 offers" in pr["title"] and "20% offer for 48h" in pr["title"]
    assert pr["value"] == pytest.approx(eng.book.elasticity.category_value("Fashion"), abs=1e-3)
    assert pr["prior"] == -2.5
    assert client.get("/lessons").json()[0]["kind"] == "price_response"   # measured outcomes rank first
    assert "price_response" not in kinds(platform="dmart")
    # Desi Living sells Fashion but only 2 other merchants taught it: below the anonymity floor
    assert "price_response" not in kinds(merchant="shopify-0001")
    closed_offer(eng, "shopify-0001", base=50, units=75)
    pr = next(l for l in client.get("/lessons", params={"merchant": "shopify-0001"}).json()
              if l["kind"] == "price_response")
    assert pr["learned_from"] == 3 and pr["applies_to"] == 1
    for l in client.get("/lessons").json():
        assert set(l) == {"id", "kind", "category", "platform", "title", "detail", "value", "prior", "evidence_n",
                          "learned_from", "applies_to", "updated_sim_ts"}
        assert l["evidence_n"] > 0 and l["learned_from"] > 0


def test_scope_filtering_never_leaks_another_merchant(network):
    eng = network
    for mid, *_ in MERCHANTS:
        fake_signal(eng, mid, revenue=100.0 * len(mid))
    closed_offer(eng, "shopify-0002")
    client = __import__("fastapi.testclient", fromlist=["TestClient"]).TestClient(create_app(eng))
    for mid, pid, name, *_ in MERCHANTS:
        others = [(m, n) for m, p, n, *_ in MERCHANTS if m != mid]
        for path in ("/platforms", "/lessons", "/attention", "/tree"):
            r = client.get(path, params={"merchant": mid})
            assert r.status_code == 200, (path, r.text)
            text = r.text
            for m, n in others:
                assert m not in text and n not in text, (mid, path, m)
        (p,) = client.get("/platforms", params={"merchant": mid}).json()
        m = eng.reg.merchant_index[mid]
        assert p["id"] == pid and p["merchants"] == 1 and p["orders_today"] == int(eng.reg.m_orders_today[m])
        assert p["offers_applied"] == eng.ledger.get(m, [0])[0] and p["orders_per_sec"] is None
        att = client.get("/attention", params={"merchant": mid}).json()
        assert {s["merchant"] for s in att} == {mid} and att[0]["platform_name"] in ("Shopify", "DMart")
    # a platform sees only its own merchants
    text = client.get("/tree", params={"platform": "dmart"}).text
    assert "dmart-0001" in text and "shopify" not in text.replace('"platform"', "")
    assert {s["platform"] for s in client.get("/attention", params={"platform": "shopify", "limit": 50}).json()} == {"shopify"}
    assert client.get("/tree", params={"merchant": "nobody"}).status_code == 404
    assert client.get("/platforms", params={"platform": "dmart", "merchant": "shopify-0001"}).status_code == 404
    assert client.get("/lessons", params={"platform": "amazon"}).status_code == 404


def test_tree_counts_add_up_and_attention_ranks_urgent_rupees_first(network):
    eng = network
    fake_signal(eng, "dmart-0001", urgency="medium", revenue=9000.0)
    fake_signal(eng, "shopify-0002", urgency="high", revenue=500.0, kind="discount")
    fake_signal(eng, "shopify-0003", urgency="medium", revenue=20000.0)
    client = __import__("fastapi.testclient", fromlist=["TestClient"]).TestClient(create_app(eng))
    t = client.get("/tree").json()
    assert t["scope"] == {"level": "hq", "id": None}
    merchants = [m for p in t["platforms"] for c in p["categories"] for m in c["merchants"]]
    assert len(merchants) == len(MERCHANTS)
    for key in ("orders_today", "open_signals"):
        assert t["root"][key] == sum(p[key] for p in t["platforms"]) == sum(m[key] for m in merchants)
        for p in t["platforms"]:
            assert p[key] == sum(m[key] for c in p["categories"] for m in c["merchants"])
    assert t["root"]["revenue_today"] == pytest.approx(sum(m["revenue_today"] for m in merchants))
    assert t["root"]["orders_today"] == int(eng.reg.m_orders_today[:eng.reg.nm].sum()) > 0
    status = {m["id"]: m["status"] for m in merchants}
    open_by = eng._open_signals()
    for mid, *_ in MERCHANTS:
        sigs = open_by.get(mid, [])
        assert status[mid] == ("urgent" if any(x["urgency"] == "high" for x in sigs) else "signal" if sigs else "ok")
    assert status["shopify-0002"] == "urgent"
    n_open = sum(len(v) for v in open_by.values())
    plats = {p["id"]: p for p in client.get("/platforms").json()}
    assert sum(p["merchants"] for p in plats.values()) == len(MERCHANTS)
    assert sum(p["orders_today"] for p in plats.values()) == t["root"]["orders_today"]
    assert sum(p["open_signals"] for p in plats.values()) == t["root"]["open_signals"] == n_open >= 3
    assert sum(p["signals_total"] for p in plats.values()) == eng.pipeline["signals_raised"]
    assert sum(c["merchants"] for c in plats["shopify"]["categories"]) == plats["shopify"]["merchants"] == 3
    assert plats["shopify"]["active_customers_today"] > 0 and plats["shopify"]["accuracy"]["wape"] is not None
    assert set(plats["shopify"]) >= {"id", "name", "kind", "color", "merchants", "customers", "series", "orders_today",
                                     "revenue_today", "orders_per_sec", "active_customers_today", "accuracy",
                                     "open_signals", "signals_total", "signals_by_kind", "offers_applied",
                                     "uplift_revenue", "hit_rate", "categories"}
    sub = client.get("/tree", params={"platform": "shopify"}).json()
    assert sub["scope"] == {"level": "platform", "id": "shopify"} and [p["id"] for p in sub["platforms"]] == ["shopify"]
    att = client.get("/attention", params={"limit": 200}).json()
    assert len(att) == n_open
    rank = [({"high": 0, "medium": 1, "low": 2}[s["urgency"]], -s["expected"]["revenue"]) for s in att]
    assert rank == sorted(rank)
    mine = [s["title"] for s in att if s["detail"] == "test"]
    assert mine.index("discount at shopify-0002") < mine.index("restock at shopify-0003") < mine.index("restock at dmart-0001")
    assert len(client.get("/attention", params={"limit": 2}).json()) == 2


def test_run_days_control_is_validated_and_published(engine):
    client = __import__("fastapi.testclient", fromlist=["TestClient"]).TestClient(create_app(engine))
    assert client.post("/control", json={"cmd": "run_days", "days": 7}).status_code == 200
    assert engine.control[-1] == {"cmd": "run_days", "days": 7}
    for bad in (0, 61, "7", 2.5, None, True):
        assert client.post("/control", json={"cmd": "run_days", "days": bad}).status_code == 400
