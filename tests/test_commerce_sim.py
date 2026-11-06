"""Live Commerce simulator: the world is deterministic and believable, the demand model has the
shapes the engine is meant to find, and the payloads match ``commerce.contract``.

Unit tests need no broker. The broker test runs the simulator in a subprocess on its own
``SB_COMMERCE_PREFIX`` against the ingest broker (skipped when it is not up):

    docker compose up -d mosquitto && uv run pytest tests/test_commerce_sim.py
"""
import json
import os
import socket
import subprocess
import sys
import time
import uuid
from datetime import date

import numpy as np
import pytest

from secondbrain.commerce import contract
from secondbrain.commerce.simulator import CATS, DAY, PIDS, Simulator, State, DemandModel, sim_date, sim_ts
from secondbrain.commerce.world import build_world

FEST = {name: (d - sim_date(0)).days for d, name in contract.FESTIVALS.items()}


@pytest.fixture(scope="module")
def world():
    return build_world(merchants=90, customers=15000, seed=11)


@pytest.fixture(scope="module")
def model(world):
    return DemandModel(world)


@pytest.fixture()
def sim(world):
    sent: list[tuple[str, str, int, bool]] = []
    s = Simulator(world, lambda *m: sent.append(m), "1day")
    s.sent = sent
    return s


def series_of(model: DemandModel, platform: str, category: str) -> np.ndarray:
    return np.flatnonzero((model.plat == PIDS.index(platform)) & (model.cat == CATS.index(category)))


def test_world_is_deterministic():
    a, b, c = build_world(40, 4000, seed=3), build_world(40, 4000, seed=3), build_world(40, 4000, seed=4)
    assert a.merchants == b.merchants and a.catalogs == b.catalogs
    assert np.array_equal(a.customer_interval, b.customer_interval)
    assert a.merchants != c.merchants
    assert DemandModel(a).day_rate(5).tolist() == DemandModel(b).day_rate(5).tolist()


def test_world_platforms_merchants_and_map(world):
    assert [(p.id, p.name) for p in world.platforms] == [("shopify", "Shopify"), ("reliance", "Reliance Retail"),
                                                        ("dmart", "DMart")]
    ms = world.merchants
    assert len({m.id for m in ms}) == len(ms) == 90 and len({m.name for m in ms}) == len(ms)
    assert {m.platform for m in ms} == {"shopify", "reliance", "dmart"}
    assert all(m.name.startswith("DMart ") for m in ms if m.platform == "dmart")
    assert all(0 <= m.x <= 1 and 0 <= m.y <= 1 for m in ms)
    north = [m.y for m in ms if m.city in ("Delhi", "Gurugram", "Noida", "Chandigarh")]
    south = [m.y for m in ms if m.city in ("Bengaluru", "Chennai", "Kochi")]
    assert north and south and max(north) < min(south)
    mumbai = next(m for m in ms if m.city == "Mumbai")
    assert mumbai.x == pytest.approx((72.88 - 68) / 29.5, abs=0.01) and mumbai.y == pytest.approx((37.5 - 19.08) / 31, abs=0.01)


def test_catalogs_are_sane(world):
    for m, cat in zip(world.merchants, world.catalogs):
        assert 15 <= len(cat) <= 45
        assert len({p.sku for p in cat}) == len(cat)
        assert all(0 < p.cost < p.price and p.lead_days >= 1 for p in cat)
        assert all(p.perishable for p in cat if p.category in ("Dairy", "Fresh"))
        if m.platform == "shopify":
            assert all(p.name.startswith(m.name) for p in cat)
    atta = [p.price for c in world.catalogs for p in c if p.name == "Aashirvaad Atta 5kg"]
    assert atta and all(250 <= x <= 320 for x in atta)


def test_customers_belong_to_merchants(world):
    assert world.n_customers == 15000 == world.pool_size.sum()
    assert np.all(np.diff(world.customer_merchant) >= 0)
    for i in (0, 45, 89):
        a, n = world.pool_start[i], world.pool_size[i]
        assert np.all(world.customer_merchant[a:a + n] == i)
    assert 0.08 < np.isfinite(world.customer_lapse).mean() < 0.16


def test_weekends_up_for_dmart_not_shopify(model):
    sun, wed = 54, 50  # 2026-12-13 (Sunday) and 2026-12-09 (Wednesday)
    assert sim_date(sun).weekday() == 6 and sim_date(wed).weekday() == 2
    dmart, shop = series_of(model, "dmart", "Staples"), series_of(model, "shopify", "Beauty")
    ratio = model.day_factor(sun) / model.day_factor(wed)
    assert np.median(ratio[dmart]) > 1.5
    assert 0.85 < np.median(ratio[shop]) < 1.15


def test_festivals_ramp_then_dip(model):
    def ratio(day, category, platform="reliance"):
        s = series_of(model, platform, category)
        return np.median(model.day_factor(day)[s] / model.day_factor(day - 28)[s])  # same weekday, before the season
    assert ratio(FEST["Dhanteras"], "Electronics") > 2.2
    assert ratio(FEST["Diwali"], "Sweets", "dmart") > 2.0
    assert ratio(FEST["Diwali"] + 2, "Electronics") < 0.85
    assert 0.85 < ratio(FEST["Dhanteras"], "Dairy", "dmart") < 1.6


def test_intraday_profile(world, model):
    st = State(model)
    assert st.expected(3 * 3600, 4 * 3600).sum() == 0  # nobody orders at 3am
    dmart = series_of(model, "dmart", "Staples")
    evening, morning = st.expected(19 * 3600, 20 * 3600)[dmart], st.expected(9 * 3600, 10 * 3600)[dmart]
    assert evening.sum() > 2.5 * morning.sum()
    assert np.allclose(st.expected(0, DAY), st.rate)


def test_offer_raises_demand_by_the_elasticity(model):
    st = State(model)
    s = int(series_of(model, "shopify", "Fashion")[0])
    e = model.elasticity[s]
    assert -3.5 < e < -1.7
    base = st.expected(10 * 3600, 22 * 3600)[s]
    st.apply_offer(s, 0.10, DAY, "sig-1")
    lifted = st.expected(10 * 3600, 22 * 3600)[s]
    assert lifted / base == pytest.approx(0.9 ** e)

    st.rate = st.rate.copy()
    st.rate[s] = 20000
    st.stock[s] = 10**9
    for offer in (False, True):
        st.t = 0.0
        st.apply_offer(s, 0.10 if offer else 0.0, DAY if offer else -1, "sig-1" if offer else None)
        lines = st.step(DAY)
        n = int((lines.series == s).sum())
        if not offer:
            plain = n
    assert n / plain == pytest.approx(0.9 ** e, rel=0.05)


def test_orders_carry_the_offer_and_discounted_price(sim):
    m = sim.world.merchants[0]
    prod = max(sim.world.catalogs[0], key=lambda p: sim.model.base[sim.sku_index[0][p.sku]])
    s = sim.sku_index[0][prod.sku]
    sim.state.rate = sim.state.rate.copy()
    sim.state.rate[s], sim.state.stock[s] = 5000, 10**9
    sim.handle({"cmd": "apply_offer", "signal_id": "sig-42", "platform": m.platform, "merchant": m.id,
                "sku": prod.sku, "discount": 0.08, "until": "2026-10-20T20:00:00", "by": "merchant"})
    for _ in range(60):
        sim.advance(1440)
    seen = [(o, json.loads(p)["sim_ts"]) for t, p, *_ in sim.sent if t == sim.topics[0]
            for o in json.loads(p)["orders"] if o["sku"] == prod.sku]
    during = [o for o, ts in seen if ts <= "2026-10-20T20:00:00"]
    after = [o for o, ts in seen if ts > "2026-10-20T20:06:00"]
    assert during and all(o["offer"] == "sig-42" and o["price"] == round(prod.price * 0.92, 2) for o in during)
    assert after and all(o["offer"] is None and o["price"] == prod.price for o in after)


def test_stockouts_cap_sales(model):
    st = State(model)
    s = 7
    st.rate = st.rate.copy()
    st.rate[s], st.stock[s], st.pending_at[s] = 1000, 3, np.inf
    lines = st.step(DAY)
    assert lines.qty[lines.series == s].sum() == 3 and st.stock[s] == 0
    assert st.stock.min() >= 0


def test_restocks_arrive_after_lead_time(model):
    st = State(model)
    s = 3
    st.stock[s], st.pending_at[s] = 0, np.inf
    st.reorder(0)
    assert st.pending_qty[s] > 0 and st.pending_at[s] >= (model.lead[s]) * DAY
    due = st.arrivals(st.pending_at[s])
    assert s in due and st.stock[s] == st.pending_qty[s]


def test_history_matches_the_model_and_hands_over_to_live(model):
    st = State(model)
    h = st.history
    assert len(h["days"]) == contract.HISTORY_DAYS and h["days"][-1] == -1
    for i in (0, 30, 55):
        d = h["days"][i]
        free = h["units"][i] == h["demand"][i]
        mean = model.day_rate(d) * (1 + model.qty_extra) * (1 - model.promo_discount(d)) ** model.elasticity
        assert h["demand"][i].sum() == pytest.approx(mean.sum(), rel=0.02)
        assert free.mean() > 0.9
    st.stock[:] = 10**9
    sold = 0
    while st.t < DAY:
        sold += st.step(min(st.t + 1440, DAY)).qty.sum()
    assert sold == pytest.approx((model.day_rate(0) * (1 + model.qty_extra)).sum(), rel=0.02)


def test_shocks_cut_demand_for_days(model):
    shocks = [(s, f) for d in range(0, 20) for s, f in model.active_shocks(d) if f < 1]
    assert shocks and all(0.6 <= f <= 0.75 for _, f in shocks)
    s, f = shocks[0]
    day = next(d for d in range(0, 20) if (s, f) in model.active_shocks(d))
    assert model.shock(day)[s] == pytest.approx(f)


def test_lapsed_customers_stop_buying(sim):
    sim.advance(3 * DAY)
    w = sim.world
    customers = [int(o["customer"][1:]) for t, p, *_ in sim.sent if t.endswith("/orders") for o in json.loads(p)["orders"]]
    lapsed = w.customer_lapse < 0
    assert lapsed.mean() > 0.03
    assert lapsed[customers].mean() < 0.005
    assert len(set(customers)) > 0.3 * w.n_customers


def test_payloads_match_the_contract(sim):
    sim.publish_setup()
    by_kind = {}
    for topic, payload, qos, retain in sim.sent:
        parsed = contract.parse_merchant_topic(topic)
        by_kind.setdefault(parsed[2] if parsed else topic, []).append((json.loads(payload), qos, retain))
    (w, qos, retain), = by_kind[contract.WORLD]
    assert set(w) == {"started_ms", "sim_start", "platforms", "merchants", "customers"} and qos == 1 and retain
    assert w["sim_start"] == contract.SIM_START and w["customers"] == sim.world.n_customers
    assert set(w["merchants"][0]) == {"id", "platform", "name", "city", "category", "x", "y"}
    assert set(w["platforms"][0]) == {"id", "name", "kind", "color"}
    cat, qos, retain = by_kind["catalog"][0]
    assert set(cat["products"][0]) == {"sku", "name", "category", "price", "cost", "stock", "lead_days", "perishable"}
    assert qos == 1 and retain and len(by_kind["catalog"]) == len(sim.world.merchants)
    hist, qos, retain = by_kind["history"][0]
    assert len(hist["days"]) == len(hist["units"]) == len(hist["prices"]) == contract.HISTORY_DAYS
    assert len(hist["units"][0]) == len(hist["skus"]) and hist["days"][-1] == "2026-10-19" and not retain

    sim.sent.clear()
    sim.advance(DAY)
    kinds = {}
    for topic, payload, qos, retain in sim.sent:
        kinds.setdefault(contract.parse_merchant_topic(topic)[2], []).append((json.loads(payload), qos))
    order_msg, qos = kinds["orders"][0]
    assert qos == 0 and set(order_msg) == {"sent_ms", "sim_ts", "orders"}
    assert set(order_msg["orders"][0]) == {"id", "customer", "sku", "qty", "price", "offer"}
    stock, qos = kinds["stock"][-1]
    assert qos == 1 and set(stock) == {"sim_ts", "stock", "restocked"}
    assert stock["sim_ts"] == "2026-10-20T23:59:59"
    clock = sim.clock_payload()
    assert {"sent_ms", "sim_ts", "speed", "running", "orders_per_sec"} <= set(clock)
    assert clock["sim_ts"] == "2026-10-21T00:00:00"


def test_guard_slows_the_clock_and_recovers(sim):
    sim.adapt(0.2)
    assert sim.factor <= 0.5
    for _ in range(100):
        sim.adapt(0.01)
    assert sim.factor == 1.0


def test_controls(sim):
    sim.handle({"cmd": "start"})
    assert sim.running
    sim.handle({"cmd": "speed", "speed": "1min"})
    assert sim.requested == "1min"
    sim.advance(3600)
    sim.handle({"cmd": "reset"})
    assert not sim.running and sim.state.t == 0
    assert any(t == contract.WORLD for t, *_ in sim.sent)


def test_run_days_runs_fast_then_pauses_on_the_day(sim):
    sim.handle({"cmd": "speed", "speed": "1min"})
    sim.advance(5 * 3600)
    start = sim.state.t
    sim.handle({"cmd": "run_days", "days": 2})
    clock = sim.clock_payload()
    assert sim.running and clock["requested"] == "1day" and not sim.catching_up
    assert clock["run_until"] == sim_ts(start + 2 * DAY)
    ticks = 0
    while sim.running and ticks < 1000:
        dt = sim.step(6 * 3600.0)   # big steps: the last one is cut to land exactly on the target
        sim.advance(dt)
        sim.finish_step()
        ticks += 1
    assert not sim.running and sim.state.t == pytest.approx(start + 2 * DAY)
    assert ticks == 8 and sim.state.day == 2
    clock = sim.clock_payload()
    assert clock["run_until"] is None and clock["requested"] == "1min" and not clock["running"]
    # a pause in the middle cancels the run and keeps the user's speed
    sim.handle({"cmd": "run_days", "days": 3})
    sim.handle({"cmd": "pause"})
    assert sim.run_until is None and sim.requested == "1min" and not sim.running
    for bad in (0, -1, "x", None, 999):
        sim.handle({"cmd": "run_days", "days": bad})
        assert sim.run_until is None and not sim.running


def _broker_up() -> bool:
    try:
        socket.create_connection((contract.INGEST_HOST, contract.INGEST_PORT), timeout=0.5).close()
        return True
    except OSError:
        return False


@pytest.mark.skipif(not _broker_up(), reason="ingest broker not running")
def test_streams_over_the_broker():
    import paho.mqtt.client as mqtt

    prefix = f"commerce-test-sim/{uuid.uuid4().hex[:8]}"
    got: dict[str, list] = {"world": [], "orders": [], "clock": [], "history": [], "catalog": []}
    retained: set[str] = set()

    def on_message(client, userdata, msg):
        if not msg.payload:
            return
        if msg.topic.endswith(("/_world", "/catalog")):
            retained.add(msg.topic)
        for kind in got:
            if msg.topic.endswith(kind):
                got[kind].append(json.loads(msg.payload))

    c = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
    c.on_message = on_message
    c.connect(contract.INGEST_HOST, contract.INGEST_PORT)
    c.subscribe(f"{prefix}/#", qos=1)
    c.loop_start()
    proc = subprocess.Popen([sys.executable, "-m", "secondbrain.commerce.simulator", "--merchants", "30",
                             "--customers", "3000", "--speed", "1day", "--autostart"],
                            env={**os.environ, "SB_COMMERCE_PREFIX": prefix},
                            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    try:
        deadline = time.time() + 20
        while time.time() < deadline and not (len(got["orders"]) > 20 and any(k["running"] for k in got["clock"])):
            time.sleep(0.1)
    finally:
        proc.terminate()
        proc.wait(timeout=10)
        c.unsubscribe(f"{prefix}/#")
        for topic in retained:  # leave nothing behind on the shared broker
            c.publish(topic, b"", qos=1, retain=True).wait_for_publish(5)
        c.loop_stop()
        c.disconnect()
    assert proc.returncode in (0, -15), proc.stderr.read().decode()[-2000:]
    assert len(got["world"]) == 1 and len(got["world"][0]["merchants"]) == 30
    assert len(got["catalog"]) == 30 and len(got["history"]) == 30
    assert got["orders"] and got["orders"][0]["orders"][0]["sku"]
    assert any(k["running"] and k["speed"] > 0 for k in got["clock"])


def test_quiet_real_time_ticks_dont_crash(sim):
    """At real time a 100 ms tick often has no orders at all, at night none for hours."""
    sim.handle({"cmd": "start"})
    for _ in range(50):
        sim.advance(0.1)  # 3am: nobody orders
    assert sim.clock_payload()["running"]
