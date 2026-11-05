"""The brain judging itself from recorded data: outcomes of its warnings and
grades for its forecasts, the same whether the sales are real or simulated."""

from datetime import date, datetime, timedelta, timezone

import numpy as np
import psycopg
import pytest
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

from scripts.generate_lumen_sales import PRODUCTS
from secondbrain import evolution
from secondbrain.forecasting import metrics, store
from secondbrain.forecasting.__main__ import run as run_forecasts
from secondbrain.models import build_models
from tests.test_forecasting import history, load_acme_sales  # noqa: F401  (fixture)

BRAIN = "root.acme.sales"
NAMES = {p.sku: p.name for p in PRODUCTS}
PRICES = {p.sku: p.price for p in PRODUCTS}


def test_grading_math_on_a_hand_made_example():
    days = [
        {"units": 10, "in_stock": True, "forecast": 12, "lo80": 8, "hi80": 14, "lo95": 6, "hi95": 16},
        {"units": 20, "in_stock": True, "forecast": 15, "lo80": 12, "hi80": 18, "lo95": 10, "hi95": 21},
        # Out of stock: zero sales say nothing about demand, so the day is left out.
        {"units": 0, "in_stock": False, "forecast": 15, "lo80": 12, "hi80": 18, "lo95": 10, "hi95": 21},
        {"units": 30, "in_stock": True, "forecast": 27, "lo80": 20, "hi80": 34, "lo95": 15, "hi95": 40},
    ]
    g = evolution.grade(days)
    assert (g["days"], g["censored_days"], g["actual"], g["forecast"], g["abs_error"]) == (3, 1, 60, 54, 10)
    assert g["wape"] == pytest.approx(10 / 60) and g["bias"] == pytest.approx(-6 / 60)
    assert g["coverage80"] == pytest.approx(2 / 3) and g["coverage95"] == 1.0
    # Same definitions as the backtest's metrics, on the in-stock days.
    actual, forecast = np.array([10.0, 20.0, 30.0]), np.array([12.0, 15.0, 27.0])
    assert g["wape"] == pytest.approx(metrics.wape(actual, forecast))
    assert g["bias"] == pytest.approx(metrics.bias(actual, forecast))

    text = evolution.describe_accuracy(g)
    assert text.startswith("Forecasts were off by 17% from real sales over 3 graded product-days (running 10% low)")
    assert "67% of days fell inside the 80% range and 100% inside the 95% range" in text
    assert "1 out-of-stock day(s) were left out" in text
    assert evolution.grade([])["wape"] is None and evolution.describe_accuracy(None) is None


def add_sales(conn, sku: str, first: date, last: date, units: int | dict, in_stock=True) -> None:
    """Sales as ingestion would record them; ``in_stock`` may be a set of out-of-stock days' complement."""
    day = first
    while day <= last:
        n = units[day] if isinstance(units, dict) else units
        ok = in_stock(day) if callable(in_stock) else in_stock
        conn.execute(
            """INSERT INTO sales_daily (brain_path, sku, day, units, revenue, price, discount, in_stock, sensitivity)
               SELECT %s::ltree, %s, %s, %s, %s, base_price, 0, %s, sensitivity FROM products
               WHERE brain_path = %s::ltree AND sku = %s""",
            (BRAIN, sku, day, n, n * PRICES[sku], ok, BRAIN, sku))
        day += timedelta(days=1)


def add_warning(conn, sku: str, status: str, as_of: date, runs_out: date, created_at: datetime) -> str:
    details = {"brain": BRAIN, "sku": sku, "name": NAMES[sku], "price": PRICES[sku], "quantity": 100,
               "supplier": "Supplier", "lead_time_days": 14, "runs_out": runs_out.isoformat(),
               "order_by": as_of.isoformat(), "as_of": as_of.isoformat()}
    return conn.execute(
        """INSERT INTO signals (company, kind, fingerprint, title, action, reason, impact, impact_value, urgency,
                                confidence, due, evidence, details, status, created_at)
           VALUES ('root.acme', 'stockout', %s, %s, 'Order', 'Low stock', 'At risk', 1000, 'high', 0.8, %s, %s, %s, %s, %s)
           RETURNING id::text""",
        (f"stockout:{sku}", f"{NAMES[sku]} runs out on {runs_out}", as_of,
         Jsonb({"products": [{"brain": BRAIN, "sku": sku}]}), Jsonb(details), status, created_at),
    ).fetchone()[0]


def test_judges_from_recorded_sales_once_the_data_covers_the_window(client, settings, history):  # noqa: F811
    """No simulation: real-looking sales rows decide the outcomes."""
    load_acme_sales(settings, history)  # through Nov 3
    run_forecasts(settings.admin_database_url, under="root.acme", horizon=56)
    _, embedder = build_models(settings)
    t0 = datetime(2026, 11, 4, 9, tzinfo=timezone.utc)
    with psycopg.connect(settings.admin_database_url) as conn:
        acted = add_warning(conn, "halo-lamp", "acted", date(2026, 11, 4), date(2026, 11, 10), t0)
        # Nobody answered, and the same run-out was warned about again two days later: one story.
        first = add_warning(conn, "cloud-throw", "expired", date(2026, 11, 4), date(2026, 11, 10), t0)
        again = add_warning(conn, "cloud-throw", "expired", date(2026, 11, 6), date(2026, 11, 11), t0 + timedelta(days=2))
        stopped = add_warning(conn, "aria-sofa", "dismissed", date(2026, 11, 4), date(2026, 11, 8), t0)
        add_sales(conn, "halo-lamp", date(2026, 11, 4), date(2026, 11, 12), 5)
        add_sales(conn, "cloud-throw", date(2026, 11, 4), date(2026, 11, 12), 0, in_stock=lambda d: d < date(2026, 11, 9))
        add_sales(conn, "aria-sofa", date(2026, 11, 4), date(2026, 11, 5), 1)  # then its data stops

    def judge():
        with psycopg.connect(settings.admin_database_url, row_factory=dict_row) as conn:
            return evolution.judge(conn, embedder, settings, "root.acme")

    # Data through Nov 12 doesn't cover a week past the Nov 10 run-out yet: nothing is judged,
    # even though cloud-throw is visibly out of stock already.
    early = judge()
    assert early["outcomes"] == [] and early["waiting"] == []

    with psycopg.connect(settings.admin_database_url) as conn:
        add_sales(conn, "halo-lamp", date(2026, 11, 13), date(2026, 11, 18), 5)
        add_sales(conn, "cloud-throw", date(2026, 11, 13), date(2026, 11, 18), 0, in_stock=False)
    judged = judge()
    assert {o["title"].split(" runs out")[0]: o["result"] for o in judged["outcomes"]} == {
        NAMES["halo-lamp"]: "prevented", NAMES["cloud-throw"]: "happened"}
    assert judged["waiting"] == [f"{NAMES['aria-sofa']}: no sales data after Nov 5, so the warning for Nov 8 can't be judged yet"]

    with psycopg.connect(settings.admin_database_url, row_factory=dict_row) as conn:
        outcomes = {r["signal_id"]: r for r in conn.execute(
            "SELECT signal_id::text, result, value::float, evaluated_on, detail FROM signal_outcomes")}
        points = {p["day"]: p["forecast"] for p in store.latest_forecast(conn, BRAIN, "cloud-throw")["points"]}
        remembered = conn.execute("SELECT count(*) AS n FROM memories WHERE source = 'signal outcome' "
                                  "AND brain_path = %s::ltree", (BRAIN,)).fetchone()["n"]
    assert set(outcomes) == {acted, again}  # the earlier, superseded warning isn't counted twice
    assert first not in outcomes and stopped not in outcomes
    assert outcomes[acted]["value"] == 1000 and outcomes[acted]["evaluated_on"] == date(2026, 11, 18)
    # Lost sales come from the forecast on the out-of-stock days (Nov 9-18), nothing having sold.
    lost = round(sum(points[(date(2026, 11, 9) + timedelta(days=i)).isoformat()] for i in range(10)))
    assert outcomes[again]["value"] == pytest.approx(-lost * PRICES["cloud-throw"])
    assert f"ran out on Nov 9, as predicted for Nov 11; about {lost:,} sales lost" in outcomes[again]["detail"]
    assert remembered == 2

    # Judged once: running it again adds nothing.
    assert judge()["outcomes"] == []


def test_refresh_grades_forecasts_and_is_idempotent(client, settings, history):  # noqa: F811
    load_acme_sales(settings, history)
    run_forecasts(settings.admin_database_url, under="root.acme", horizon=56)
    llm, embedder = build_models(settings)
    with psycopg.connect(settings.admin_database_url, row_factory=dict_row) as conn:
        forecast = {p["day"]: p for p in store.latest_forecast(conn, BRAIN, "halo-lamp")["points"]}
        store.save_inventory(conn, BRAIN, [
            {"sku": "halo-lamp", "as_of": date(2026, 11, 14), "on_hand": 400, "on_order": 0, "order_eta": None,
             "supplier": "Lampworks", "lead_time_days": 14, "unit_cost": 150}])
        # Ten days of "real" sales; halo-lamp was out of stock on two of them.
        days = [date(2026, 11, 4) + timedelta(days=i) for i in range(10)]
        units = {d: max(0, int(forecast[d.isoformat()]["forecast"]) + (3 if d.day % 2 else -2)) for d in days}
        out = {date(2026, 11, 8), date(2026, 11, 9)}
        add_sales(conn, "halo-lamp", days[0], days[-1], units, in_stock=lambda d: d not in out)
        for sku in ("cloud-throw", "aria-sofa"):
            add_sales(conn, sku, days[0], days[-1], 2)

    first = evolution.refresh(settings, llm, embedder, "root.acme")
    assert first["data_through"] == "2026-11-13" and first["forecasted"] == [BRAIN]
    assert first["scored"] == 3 and first["signals_recalculated"]
    assert first["accuracy_text"].startswith("Forecasts were off by ")

    with psycopg.connect(settings.admin_database_url, row_factory=dict_row) as conn:
        row = conn.execute("""SELECT a.* FROM forecast_accuracy a JOIN forecasts f ON f.id = a.forecast_id
                              WHERE a.sku = 'halo-lamp' AND f.cutoff = '2026-11-03'""").fetchone()
        graded = [d for d in days if d not in out]
        actual = np.array([units[d] for d in graded], dtype=float)
        predicted = np.array([forecast[d.isoformat()]["forecast"] for d in graded])
        assert (row["days"], row["censored_days"], row["complete"]) == (8, 2, False)
        assert row["scored_through"] == date(2026, 11, 13)
        assert row["wape"] == pytest.approx(metrics.wape(actual, predicted), rel=1e-5)
        assert row["bias"] == pytest.approx(metrics.bias(actual, predicted), rel=1e-5)
        accuracy = evolution.realized_accuracy(conn, BRAIN, "halo-lamp")
        assert accuracy["days"] == 8 and accuracy["wape"] == pytest.approx(row["wape"], rel=1e-5)
        assert evolution.realized_accuracy(conn, BRAIN, "no-such-sku") is None

    def state():
        with psycopg.connect(settings.admin_database_url) as conn:
            return [conn.execute(sql).fetchall() for sql in (
                "SELECT id FROM forecasts ORDER BY id",
                "SELECT id, status FROM signals ORDER BY id",
                "SELECT * FROM forecast_accuracy ORDER BY forecast_id",
                "SELECT * FROM evolution_state",
                "SELECT count(*) FROM memories",
            )]

    before = state()
    second = evolution.refresh(settings, llm, embedder, "root.acme")
    assert (second["scored"], second["outcomes"], second["expired"], second["forecasted"],
            second["signals_recalculated"]) == (0, [], 0, [], False)
    assert state() == before
    # Ingestion can trigger the same refresh over the admin API.
    from tests.conftest import ADMIN_TOKEN

    assert client.post("/admin/refresh", json={"company": "root.acme"}).status_code == 401
    third = client.post("/admin/refresh", json={"company": "root.acme"}, headers={"X-Admin-Token": ADMIN_TOKEN}).json()
    assert third["scored"] == 0 and not third["signals_recalculated"] and state() == before

    # The Explore page's evolution tab shows the grade to people who can see the product, not to others.
    body = client.get("/brain/evolution", headers={"X-User-Id": "asha"}).json()
    assert body["accuracy"]["summary"].startswith("Forecasts were off by ")
    assert body["accuracy"]["weeks"][0]["forecasts"]["days"] >= 8
    hidden = client.get("/brain/evolution", headers={"X-User-Id": "meera"}).json()
    assert hidden["accuracy"]["recent"] is None and hidden["accuracy"]["weeks"] == []


def _acme_with_stock(settings, history):  # noqa: F811
    load_acme_sales(settings, history)
    run_forecasts(settings.admin_database_url, under="root.acme", horizon=56)
    with psycopg.connect(settings.admin_database_url) as conn:
        daily = {sku: sum(p["forecast"] for p in store.latest_forecast(conn, "root.acme.sales", sku)["points"]) / 56
                 for sku in ("halo-lamp", "cloud-throw")}
        store.save_inventory(conn, "root.acme.sales", [
            {"sku": "halo-lamp", "as_of": date(2026, 11, 4), "on_hand": int(daily["halo-lamp"] * 30), "on_order": 0,
             "order_eta": None, "supplier": "Lampworks", "lead_time_days": 14, "unit_cost": 150},
            {"sku": "cloud-throw", "as_of": date(2026, 11, 4), "on_hand": int(daily["cloud-throw"] * 25), "on_order": 0,
             "order_eta": None, "supplier": "Weavers", "lead_time_days": 14, "unit_cost": 46},
        ])


def test_following_the_brain_beats_ignoring_it_from_the_same_start(client, settings, history):  # noqa: F811
    from secondbrain import demo, signals

    _acme_with_stock(settings, history)
    llm, embedder = build_models(settings)
    signals.run(settings, llm, "root.acme")
    with psycopg.connect(settings.admin_database_url, row_factory=dict_row) as conn:
        demo.save(conn)

    ignored = evolution.advance(settings, llm, embedder, "root.acme", days=56, policy="ignore")
    with psycopg.connect(settings.admin_database_url, row_factory=dict_row) as conn:
        restored = demo.restore(conn)
        assert restored["today"] == "2026-11-04" and restored["outcome_memories_removed"] > 0
        assert conn.execute("SELECT count(*) AS n FROM signal_outcomes").fetchone()["n"] == 0
    followed = evolution.advance(settings, llm, embedder, "root.acme", days=56, policy="follow")

    assert ignored["before"] == followed["before"]
    assert ignored["after"]["came_true"] >= 1 and ignored["after"]["lost"] > 0 and ignored["acted"] == 0
    assert followed["acted"] >= 2 and followed["after"]["prevented"] >= 1
    assert followed["after"]["lost"] < ignored["after"]["lost"]
    lamp = followed["products"]["halo-lamp"]
    assert len(lamp["days"]) == len(lamp["stock"]) == len(lamp["demand"]) == 56 and lamp["arrived_on"]
    assert ignored["products"]["halo-lamp"]["out_of_stock_days"] > lamp["out_of_stock_days"]


def test_simulate_endpoint_takes_a_policy_and_reset_needs_a_snapshot(client):
    from tests.conftest import ADMIN_TOKEN

    headers = {"X-Admin-Token": ADMIN_TOKEN}
    assert client.post("/admin/simulate", json={"days": 7, "policy": "maybe"}, headers=headers).status_code == 422
    assert client.post("/admin/demo/reset", headers=headers).status_code == 409
