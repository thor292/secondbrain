from datetime import date, timedelta

import numpy as np
import psycopg
import pytest

from scripts.generate_lumen_sales import PRODUCTS, generate, holiday_share
from secondbrain.forecasting import backtest, calendar, metrics, nowcast, store
from secondbrain.forecasting.__main__ import run as run_forecasts
from secondbrain.sensitivity import Sensitivity

SKUS = ["halo-lamp", "cloud-throw", "aria-sofa"]  # a subset keeps the tests quick


@pytest.fixture(scope="module")
def history():
    rows, promos = generate()
    return [r for r in rows if r["sku"] in SKUS], [vars(p) for p in promos]


@pytest.fixture(scope="module")
def results(history):
    rows, promos = history
    return backtest.run(rows, promos, horizon=28)


def test_retail_calendar():
    assert [calendar.black_friday(y) for y in (2024, 2025, 2026)] == [
        date(2024, 11, 29), date(2025, 11, 28), date(2026, 11, 27)]
    assert calendar.cyber_monday(2026) == date(2026, 11, 30)
    assert calendar.memorial_day(2026) == date(2026, 5, 25)
    assert calendar.labor_day(2026) == date(2026, 9, 7)


def test_metrics():
    actual, forecast = np.array([10.0, 0.0, 30.0]), np.array([12.0, 1.0, 27.0])
    assert metrics.wape(actual, forecast) == pytest.approx(6 / 40)
    assert metrics.bias(actual, forecast) == pytest.approx(0.0)
    assert metrics.mase(actual, forecast, np.array([1.0, 3.0, 1.0, 3.0]), season=1) == pytest.approx(1.0)
    assert metrics.smape(np.zeros(2), np.zeros(2)) == 0.0


def test_generator_is_deterministic_and_seasonal():
    rows, _ = generate()
    again, _ = generate()
    assert rows == again
    assert {r["sku"] for r in rows} == {p.sku for p in PRODUCTS}
    lamp = {r["day"]: r["units"] for r in rows if r["sku"] == "halo-lamp"}
    october = np.mean([lamp[date(2025, 10, d)] for d in range(1, 29)])
    assert lamp[calendar.black_friday(2025)] > 2.5 * october
    assert 0.15 < holiday_share(rows, 2025) < 0.35
    sofa_stockout = [r for r in rows if r["sku"] == "aria-sofa" and not r["in_stock"]]
    assert sofa_stockout and all(r["units"] == 0 for r in sofa_stockout)


def test_backtest_windows_include_last_years_season():
    plan = backtest.windows(date(2026, 11, 3), date(2024, 9, 1), 56)
    labels = [w.label for w in plan]
    assert labels == ["recent 1", "recent 2", "recent 3", "same weeks last year"]
    assert plan[-1].origin == date(2025, 11, 4)
    # Too little history for a window a year back: only recent ones remain.
    assert all(w.label.startswith("recent") for w in backtest.windows(date(2026, 11, 3), date(2026, 1, 1), 28))


def test_backtest_picks_a_model_per_product(results):
    assert [r.sku for r in results] == sorted(SKUS)
    for r in results:
        assert set(r.scores) == {"seasonal_naive", "year_over_year", "ets", "gradient_boosting"}
        best = r.scores[r.model]["wape"]
        assert best == min(s["wape"] for s in r.scores.values())
        # The chosen model must beat repeating last week.
        assert best < r.scores["seasonal_naive"]["wape"]
        assert r.model in r.reason and "same weeks last year" in r.reason
        assert len(r.scores[r.model]["windows"]) == 4


def test_forecast_has_intervals_and_sees_black_friday_coming(results):
    lamp = next(r for r in results if r.sku == "halo-lamp")
    assert len(lamp.points) == 28
    assert lamp.points[0]["day"] == "2026-11-04"
    for p in lamp.points:
        assert p["lo95"] <= p["lo80"] <= p["forecast"] <= p["hi80"] <= p["hi95"]
    by_day = {p["day"]: p["forecast"] for p in lamp.points}
    if lamp.model in ("gradient_boosting", "year_over_year"):
        assert by_day["2026-11-27"] > 1.5 * by_day["2026-11-13"]


def test_nowcast_moves_the_rest_of_the_forecast():
    points = [{"day": (date(2026, 11, 4) + timedelta(days=i)).isoformat(),
               "forecast": 10.0, "lo80": 8.0, "hi80": 12.0} for i in range(10)]
    unchanged, none = nowcast.adjust(points, {})
    assert none.factor == 1.0 and unchanged == points
    actuals = {date(2026, 11, 4) + timedelta(days=i): 20 for i in range(7)}
    ahead, adj = nowcast.adjust(points, actuals)
    assert adj.days_observed == 7 and adj.factor == pytest.approx(1.5)
    assert len(ahead) == 3 and ahead[0]["forecast"] == 15.0 and ahead[0]["hi80"] == 18.0


def load_acme_sales(settings, history, sensitivity=Sensitivity.INTERNAL):
    rows, promos = history
    with psycopg.connect(settings.admin_database_url) as conn:
        store.save_history(conn, "root.acme.sales", sensitivity,
                           [(p.sku, p.name, p.category, p.price) for p in PRODUCTS if p.sku in SKUS],
                           rows, promos)


def test_forecast_api_respects_permissions(client, settings, history):
    load_acme_sales(settings, history)
    missing = client.get("/forecast", params={"product": "halo-lamp"}, headers={"X-User-Id": "asha"})
    assert missing.status_code == 404 and "No forecast" in missing.json()["detail"]

    run_forecasts(settings.admin_database_url, under="root.acme", horizon=28)

    response = client.get("/forecast", params={"product": "Halo", "horizon": 14},
                          headers={"X-User-Id": "asha"})
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["product"] == {"brain": "root.acme.sales", "sku": "halo-lamp", "name": "Halo brass floor lamp"}
    assert len(body["forecast"]) == 14 and body["trained_through"] == "2026-11-03"
    assert body["model"] in body["backtest"] and body["reason"]
    assert body["adjustment"]["factor"] == 1.0
    assert len(body["history"]) == 28

    products = client.get("/forecast/products", headers={"X-User-Id": "ravi"}).json()
    assert {p["sku"] for p in products} == set(SKUS)

    # Finance, HR and another company can't see Sales' numbers.
    for user in ("meera", "sara", "lee"):
        hidden = client.get("/forecast", params={"product": "halo-lamp"}, headers={"X-User-Id": user})
        assert hidden.status_code == 404
        assert client.get("/forecast/products", headers={"X-User-Id": user}).json() == []
    assert client.get("/forecast", params={"product": "halo-lamp"},
                      headers={"X-User-Id": "nobody"}).status_code == 401
    too_far = client.get("/forecast", params={"product": "halo-lamp", "horizon": 60},
                         headers={"X-User-Id": "asha"})
    assert too_far.status_code == 400


def test_restricted_sales_need_a_restricted_grant(client, settings, history):
    load_acme_sales(settings, history, Sensitivity.RESTRICTED)
    asha = client.get("/forecast/products", headers={"X-User-Id": "asha"}).json()
    vikram = client.get("/forecast/products", headers={"X-User-Id": "vikram"}).json()
    assert asha == [] and len(vikram) == len(SKUS)


def test_app_role_cannot_write_sales(client, settings):
    with psycopg.connect(settings.app_database_url) as conn:
        with pytest.raises(psycopg.errors.InsufficientPrivilege):
            conn.execute("DELETE FROM sales_daily")
