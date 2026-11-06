from datetime import date, timedelta

import numpy as np
import pandas as pd
import psycopg
import pytest

from scripts.generate_sales import PRODUCTS, festive_share, generate
from secondbrain.forecasting import backtest, calendar, metrics, nowcast, store
from secondbrain.forecasting.__main__ import run as run_forecasts
from secondbrain.sensitivity import Sensitivity

SKUS = ["brass-floor-lamp", "kantha-throw", "sheesham-sofa"]  # a subset keeps the tests quick


@pytest.fixture(scope="module")
def history():
    rows, promos = generate()
    return [r for r in rows if r["sku"] in SKUS], [vars(p) for p in promos]


@pytest.fixture(scope="module")
def results(history):
    rows, promos = history
    return backtest.run(rows, promos, horizon=28)


def test_indian_retail_calendar():
    assert [calendar.diwali(y) for y in (2024, 2025, 2026, 2027)] == [
        date(2024, 11, 1), date(2025, 10, 20), date(2026, 11, 8), date(2027, 10, 29)]
    assert [calendar.dhanteras(y) for y in (2024, 2025, 2026)] == [
        date(2024, 10, 29), date(2025, 10, 18), date(2026, 11, 6)]
    assert [calendar.holi(y) for y in (2025, 2026)] == [date(2025, 3, 14), date(2026, 3, 4)]
    assert calendar.raksha_bandhan(2026) == date(2026, 8, 28)
    assert calendar.festive_sale_start(2026) == date(2026, 10, 3)
    assert calendar.shipping_cutoff(2026) == date(2026, 11, 4)  # Dhanteras minus 2 days
    # Distances are to the nearest occurrence, negative before it.
    assert calendar.days_to_diwali(date(2026, 11, 5)) == -3
    assert calendar.days_to_diwali(date(2025, 10, 25)) == 5
    assert calendar.days_to_diwali(date(2026, 1, 10)) == 82  # last October's Diwali is nearer
    assert calendar.in_diwali_peak(date(2025, 10, 18)) and not calendar.in_diwali_peak(date(2025, 11, 5))
    assert calendar.is_wedding_season(date(2026, 11, 20)) and calendar.is_wedding_season(date(2027, 2, 10))
    assert not calendar.is_wedding_season(date(2026, 7, 1))
    # A year back is 52 weeks, except around Diwali: the same distance from last year's Diwali.
    assert calendar.same_day_last_year(date(2026, 11, 5)) == date(2025, 10, 17)
    assert calendar.same_day_last_year(date(2026, 7, 6)) == date(2025, 7, 7)


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
    units = lambda sku: {r["day"]: r["units"] for r in rows if r["sku"] == sku}
    mean = lambda series, first, last: np.mean([u for d, u in series.items() if first <= d <= last])
    lamp, diya, sofa = units("brass-floor-lamp"), units("diya-candle-set"), units("sheesham-sofa")
    # Brass is bought on Dhanteras; Diwali itself is quiet.
    september = mean(lamp, date(2025, 9, 1), date(2025, 9, 20))
    assert lamp[calendar.dhanteras(2025)] > 2.5 * september
    assert lamp[calendar.diwali(2025)] < 0.5 * lamp[calendar.dhanteras(2025)]
    # Gifting builds through the festive season, then the lull after Diwali.
    assert mean(diya, date(2025, 10, 21), date(2025, 10, 30)) < 0.6 * mean(diya, date(2025, 10, 8), date(2025, 10, 16))
    # Furniture sells before Diwali, not in the monsoon.
    assert mean(sofa, date(2025, 9, 22), date(2025, 10, 9)) > 1.3 * mean(sofa, date(2025, 6, 15), date(2025, 7, 31))
    # Sunday is the busiest day.
    throw = units("kantha-throw")
    by_weekday = [np.mean([u for d, u in throw.items() if d.weekday() == i]) for i in range(7)]
    assert int(np.argmax(by_weekday)) == 6
    assert 0.1 < festive_share(rows, 2025) < 0.25
    sofa_stockout = [r for r in rows if r["sku"] == "sheesham-sofa" and not r["in_stock"]]
    assert sofa_stockout and all(r["units"] == 0 for r in sofa_stockout)


def test_backtest_windows_include_last_years_season():
    plan = backtest.windows(date(2026, 11, 5), date(2024, 9, 1), 56)
    labels = [w.label for w in plan]
    assert labels == ["recent 1", "recent 2", "recent 3", "same weeks last year"]
    # Aligned on Diwali: three days before Dhanteras 2025, so the window holds last year's peak.
    assert plan[-1].origin == date(2025, 10, 17)
    # A long horizon: the year-ago window replaces the recent one it overlaps.
    assert [w.label for w in backtest.windows(date(2026, 11, 5), date(2024, 9, 1), 120)] == [
        "recent 1", "recent 2", "same weeks last year"]
    # Too little history for a window a year back: only recent ones remain.
    assert all(w.label.startswith("recent") for w in backtest.windows(date(2026, 11, 5), date(2026, 1, 1), 28))


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


def test_forecast_has_intervals_and_sees_diwali_coming(results, history):
    lamp = next(r for r in results if r.sku == "brass-floor-lamp")
    assert len(lamp.points) == 28
    assert lamp.points[0]["day"] == "2026-11-06"  # Dhanteras
    for p in lamp.points:
        assert p["lo95"] <= p["lo80"] <= p["forecast"] <= p["hi80"] <= p["hi95"]
    # Whatever model won, it must see the Dhanteras peak and the lull after Diwali.
    for r in results:
        if r.sku == "sheesham-sofa":
            continue  # furniture sells before Diwali, not on it
        by_day = {p["day"]: p["forecast"] for p in r.points}
        after = np.mean([by_day[f"2026-11-{d}"] for d in range(10, 17)])
        assert by_day["2026-11-06"] > 1.8 * after, r.sku
        # Dhanteras stands above the busy festive weeks before it, not just above the lull after.
        recent = np.mean([x["units"] for x in history[0] if x["sku"] == r.sku and x["day"] > date(2026, 10, 22)])
        assert by_day["2026-11-06"] > recent, r.sku


def test_nowcast_moves_the_rest_of_the_forecast():
    points = [{"day": (date(2026, 11, 6) + timedelta(days=i)).isoformat(),
               "forecast": 10.0, "lo80": 8.0, "hi80": 12.0} for i in range(10)]
    unchanged, none = nowcast.adjust(points, {})
    assert none.factor == 1.0 and unchanged == points
    actuals = {date(2026, 11, 6) + timedelta(days=i): 20 for i in range(7)}
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
    missing = client.get("/forecast", params={"product": "brass-floor-lamp"}, headers={"X-User-Id": "asha"})
    assert missing.status_code == 404 and "No forecast" in missing.json()["detail"]

    run_forecasts(settings.admin_database_url, under="root.acme", horizon=28)

    response = client.get("/forecast", params={"product": "Moradabad", "horizon": 14},
                          headers={"X-User-Id": "asha"})
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["product"] == {"brain": "root.acme.sales", "sku": "brass-floor-lamp", "name": "Moradabad brass floor lamp"}
    assert len(body["forecast"]) == 14 and body["trained_through"] == "2026-11-05"
    assert body["model"] in body["backtest"] and body["reason"]
    assert body["adjustment"]["factor"] == 1.0
    quality = body["quality"]
    assert quality["grade"] in ("good", "fair", "weak") and quality["summary"]
    assert 0 < quality["coverage80"] <= 1 and quality["history_days"] > 700 and quality["warnings"] == []
    assert len(body["history"]) == 28

    products = client.get("/forecast/products", headers={"X-User-Id": "ravi"}).json()
    assert {p["sku"] for p in products} == set(SKUS)

    # Finance, HR and another company can't see Sales' numbers.
    for user in ("meera", "sara", "lee"):
        hidden = client.get("/forecast", params={"product": "brass-floor-lamp"}, headers={"X-User-Id": user})
        assert hidden.status_code == 404
        assert client.get("/forecast/products", headers={"X-User-Id": user}).json() == []
    assert client.get("/forecast", params={"product": "brass-floor-lamp"},
                      headers={"X-User-Id": "nobody"}).status_code == 401
    too_far = client.get("/forecast", params={"product": "brass-floor-lamp", "horizon": 60},
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


# ---------- messy real-world data ----------

from secondbrain.forecasting import explain, intervals, prepare  # noqa: E402

END = date(2026, 11, 5)


def product_rows(sku, first, mean, seed, end=END, **days):
    """Daily rows for one product: ``mean`` units a day with a weekly rhythm and Poisson noise.

    days: stockout={day,...}, missing={day,...}, extra={day: units}.
    """
    rng = np.random.default_rng(seed)
    rows, day = [], first
    while day <= end:
        units = int(rng.poisson(mean * (1.2 if day.weekday() >= 5 else 0.92)))
        in_stock = day not in days.get("stockout", set())
        if day not in days.get("missing", set()):
            rows.append({"sku": sku, "day": day, "units": (units if in_stock else 0) + days.get("extra", {}).get(day, 0),
                         "discount": 0.0, "in_stock": in_stock})
        day += timedelta(days=1)
    return rows


def assert_valid(points, horizon):
    assert len(points) == horizon
    for p in points:
        values = [p["lo95"], p["lo80"], p["forecast"], p["hi80"], p["hi95"]]
        assert all(np.isfinite(v) and v >= 0 for v in values), p
        assert values == sorted(values), p


@pytest.fixture(scope="module")
def messy():
    """A small shop with a regular seller, two new products, a slow mover and a discontinued one."""
    rows = [r for r in generate()[0] if r["sku"] == "brass-floor-lamp"]
    rows += product_rows("new-3w", END - timedelta(days=20), 12, 1)
    rows += product_rows("new-3m", END - timedelta(days=89), 20, 2)
    rows += [r if r["units"] == 0 else {**r, "units": int(r["units"] * 2)}
             for r in product_rows("slow", date(2025, 6, 1), 0.15, 3)]
    rows += [r if r["day"] < END - timedelta(days=60) else {**r, "units": 0}
             for r in product_rows("gone", date(2025, 6, 1), 9, 4)]
    results = backtest.run(rows, [], horizon=28)
    return {r.sku: r for r in results}


def test_every_forecast_is_finite_ordered_and_explained(messy):
    assert set(messy) == {"brass-floor-lamp", "new-3w", "new-3m", "slow", "gone"}
    for r in messy.values():
        assert_valid(r.points, 28)
        assert r.model in r.scores and r.reason
        q = r.quality
        assert q["grade"] in ("good", "fair", "weak") and q["summary"] and isinstance(q["warnings"], list)
        assert {"history_days", "wape", "coverage80", "coverage95", "pattern"} <= set(q)


def test_short_histories_use_only_models_that_fit(messy):
    new = messy["new-3w"]
    assert new.model == "cold_start" and new.quality["grade"] == "weak"
    level = np.mean([p["forecast"] for p in new.points])
    assert 6 < level < 24  # its recent average of ~12 a day, shaped by the shop
    three_months = messy["new-3m"]
    assert "year_over_year" not in three_months.scores and three_months.model != "cold_start"
    assert three_months.quality["grade"] != "good"
    # A new product must not knock gradient boosting out for everyone else.
    assert "gradient_boosting" in messy["brass-floor-lamp"].scores


def test_intermittent_products_are_chosen_by_squared_error(messy):
    slow = messy["slow"]
    assert slow.quality["pattern"] == "intermittent"
    assert {"croston", "tsb", "adida"} <= set(slow.scores)
    rmsse = {m: s["rmsse"] for m, s in slow.scores.items()}
    assert rmsse[slow.model] == min(rmsse.values())
    # The average rate (~0.3 a day), not the zero a median-loving metric would pick.
    assert 0.1 < np.mean([p["forecast"] for p in slow.points]) < 0.8


def test_discontinued_products_are_not_forecast_forever(messy):
    gone = messy["gone"]
    assert gone.model == "discontinued" and all(p["forecast"] == 0 for p in gone.points)
    assert gone.quality["grade"] == "weak" and "discontinued" in gone.quality["summary"]
    # Leaving the feed counts the same as reporting zeros.
    rows = product_rows("a", date(2025, 6, 1), 9, 5) + [r for r in product_rows("b", date(2025, 6, 1), 9, 6)
                                                        if r["day"] < END - timedelta(days=60)]
    left = {r.sku: r for r in backtest.run(rows, [], horizon=14)}
    assert left["b"].model == "discontinued" and left["a"].model != "discontinued"


def test_stockouts_are_censored_demand_not_zero_sales():
    rng = np.random.default_rng(9)
    out = {date(2025, 6, 1) + timedelta(days=int(d)) for d in rng.choice(520, 160, replace=False)}
    out |= {END - timedelta(days=i) for i in range(10)}  # and sold out right now
    rows = product_rows("x", date(2025, 6, 1), 20, 7, stockout=out)
    data = prepare.panel(rows)
    assert not data.loc[~data["in_stock"], "known"].any()
    assert data.loc[~data["in_stock"], "demand"].mean() > 15  # filled, not zero
    result = backtest.run(rows, [], horizon=28)[0]
    assert 16 < np.mean([p["forecast"] for p in result.points]) < 25
    assert any(w.startswith("Out of stock for the last") for w in result.quality["warnings"])
    # The same zeros recorded as in-stock days would drag the forecast down.
    naive = backtest.run([{**r, "in_stock": True} for r in rows], [], horizon=28)[0]
    assert np.mean([p["forecast"] for p in naive.points]) < np.mean([p["forecast"] for p in result.points])


def test_gaps_and_bulk_orders_are_set_aside():
    first = date(2025, 6, 1)
    outage = {date(2026, 3, d) for d in range(1, 8)}
    missing = {date(2026, 5, 5), date(2026, 5, 6)}
    a = product_rows("a", first, 30, 1, missing=outage | missing, extra={date(2026, 9, 1): 500})
    b = product_rows("b", first, 2, 2, missing=outage)  # reports its zero days
    sparse = [r for r in product_rows("c", first, 0.5, 3, missing=outage) if r["units"] > 0]  # omits zero days
    data = prepare.panel(a + b + sparse)
    day = lambda sku, d: data[(data["sku"] == sku) & (data["day"] == pd.Timestamp(d))].iloc[0]
    assert not day("a", date(2026, 3, 3))["observed"]  # the whole feed was down
    assert not day("a", date(2026, 5, 5))["observed"]  # nearly every day is there, so a gap is unknown
    sold = {r["day"] for r in sparse}
    quiet = next(d for d in (min(sold) + timedelta(days=i) for i in range(300)) if d not in outage | sold)
    assert day("c", quiet)["observed"] and day("c", quiet)["units"] == 0  # a feed without zero rows
    spike = day("a", date(2026, 9, 1))
    assert spike["outlier"] and not spike["known"] and spike["demand"] < 100
    assert data["day"].groupby(data["sku"]).diff().dropna().dt.days.eq(1).all()  # no holes left
    result = {r.sku: r for r in backtest.run(a + b + sparse, [], horizon=28)}
    assert 22 < np.mean([p["forecast"] for p in result["a"].points]) < 40
    assert any("one-off spike" in w for w in result["a"].quality["warnings"])


def test_promotion_and_diwali_spikes_are_not_outliers():
    rows = [r for r in generate()[0] if r["sku"] in ("diya-candle-set", "brass-floor-lamp")]
    data = prepare.panel(rows)
    assert not data["outlier"].any()


def test_intervals_hold_about_as_often_as_they_claim():
    rng = np.random.default_rng(1)
    frames = []
    for w in range(4):
        f = rng.uniform(2, 60, 120)
        y = rng.poisson(rng.gamma(10, f / 10))
        frames.append(pd.DataFrame({"window": f"w{w}", "step": np.arange(1, 121), "day": pd.NaT,
                                    "units": y.astype(float), "forecast": f}))
    cal = intervals.calibrate(pd.concat(frames, ignore_index=True))
    assert 0.75 <= cal.coverage[80] <= 0.88 and 0.92 <= cal.coverage[95] <= 0.99
    # Fresh data from the same process lands inside about as often.
    hits = []
    for _ in range(20):
        f = rng.uniform(2, 60, 120)
        y = rng.poisson(rng.gamma(10, f / 10))
        pts = intervals.points(pd.date_range("2027-01-01", periods=120), f, cal)
        hits += [p["lo80"] <= v <= p["hi80"] for p, v in zip(pts, y)]
    assert 0.74 <= np.mean(hits) <= 0.88
    # Garbage in, still valid points out.
    assert_valid(intervals.points(pd.date_range("2027-01-01", periods=3), np.array([np.nan, -4, np.inf]), cal), 3)


def test_drivers_skip_last_year_when_there_is_none():
    cutoff = date(2026, 11, 3)
    sales = [{"day": cutoff - timedelta(days=i), "units": 10, "in_stock": True} for i in range(60)]
    points = [{"day": (cutoff + timedelta(days=i)).isoformat(), "forecast": 10.0} for i in range(1, 29)]
    names = [d["name"] for d in explain.drivers(sales, [], cutoff, points)]
    assert "Recent level" in names
    assert "Same weeks last year" not in names and "Growth" not in names and "Diwali effect" not in names
    assert [d["name"] for d in explain.drivers([], [], cutoff, [])] == ["Recent level"]


def test_drivers_measure_last_diwali_and_name_this_years():
    cutoff = date(2026, 11, 5)
    sales = [{"day": r["day"], "units": r["units"], "in_stock": r["in_stock"]}
             for r in generate()[0] if r["sku"] == "diya-candle-set" and r["day"] > cutoff - timedelta(days=550)]
    points = [{"day": (cutoff + timedelta(days=i)).isoformat(), "forecast": 100.0} for i in range(1, 29)]
    diwali = next(d for d in explain.drivers(sales, [], cutoff, points) if d["name"] == "Diwali effect")
    assert float(diwali["value"].split("x")[0]) > 1.2  # against festive-sale weeks, already busy
    assert "Dhanteras to Diwali 2025 (Oct 18–Oct 20)" in diwali["explanation"]
    assert "This year Dhanteras is Nov 6 and Diwali Nov 8" in diwali["explanation"]
