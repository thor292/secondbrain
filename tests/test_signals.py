from datetime import date, timedelta

import psycopg

from secondbrain import signals
from secondbrain.forecasting import store
from secondbrain.forecasting.__main__ import run as run_forecasts
from tests.test_forecasting import history, load_acme_sales  # noqa: F401  (fixture)


def test_projected_stock_adds_incoming_order_on_arrival():
    points = [{"day": (date(2026, 11, 4) + timedelta(days=i)).isoformat(), "forecast": 10.0} for i in range(5)]
    projection = signals.project_stock(25, 50, date(2026, 11, 6), points)
    assert [round(s) for _, s in projection] == [15, 5, 45, 35, 25]


def setup_acme(client, settings, history):  # noqa: F811
    load_acme_sales(settings, history)
    run_forecasts(settings.admin_database_url, under="root.acme", horizon=56)
    with psycopg.connect(settings.admin_database_url) as conn:
        daily = {}
        for sku in ("halo-lamp", "cloud-throw"):
            points = store.latest_forecast(conn, "root.acme.sales", sku)["points"]
            daily[sku] = sum(p["forecast"] for p in points) / len(points)
        store.save_inventory(conn, "root.acme.sales", [
            # Runs out within days, long lead time: must be ordered at once.
            {"sku": "halo-lamp", "as_of": date(2026, 11, 4), "on_hand": int(daily["halo-lamp"] * 5), "on_order": 0,
             "order_eta": None, "supplier": "Lampworks", "lead_time_days": 28, "unit_cost": 150},
            {"sku": "aria-sofa", "as_of": date(2026, 11, 4), "on_hand": 0, "on_order": 50,
             "order_eta": date(2026, 12, 2), "supplier": "Sofa Co", "lead_time_days": 42, "unit_cost": 980},
            {"sku": "cloud-throw", "as_of": date(2026, 11, 4), "on_hand": int(daily["cloud-throw"] * 400), "on_order": 0,
             "order_eta": None, "supplier": "Weavers", "lead_time_days": 21, "unit_cost": 46},
        ])
    # A note in Sales for the business signals to cite.
    client.post("/remember", json={"content": "Bluebird Stores wants a bulk order of throws before December"},
                headers={"X-User-Id": "asha"})
    client.post("/consolidate", headers={"X-User-Id": "asha"})


def get_signals(client, user, **params):
    response = client.get("/signals", params=params, headers={"X-User-Id": user})
    assert response.status_code == 200, response.text
    return response.json()


def test_stock_and_business_signals_with_permissions_and_feedback(client, settings, history):  # noqa: F811
    from secondbrain.models import build_models

    setup_acme(client, settings, history)
    llm, _ = build_models(settings)
    signals.run(settings, llm, "root.acme")

    asha = {s["kind"]: s for s in get_signals(client, "asha")}
    assert set(asha) == {"stockout", "backorder", "excess_stock", "business"}

    stockout = asha["stockout"]
    assert stockout["title"].startswith("Halo brass floor lamp runs out on")
    assert "today and ask them to expedite" in stockout["action"]  # 5 days of stock, 28-day lead time
    assert stockout["urgency"] == "high" and stockout["impact_value"] > 0
    assert stockout["evidence"]["products"] == [{"brain": "root.acme.sales", "sku": "halo-lamp"}]

    assert asha["backorder"]["title"] == "Aria linen sofa is out of stock until Dec 2"
    assert "pre-orders" in asha["backorder"]["action"]
    assert "days of stock" in asha["excess_stock"]["title"] and asha["excess_stock"]["urgency"] == "low"

    # Only the signal that cites evidence survives; it cites Asha's Sales note.
    business = [s for s in get_signals(client, "asha") if s["kind"] == "business"]
    assert len(business) == 1
    assert "Bluebird" in business[0]["evidence"]["notes"][0]["content"]

    # Finance and HR can't see Sales' products or notes; another company sees nothing of Acme's.
    for user in ("meera", "sara", "lee"):
        assert [s for s in get_signals(client, user) if s["company"] == "root.acme" and s["kind"] != "business"] == []
        assert not any(s["kind"] == "business" and s["evidence"].get("memories") for s in get_signals(client, user))
    assert client.get("/signals", headers={"X-User-Id": "nobody"}).status_code == 401

    # People who can't see a signal can't act on it either.
    assert client.post(f"/signals/{stockout['id']}/feedback", json={"status": "acted"},
                       headers={"X-User-Id": "meera"}).status_code == 404
    assert client.post(f"/signals/{stockout['id']}/feedback", json={"status": "acted", "note": "ordered 600"},
                       headers={"X-User-Id": "asha"}).json() == {"ok": True}

    # Acted-on signals aren't raised again on the next run, and stay in the history.
    signals.run(settings, llm, "root.acme")
    assert "stockout" not in {s["kind"] for s in get_signals(client, "asha")}
    closed = [s for s in get_signals(client, "asha", include_closed=True) if s["kind"] == "stockout"]
    assert closed[0]["status"] == "acted" and closed[0]["feedback_by"] == "asha" and closed[0]["feedback_note"] == "ordered 600"


def test_app_role_cannot_create_signals(client, settings):
    with psycopg.connect(settings.app_database_url) as conn:
        conn.execute("SELECT set_config('app.user_id', 'asha', true)")
        try:
            conn.execute("INSERT INTO signals (company, kind, fingerprint, title, action, reason, impact, urgency, "
                         "confidence, evidence) VALUES ('root.acme', 'business', 'x', 'x', 'x', 'x', 'x', 'low', 0.5, '{}')")
            raised = False
        except psycopg.errors.InsufficientPrivilege:
            raised = True
    assert raised


def test_time_passing_judges_outcomes_and_the_brain_remembers(client, settings, history):  # noqa: F811
    from secondbrain import evolution
    from secondbrain.models import build_models

    load_acme_sales(settings, history)
    run_forecasts(settings.admin_database_url, under="root.acme", horizon=56)
    with psycopg.connect(settings.admin_database_url) as conn:
        daily = {sku: sum(p["forecast"] for p in store.latest_forecast(conn, "root.acme.sales", sku)["points"]) / 56
                 for sku in ("halo-lamp", "cloud-throw")}
        store.save_inventory(conn, "root.acme.sales", [
            # Runs out in ~40 days; a 14-day lead time leaves time to reorder.
            {"sku": "halo-lamp", "as_of": date(2026, 11, 4), "on_hand": int(daily["halo-lamp"] * 40), "on_order": 0,
             "order_eta": None, "supplier": "Lampworks", "lead_time_days": 14, "unit_cost": 150},
            # Runs out in ~20 days and will be ignored.
            {"sku": "cloud-throw", "as_of": date(2026, 11, 4), "on_hand": int(daily["cloud-throw"] * 20), "on_order": 0,
             "order_eta": None, "supplier": "Weavers", "lead_time_days": 21, "unit_cost": 46},
        ])
    llm, embedder = build_models(settings)
    signals.run(settings, llm, "root.acme")
    by_sku = {s["evidence"]["products"][0]["sku"]: s for s in get_signals(client, "ravi") if s["kind"] == "stockout"}
    client.post(f"/signals/{by_sku['halo-lamp']['id']}/feedback", json={"status": "acted"}, headers={"X-User-Id": "ravi"})
    client.post(f"/signals/{by_sku['cloud-throw']['id']}/feedback", json={"status": "dismissed", "note": "not a priority"},
                headers={"X-User-Id": "ravi"})

    result = evolution.advance(settings, llm, embedder, "root.acme", days=45)
    outcomes = {o["title"].split(" runs out")[0]: o["result"] for o in result["outcomes"]}
    assert outcomes == {"Halo brass floor lamp": "prevented", "Cloud waffle throw": "happened"}
    assert any("on order arrived" in a for a in result["products"]["halo-lamp"]["arrivals"])
    assert result["products"]["cloud-throw"]["out_of_stock_days"] > 0

    closed = {s["evidence"]["products"][0]["sku"]: s for s in get_signals(client, "ravi", include_closed=True)
              if s["kind"] == "stockout" and s["status"] != "open"}
    assert closed["halo-lamp"]["outcome"] == "prevented" and closed["cloud-throw"]["outcome"] == "happened"

    # The brain remembers what happened, visible to the people who can see the product.
    with psycopg.connect(settings.admin_database_url) as conn:
        remembered = [r[0] for r in conn.execute(
            "SELECT content FROM memories WHERE source = 'signal outcome' AND brain_path = 'root.acme.sales'")]
        as_of = conn.execute("SELECT max(as_of) FROM inventory WHERE brain_path = 'root.acme.sales'").fetchone()[0]
    assert len(remembered) == 2 and all(m.startswith("Outcome: ") for m in remembered)
    assert as_of == date(2026, 11, 4) + timedelta(days=45)
    # The order Ravi's action placed arrived once and isn't counted again.
    with psycopg.connect(settings.admin_database_url) as conn:
        on_order = conn.execute("SELECT on_order FROM inventory WHERE sku = 'halo-lamp' AND brain_path = 'root.acme.sales'").fetchone()[0]
    assert on_order == 0

    # New signals carry the track record: the one ignored warning came true.
    fresh = [s for s in get_signals(client, "ravi") if s["kind"] == "stockout"]
    for s in fresh:
        assert "Track record here: 1 of 1 ignored run-out warnings came true" in s["evidence"]["facts"]
    # Outcomes stay hidden from people who can't see the product.
    assert all(s["company"] != "root.acme" or s["kind"] == "business" for s in get_signals(client, "meera", include_closed=True))


def test_evolution_and_hq_endpoints(client):
    from tests.conftest import ADMIN_TOKEN

    body = client.get("/brain/evolution", headers={"X-User-Id": "asha"}).json()
    assert {"today", "outcomes", "feedback", "saved", "lost", "events"} <= set(body)
    assert client.get("/brain/evolution", headers={"X-User-Id": "nobody"}).status_code == 401
    assert client.get("/admin/hq").status_code == 401
    assert client.post("/admin/simulate", json={"days": 7}).status_code == 401
    assert client.post("/admin/simulate", json={"days": 500}, headers={"X-Admin-Token": ADMIN_TOKEN}).status_code == 400
    hq = client.get("/admin/hq", headers={"X-Admin-Token": ADMIN_TOKEN}).json()
    names = {c["name"] for c in hq["companies"]}
    assert {"Acme Retail", "Globex Corp"} <= names
    # Aggregates only: no company's note or document text reaches HQ.
    assert "24 days" not in str(hq["companies"]) and "18 percent" not in str(hq)
    assert "SecondBrain HQ" in client.get("/hq").text
