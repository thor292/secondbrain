"""Real data in: contract, idempotent processing, in-stock days, auth, dead letters, CSV, MQTT."""

import json
import socket
import threading
import time
from datetime import date, timedelta

import psycopg
import pytest
from fastapi.testclient import TestClient

from secondbrain import admin
from secondbrain.api import create_app
from secondbrain.ingest import contract, processor
from secondbrain.ingest.mqtt import Bridge, MqttSettings, parse_topic
from secondbrain.ingest.pipeline import RefreshScheduler
from secondbrain.schema import create_schema
from tests.conftest import ADMIN_TOKEN, EXAMPLES

ADMIN = {"X-Admin-Token": ADMIN_TOKEN}
LAMP = {"name": "Halo brass floor lamp", "category": "lighting", "price": 289}


@pytest.fixture()
def api(settings, monkeypatch):
    # Refreshing forecasts is tested on its own; here data only needs to land.
    monkeypatch.setenv("SB_INGEST_AUTO_REFRESH", "false")
    create_schema(settings, reset=True)
    with psycopg.connect(settings.admin_database_url) as conn:
        for company in ("acme", "globex"):
            admin.onboard(conn, json.loads((EXAMPLES / company / "company.json").read_text()))
    app = create_app(settings)
    with TestClient(app) as client:
        yield client


@pytest.fixture()
def db(settings, api):
    with psycopg.connect(settings.admin_database_url, autocommit=True) as conn:
        yield conn


def new_key(api, company="acme", scope=None) -> dict:
    response = api.post("/admin/ingest/keys", json={"company": company, "name": "ERP", "scope": scope}, headers=ADMIN)
    assert response.status_code == 200, response.text
    return {"Authorization": f"Bearer {response.json()['key']}"}


def msg(event_id, data, company="acme", brain="sales"):
    return {"schema_version": 1, "event_id": event_id, "company": company, "brain": brain, "data": data}


def line(order, line_id, qty, ts="2026-11-04T10:00:00-05:00", **extra):
    return {"order_id": order, "line_id": line_id, "sku": "halo-lamp", "quantity": qty, "unit_price": 289,
            "ts": ts, **extra}


def post(api, key, kind, body, status=200):
    response = api.post(f"/ingest/{kind}", json=body, headers=key)
    assert response.status_code == status, response.text
    return response.json()


def day_row(db, day, sku="halo-lamp", brain="root.acme.sales"):
    return db.execute("""SELECT units, revenue::float, price::float, discount, in_stock FROM sales_daily
                         WHERE brain_path = %s::ltree AND sku = %s AND day = %s""", (brain, sku, day)).fetchone()


def test_contract_normalizes_addresses_returns_and_discounts():
    m = contract.parse("sale", msg("e1", line("o1", "1", -2, discount_amount=57.8), company="root.acme"))
    assert (m.company, m.brain) == ("root.acme", "root.acme.sales")
    assert m.record.kind == "return" and m.record.quantity == 2 and m.record.discount == pytest.approx(0.1)
    assert m.record.business_day == date(2026, 11, 4)  # the day in the business's own offset, not UTC
    assert contract.brain_path("root.acme", "_") == "root.acme"
    assert contract.brain_path("root.acme", "sales.emea") == "root.acme.sales.emea"
    with pytest.raises(ValueError):
        contract.brain_path("root.acme", "root.globex.sales")
    with pytest.raises(ValueError, match="qty"):  # typos are errors, not silently ignored fields
        contract.parse("sale", msg("e2", {**line("o1", "1", 1), "qty": 3}))
    snap = {"sku": "x", "on_hand": 1}
    assert contract.parse("inventory", msg("e3", {**snap, "as_of": "2026-11-04"})).record.business_day == date(2026, 11, 4)
    timed = contract.parse("inventory", msg("e4", {**snap, "as_of": "2026-11-04T23:30:00-08:00"})).record
    assert timed.business_day == date(2026, 11, 4) and timed.at.tzinfo is not None
    with pytest.raises(ValueError):
        contract.parse("inventory", msg("e5", {**snap, "as_of": "2026-11-04T10:00:00"}))  # no offset


def test_sale_events_aggregate_and_replays_never_double_count(api, db):
    key = new_key(api)
    first = [msg("s1", {**line("o1", "1", 2), "product": LAMP}),
             msg("s2", line("o2", "1", 1, discount=0.2)),
             msg("s3", line("o3", "1", 4, ts="2026-11-05T09:00:00-05:00"))]
    assert post(api, key, "sale", first) == {"processed": 3, "duplicates": 0, "rejected": []}
    # The same batch again (a retry), and the same order line under a new event id.
    assert post(api, key, "sale", first)["duplicates"] == 3
    assert post(api, key, "sale", msg("s1-retry", line("o1", "1", 2)))["processed"] == 1
    # A return is recorded but doesn't reduce demand.
    assert post(api, key, "sale", msg("r1", line("o1", "1", -1, kind="return")))["processed"] == 1

    units, revenue, price, discount, in_stock = day_row(db, date(2026, 11, 4))
    assert (units, in_stock) == (3, True)
    assert revenue == pytest.approx(2 * 289 + 289 * 0.8)
    assert price == pytest.approx(revenue / 3, abs=0.01) and discount == pytest.approx(0.2 / 3)
    assert day_row(db, date(2026, 11, 5))[0] == 4
    assert db.execute("SELECT count(*) FROM sale_lines").fetchone()[0] == 4

    # A daily close sets the day's totals; gaps between days are filled with zero-sale days.
    post(api, key, "daily_sales", msg("d1", {"sku": "halo-lamp", "day": "2026-11-08", "units": 5, "revenue": 1300}))
    assert day_row(db, date(2026, 11, 8))[:3] == (5, 1300.0, 260.0)
    assert processor.fill_gaps(db, "root.acme") == 2
    assert day_row(db, date(2026, 11, 6))[:1] == (0,)


def test_in_stock_follows_snapshots_and_stockout_events(api, db):
    key = new_key(api)
    d = lambda n: date(2026, 11, n)  # noqa: E731
    first = {"sku": "halo-lamp", "product": LAMP, "supplier": "Nordlys", "lead_time_days": 28, "unit_cost": 150}

    def snap(event_id, day, on_hand, **extra):
        data = {"sku": "halo-lamp", "as_of": f"2026-11-{day:02d}T20:00:00-05:00", "on_hand": on_hand, **extra}
        result = post(api, key, "inventory", msg(event_id, data))
        assert result["rejected"] == [], result

    snap("i1", 1, 12, **first)
    post(api, key, "sale", msg("s1", line("o1", "1", 2, ts="2026-11-02T10:00:00-05:00")))
    snap("i2", 2, 0)  # ran out on the 2nd
    snap("i4", 4, 0)  # still out on the 4th
    snap("i6", 6, 30, on_order=0)  # back on the 6th: it started that day at zero
    assert [day_row(db, d(n))[4] for n in range(2, 7)] == [False] * 5
    assert day_row(db, d(2))[0] == 2  # the 2nd's sales are kept, the day just isn't full demand
    assert db.execute("SELECT on_hand, as_of FROM inventory WHERE sku = 'halo-lamp'").fetchone() == (30, d(6))

    # An older snapshot arriving late doesn't roll stock back, but its zero day still counts.
    result = post(api, key, "inventory", msg("i0", {"sku": "halo-lamp", "as_of": "2026-10-30", "on_hand": 0}))
    assert result["processed"] == 1
    assert db.execute("SELECT on_hand FROM inventory WHERE sku = 'halo-lamp'").fetchone() == (30,)
    assert day_row(db, d(1) - timedelta(days=2))[4] is False

    # Stockout events: out on the 8th, back on the 10th.
    post(api, key, "stockout", msg("o8", {"sku": "halo-lamp", "at": "2026-11-08T15:00:00-05:00"}))
    assert db.execute("SELECT on_hand FROM inventory WHERE sku = 'halo-lamp'").fetchone() == (0,)
    post(api, key, "stockout", msg("b10", {"sku": "halo-lamp", "state": "back_in_stock", "at": "2026-11-10T08:00:00-05:00"}))
    assert [day_row(db, d(n))[4] for n in (8, 9, 10)] == [False, False, False]
    post(api, key, "sale", msg("s11", line("o9", "1", 1, ts="2026-11-11T10:00:00-05:00")))
    assert day_row(db, d(11))[4] is True and day_row(db, d(7)) is None


def test_purchase_orders_and_receipts_move_stock(api, db):
    key = new_key(api)
    post(api, key, "inventory", msg("i1", {"sku": "halo-lamp", "product": LAMP, "as_of": "2026-11-01T20:00:00-05:00",
                                           "on_hand": 0, "supplier": "Nordlys", "lead_time_days": 28, "unit_cost": 150}))
    stock = lambda: db.execute("SELECT on_hand, on_order, order_eta FROM inventory").fetchone()  # noqa: E731
    po = {"po_id": "PO-1", "sku": "halo-lamp", "quantity": 100, "eta": "2026-11-20",
          "ordered_at": "2026-11-02T09:00:00-05:00"}
    post(api, key, "purchase_order", msg("po1", po))
    assert stock() == (0, 100, date(2026, 11, 20))
    post(api, key, "purchase_order", msg("po1-again", po))  # same order resent: not counted twice
    assert stock() == (0, 100, date(2026, 11, 20))
    post(api, key, "goods_receipt", msg("gr1", {"po_id": "PO-1", "sku": "halo-lamp", "quantity": 60,
                                                "received_at": "2026-11-18T11:00:00-05:00"}))
    assert stock() == (60, 40, date(2026, 11, 20))
    post(api, key, "purchase_order", msg("po1-cancel", {**po, "status": "cancelled",
                                                        "updated_at": "2026-11-19T09:00:00-05:00"}))
    assert stock() == (60, 0, None)
    # Out of stock from the 1st until the goods arrived on the 18th.
    assert day_row(db, date(2026, 11, 10))[4] is False and day_row(db, date(2026, 11, 19)) is None


def test_keys_only_write_inside_their_company_and_scope(api, db):
    acme = new_key(api)
    sales_only = new_key(api, scope="sales")
    product = {"sku": "x1", **LAMP}
    assert post(api, acme, "product", msg("p1", product, company="globex", brain="_"), status=403)
    assert post(api, sales_only, "product", msg("p2", product, brain="finance"), status=403)
    # A mixed batch is refused whole: nothing from it is written.
    post(api, acme, "product", [msg("p3", product), msg("p4", product, company="globex", brain="_")], status=403)
    assert db.execute("SELECT count(*) FROM products").fetchone() == (0,)
    assert post(api, sales_only, "product", msg("p5", product))["processed"] == 1

    assert api.post("/ingest/product", json=msg("p6", product)).status_code == 401
    assert api.post("/ingest/product", json=msg("p6", product), headers={"Authorization": "Bearer sbk_nope"}).status_code == 401
    assert api.post("/admin/ingest/keys", json={"company": "acme", "name": "x"}).status_code == 401
    key_id = api.get("/admin/ingest/keys", params={"company": "acme"}, headers=ADMIN).json()[0]["id"]
    assert api.delete(f"/admin/ingest/keys/{key_id}", headers=ADMIN).status_code == 200
    assert api.post("/ingest/product", json=msg("p7", product), headers=acme).status_code == 401
    assert api.post("/ingest/nonsense", json=msg("p8", product), headers=sales_only).status_code == 404


def test_bad_messages_are_dead_lettered_and_can_be_resent(api, db):
    key = new_key(api)
    unknown = msg("s1", line("o1", "1", 1))  # unknown SKU, no product details
    broken = msg("s2", {"order_id": "o2", "sku": "halo-lamp"})
    result = post(api, key, "sale", [unknown, broken])
    assert result["processed"] == 0 and [r["event_id"] for r in result["rejected"]] == ["s1", "s2"]
    assert "unknown sku 'halo-lamp'" in result["rejected"][0]["error"]
    assert "quantity: Field required" in result["rejected"][1]["error"]
    errors = api.get("/admin/ingest/errors", params={"company": "acme"}, headers=ADMIN).json()
    assert {e["event_id"] for e in errors} == {"s1", "s2"} and errors[0]["payload"]["company"] == "acme"
    assert db.execute("SELECT count(*) FROM sales_daily").fetchone() == (0,)

    # Fixed upstream: the product arrives, the same event is resent and now applies.
    post(api, key, "product", msg("p1", {"sku": "halo-lamp", **LAMP, "sensitivity": "restricted"}))
    assert post(api, key, "sale", unknown)["processed"] == 1
    assert db.execute("SELECT status, attempts FROM ingest_events WHERE event_id = 's1'").fetchone() == ("processed", 2)
    assert day_row(db, date(2026, 11, 4))[0] == 1
    # Sales carry the product's sensitivity, and follow it when it changes.
    level = lambda: db.execute("SELECT sensitivity FROM sales_daily").fetchone()[0]  # noqa: E731
    assert level() == 2
    post(api, key, "product", msg("p2", {"sku": "halo-lamp", **LAMP, "sensitivity": "internal"}))
    assert level() == 1


def test_csv_backfill_is_idempotent(api, db):
    key = new_key(api)

    def upload(kind, text):
        response = api.post(f"/ingest/{kind}/csv", data={"brain": "sales"}, headers=key,
                            files={"file": (f"{kind}.csv", text.encode(), "text/csv")})
        assert response.status_code == 200, response.text
        return response.json()

    products = "sku,name,category,price,sensitivity\nhalo-lamp,Halo lamp,lighting,289,internal\nrug,Jute rug,textiles,349,\n"
    assert upload("product", products)["processed"] == 2
    days = ["sku,day,units,revenue,in_stock"] + [
        f"halo-lamp,{date(2026, 10, 1) + timedelta(days=i)},{10 + i},{(10 + i) * 289},{'false' if i == 3 else 'true'}"
        for i in range(10)]
    assert upload("daily_sales", "\n".join(days))["processed"] == 10
    assert upload("daily_sales", "\n".join(days)) == {"processed": 0, "duplicates": 10, "rejected": []}
    stock = ("sku,as_of,on_hand,supplier,lead_time_days,unit_cost\n"
             "rug,2026-10-10,0,Weavers,35,180\nrug,2026-10-05,40,Weavers,35,180\n")  # applied oldest first
    assert upload("inventory", stock)["processed"] == 2
    assert db.execute("SELECT on_hand, as_of FROM inventory WHERE sku = 'rug'").fetchone() == (0, date(2026, 10, 10))
    assert db.execute("SELECT sum(units), count(*) FILTER (WHERE NOT in_stock) FROM sales_daily WHERE sku = 'halo-lamp'").fetchone() == (145, 1)
    bad = upload("daily_sales", "sku,day,units\nghost,2026-10-01,3\n")
    assert len(bad["rejected"]) == 1 and "unknown sku 'ghost'" in bad["rejected"][0]["error"]


def test_mqtt_handler(api, db, settings):
    published, refreshed = [], []
    bridge = Bridge(settings, on_processed=refreshed.append, publish=lambda t, p: published.append((t, json.loads(p))))
    product = json.dumps(msg("p1", {"sku": "halo-lamp", **LAMP})).encode()
    assert bridge.handle("sb/v1/acme/sales/product", product) is True
    assert refreshed == ["root.acme"] and published == []
    batch = json.dumps([msg("s1", line("o1", "1", 2)), msg("s2", line("o2", "1", 1))]).encode()
    assert bridge.handle("sb/v1/acme/sales/sale", batch) and bridge.handle("sb/v1/acme/sales/sale", batch)
    assert day_row(db, date(2026, 11, 4))[0] == 3  # redelivered batch counted once

    # Topic and payload must agree; bad JSON and bad topics are dead-lettered and still acked.
    assert bridge.handle("sb/v1/globex/_/product", product) is True
    assert published[-1][0] == "sb/v1/globex/_errors" and "does not match" in published[-1][1]["error"]
    assert bridge.handle("sb/v1/acme/sales/sale", b"{not json") is True
    assert published[-1][0] == "sb/v1/acme/_errors" and "unreadable" in published[-1][1]["error"]
    assert bridge.handle("sb/v1/acme/sales/teleport", product) is True
    rejected = db.execute("SELECT count(*) FROM ingest_events WHERE status = 'rejected'").fetchone()[0]
    assert rejected == 3

    assert parse_topic("sb/v1/acme/sales.emea/sale").brain == "root.acme.sales.emea"
    assert parse_topic("sb/v1/acme/_/promotion").brain == "root.acme"
    # Database down: not acked, so the broker redelivers later.
    down = Bridge(settings.model_copy(update={"admin_database_url": "postgresql://x:y@localhost:1/none"}))
    assert down.handle("sb/v1/acme/sales/product", product) is False


def test_refresh_is_debounced_per_company():
    calls = []
    scheduler = RefreshScheduler(calls.append, min_interval=0.3)
    scheduler.mark("root.acme")
    scheduler.mark("root.globex")
    assert scheduler.run_due() == ["root.acme", "root.globex"]
    scheduler.mark("root.acme")  # more data during the quiet period: refreshed when it ends, not before
    assert scheduler.run_due() == []
    time.sleep(0.35)
    assert scheduler.run_due() == ["root.acme"] and scheduler.run_due() == []
    assert calls == ["root.acme", "root.globex", "root.acme"]


def test_refresh_company_forecasts_and_signals_from_ingested_stock(api, db, settings):
    from scripts.generate_sales import generate
    from secondbrain.ingest.pipeline import refresh_company
    from secondbrain.models import build_models

    key = new_key(api)
    rows, _ = generate()
    lamp = [r for r in rows if r["sku"] == "brass-floor-lamp"]  # sent as halo-lamp below
    text = "sku,day,units,revenue,discount,in_stock\n" + "\n".join(
        f"halo-lamp,{r['day']},{r['units']},{r['revenue']},{r['discount']},{str(r['in_stock']).lower()}" for r in lamp)
    post(api, key, "product", msg("p1", {"sku": "halo-lamp", **LAMP}))
    api.post("/ingest/daily_sales/csv", data={"brain": "sales"}, headers=key,
             files={"file": ("sales.csv", text.encode(), "text/csv")})
    post(api, key, "inventory", msg("i1", {"sku": "halo-lamp", "as_of": "2026-11-04", "on_hand": 40,
                                           "supplier": "Nordlys", "lead_time_days": 28, "unit_cost": 150}))
    refresh_company(settings, *build_models(settings), "root.acme")
    assert db.execute("SELECT count(*) FROM forecasts WHERE sku = 'halo-lamp'").fetchone() == (1,)
    kinds = [r[0] for r in db.execute("SELECT kind FROM signals WHERE company = 'root.acme'").fetchall()]
    assert "stockout" in kinds


def _broker_up(port: int = 51883) -> bool:
    try:
        socket.create_connection(("localhost", port), timeout=0.5).close()
        return True
    except OSError:
        return False


@pytest.mark.skipif(not _broker_up(), reason="mosquitto not running on localhost:51883")
def test_mqtt_bridge_end_to_end_with_mosquitto(api, db, settings):
    import paho.mqtt.client as mqtt
    from paho.mqtt.enums import CallbackAPIVersion

    from secondbrain.ingest import mqtt as bridge_module

    m = MqttSettings(_env_file=None, host="localhost", port=51883, tls=False,
                     client_id=f"sb-test-{time.time_ns()}", topic_prefix=f"sbtest{time.time_ns()}/v1")
    bridge = Bridge(settings, m.topic_prefix)
    client = bridge_module.build_client(m, bridge)
    client.connect(m.host, m.port)
    client.loop_start()
    try:
        publisher = mqtt.Client(CallbackAPIVersion.VERSION2)
        publisher.connect("localhost", 51883)
        publisher.loop_start()
        time.sleep(0.5)  # let the bridge subscribe
        for event in (msg("p1", {"sku": "halo-lamp", **LAMP}),):
            publisher.publish(f"{m.topic_prefix}/acme/sales/product", json.dumps(event), qos=1).wait_for_publish()
        publisher.publish(f"{m.topic_prefix}/acme/sales/sale", json.dumps(msg("s1", line("o1", "1", 3))), qos=1).wait_for_publish()
        deadline = time.time() + 5
        while time.time() < deadline and day_row(db, date(2026, 11, 4)) is None:
            time.sleep(0.1)
        assert day_row(db, date(2026, 11, 4))[0] == 3
        publisher.loop_stop()
        publisher.disconnect()
    finally:
        client.loop_stop()
        client.disconnect()


def test_threads_are_not_leaked_by_the_router(api):
    # Auto refresh is off in these tests, so ingesting must not start the refresh thread.
    key = new_key(api)
    post(api, key, "product", msg("p1", {"sku": "halo-lamp", **LAMP}))
    assert not any(t.name == "ingest-refresh" for t in threading.enumerate())
