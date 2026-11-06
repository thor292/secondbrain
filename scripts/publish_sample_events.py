"""Publish a realistic day of Desi Living's business events, to demo live data arriving.

Plays one business day in Merchandising, in order: a planned promotion,
customer orders through the day (one to three lines each, the bestsellers
most often), a return, a purchase order, then the end-of-day stock count per
product. A product that sells out gets a stockout event, so its day is
marked out of stock. Stock starts from the current inventory in the
database, and the day defaults to the day after the last sales day there.

    # dev broker from docker compose, with the bridge running:
    #   SB_MQTT_PORT=51883 SB_MQTT_TLS=false uv run python -m secondbrain.ingest.mqtt
    uv run python scripts/publish_sample_events.py
    uv run python scripts/publish_sample_events.py --orders 80 --delay 0.5   # slower, for watching
    uv run python scripts/publish_sample_events.py --http http://localhost:8765 --key sbk_...
    uv run python scripts/publish_sample_events.py --dry-run                 # print, send nothing

Event ids come from ``--run`` (default: the day), so running it twice for the
same day replays the same events, and SecondBrain ignores the repeats.
"""

import argparse
import json
import time
from datetime import date, datetime, time as clock, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import psycopg

from generate_sales import PRODUCTS
from secondbrain.config import get_settings

COMPANY_SPEC = Path(__file__).parent.parent / "examples" / "desiliving" / "company.json"
DEPARTMENT = "merchandising"
REORDER = "brass-floor-lamp"  # the day's purchase order


def company_brain() -> tuple[str, str]:
    spec = json.loads(COMPANY_SPEC.read_text())
    if DEPARTMENT not in {d["id"] for d in spec["departments"]}:
        raise SystemExit(f"{COMPANY_SPEC} has no {DEPARTMENT} department")
    return spec["id"], f"root.{spec['id']}.{DEPARTMENT}"


def current_state(brain: str) -> tuple[dict, dict, date | None]:
    """Products (sku → name, price) and stock (sku → on_hand, supplier, on_order, order_eta) from the database,
    and the last sales day."""
    try:
        with psycopg.connect(get_settings().admin_database_url, connect_timeout=3) as conn:
            products = {r[0]: (r[1], float(r[2])) for r in conn.execute(
                "SELECT sku, name, base_price FROM products WHERE brain_path = %s::ltree", (brain,))}
            stock = {r[0]: r[1:] for r in conn.execute(
                """SELECT sku, on_hand, supplier, on_order, order_eta FROM inventory
                   WHERE brain_path = %s::ltree""", (brain,))}
            last = conn.execute("SELECT max(day) FROM sales_daily WHERE brain_path = %s::ltree", (brain,)).fetchone()[0]
    except psycopg.OperationalError:
        products, stock, last = {}, {}, None
    if not products:
        products = {p.sku: (p.name, p.price) for p in PRODUCTS}
    return products, stock, last


def expected_units(brain: str, day: date) -> dict[str, float]:
    """What each product's latest forecast expects to sell on ``day`` (empty without forecasts)."""
    try:
        with psycopg.connect(get_settings().admin_database_url, connect_timeout=3) as conn:
            return {r[0]: float(r[1]) for r in conn.execute(
                """SELECT DISTINCT ON (f.sku) f.sku, p.forecast
                   FROM forecasts f CROSS JOIN LATERAL jsonb_to_recordset(f.points) AS p(day date, forecast float8)
                   WHERE f.brain_path = %s::ltree AND p.day = %s ORDER BY f.sku, f.created_at DESC""",
                (brain, day))}
    except psycopg.OperationalError:
        return {}


def events(company: str, day: date, orders: int, run: str, tz: ZoneInfo, products: dict, stock: dict,
           seed: int, expected: dict[str, float] | None = None) -> list[tuple[str, dict]]:
    """(kind, message) pairs in the order they happen.

    With ``expected`` (units per product from the forecast), each product
    sells about that many units, so the live day continues the history at its
    real scale; otherwise ``orders`` orders of one to three lines."""
    rng = np.random.default_rng(seed)
    n = 0

    def message(kind: str, data: dict) -> tuple[str, dict]:
        nonlocal n
        n += 1
        return kind, {"schema_version": 1, "event_id": f"demo-{run}-{n:05d}", "company": company,
                      "brain": DEPARTMENT, "data": data}

    def at(hour: float) -> str:
        return datetime.combine(day, clock(0), tz).replace(hour=int(hour), minute=int(hour % 1 * 60)).isoformat()

    out = [message("promotion", {"promo_id": f"wedding-{day.year}", "name": "Wedding season sale",
                                 "starts": str(day + timedelta(days=10)), "ends": str(day + timedelta(days=17)),
                                 "discount": 0.15})]
    skus = [p.sku for p in PRODUCTS if p.sku in products]
    weights = np.array([p.daily_units for p in PRODUCTS if p.sku in products])
    on_hand = {sku: stock[sku][0] if sku in stock else 60 for sku in skus}
    # A snapshot replaces the whole position, so it repeats what's already on order.
    on_order = {sku: (stock[sku][2], stock[sku][3]) for sku in stock}
    sold_out: set[str] = set()
    # Busier in the evening, like the shop's real traffic.
    slots = np.arange(7, 23.5, 0.25)
    busy = np.where(slots >= 18, 2.0, 1.0)
    if expected:
        lines = []
        for sku in skus:
            units = int(rng.poisson(max(expected.get(sku, 0.0), 0.0)))
            while units > 0:
                qty = min(units, 1 + int(rng.poisson(0.3)))
                lines.append((sku, qty))
                units -= qty
        rng.shuffle(lines)
        hours = np.sort(rng.choice(slots, size=len(lines), p=busy / busy.sum()))
        plan = [(i, 0, hour, sku, qty) for i, (hour, (sku, qty)) in enumerate(zip(hours, lines))]
    else:
        hours = np.sort(rng.choice(slots, size=orders, p=busy / busy.sum()))
        plan = [(i, line, hour, str(rng.choice(skus, p=weights / weights.sum())), 1 + int(rng.poisson(0.3)))
                for i, hour in enumerate(hours) for line in range(int(rng.integers(1, 4)))]
    for i, line, hour, sku, wanted in plan:
        if on_hand[sku] <= 0:
            continue
        qty = int(min(on_hand[sku], wanted))
        on_hand[sku] -= qty
        out.append(message("sale", {"order_id": f"DL-{run}-{i + 1:04d}", "line_id": str(line + 1), "sku": sku,
                                    "quantity": qty, "unit_price": products[sku][1], "ts": at(hour)}))
        if on_hand[sku] == 0 and sku in stock:
            sold_out.add(sku)
            out.append(message("stockout", {"sku": sku, "state": "out", "at": at(hour)}))
    first_sale = next(m for k, m in out if k == "sale")["data"]
    out.append(message("sale", {**first_sale, "quantity": 1, "kind": "return", "ts": at(21)}))
    if REORDER in stock:
        eta = day + timedelta(days=28)
        qty, current_eta = on_order[REORDER]
        on_order[REORDER] = (qty + 400, min(current_eta, eta) if qty and current_eta else eta)
        out.append(message("purchase_order", {"po_id": f"PO-{run}-1", "sku": REORDER, "quantity": 400,
                                              "supplier": stock[REORDER][1], "ordered_at": at(16),
                                              "eta": str(day + timedelta(days=28))}))
    for sku in skus:
        if sku in stock:  # the first count of an untracked product would also need supplier and costs
            qty, eta = on_order[sku]
            out.append(message("inventory", {"sku": sku, "as_of": at(23.75), "on_hand": on_hand[sku],
                                             "on_order": qty, "order_eta": str(eta) if qty and eta else None}))
    return out


def send_mqtt(args, items: list[tuple[str, dict]]) -> None:
    import paho.mqtt.client as mqtt
    from paho.mqtt.enums import CallbackAPIVersion

    client = mqtt.Client(CallbackAPIVersion.VERSION2, client_id=f"desiliving-demo-{time.time_ns()}")
    if args.username:
        client.username_pw_set(args.username, args.password)
    if args.tls:
        client.tls_set(ca_certs=args.ca_file)
    client.connect(args.host, args.port)
    client.loop_start()
    try:
        for kind, m in items:
            topic = f"sb/v1/{m['company']}/{m['brain']}/{kind}"
            client.publish(topic, json.dumps(m), qos=1).wait_for_publish(timeout=10)
            print(f"{topic}  {m['event_id']}")
            time.sleep(args.delay)
    finally:
        client.loop_stop()
        client.disconnect()


def send_http(args, items: list[tuple[str, dict]]) -> None:
    import httpx

    for kind, m in items:
        response = httpx.post(f"{args.http.rstrip('/')}/ingest/{kind}", json=m,
                              headers={"Authorization": f"Bearer {args.key}"}, timeout=30)
        response.raise_for_status()
        print(f"{kind:<15} {m['event_id']}  {response.json()}")
        time.sleep(args.delay)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--day", type=date.fromisoformat, help="business day (default: after the last sales day)")
    parser.add_argument("--orders", type=int, default=40, help="orders when there are no forecasts")
    parser.add_argument("--orders-only", action="store_true", help="send --orders orders, ignoring the forecast")
    parser.add_argument("--delay", type=float, default=0.01, help="seconds between messages")
    parser.add_argument("--run", help="event id prefix (default: the day); reuse it to replay")
    parser.add_argument("--tz", default="Asia/Kolkata", help="the business's time zone")
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=51883)
    parser.add_argument("--tls", action="store_true", help="MQTTS (production brokers, port 8883)")
    parser.add_argument("--ca-file")
    parser.add_argument("--username")
    parser.add_argument("--password")
    parser.add_argument("--http", help="send over HTTP to this SecondBrain URL instead of MQTT")
    parser.add_argument("--key", help="ingest key, with --http")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    company, brain = company_brain()
    products, stock, last = current_state(brain)
    day = args.day or (last + timedelta(days=1) if last else date.today())
    expected = {} if args.orders_only else expected_units(brain, day)
    items = events(company, day, args.orders, args.run or day.isoformat(), ZoneInfo(args.tz), products, stock,
                   args.seed, expected)
    if args.dry_run:
        for kind, m in items:
            print(kind, json.dumps(m))
    elif args.http:
        if not args.key:
            raise SystemExit("--http needs --key (create one: uv run python -m secondbrain.ingest key create desiliving demo)")
        send_http(args, items)
    else:
        send_mqtt(args, items)
    counts = {k: sum(1 for kind, _ in items if kind == k) for k in dict.fromkeys(k for k, _ in items)}
    print(f"\n{len(items)} events for {brain} on {day}: {counts}")


if __name__ == "__main__":
    main()
