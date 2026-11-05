"""Validate, deduplicate and apply one message, exactly once.

Each message runs in its own transaction on an autocommit admin connection:
the event id is claimed in ``ingest_events`` and the business tables are
written in the same transaction, so a replay either finds the claim (and
does nothing) or the first attempt never committed (and it runs again).
Two workers given the same message at once serialize on the claim's unique
key; the second sees a duplicate.

A message that can't be applied (invalid, unknown SKU without product
details, a brain outside its company) is rolled back and recorded as
'rejected': the dead-letter queue. Database outages are not the message's
fault and are raised, so the caller retries instead of dead-lettering.

What each kind does:

- product: upsert; a changed sensitivity is copied to the product's sales,
  stock and forecasts, which carry it for row-level security.
- sale: stored as an order line (keyed by order, line and sale/return, so a
  resent line never counts twice) and added to that day's ``sales_daily``.
  Returns are kept as lines but don't reduce the day's units: forecasting
  needs what customers ordered.
- daily_sales: sets the day's totals (an end-of-day close or a backfill).
- inventory: replaces the stock position if it's newer than the one applied.
- stockout: marks the product out of stock (or back in stock).
- purchase_order / goods_receipt: move units to on order, then to on hand,
  when they happened after the last stock snapshot (earlier ones are already
  in it).
- promotion: upsert by the client's promo id, or remove when cancelled.

``sales_daily.in_stock`` is false for every day on which stock was at zero
at some point: the day a snapshot or stockout says on hand hit 0, every day
until it's back, and the day it comes back. Forecasting ignores those days'
sales as demand, and outcome judging reads them as run-outs.
"""

from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal
from typing import Literal

import psycopg
from psycopg.types.json import Jsonb
from pydantic import ValidationError

from secondbrain.ingest import contract
from secondbrain.ingest.contract import Message
from secondbrain.sensitivity import Sensitivity

Status = Literal["processed", "duplicate", "rejected"]


@dataclass
class Outcome:
    status: Status
    event_id: str | None = None
    company: str | None = None
    error: str | None = None
    note: str | None = None  # e.g. "stale snapshot ignored"


class Rejected(Exception):
    """The message can't be applied; it goes to the dead-letter table."""


def process(conn: psycopg.Connection, kind: str, raw: object, source: str,
            expect_company: str | None = None, expect_brain: str | None = None) -> Outcome:
    """Apply one message. ``conn`` must be an autocommit admin connection.

    ``expect_company``/``expect_brain`` (paths) come from where the message
    arrived (MQTT topic); a payload naming anything else is rejected.
    """
    try:
        msg = contract.parse(kind, raw)
        if expect_company and msg.company != expect_company:
            raise Rejected(f"payload company {msg.company} does not match {expect_company} from the topic")
        if expect_brain and msg.brain != expect_brain:
            raise Rejected(f"payload brain {msg.brain} does not match {expect_brain} from the topic")
    except (ValueError, Rejected) as exc:
        # Filed under the company it arrived for, which isn't always the one it names.
        return dead_letter(conn, kind, raw, source, _error_text(exc), company=expect_company)

    try:
        with conn.transaction():
            if not _claim(conn, msg, source):
                return Outcome("duplicate", msg.envelope.event_id, msg.company)
            note = _apply(conn, msg)
        return Outcome("processed", msg.envelope.event_id, msg.company, note=note)
    except (Rejected, psycopg.DataError, psycopg.IntegrityError) as exc:
        return dead_letter(conn, kind, raw, source, str(exc).strip(), msg)


def _error_text(exc: Exception) -> str:
    if isinstance(exc, ValidationError):
        return "; ".join(f"{'.'.join(map(str, e['loc'])) or 'message'}: {e['msg']}" for e in exc.errors())
    return str(exc)


def _claim(conn: psycopg.Connection, msg: Message, source: str) -> bool:
    """Record the event as processed; False if it already was. Rejected events may be retried."""
    return conn.execute(
        """INSERT INTO ingest_events (company, event_id, kind, brain, source, status)
           VALUES (%s, %s, %s, %s, %s, 'processed')
           ON CONFLICT (company, event_id) DO UPDATE
           SET status = 'processed', error = NULL, payload = NULL, kind = excluded.kind,
               brain = excluded.brain, source = excluded.source,
               attempts = ingest_events.attempts + 1, received_at = now()
           WHERE ingest_events.status = 'rejected'
           RETURNING id""",
        (msg.company, msg.envelope.event_id, msg.kind, msg.brain, source),
    ).fetchone() is not None


def dead_letter(conn: psycopg.Connection, kind: str, raw: object, source: str, error: str,
                msg: Message | None = None, company: str | None = None) -> Outcome:
    """Record a message that can't be applied, with the reason, for fixing and resending."""
    raw_dict = raw if isinstance(raw, dict) else {}
    company = msg.company if msg else company or _str(raw_dict.get("company"))
    if company and not msg:
        try:
            company = contract.company_path(company)
        except ValueError:
            pass
    event_id = msg.envelope.event_id if msg else _str(raw_dict.get("event_id"))
    brain = msg.brain if msg else _str(raw_dict.get("brain"))
    with conn.transaction():
        conn.execute(
            """INSERT INTO ingest_events (company, event_id, kind, brain, source, status, error, payload)
               VALUES (%s, %s, %s, %s, %s, 'rejected', %s, %s)
               ON CONFLICT (company, event_id) DO UPDATE
               SET error = excluded.error, payload = excluded.payload, attempts = ingest_events.attempts + 1,
                   received_at = now()
               WHERE ingest_events.status = 'rejected'""",
            (company, event_id, kind, brain, source, error[:2000], Jsonb(raw)),
        )
    return Outcome("rejected", event_id, company, error)


def _str(value: object) -> str | None:
    return value[:200] if isinstance(value, str) else None


def _apply(conn: psycopg.Connection, msg: Message) -> str | None:
    kind = conn.execute("SELECT kind FROM brains WHERE path = %s::ltree", (msg.brain,)).fetchone()
    if kind is None:
        raise Rejected(f"no brain {msg.brain}; onboard the company and its departments first")
    return HANDLERS[msg.kind](conn, msg.brain, msg.record)


# --- products -------------------------------------------------------------

def _product(conn: psycopg.Connection, brain: str, p: contract.Product) -> None:
    level = int(Sensitivity.parse(p.sensitivity))
    conn.execute(
        """INSERT INTO products (brain_path, sku, name, category, base_price, sensitivity)
           VALUES (%s::ltree, %s, %s, %s, %s, %s)
           ON CONFLICT (brain_path, sku) DO UPDATE SET name = excluded.name, category = excluded.category,
               base_price = excluded.base_price, sensitivity = excluded.sensitivity""",
        (brain, p.sku, p.name, p.category, p.price, level),
    )
    for table in ("sales_daily", "inventory", "forecasts"):
        conn.execute(f"""UPDATE {table} SET sensitivity = %s
                         WHERE brain_path = %s::ltree AND sku = %s AND sensitivity <> %s""",
                     (level, brain, p.sku, level))


def _known_product(conn: psycopg.Connection, brain: str, sku: str, info: contract.ProductInfo | None) -> Decimal:
    """The product's list price, creating the product from ``info`` when it's new."""
    row = conn.execute("SELECT base_price FROM products WHERE brain_path = %s::ltree AND sku = %s",
                       (brain, sku)).fetchone()
    if row:
        return row[0]
    if info is None:
        raise Rejected(f"unknown sku {sku!r} in {brain}: send a product message first, "
                       f"or include \"product\": {{name, category, price}}")
    _product(conn, brain, contract.Product(sku=sku, **info.model_dump()))
    return info.price


# --- sales ----------------------------------------------------------------

UPSERT_DAY = """
INSERT INTO sales_daily (brain_path, sku, day, units, revenue, price, discount, in_stock, sensitivity)
SELECT %(brain)s::ltree, %(sku)s, %(day)s, %(units)s, %(revenue)s, %(price)s, %(discount)s,
       coalesce(%(in_stock)s::boolean, true), sensitivity
FROM products WHERE brain_path = %(brain)s::ltree AND sku = %(sku)s
ON CONFLICT (brain_path, sku, day) DO UPDATE SET
"""


def _sale(conn: psycopg.Connection, brain: str, s: contract.Sale) -> str | None:
    _known_product(conn, brain, s.sku, s.product)
    day = s.business_day
    added = conn.execute(
        """INSERT INTO sale_lines (brain_path, order_id, line_id, kind, sku, day, ts, quantity, unit_price, discount)
           VALUES (%s::ltree, %s, %s, %s, %s, %s, %s, %s, %s, %s) ON CONFLICT DO NOTHING""",
        (brain, s.order_id, s.line_id, s.kind, s.sku, day, s.ts, s.quantity, s.unit_price, s.discount),
    ).rowcount
    if not added:
        return "order line already received under another event id"
    if s.kind == "return":
        return None
    price = (s.unit_price * Decimal(1 - s.discount)).quantize(Decimal("0.01"))
    conn.execute(
        UPSERT_DAY + """
            units = sales_daily.units + excluded.units,
            revenue = sales_daily.revenue + excluded.revenue,
            price = round((sales_daily.revenue + excluded.revenue) / (sales_daily.units + excluded.units), 2),
            discount = (sales_daily.discount * sales_daily.units + excluded.discount * excluded.units)
                       / (sales_daily.units + excluded.units)""",
        {"brain": brain, "sku": s.sku, "day": day, "units": s.quantity, "revenue": price * s.quantity,
         "price": price, "discount": s.discount, "in_stock": None},
    )
    return None


def _daily_sales(conn: psycopg.Connection, brain: str, d: contract.DailySales) -> None:
    list_price = _known_product(conn, brain, d.sku, d.product)
    cents = Decimal("0.01")
    if d.price is not None:
        price = d.price
        revenue = d.revenue if d.revenue is not None else price * d.units
    elif d.revenue is not None and d.units:
        revenue, price = d.revenue, d.revenue / d.units
    else:
        price = list_price * Decimal(1 - d.discount)
        revenue = d.revenue if d.revenue is not None else price * d.units
    conn.execute(
        UPSERT_DAY + """
            units = excluded.units, revenue = excluded.revenue, price = excluded.price,
            discount = excluded.discount, in_stock = coalesce(%(in_stock)s::boolean, sales_daily.in_stock)""",
        {"brain": brain, "sku": d.sku, "day": d.day, "units": d.units, "revenue": revenue.quantize(cents),
         "price": price.quantize(cents), "discount": d.discount, "in_stock": d.in_stock},
    )


# --- stock ----------------------------------------------------------------

def _state(conn: psycopg.Connection, brain: str, sku: str) -> tuple[datetime | None, date | None]:
    conn.execute("INSERT INTO stock_state (brain_path, sku) VALUES (%s::ltree, %s) ON CONFLICT DO NOTHING",
                 (brain, sku))
    return conn.execute(
        "SELECT observed_at, out_since FROM stock_state WHERE brain_path = %s::ltree AND sku = %s FOR UPDATE",
        (brain, sku),
    ).fetchone()


def _set_state(conn: psycopg.Connection, brain: str, sku: str, **values) -> None:
    sets = ", ".join(f"{k} = %({k})s" for k in values)
    conn.execute(f"UPDATE stock_state SET {sets} WHERE brain_path = %(brain)s::ltree AND sku = %(sku)s",
                 {**values, "brain": brain, "sku": sku})


def _mark_out(conn: psycopg.Connection, brain: str, sku: str, first: date, last: date) -> None:
    """Mark days out of stock, creating zero-sale days that don't exist yet."""
    conn.execute(
        """INSERT INTO sales_daily (brain_path, sku, day, units, revenue, price, discount, in_stock, sensitivity)
           SELECT p.brain_path, p.sku, d::date, 0, 0, p.base_price, 0, false, p.sensitivity
           FROM products p, generate_series(%s::date, %s::date, interval '1 day') d
           WHERE p.brain_path = %s::ltree AND p.sku = %s
           ON CONFLICT (brain_path, sku, day) DO UPDATE SET in_stock = false""",
        (first, last, brain, sku),
    )


def _went_out(conn: psycopg.Connection, brain: str, sku: str, day: date, current: bool) -> None:
    """Stock hit zero on ``day``. ``current``: this is the latest thing we know about the product."""
    _, out_since = _state(conn, brain, sku)
    if not current:
        _mark_out(conn, brain, sku, day, day)
        return
    start = min(out_since, day) if out_since else day
    _mark_out(conn, brain, sku, start, max(day, start))
    _set_state(conn, brain, sku, out_since=start)


def _came_back(conn: psycopg.Connection, brain: str, sku: str, day: date) -> None:
    """Back in stock on ``day``: it started that day at zero, so the whole run up to it is marked."""
    _, out_since = _state(conn, brain, sku)
    if out_since and out_since <= day:
        _mark_out(conn, brain, sku, out_since, day)
    if out_since:
        _set_state(conn, brain, sku, out_since=None)


def _inventory_row(conn: psycopg.Connection, brain: str, sku: str):
    return conn.execute(
        """SELECT as_of, on_hand, on_order, order_eta FROM inventory
           WHERE brain_path = %s::ltree AND sku = %s FOR UPDATE""",
        (brain, sku),
    ).fetchone()


def _inventory(conn: psycopg.Connection, brain: str, i: contract.InventorySnapshot) -> str | None:
    _known_product(conn, brain, i.sku, i.product)
    observed_at, _ = _state(conn, brain, i.sku)
    current = _inventory_row(conn, brain, i.sku)
    day = i.business_day
    newer = (observed_at is None or i.at > observed_at) and (current is None or day >= current[0])
    if i.on_hand == 0:
        _went_out(conn, brain, i.sku, day, newer)
    if not newer:
        return "older than the stock position already applied; only its out-of-stock day was used"
    if current is None and None in (i.supplier, i.lead_time_days, i.unit_cost):
        raise Rejected(f"first stock snapshot for {i.sku!r} needs supplier, lead_time_days and unit_cost")
    values = {"brain": brain, "sku": i.sku, "day": day, "on_hand": i.on_hand, "on_order": i.on_order,
              "eta": i.order_eta if i.on_order else None, "supplier": i.supplier, "lead": i.lead_time_days,
              "cost": i.unit_cost}
    if current is None:
        conn.execute(
            """INSERT INTO inventory (brain_path, sku, as_of, on_hand, on_order, order_eta, supplier,
                                      lead_time_days, unit_cost, sensitivity)
               SELECT %(brain)s::ltree, %(sku)s, %(day)s, %(on_hand)s, %(on_order)s, %(eta)s, %(supplier)s,
                      %(lead)s, %(cost)s, sensitivity
               FROM products WHERE brain_path = %(brain)s::ltree AND sku = %(sku)s""",
            values,
        )
    else:
        conn.execute(
            """UPDATE inventory SET as_of = %(day)s, on_hand = %(on_hand)s, on_order = %(on_order)s,
                      order_eta = %(eta)s, supplier = coalesce(%(supplier)s, supplier),
                      lead_time_days = coalesce(%(lead)s, lead_time_days), unit_cost = coalesce(%(cost)s, unit_cost)
               WHERE brain_path = %(brain)s::ltree AND sku = %(sku)s""",
            values,
        )
    _set_state(conn, brain, i.sku, observed_at=i.at)
    if i.on_hand > 0:
        _came_back(conn, brain, i.sku, day)
    return None


def _stockout(conn: psycopg.Connection, brain: str, s: contract.Stockout) -> str | None:
    _known_product(conn, brain, s.sku, s.product)
    observed_at, _ = _state(conn, brain, s.sku)
    newer = observed_at is None or s.at > observed_at
    day = s.business_day
    if s.state == "out":
        _went_out(conn, brain, s.sku, day, newer)
        if newer:
            conn.execute(
                """UPDATE inventory SET on_hand = 0, as_of = greatest(as_of, %s)
                   WHERE brain_path = %s::ltree AND sku = %s""",
                (day, brain, s.sku),
            )
            _set_state(conn, brain, s.sku, observed_at=s.at)
    elif newer:
        _came_back(conn, brain, s.sku, day)
    return None if newer else "older than the stock position already applied"


def _open_eta(conn: psycopg.Connection, brain: str, sku: str) -> date | None:
    row = conn.execute(
        """SELECT min(eta) FROM purchase_orders WHERE brain_path = %s::ltree AND sku = %s
           AND status = 'placed' AND received < quantity""",
        (brain, sku),
    ).fetchone()
    return row[0]


def _purchase_order(conn: psycopg.Connection, brain: str, po: contract.PurchaseOrder) -> str | None:
    _known_product(conn, brain, po.sku, po.product)
    old = conn.execute(
        """SELECT quantity, received, status FROM purchase_orders
           WHERE brain_path = %s::ltree AND po_id = %s AND sku = %s FOR UPDATE""",
        (brain, po.po_id, po.sku),
    ).fetchone()
    was_open = max(old[0] - old[1], 0) if old and old[2] == "placed" else 0
    received = old[1] if old else 0
    now_open = max(po.quantity - received, 0) if po.status == "placed" else 0
    conn.execute(
        """INSERT INTO purchase_orders (brain_path, po_id, sku, quantity, eta, supplier, ordered_at, status)
           VALUES (%s::ltree, %s, %s, %s, %s, %s, %s, %s)
           ON CONFLICT (brain_path, po_id, sku) DO UPDATE SET quantity = excluded.quantity, eta = excluded.eta,
               supplier = coalesce(excluded.supplier, purchase_orders.supplier), status = excluded.status""",
        (brain, po.po_id, po.sku, po.quantity, po.eta, po.supplier, po.ordered_at, po.status),
    )
    observed_at, _ = _state(conn, brain, po.sku)
    if _inventory_row(conn, brain, po.sku) is None:
        return "recorded; no stock position yet, the first inventory snapshot should include it"
    if observed_at is not None and (po.updated_at or po.ordered_at) <= observed_at:
        return "recorded; it happened before the last stock snapshot, which already includes it"
    change = now_open - was_open
    conn.execute(
        """UPDATE inventory SET on_order = greatest(on_order + %s, 0),
                  order_eta = CASE WHEN greatest(on_order + %s, 0) = 0 THEN NULL
                                   ELSE coalesce(%s, order_eta) END
           WHERE brain_path = %s::ltree AND sku = %s""",
        (change, change, _open_eta(conn, brain, po.sku), brain, po.sku),
    )
    return None


def _goods_receipt(conn: psycopg.Connection, brain: str, g: contract.GoodsReceipt) -> str | None:
    _known_product(conn, brain, g.sku, g.product)
    if g.po_id:
        conn.execute(
            """UPDATE purchase_orders SET received = least(received + %s, quantity)
               WHERE brain_path = %s::ltree AND po_id = %s AND sku = %s""",
            (g.quantity, brain, g.po_id, g.sku),
        )
    observed_at, _ = _state(conn, brain, g.sku)
    if _inventory_row(conn, brain, g.sku) is None:
        return "recorded; no stock position yet"
    if observed_at is not None and g.received_at <= observed_at:
        return "received before the last stock snapshot, which already includes it"
    day = g.business_day
    conn.execute(
        """UPDATE inventory SET on_hand = on_hand + %(q)s, on_order = greatest(on_order - %(q)s, 0),
                  order_eta = CASE WHEN on_order - %(q)s <= 0 THEN NULL ELSE coalesce(%(eta)s, order_eta) END,
                  as_of = greatest(as_of, %(day)s)
           WHERE brain_path = %(brain)s::ltree AND sku = %(sku)s""",
        {"q": g.quantity, "eta": _open_eta(conn, brain, g.sku), "day": day, "brain": brain, "sku": g.sku},
    )
    _came_back(conn, brain, g.sku, day)
    return None


# --- promotions -----------------------------------------------------------

def _promotion(conn: psycopg.Connection, brain: str, p: contract.Promotion) -> None:
    mapped = conn.execute(
        "SELECT promotion_id FROM promotion_ids WHERE brain_path = %s::ltree AND promo_id = %s",
        (brain, p.promo_id),
    ).fetchone()
    if p.cancelled:
        if mapped:
            conn.execute("DELETE FROM promotions WHERE id = %s", mapped)
        return
    level = conn.execute(
        "SELECT sensitivity FROM products WHERE brain_path = %s::ltree AND sku = %s", (brain, p.sku)
    ).fetchone() if p.sku else None
    values = (p.sku, p.name, p.starts, p.ends, p.discount, level[0] if level else int(Sensitivity.INTERNAL))
    if mapped:
        conn.execute("""UPDATE promotions SET sku = %s, name = %s, starts = %s, ends = %s, discount = %s,
                        sensitivity = %s WHERE id = %s""", (*values, mapped[0]))
        return
    new_id = conn.execute(
        """INSERT INTO promotions (brain_path, sku, name, starts, ends, discount, sensitivity)
           VALUES (%s::ltree, %s, %s, %s, %s, %s, %s) RETURNING id""",
        (brain, *values),
    ).fetchone()[0]
    conn.execute("INSERT INTO promotion_ids (brain_path, promo_id, promotion_id) VALUES (%s::ltree, %s, %s)",
                 (brain, p.promo_id, new_id))


HANDLERS = {
    "product": _product,
    "sale": _sale,
    "daily_sales": _daily_sales,
    "inventory": _inventory,
    "stockout": _stockout,
    "purchase_order": _purchase_order,
    "goods_receipt": _goods_receipt,
    "promotion": _promotion,
}


def fill_gaps(conn: psycopg.Connection, under: str) -> int:
    """Add zero-sale days so every product's series runs without gaps to its brain's last day.

    Order lines only create days on which something sold; the forecasting
    models expect one row per day. Returns the number of days added.
    """
    return conn.execute(
        """WITH span AS (
               SELECT s.brain_path, s.sku, min(s.day) AS first,
                      max(max(s.day)) OVER (PARTITION BY s.brain_path) AS last
               FROM sales_daily s WHERE s.brain_path <@ %s::ltree GROUP BY s.brain_path, s.sku)
           INSERT INTO sales_daily (brain_path, sku, day, units, revenue, price, discount, in_stock, sensitivity)
           SELECT p.brain_path, p.sku, d::date, 0, 0, p.base_price, 0, true, p.sensitivity
           FROM span JOIN products p USING (brain_path, sku),
                generate_series(span.first, span.last, interval '1 day') d
           ON CONFLICT (brain_path, sku, day) DO NOTHING""",
        (under,),
    ).rowcount
