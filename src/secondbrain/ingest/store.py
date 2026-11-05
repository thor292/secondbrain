"""Tables behind ingestion.

System tables, written and read on the admin connection only: the app role
gets no grant, so nothing here is visible to company users. The business
data itself lands in the forecasting tables (products, sales_daily,
inventory, promotions), where row-level security already applies.

Created with ``IF NOT EXISTS`` so an existing database (the running demo)
can gain them without a reset: ``uv run python -m secondbrain.ingest schema``.
"""

import psycopg

SCHEMA_SQL = """
-- Every message we've seen, by its idempotency key. 'processed' rows make
-- replays no-ops; 'rejected' rows are the dead-letter queue (payload kept so
-- the message can be fixed and resent). A rejected event may be resent with
-- the same event_id and is then processed normally.
CREATE TABLE IF NOT EXISTS ingest_events (
    id            bigserial PRIMARY KEY,
    company       text,  -- as the message gave it, so even malformed messages are kept
    event_id      text,
    kind          text,
    brain         text,
    source        text NOT NULL,  -- http, csv, mqtt
    status        text NOT NULL CHECK (status IN ('processed', 'rejected')),
    error         text,
    payload       jsonb,  -- kept for rejected messages only
    attempts      integer NOT NULL DEFAULT 1,
    received_at   timestamptz NOT NULL DEFAULT now(),
    UNIQUE (company, event_id)
);
CREATE INDEX IF NOT EXISTS ingest_events_status_idx ON ingest_events (company, status, received_at DESC);

-- Per-company ingest API keys. Only a SHA-256 of the key is stored: keys are
-- 256 random bits, so a fast hash is enough (no password stretching needed).
CREATE TABLE IF NOT EXISTS ingest_keys (
    id          uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    company     ltree NOT NULL,
    scope       ltree NOT NULL,  -- the subtree this key may write to (the company, or one department)
    name        text NOT NULL,
    prefix      text NOT NULL,  -- first characters of the key, to recognise it in a list
    key_hash    text NOT NULL UNIQUE,
    created_at  timestamptz NOT NULL DEFAULT now(),
    revoked_at  timestamptz,
    CHECK (scope <@ company)
);

-- Order lines as received. Lines are keyed by (order, line, sale/return), so
-- a publisher that resends a line under a new event_id still can't double count.
CREATE TABLE IF NOT EXISTS sale_lines (
    brain_path   ltree NOT NULL,
    order_id     text NOT NULL,
    line_id      text NOT NULL,
    kind         text NOT NULL CHECK (kind IN ('sale', 'return')),
    sku          text NOT NULL,
    day          date NOT NULL,
    ts           timestamptz NOT NULL,
    quantity     integer NOT NULL CHECK (quantity > 0),
    unit_price   numeric(12, 4) NOT NULL,
    discount     real NOT NULL,
    PRIMARY KEY (brain_path, order_id, line_id, kind)
);

-- What ingestion knows about each product's stock beyond the inventory row:
-- when the last applied observation was taken (events from before it are
-- already reflected in it), and since which day the product has been out of
-- stock, so a run of out-of-stock days is marked when it ends.
CREATE TABLE IF NOT EXISTS stock_state (
    brain_path   ltree NOT NULL,
    sku          text NOT NULL,
    observed_at  timestamptz,
    out_since    date,
    PRIMARY KEY (brain_path, sku)
);

CREATE TABLE IF NOT EXISTS purchase_orders (
    brain_path   ltree NOT NULL,
    po_id        text NOT NULL,
    sku          text NOT NULL,
    quantity     integer NOT NULL,
    received     integer NOT NULL DEFAULT 0,
    eta          date,
    supplier     text,
    ordered_at   timestamptz NOT NULL,
    status       text NOT NULL CHECK (status IN ('placed', 'cancelled')),
    PRIMARY KEY (brain_path, po_id, sku)
);

-- The client's promotion id for each promotions row, so updates replace it.
CREATE TABLE IF NOT EXISTS promotion_ids (
    brain_path    ltree NOT NULL,
    promo_id      text NOT NULL,
    promotion_id  integer NOT NULL REFERENCES promotions(id) ON DELETE CASCADE,
    PRIMARY KEY (brain_path, promo_id)
);
"""

DROP_SQL = """
DROP TABLE IF EXISTS promotion_ids, purchase_orders, stock_state, sale_lines, ingest_keys, ingest_events CASCADE;
"""


def ensure_schema(conn: psycopg.Connection) -> None:
    conn.execute(SCHEMA_SQL)


def dead_letters(conn: psycopg.Connection, company: str | None = None, limit: int = 50) -> list[tuple]:
    """Most recent rejected messages, optionally for one company (as sent: id or path)."""
    return conn.execute(
        """SELECT received_at, company, event_id, kind, brain, source, attempts, error, payload
           FROM ingest_events WHERE status = 'rejected'
             AND (%(c)s::text IS NULL OR company IN (%(c)s, 'root.' || %(c)s, replace(%(c)s, 'root.', '')))
           ORDER BY received_at DESC LIMIT %(n)s""",
        {"c": company, "n": limit},
    ).fetchall()
