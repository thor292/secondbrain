"""Sales history, promotions and forecasts in PostgreSQL.

Sales belong to a brain, like memories, and the same row-level security
decides who may read them: ``sb_can_see(brain_path, sensitivity)``. Loading
history and storing forecasts are system jobs on the admin connection; the
app role can only read.

``sales_daily`` becomes a TimescaleDB hypertable when the extension is
installed, and stays a plain table otherwise. Queries are the same either way.
"""

from datetime import date

import psycopg
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

SCHEMA_SQL = """
CREATE TABLE products (
    brain_path   ltree NOT NULL REFERENCES brains(path),
    sku          text NOT NULL,
    name         text NOT NULL,
    category     text NOT NULL,
    base_price   numeric(10, 2) NOT NULL,
    sensitivity  smallint NOT NULL CHECK (sensitivity BETWEEN 0 AND 2),
    PRIMARY KEY (brain_path, sku)
);

CREATE TABLE sales_daily (
    brain_path   ltree NOT NULL,
    sku          text NOT NULL,
    day          date NOT NULL,
    units        integer NOT NULL CHECK (units >= 0),
    revenue      numeric(12, 2) NOT NULL,
    price        numeric(10, 2) NOT NULL,  -- selling price after discount
    discount     real NOT NULL DEFAULT 0,
    -- Days out of stock record zero sales that say nothing about demand.
    in_stock     boolean NOT NULL DEFAULT true,
    -- Copied from the product so row-level security needs no join.
    sensitivity  smallint NOT NULL CHECK (sensitivity BETWEEN 0 AND 2),
    PRIMARY KEY (brain_path, sku, day),
    FOREIGN KEY (brain_path, sku) REFERENCES products
);

-- Past and planned promotions. Planned ones are known in advance, so models
-- that use them see a sale coming.
CREATE TABLE promotions (
    id           serial PRIMARY KEY,
    brain_path   ltree NOT NULL REFERENCES brains(path),
    sku          text,  -- NULL: sitewide
    name         text NOT NULL,
    starts       date NOT NULL,
    ends         date NOT NULL,
    discount     real NOT NULL CHECK (discount >= 0 AND discount < 1),
    sensitivity  smallint NOT NULL CHECK (sensitivity BETWEEN 0 AND 2)
);

CREATE TABLE forecasts (
    id           uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    brain_path   ltree NOT NULL,
    sku          text NOT NULL,
    created_at   timestamptz NOT NULL DEFAULT now(),
    cutoff       date NOT NULL,  -- last day of history the models saw
    model        text NOT NULL,
    reason       text NOT NULL,
    points       jsonb NOT NULL,  -- [{day, forecast, lo80, hi80, lo95, hi95}]
    backtest     jsonb NOT NULL,  -- {model: {wape, mase, smape, bias, rmsse, windows}}
    -- How far to trust it: {grade, summary, pattern, history_days, wape, bias,
    -- coverage80, coverage95, backtest_windows, backtest_days, warnings}
    quality      jsonb,
    sensitivity  smallint NOT NULL CHECK (sensitivity BETWEEN 0 AND 2),
    FOREIGN KEY (brain_path, sku) REFERENCES products
);

-- Stock position per product: what is on hand, what is on order and when it
-- arrives, and how long the supplier takes. Decision signals compare it with
-- the forecast to find run-outs and excess stock.
CREATE TABLE inventory (
    brain_path      ltree NOT NULL,
    sku             text NOT NULL,
    as_of           date NOT NULL,
    on_hand         integer NOT NULL CHECK (on_hand >= 0),
    on_order        integer NOT NULL DEFAULT 0 CHECK (on_order >= 0),
    order_eta       date,
    supplier        text NOT NULL,
    lead_time_days  integer NOT NULL CHECK (lead_time_days > 0),
    unit_cost       numeric(10, 2) NOT NULL,
    sensitivity     smallint NOT NULL CHECK (sensitivity BETWEEN 0 AND 2),
    PRIMARY KEY (brain_path, sku),
    FOREIGN KEY (brain_path, sku) REFERENCES products
);

CREATE INDEX promotions_brain_idx ON promotions USING gist (brain_path);
CREATE INDEX forecasts_latest_idx ON forecasts (brain_path, sku, created_at DESC);

ALTER TABLE products ENABLE ROW LEVEL SECURITY;
ALTER TABLE products FORCE ROW LEVEL SECURITY;
ALTER TABLE sales_daily ENABLE ROW LEVEL SECURITY;
ALTER TABLE sales_daily FORCE ROW LEVEL SECURITY;
ALTER TABLE promotions ENABLE ROW LEVEL SECURITY;
ALTER TABLE promotions FORCE ROW LEVEL SECURITY;
ALTER TABLE inventory ENABLE ROW LEVEL SECURITY;
ALTER TABLE inventory FORCE ROW LEVEL SECURITY;
ALTER TABLE forecasts ENABLE ROW LEVEL SECURITY;
ALTER TABLE forecasts FORCE ROW LEVEL SECURITY;

CREATE POLICY products_read ON products FOR SELECT USING (sb_can_see(brain_path, sensitivity));
CREATE POLICY sales_read ON sales_daily FOR SELECT USING (sb_can_see(brain_path, sensitivity));
CREATE POLICY promotions_read ON promotions FOR SELECT USING (sb_can_see(brain_path, sensitivity));
CREATE POLICY forecasts_read ON forecasts FOR SELECT USING (sb_can_see(brain_path, sensitivity));
CREATE POLICY inventory_read ON inventory FOR SELECT USING (sb_can_see(brain_path, sensitivity));

GRANT SELECT ON products, sales_daily, promotions, forecasts, inventory TO secondbrain_app;
"""

DROP_SQL = "DROP TABLE IF EXISTS inventory, forecasts, promotions, sales_daily, products CASCADE;"


def create_schema(conn: psycopg.Connection) -> bool:
    """Create the tables; returns whether sales_daily is a TimescaleDB hypertable."""
    conn.execute(SCHEMA_SQL)
    if conn.execute("SELECT 1 FROM pg_available_extensions WHERE name = 'timescaledb'").fetchone() is None:
        return False
    try:
        # The extension can be installed but not preloaded; then stay on plain PostgreSQL.
        with conn.transaction():
            conn.execute("CREATE EXTENSION IF NOT EXISTS timescaledb")
            conn.execute("SELECT create_hypertable('sales_daily', by_range('day', INTERVAL '3 months'))")
    except psycopg.Error:
        return False
    return True


def migrate(conn: psycopg.Connection) -> None:
    """Bring a database created before forecast quality was stored up to date."""
    conn.execute("ALTER TABLE forecasts ADD COLUMN IF NOT EXISTS quality jsonb")


def save_history(
    conn: psycopg.Connection,
    brain_path: str,
    sensitivity: int,
    products: list[tuple[str, str, str, float]],
    rows: list[dict],
    promotions: list[dict],
) -> None:
    """Replace a brain's products, daily sales and promotions."""
    if conn.execute("SELECT 1 FROM brains WHERE path = %s::ltree", (brain_path,)).fetchone() is None:
        raise ValueError(f"No brain {brain_path!r}; onboard the company first")
    for table in ("inventory", "forecasts", "sales_daily", "promotions", "products"):
        conn.execute(f"DELETE FROM {table} WHERE brain_path = %s::ltree", (brain_path,))
    conn.cursor().executemany(
        """INSERT INTO products (brain_path, sku, name, category, base_price, sensitivity)
           VALUES (%s::ltree, %s, %s, %s, %s, %s)""",
        [(brain_path, *p, int(sensitivity)) for p in products],
    )
    with conn.cursor().copy(
        """COPY sales_daily (brain_path, sku, day, units, revenue, price, discount, in_stock, sensitivity)
           FROM STDIN"""
    ) as copy:
        for r in rows:
            copy.write_row((brain_path, r["sku"], r["day"], r["units"], r["revenue"], r["price"],
                            r["discount"], r["in_stock"], int(sensitivity)))
    conn.cursor().executemany(
        """INSERT INTO promotions (brain_path, sku, name, starts, ends, discount, sensitivity)
           VALUES (%s::ltree, %s, %s, %s, %s, %s, %s)""",
        [(brain_path, p["sku"], p["name"], p["starts"], p["ends"], p["discount"], int(sensitivity))
         for p in promotions],
    )


def series_brains(conn: psycopg.Connection, under: str = "root") -> list[str]:
    """Brains that have sales history, within a subtree."""
    rows = conn.execute(
        "SELECT DISTINCT brain_path::text FROM products WHERE brain_path <@ %s::ltree ORDER BY 1",
        (under,),
    ).fetchall()
    return [r[0] for r in rows]


def load_sales(conn: psycopg.Connection, brain_path: str) -> list[dict]:
    with conn.cursor(row_factory=dict_row) as cur:
        return cur.execute(
            """SELECT sku, day, units, discount, in_stock FROM sales_daily
               WHERE brain_path = %s::ltree ORDER BY sku, day""",
            (brain_path,),
        ).fetchall()


def load_promotions(conn: psycopg.Connection, brain_path: str) -> list[dict]:
    with conn.cursor(row_factory=dict_row) as cur:
        return cur.execute(
            "SELECT sku, name, starts, ends, discount FROM promotions WHERE brain_path = %s::ltree",
            (brain_path,),
        ).fetchall()


def save_forecast(conn: psycopg.Connection, brain_path: str, sku: str, cutoff: date,
                  model: str, reason: str, points: list[dict], backtest: dict,
                  quality: dict | None = None) -> None:
    conn.execute(
        """INSERT INTO forecasts (brain_path, sku, cutoff, model, reason, points, backtest, quality, sensitivity)
           SELECT %s::ltree, %s, %s, %s, %s, %s, %s, %s, sensitivity FROM products
           WHERE brain_path = %s::ltree AND sku = %s""",
        (brain_path, sku, cutoff, model, reason, Jsonb(points), Jsonb(backtest),
         Jsonb(quality) if quality is not None else None, brain_path, sku),
    )


# Reads below run on the app connection as a user, so row-level security filters them.

def visible_products(conn: psycopg.Connection) -> list[dict]:
    with conn.cursor(row_factory=dict_row) as cur:
        return cur.execute(
            """SELECT p.brain_path::text AS brain, p.sku, p.name, p.category,
                      p.base_price::float AS base_price,
                      (SELECT max(f.cutoff) FROM forecasts f
                       WHERE f.brain_path = p.brain_path AND f.sku = p.sku) AS forecast_cutoff
               FROM products p ORDER BY p.brain_path, p.sku"""
        ).fetchall()


def find_products(conn: psycopg.Connection, product: str, brain: str | None = None) -> list[dict]:
    """Products this user can see matching a SKU or a name."""
    with conn.cursor(row_factory=dict_row) as cur:
        return cur.execute(
            """SELECT brain_path::text AS brain, sku, name FROM products
               WHERE (sku = %(p)s OR name ILIKE '%%' || %(p)s || '%%')
                 AND (%(brain)s::ltree IS NULL OR brain_path = %(brain)s::ltree)
               ORDER BY sku = %(p)s DESC, brain_path, sku""",
            {"p": product, "brain": brain},
        ).fetchall()


def latest_forecast(conn: psycopg.Connection, brain_path: str, sku: str) -> dict | None:
    with conn.cursor(row_factory=dict_row) as cur:
        return cur.execute(
            """SELECT created_at, cutoff, model, reason, points, backtest, quality FROM forecasts
               WHERE brain_path = %s::ltree AND sku = %s ORDER BY created_at DESC LIMIT 1""",
            (brain_path, sku),
        ).fetchone()


def recent_sales(conn: psycopg.Connection, brain_path: str, sku: str, since: date) -> list[dict]:
    with conn.cursor(row_factory=dict_row) as cur:
        return cur.execute(
            """SELECT day, units, in_stock FROM sales_daily
               WHERE brain_path = %s::ltree AND sku = %s AND day >= %s ORDER BY day""",
            (brain_path, sku, since),
        ).fetchall()


def save_inventory(conn: psycopg.Connection, brain_path: str, rows: list[dict]) -> None:
    """Replace a brain's stock positions (system job, admin connection)."""
    conn.execute("DELETE FROM inventory WHERE brain_path = %s::ltree", (brain_path,))
    conn.cursor().executemany(
        """INSERT INTO inventory (brain_path, sku, as_of, on_hand, on_order, order_eta, supplier,
                                  lead_time_days, unit_cost, sensitivity)
           SELECT %(brain)s::ltree, %(sku)s, %(as_of)s, %(on_hand)s, %(on_order)s, %(order_eta)s, %(supplier)s,
                  %(lead_time_days)s, %(unit_cost)s, sensitivity
           FROM products WHERE brain_path = %(brain)s::ltree AND sku = %(sku)s""",
        [{**r, "brain": brain_path} for r in rows],
    )


def inventory(conn: psycopg.Connection, brain_path: str, sku: str) -> dict | None:
    with conn.cursor(row_factory=dict_row) as cur:
        return cur.execute(
            """SELECT as_of, on_hand, on_order, order_eta, supplier, lead_time_days, unit_cost::float AS unit_cost
               FROM inventory WHERE brain_path = %s::ltree AND sku = %s""",
            (brain_path, sku),
        ).fetchone()
