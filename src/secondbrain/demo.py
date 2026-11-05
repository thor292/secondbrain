"""Demo only: save the business state once, and put it back with one click.

Moving time forward changes sales, stock, forecasts, signals and outcomes. To
show the same starting point twice (once following the brain, once ignoring
it), the loader saves those tables into a ``demo_snapshot`` schema, and
``restore`` copies them back. Knowledge (documents, notes, people) is never
touched; only the outcome memories added since the snapshot are removed.

    uv run python -m secondbrain.demo save
    uv run python -m secondbrain.demo restore
"""

import argparse

import psycopg
from psycopg.rows import dict_row, tuple_row

from secondbrain.config import get_settings

# In foreign-key order: a table comes after the tables it references.
TABLES = [
    "promotions", "promotion_ids", "sales_daily", "inventory", "forecasts", "forecast_accuracy",
    "signals", "signal_outcomes", "evolution_state",
    "ingest_events", "sale_lines", "stock_state", "purchase_orders",
]


class NoSnapshot(Exception):
    pass


def _existing(conn: psycopg.Connection, schema: str) -> list[str]:
    with conn.cursor(row_factory=tuple_row) as cur:
        present = {r[0] for r in cur.execute(
            "SELECT table_name FROM information_schema.tables WHERE table_schema = %s", (schema,))}
    return [t for t in TABLES if t in present]


def save(conn: psycopg.Connection) -> dict:
    conn.execute("DROP SCHEMA IF EXISTS demo_snapshot CASCADE")
    conn.execute("CREATE SCHEMA demo_snapshot")
    tables = _existing(conn, "public")
    for t in tables:
        conn.execute(f"CREATE TABLE demo_snapshot.{t} AS TABLE public.{t}")
    conn.execute("""CREATE TABLE demo_snapshot.outcome_memories AS
                    SELECT id FROM memories WHERE source = 'signal outcome'""")
    conn.commit()
    return {"saved": tables}


def restore(conn: psycopg.Connection) -> dict:
    tables = _existing(conn, "demo_snapshot")
    if not tables:
        raise NoSnapshot("No demo snapshot saved yet: run `uv run python -m secondbrain.demo save` after loading the examples.")
    conn.execute("TRUNCATE " + ", ".join(f"public.{t}" for t in tables) + " CASCADE")
    # Outcome memories written after the snapshot (and links to them) go.
    stale = "SELECT id FROM memories WHERE source = 'signal outcome' AND id NOT IN (SELECT id FROM demo_snapshot.outcome_memories)"
    conn.execute(f"UPDATE memories SET superseded_by = NULL WHERE superseded_by IN ({stale})")
    conn.execute(f"UPDATE memories SET amended_by = NULL WHERE amended_by IN ({stale})")
    for link in ("memory_entities", "entity_relations", "entity_extractions"):
        conn.execute(f"DELETE FROM {link} WHERE memory_id IN ({stale})")
    removed = conn.execute(f"DELETE FROM memories WHERE id IN ({stale})").rowcount
    for t in tables:
        conn.execute(f"INSERT INTO public.{t} SELECT * FROM demo_snapshot.{t}")
    # Keep serial ids ahead of the restored rows.
    conn.execute("SELECT setval(pg_get_serial_sequence('promotions', 'id'), coalesce(max(id), 1)) FROM promotions")
    with conn.cursor(row_factory=tuple_row) as cur:
        today = cur.execute("SELECT max(as_of) FROM inventory").fetchone()[0]
    conn.commit()
    return {"restored": tables, "outcome_memories_removed": removed, "today": today.isoformat() if today else None}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("action", choices=["save", "restore"])
    args = parser.parse_args()
    with psycopg.connect(get_settings().admin_database_url, row_factory=dict_row) as conn:
        print(save(conn) if args.action == "save" else restore(conn))


if __name__ == "__main__":
    main()
