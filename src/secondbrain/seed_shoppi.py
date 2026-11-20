"""Load the Shoppi sample merchants (examples/shoppi) into an empty SecondBrain database.

For a fresh deployment (e.g. Cloud SQL), run once as a job with the same image and settings
as the API:

    python -m secondbrain.seed_shoppi              # everything
    python -m secondbrain.seed_shoppi --no-ai      # skip document embeddings and signals (no LiteLLM)

or, on a running service, with the admin token (uses the service's own settings and network):

    curl -X POST <service-url>/admin/seed/shoppi -H "X-Admin-Token: <token>"

Steps: create the schema if the database is empty; each merchant becomes a company brain (root.<slug>) with a
Merchandising department and a <slug>_owner login; 400 days of daily sales, sale periods
(as promotions) and current stock go into root.<slug>.merchandising; the catalog-and-trends
document is imported there; then forecasts and decision signals are calculated.

Safe to run again: onboarding updates in place, history is replaced, and a document already
imported into a brain is not imported twice.
"""

import argparse
import csv
from collections import Counter, defaultdict
from datetime import date, timedelta
from pathlib import Path

import psycopg

from secondbrain import admin, schema
from secondbrain.config import get_settings
from secondbrain.forecasting import store
from secondbrain.sensitivity import Sensitivity

DATA = Path(__file__).resolve().parents[2] / "examples" / "shoppi"
LEAD_DAYS = {"haldirams": 3, "bigbasket": 2, "fabindia": 14, "bata": 10}
SUPPLIER = {"haldirams": "Own factory", "bigbasket": "Regional suppliers",
            "fabindia": "Artisan clusters", "bata": "Own factory"}


def _read(name: str) -> list[dict]:
    with open(DATA / name, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _promotions(sku: str, days: list[dict], regular: float) -> list[dict]:
    """Runs of days sold at least 15% under the regular price, as promotions."""
    out, start = [], None
    for i, h in enumerate(days + [None]):
        on_sale = h is not None and float(h["price"]) <= regular * 0.85
        if on_sale and start is None:
            start = i
        elif not on_sale and start is not None:
            first, last = days[start], days[i - 1]
            out.append({"sku": sku, "name": "Price cut" if i == len(days) else "Sale",
                        "starts": date.fromisoformat(first["date"]), "ends": date.fromisoformat(last["date"]),
                        "discount": round(1 - float(first["price"]) / regular, 2)})
            start = None
    return out


def load_merchants(conn: psycopg.Connection, log=print) -> list[str]:
    """Onboard every merchant and save its sales history and stock. Returns the company brains."""
    merchants, products = _read("merchants.csv"), _read("products.csv")
    by_sku = defaultdict(list)
    for h in _read("daily_history.csv"):
        by_sku[h["sku"]].append(h)
    level = Sensitivity.parse("internal")
    companies = []
    for m in merchants:
        slug = m["slug"]
        company = admin.onboard(conn, {
            "id": slug, "name": m["name"], "sharing": "patterns",
            "departments": [{"id": "merchandising", "name": "Merchandising"}],
            "users": [{"id": f"{slug}_owner", "name": f"{m['name']} owner", "role": "executive"}],
        })
        brain = f"{company}.merchandising"
        catalog, rows, promos, stock = [], [], [], []
        for p in (p for p in products if p["merchant_id"] == m["id"]):
            days = by_sku[p["sku"]]
            regular = Counter(float(h["price"]) for h in days).most_common(1)[0][0]  # price on most days
            catalog.append((p["sku"], p["title"], p["category"], regular))
            for h in days:
                price = float(h["price"])
                rows.append({"sku": p["sku"], "day": date.fromisoformat(h["date"]),
                             "units": int(h["units_sold"]), "revenue": float(h["revenue"]), "price": price,
                             "discount": round(max(0.0, 1 - price / regular), 4),
                             "in_stock": int(h["lost_sales"]) == 0 and int(h["closing_stock"]) > 0})
            promos += _promotions(p["sku"], days, regular)
            stock.append({"sku": p["sku"], "as_of": date.fromisoformat(days[-1]["date"]) + timedelta(days=1),
                          "on_hand": int(p["inventory"]), "on_order": 0, "order_eta": None,
                          "supplier": SUPPLIER.get(slug, "Supplier"), "lead_time_days": LEAD_DAYS.get(slug, 7),
                          "unit_cost": float(p["cost"])})
        store.save_history(conn, brain, level, catalog, rows, promos)
        store.save_inventory(conn, brain, stock)
        conn.commit()
        companies.append(company)
        log(f"{brain}: {len(catalog)} products, {len(rows)} daily sales, {len(promos)} promotions")
    return companies


def import_docs(conn: psycopg.Connection, settings, embedder, log=print) -> None:
    for folder in sorted(p for p in (DATA / "docs").iterdir() if p.is_dir()):
        brain = f"root.{folder.name}.merchandising"
        docs = [(name, text) for name, text in admin.read_documents(folder)
                if conn.execute("SELECT 1 FROM memories WHERE brain_path = %s::ltree AND source = %s LIMIT 1",
                                (brain, name)).fetchone() is None]
        count = admin.import_documents(conn, embedder, settings, brain, docs, Sensitivity.parse("internal"))
        conn.commit()
        log(f"{brain}: imported {count} document chunks" if docs else f"{brain}: documents already imported")


def run(settings, ai: bool = True, log=print) -> None:
    """Load everything. ``log`` gets one line per step. Only one run at a time per database."""
    with psycopg.connect(settings.admin_database_url, autocommit=True) as lock:
        if not lock.execute("SELECT pg_try_advisory_lock(hashtext('secondbrain.seed_shoppi'))").fetchone()[0]:
            raise RuntimeError("A Shoppi data load is already running")
        _run(settings, ai, log)


def _run(settings, ai: bool, log) -> None:
    with psycopg.connect(settings.admin_database_url) as conn:
        fresh = conn.execute("SELECT to_regclass('brains') IS NULL").fetchone()[0]
    if fresh:
        schema.create_schema(settings)
    schema.ensure_relationship_schema(settings)
    with psycopg.connect(settings.admin_database_url) as conn:
        changed = schema.let_owner_through(conn)
    log(("Schema created." if fresh else "Schema already there.")
        + (f" Owner may write past row-level security on {changed} tables." if changed else ""))

    with psycopg.connect(settings.admin_database_url) as conn:
        companies = load_merchants(conn, log)

    from secondbrain.forecasting.__main__ import run as run_forecasts
    for company in companies:
        results = run_forecasts(settings.admin_database_url, under=company)
        log(f"{company}: forecasts for {sum(len(f) for f in results.values())} products")

    if not ai:
        log("Skipped documents and signals (no AI).")
        return
    from secondbrain import signals
    from secondbrain.models import build_models
    llm, embedder = build_models(settings)
    with psycopg.connect(settings.admin_database_url) as conn:
        import_docs(conn, settings, embedder, log)
    for company, found in signals.run(settings, llm).items():
        log(f"{company}: {len(found)} open signals")
    log("Done.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--no-ai", action="store_true", help="skip document embeddings and signals (no LiteLLM calls)")
    args = parser.parse_args()
    settings = get_settings()
    settings.check_required()
    run(settings, ai=not args.no_ai, log=lambda line: print(line, flush=True))


if __name__ == "__main__":
    main()
