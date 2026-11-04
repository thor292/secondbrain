"""Generate Lumen Home's current stock positions (sample data).

Stock is set against the latest forecasts so the demo shows the situations a
merchant actually meets: a bestseller that runs out during the holidays, one
that runs out right before Black Friday, a product on backorder (the Aria
sofa, as in Omar's note), one with far too much stock, and healthy ones.
Run after the forecasting job:

    uv run python scripts/generate_lumen_inventory.py
"""

from datetime import date, timedelta

import psycopg
from psycopg.rows import dict_row

from secondbrain.config import get_settings
from secondbrain.forecasting import store

BRAIN = "root.lumen.merchandising"
COST_SHARE = 0.52  # unit cost as a share of the list price (gross margin is about 46%)

# sku: (supplier, lead time in days, situation)
# situation: ("runs_out", date) | ("cover_days", n) | ("backorder", units on order, arrival)
PLAN = {
    "halo-lamp": ("Nordlys", 28, ("runs_out", date(2026, 12, 14))),
    "candle-trio": ("Ember Candle Co.", 14, ("runs_out", date(2026, 11, 26))),
    "cloud-throw": ("Gujarat textile cooperative", 21, ("runs_out", date(2026, 12, 20))),
    "aria-sofa": ("Casa Forma", 42, ("backorder", 120, date(2026, 12, 2))),
    "oak-side-table": ("Casa Forma", 42, ("cover_days", 240)),
    "jute-rug-5x8": ("Gujarat textile cooperative", 35, ("cover_days", 95)),
    "linen-duvet": ("Gujarat textile cooperative", 35, ("cover_days", 110)),
    "orb-table-lamp": ("Nordlys", 28, ("cover_days", 100)),
}


def main() -> None:
    settings = get_settings()
    with psycopg.connect(settings.admin_database_url, row_factory=dict_row) as conn:
        prices = {r["sku"]: float(r["base_price"]) for r in conn.execute(
            "SELECT sku, base_price FROM products WHERE brain_path = %s::ltree", (BRAIN,))}
        rows = []
        for sku, (supplier, lead, situation) in PLAN.items():
            forecast = store.latest_forecast(conn, BRAIN, sku)
            if forecast is None:
                raise SystemExit("Run the forecasting job first: uv run python -m secondbrain.forecasting")
            points = forecast["points"]
            as_of = forecast["cutoff"] + timedelta(days=1)
            daily = sum(p["forecast"] for p in points) / len(points)
            on_order, eta = 0, None
            if situation[0] == "runs_out":
                on_hand = int(sum(p["forecast"] for p in points if date.fromisoformat(p["day"]) < situation[1]))
            elif situation[0] == "cover_days":
                on_hand = int(daily * situation[1])
            else:
                on_hand, on_order, eta = 0, situation[1], situation[2]
            rows.append({"sku": sku, "as_of": as_of, "on_hand": on_hand, "on_order": on_order, "order_eta": eta,
                         "supplier": supplier, "lead_time_days": lead, "unit_cost": round(prices[sku] * COST_SHARE, 2)})
        store.save_inventory(conn, BRAIN, rows)
    for r in rows:
        extra = f", {r['on_order']} on order arriving {r['order_eta']}" if r["on_order"] else ""
        print(f"{r['sku']:<16} {r['on_hand']:>6} on hand{extra} ({r['supplier']}, {r['lead_time_days']} days)")


if __name__ == "__main__":
    main()
