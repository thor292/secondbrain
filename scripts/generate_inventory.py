"""Generate Desi Living's current stock positions (sample data).

Stock is set against the latest forecasts so the demo shows the situations a
merchant actually meets on Dhanteras: the brass diya sets that run out
on Diwali itself, a brass lamp that runs out in the wedding season, throws that
run out in December, the sheesham sofa on backorder from Saharanpur, a side
table with far too much stock, and healthy ones. Lead times are from the
workshops' towns (Moradabad brass, Jaipur block prints, Panipat weaves,
Saharanpur wood, Khurja ceramics) to the Hoskote warehouse.
Run after the forecasting job:

    uv run python scripts/generate_inventory.py
"""

from datetime import date, timedelta

import psycopg
from psycopg.rows import dict_row

from secondbrain.config import get_settings
from secondbrain.forecasting import store

BRAIN = "root.desiliving.merchandising"
COST_SHARE = 0.52  # unit cost as a share of the list price (gross margin is about 48%)

# sku: (supplier, lead time in days, situation)
# situation: ("runs_out", date) | ("cover_days", n) | ("backorder", units on order, arrival)
PLAN = {
    "brass-floor-lamp": ("Noor Metalcraft", 24, ("runs_out", date(2026, 11, 28))),
    "diya-candle-set": ("Noor Metalcraft", 14, ("runs_out", date(2026, 11, 8))),
    "kantha-throw": ("Panipat Looms", 18, ("runs_out", date(2026, 12, 12))),
    "sheesham-sofa": ("Saharanpur Woodcraft", 35, ("backorder", 90, date(2026, 11, 30))),
    "sheesham-side-table": ("Saharanpur Woodcraft", 35, ("cover_days", 240)),
    "dhurrie-rug-5x8": ("Panipat Looms", 21, ("cover_days", 95)),
    "blockprint-bedsheet": ("Sanganer Prints", 16, ("cover_days", 110)),
    "ceramic-lamp": ("Khurja Ceramics Co.", 20, ("cover_days", 100)),
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
        print(f"{r['sku']:<20} {r['on_hand']:>6} on hand{extra} ({r['supplier']}, {r['lead_time_days']} days)")


if __name__ == "__main__":
    main()
