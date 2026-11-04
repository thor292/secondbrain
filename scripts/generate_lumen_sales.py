"""Generate Lumen Home's daily sales history and promotion calendar.

Synthetic but shaped like a real online home decor shop: weekly rhythm
(Sunday and Monday evenings are the busiest), yearly seasons per category,
18% growth a year, Thanksgiving week and Black Friday spikes, a December gift
season that stops at the shipping cutoff, promotions with price-sensitive
demand, a stockout, and noisy counts. Seeded, so every run gives the same data.

Loads into the database (the company must be onboarded first), and can also
write a CSV:

    uv run python scripts/generate_lumen_sales.py
    uv run python scripts/generate_lumen_sales.py --csv data/lumen_sales.csv --no-db
"""

import argparse
import csv
import math
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import psycopg

from secondbrain.config import get_settings
from secondbrain.forecasting import calendar
from secondbrain.forecasting.store import save_history
from secondbrain.sensitivity import Sensitivity

BRAIN = "root.lumen.merchandising"
START = date(2024, 9, 1)
# Fixed rather than "yesterday", so the data is the same whenever it's generated.
END = date(2026, 11, 3)
SEED = 2026
YEARLY_GROWTH = 1.18  # Q3 FY2026 revenue was up 18% year over year


@dataclass
class Product:
    sku: str
    name: str
    category: str
    price: float
    daily_units: float  # average units a day at the start of the history
    season_peak: int  # day of the year when the category sells best
    season_strength: float
    gift: float  # how much the December gift season lifts it
    elasticity: float  # demand lift per unit of discount: (1 - discount) ** -elasticity


PRODUCTS = [
    Product("aria-sofa", "Aria linen sofa", "furniture", 1890, 5, 15, 0.20, 0.0, 1.6),
    Product("halo-lamp", "Halo brass floor lamp", "lighting", 289, 34, 330, 0.30, 0.6, 2.0),
    Product("jute-rug-5x8", "Woven jute rug 5x8", "textiles", 349, 26, 100, 0.25, 0.2, 1.8),
    Product("cloud-throw", "Cloud waffle throw", "textiles", 89, 105, 340, 0.40, 1.4, 2.4),
    Product("linen-duvet", "Stonewashed linen duvet set", "textiles", 229, 38, 300, 0.25, 0.8, 2.0),
    Product("orb-table-lamp", "Orb ceramic table lamp", "lighting", 149, 28, 320, 0.30, 0.9, 2.2),
    Product("oak-side-table", "Oak side table", "furniture", 399, 9, 120, 0.15, 0.1, 1.5),
    Product("candle-trio", "Ember scented candle trio", "decor", 42, 70, 345, 0.45, 2.0, 2.6),
]

WEEKLY = [1.08, 1.0, 0.96, 0.94, 0.88, 0.98, 1.16]  # Monday .. Sunday

# Out of stock while the Casa Forma container was held at customs.
STOCKOUTS = {"aria-sofa": (date(2025, 8, 4), date(2025, 8, 24))}

# Overdispersion of daily counts: smaller means noisier than Poisson.
NOISE_SHAPE = 18.0


@dataclass
class Promotion:
    name: str
    starts: date
    ends: date
    discount: float
    sku: str | None = None  # None means sitewide


def promotions(start: date, end: date) -> list[Promotion]:
    """Past promotions plus the planned ones up to the end of next year's holiday."""
    promos = []
    for year in range(start.year, end.year + 2):
        bf = calendar.black_friday(year)
        promos += [
            Promotion("Memorial Day sale", calendar.memorial_day(year) - timedelta(days=3),
                      calendar.memorial_day(year), 0.15),
            Promotion("Labor Day sale", calendar.labor_day(year) - timedelta(days=3),
                      calendar.labor_day(year), 0.15),
            # Holiday Campaign Plan: sitewide 20% off from Black Friday to the Tuesday after.
            Promotion("Black Friday sitewide", bf, bf + timedelta(days=4), 0.20),
            Promotion("End of year sale", date(year, 12, 26), date(year, 12, 31), 0.25),
        ]
    promos += [
        Promotion("Spring lighting week", date(2025, 3, 10), date(2025, 3, 16), 0.15, "halo-lamp"),
        Promotion("Throw flash sale", date(2025, 2, 3), date(2025, 2, 6), 0.25, "cloud-throw"),
        Promotion("Rug refresh", date(2025, 4, 21), date(2025, 4, 27), 0.20, "jute-rug-5x8"),
        Promotion("Sofa summer offer", date(2025, 6, 9), date(2025, 6, 22), 0.10, "aria-sofa"),
        Promotion("Spring lighting week", date(2026, 3, 9), date(2026, 3, 15), 0.15, "halo-lamp"),
        Promotion("Rug refresh", date(2026, 4, 20), date(2026, 4, 26), 0.20, "jute-rug-5x8"),
        Promotion("Candle autumn bundle", date(2026, 10, 12), date(2026, 10, 18), 0.20, "candle-trio"),
    ]
    return [p for p in promos if p.ends >= start]


def discount_on(promos: list[Promotion], sku: str, day: date) -> float:
    return max((p.discount for p in promos
                if p.starts <= day <= p.ends and p.sku in (None, sku)), default=0.0)


def holiday_lift(product: Product, day: date) -> float:
    """Traffic beyond what the discount alone explains."""
    to_bf = calendar.days_to_black_friday(day)
    # Holiday shopping builds through November, then the sale itself.
    lift = 1.3 if day.month in (11, 12) else 1.0
    if -7 <= to_bf < -1:
        lift = 1.1  # people hold off in Thanksgiving week, waiting for the sale
    elif to_bf == -1:
        lift = 0.8  # Thanksgiving Day
    elif to_bf in (0, 3):
        lift = 3.2  # Black Friday, Cyber Monday
    elif to_bf in (1, 2):
        lift = 2.5
    elif to_bf == 4:
        lift = 1.9
    if day.month == 12:
        if day.day <= 17:  # last day for guaranteed Christmas delivery
            lift *= 1.2 + 0.5 * product.gift
        elif day.day <= 24:
            lift *= 0.75
        elif day.day == 25:
            lift *= 0.35
    return lift


def expected_units(product: Product, day: date, discount: float) -> float:
    years = (day - START).days / 365.25
    growth = YEARLY_GROWTH ** years
    season = 1 + product.season_strength * math.cos(
        2 * math.pi * (day.timetuple().tm_yday - product.season_peak) / 365.25
    )
    price_lift = (1 - discount) ** -product.elasticity
    return (product.daily_units * growth * WEEKLY[day.weekday()] * season
            * holiday_lift(product, day) * price_lift)


def generate(start: date = START, end: date = END, seed: int = SEED) -> tuple[list[dict], list[Promotion]]:
    """Daily rows (sku, day, units, revenue, price, discount, in_stock) and promotions."""
    rng = np.random.default_rng(seed)
    promos = promotions(start, end)
    rows = []
    for product in PRODUCTS:
        out = STOCKOUTS.get(product.sku)
        day = start
        while day <= end:
            discount = discount_on(promos, product.sku, day)
            in_stock = not (out and out[0] <= day <= out[1])
            mean = expected_units(product, day, discount)
            # Gamma-Poisson (negative binomial): real sales vary more than Poisson.
            units = int(rng.poisson(rng.gamma(NOISE_SHAPE, mean / NOISE_SHAPE))) if in_stock else 0
            price = round(product.price * (1 - discount), 2)
            rows.append({
                "sku": product.sku, "day": day, "units": units,
                "revenue": round(units * price, 2), "price": price,
                "discount": discount, "in_stock": in_stock,
            })
            day += timedelta(days=1)
    return rows, promos


def holiday_share(rows: list[dict], year: int) -> float:
    """Share of the year's revenue from November and December."""
    total = sum(r["revenue"] for r in rows if r["day"].year == year)
    holiday = sum(r["revenue"] for r in rows if r["day"].year == year and r["day"].month >= 11)
    return holiday / total if total else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--brain", default=BRAIN, help="brain the sales belong to")
    parser.add_argument("--sensitivity", default="internal", choices=["public", "internal", "restricted"])
    parser.add_argument("--csv", type=Path, help="also write the rows to this CSV file")
    parser.add_argument("--no-db", action="store_true", help="don't load into the database")
    args = parser.parse_args()

    rows, promos = generate()
    if args.csv:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with args.csv.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print(f"Wrote {len(rows)} rows to {args.csv}")
    if not args.no_db:
        with psycopg.connect(get_settings().admin_database_url) as conn:
            save_history(conn, args.brain, Sensitivity.parse(args.sensitivity),
                         [(p.sku, p.name, p.category, p.price) for p in PRODUCTS],
                         rows, [vars(p) for p in promos])
        print(f"Loaded {len(rows)} daily sales rows and {len(promos)} promotions into {args.brain}")
    revenue = sum(r["revenue"] for r in rows if r["day"].year == 2025)
    print(f"2025 revenue of these {len(PRODUCTS)} products: ${revenue / 1e6:.1f}M; "
          f"November and December: {holiday_share(rows, 2025):.0%} of it")


if __name__ == "__main__":
    main()
