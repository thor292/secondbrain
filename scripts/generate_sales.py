"""Generate Desi Living's daily sales history and promotion calendar.

Synthetic but shaped like a real Indian D2C home decor brand (own Shopify
store, JioMart, DMart shelves): a weekly rhythm of evening and weekend
shopping (Sunday is the busiest day), yearly seasons per category, 18% growth
a year, and the Indian calendar:

- Dhanteras and Diwali are the year's peak. Gifting lifts diya sets, brass
  and ceramic lamps and throws; brass is bought on Dhanteras itself (it is
  auspicious); furniture sells in the weeks before Diwali, while it can
  still be delivered. Parcels ordered after the shipping cutoff (Dhanteras
  minus 2 days) arrive after Diwali, so online gift orders dip just before
  Dhanteras. Diwali itself is quiet, and the days after it are a lull.
- The marketplaces' Festive Sale from late September, Independence Day and
  Republic Day sales, wedding seasons (mid-Nov to mid-Dec, mid-Jan to Feb)
  for furniture and textiles, small gifting lifts before Raksha Bandhan and
  Holi, and the monsoon slowing furniture deliveries (mid-June to mid-Sept).
- Promotions with price-sensitive demand, a stockout, and noisy counts.

Seeded, so every run gives the same data. Loads into the database (the
company must be onboarded first), and can also write a CSV:

    uv run python scripts/generate_sales.py
    uv run python scripts/generate_sales.py --csv data/desiliving_sales.csv --no-db
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
from secondbrain.money import inr_short
from secondbrain.sensitivity import Sensitivity

BRAIN = "root.desiliving.merchandising"
START = date(2024, 9, 1)
# Fixed rather than "yesterday", so the data is the same whenever it's generated.
# The demo's today is Dhanteras, 6 Nov 2026.
END = date(2026, 11, 5)
SEED = 2026
YEARLY_GROWTH = 1.18  # Q2 FY27 revenue was up 18% year over year


@dataclass
class Product:
    sku: str
    name: str
    category: str
    price: float  # rupees
    daily_units: float  # typical units a day at the start of the history
    season_peak: int  # day of the year when the category sells best (outside festivals)
    season_strength: float
    gift: float  # how much Diwali gifting lifts it
    elasticity: float  # demand lift per unit of discount: (1 - discount) ** -elasticity
    wedding: float = 0.0  # how much the wedding seasons lift it
    brass: float = 0.0  # bought on Dhanteras itself (brass and metal are auspicious)


PRODUCTS = [
    Product("sheesham-sofa", "Sheesham & linen 3-seater sofa", "furniture", 48900, 5, 330, 0.15, 0.2, 1.6, wedding=1.0),
    Product("brass-floor-lamp", "Moradabad brass floor lamp", "lighting", 7490, 34, 320, 0.20, 1.0, 2.0, brass=1.0),
    Product("dhurrie-rug-5x8", "Handwoven cotton dhurrie 5x8", "textiles", 6990, 26, 300, 0.20, 0.5, 1.8, wedding=0.5),
    Product("kantha-throw", "Kantha cotton throw", "textiles", 1890, 105, 355, 0.35, 1.2, 2.4, wedding=0.3),
    Product("blockprint-bedsheet", "Jaipur block-print bedsheet set", "textiles", 3490, 38, 120, 0.15, 0.7, 2.0,
            wedding=0.9),
    Product("ceramic-lamp", "Khurja ceramic table lamp", "lighting", 2790, 28, 320, 0.20, 0.9, 2.2, wedding=0.2),
    Product("sheesham-side-table", "Sheesham wood side table", "furniture", 8990, 9, 330, 0.15, 0.3, 1.5, wedding=0.8),
    Product("diya-candle-set", "Brass diya & sandalwood candle set", "decor", 899, 70, 300, 0.10, 2.0, 2.6, brass=1.0),
]

# Evenings and weekends: Saturday and above all Sunday.
WEEKLY = [0.93, 0.90, 0.91, 0.93, 0.98, 1.12, 1.23]  # Monday .. Sunday

# Out of stock while Saharanpur Woodcraft's monsoon-delayed consignment was held up.
STOCKOUTS = {"sheesham-sofa": (date(2025, 8, 4), date(2025, 8, 24))}

# Overdispersion of daily counts: smaller means noisier than Poisson.
NOISE_SHAPE = 18.0

# Monsoon: furniture deliveries slow down and fewer people redo their homes.
MONSOON = ((6, 15), (9, 15))
MONSOON_LIFT = {"furniture": 0.75, "textiles": 0.92}  # others 0.95
# Scales the whole shop so revenue lands on the brand's numbers (Q2 FY27 ~ ₹11.9 crore).
LEVEL = 0.832


FESTIVE_SALE_DAYS = 10  # 3-12 Oct 2026, with Ghar Club early access on 2 Oct
# Pricing policy: festive discounts are capped at 20%, the sofa at 15%.
MAX_DISCOUNT = 0.20
DISCOUNT_CAP = {"sheesham-sofa": 0.15}


@dataclass
class Promotion:
    name: str
    starts: date
    ends: date
    discount: float
    sku: str | None = None  # None means sitewide


def promotions(start: date, end: date) -> list[Promotion]:
    """Past promotions plus the planned ones up to the end of next year's festive season."""
    promos = []
    for year in range(start.year, end.year + 2):
        sale = calendar.festive_sale_start(year)
        promos += [
            Promotion("Republic Day sale", date(year, 1, 24), calendar.republic_day(year), 0.15),
            Promotion("Independence Day sale", date(year, 8, 13), date(year, 8, 16), 0.15),
            # Ghar Club loyalty members get the Festive Sale a day early ...
            Promotion("Ghar Club early access", sale - timedelta(days=1), sale - timedelta(days=1), 0.10),
            # ... then the Shopify / JioMart Festive Sale: 20% sitewide for ten days ...
            Promotion("Festive Sale", sale, sale + timedelta(days=FESTIVE_SALE_DAYS - 1), 0.20),
            # ... then 10% on Diwali gifting until the last day parcels arrive in time.
            Promotion("Diwali gifting offer", sale + timedelta(days=FESTIVE_SALE_DAYS),
                      calendar.shipping_cutoff(year), 0.10),
            Promotion("Shaadi season offer", date(year, 11, 20), date(year, 12, 5), 0.10, "sheesham-sofa"),
            Promotion("Shaadi season offer", date(year, 11, 20), date(year, 12, 5), 0.10, "sheesham-side-table"),
            Promotion("Year-end sale", date(year, 12, 26), date(year, 12, 31), 0.15),
        ]
    promos += [
        Promotion("Holi bedsheet week", date(2025, 3, 8), date(2025, 3, 13), 0.20, "blockprint-bedsheet"),
        Promotion("Throw flash sale", date(2025, 2, 3), date(2025, 2, 6), 0.20, "kantha-throw"),
        Promotion("Dhurrie refresh", date(2025, 4, 21), date(2025, 4, 27), 0.20, "dhurrie-rug-5x8"),
        Promotion("Sofa monsoon offer", date(2025, 6, 9), date(2025, 6, 22), 0.10, "sheesham-sofa"),
        Promotion("Monsoon lighting week", date(2025, 7, 14), date(2025, 7, 20), 0.15, "ceramic-lamp"),
        Promotion("Holi bedsheet week", date(2026, 2, 26), date(2026, 3, 3), 0.20, "blockprint-bedsheet"),
        Promotion("Brass lamp week", date(2026, 3, 9), date(2026, 3, 15), 0.15, "brass-floor-lamp"),
        Promotion("Dhurrie refresh", date(2026, 4, 20), date(2026, 4, 26), 0.20, "dhurrie-rug-5x8"),
        Promotion("Diya gifting bundle", date(2026, 10, 12), date(2026, 10, 18), 0.20, "diya-candle-set"),
    ]
    return [p for p in promos if p.ends >= start]


def discount_on(promos: list[Promotion], sku: str, day: date) -> float:
    best = max((p.discount for p in promos
                if p.starts <= day <= p.ends and p.sku in (None, sku)), default=0.0)
    return min(best, DISCOUNT_CAP.get(sku, MAX_DISCOUNT))


def diwali_lift(product: Product, day: date) -> float:
    """The festive season around Dhanteras and Diwali, beyond what discounts explain."""
    to = calendar.days_to_diwali(day)
    year = (day - timedelta(days=to)).year
    sale, cutoff = calendar.festive_sale_start(year), calendar.shipping_cutoff(year)
    dhanteras, diwali = calendar.dhanteras(year), calendar.diwali(year)
    furniture = product.category == "furniture"
    if sale <= day < dhanteras:
        if furniture:  # bought to be delivered and set up before Diwali
            return 1.45 if day <= cutoff - timedelta(days=7) else 1.15 if day <= cutoff else 0.7
        progress = (day - sale).days / max((dhanteras - sale).days, 1)
        lift = 1.15 + (0.3 + 1.0 * product.gift) * progress ** 2
        # Past the shipping cutoff a parcel won't arrive in time: online gift orders dip.
        return lift * 0.7 if day > cutoff else lift
    if day == dhanteras:  # the year's peak: above the festive run-up, and brass most of all
        return 0.6 if furniture else (1.45 + product.gift) * (1.4 + 0.6 * product.brass)
    if dhanteras < day < diwali:  # Choti Diwali: last-minute buys in DMart stores
        return 0.5 if furniture else 1.1 + 0.5 * product.gift
    if day == diwali:
        return 0.5
    if 0 < to <= 12:  # the lull after Diwali, recovering
        return 0.6 + 0.03 * to
    return 1.0


def calendar_lift(product: Product, day: date) -> float:
    """Every calendar effect on a day: Diwali, weddings, Holi, Raksha Bandhan, national sales, monsoon."""
    lift = diwali_lift(product, day)
    if calendar.is_wedding_season(day):
        lift *= 1 + 0.35 * product.wedding
    for to in (calendar.days_to_holi(day), calendar.days_to_raksha_bandhan(day)):
        if -7 <= to <= -1:
            lift *= 1 + 0.2 * product.gift  # small gifting lift in the week before
        elif to == 0:
            lift *= 0.8  # people celebrate rather than shop
    if -2 <= calendar.days_to_national_sale(day) <= 0:
        lift *= 1.15  # the sale's banners bring traffic beyond its discount
    if MONSOON[0] <= (day.month, day.day) <= MONSOON[1]:
        lift *= MONSOON_LIFT.get(product.category, 0.95)
    return lift


def expected_units(product: Product, day: date, discount: float) -> float:
    years = (day - START).days / 365.25
    growth = YEARLY_GROWTH ** years
    season = 1 + product.season_strength * math.cos(
        2 * math.pi * (day.timetuple().tm_yday - product.season_peak) / 365.25
    )
    price_lift = (1 - discount) ** -product.elasticity
    return (LEVEL * product.daily_units * growth * WEEKLY[day.weekday()] * season
            * calendar_lift(product, day) * price_lift)


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


def festive_share(rows: list[dict], year: int) -> float:
    """Share of the calendar year's revenue from the festive season: Festive Sale start through Diwali."""
    first, last = calendar.festive_sale_start(year), calendar.diwali(year)
    total = sum(r["revenue"] for r in rows if r["day"].year == year)
    festive = sum(r["revenue"] for r in rows if first <= r["day"] <= last)
    return festive / total if total else 0.0


holiday_share = festive_share  # the old name


def revenue_between(rows: list[dict], first: date, last: date) -> float:
    return sum(r["revenue"] for r in rows if first <= r["day"] <= last)


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
    q2 = revenue_between(rows, date(2026, 7, 1), date(2026, 9, 30))
    q2_before = revenue_between(rows, date(2025, 7, 1), date(2025, 9, 30))
    fy26 = revenue_between(rows, date(2025, 4, 1), date(2026, 3, 31))
    print(f"FY26 revenue of these {len(PRODUCTS)} products: {inr_short(fy26)}; "
          f"Q2 FY27 (Jul-Sep 2026): {inr_short(q2)}, {q2 / q2_before - 1:+.0%} year over year; "
          f"Festive Sale to Diwali 2025: {festive_share(rows, 2025):.0%} of 2025")


if __name__ == "__main__":
    main()
