"""How reliable are the forecasts? An out-of-sample check on clean and messy data.

Every number here is measured on days the forecaster never saw: history is
cut at a date, the whole pipeline runs (preparation, backtest, model choice,
calibrated ranges), and the forecast is compared with what happened next.

(a) **Desi Living** (the synthetic demo shop): the demo's own setting (cut on
    5 Nov 2026, the day before Dhanteras, 120 days ahead; the generator is run
    four months further, so this history is a different random draw of the
    same shop), then the demo data cut on 6 Jul 2026 (120 days) and on 15 Oct
    2025 (56 days, three days before Dhanteras: a Diwali with only one earlier
    Diwali to learn from).
(b) **A messy shop**: three regular Desi Living-like products next to products with
    the problems real feeds have: launched 3 weeks / 3 months / 9 months ago,
    sporadic sales, frequent stockouts, 500-unit bulk orders, gaps in the
    feed, a jump in the sales level, a discontinued product (still reported
    with zero rows, and one that left the feed) and heavy promotions. Their
    true demand is known, including on stockout days, so they are scored
    against demand rather than against censored sales.

For each product: the chosen model, its grade, and per model the WAPE, bias
and how often the 80% and 95% ranges held; then the median WAPE per product
against repeating last week (seasonal naive). Then sanity assertions; the script
exits non-zero if one fails.

    uv run python scripts/forecast_reliability.py
"""

import math
import sys
import time
from datetime import date, timedelta

import numpy as np

from scripts.generate_sales import (PRODUCTS, WEEKLY, Product, discount_on, expected_units, generate,
                                    promotions)
from secondbrain.forecasting import backtest, metrics

SEED = 11
CUTOFF = date(2026, 10, 10)  # messy shop: the forecast reaches past Dhanteras and Diwali
HORIZON = 56
SHOP_START = date(2024, 9, 1)


def _score(points: list[dict], truth: dict[date, float]) -> dict:
    days = [p for p in points if date.fromisoformat(p["day"]) in truth]
    a = np.array([truth[date.fromisoformat(p["day"])] for p in days], float)
    f = np.array([p["forecast"] for p in days], float)
    cover = {level: float(np.mean([p[f"lo{level}"] <= y <= p[f"hi{level}"] for p, y in zip(days, a)]))
             for level in (80, 95)}
    return {"wape": metrics.wape(a, f), "bias": metrics.bias(a, f), "cov80": cover[80], "cov95": cover[95],
            "actual": float(a.mean()) if len(a) else 0.0, "forecast": float(f.mean()) if len(f) else 0.0}


def _check(points: list[dict]) -> None:
    for p in points:
        values = [p["lo95"], p["lo80"], p["forecast"], p["hi80"], p["hi95"]]
        assert all(math.isfinite(v) and v >= 0 for v in values), p
        assert values == sorted(values), p


# ---------- (b) the messy shop ----------

def _draw(rng: np.random.Generator, mean: float) -> int:
    return int(rng.poisson(rng.gamma(18.0, max(mean, 1e-9) / 18.0)))


def messy_shop(seed: int = SEED) -> tuple[list[dict], list[dict], dict[str, dict[date, float]], dict[str, str]]:
    """Rows up to the cutoff, promotions, true demand after it, and what each product tests."""
    rng = np.random.default_rng(seed)
    end = CUTOFF + timedelta(days=HORIZON)
    promos = promotions(SHOP_START, end)
    heavy = [{"name": "Flash sale", "starts": d, "ends": d + timedelta(days=5),
              "discount": float(rng.choice([0.2, 0.3, 0.4])), "sku": "promo-heavy"}
             for d in (SHOP_START + timedelta(days=21 * k + 9) for k in range(60)) if d <= end]
    promo_rows = [vars(p) for p in promos] + heavy

    def like(sku: str, base: Product, scale: float = 1.0):
        return lambda day, disc: scale * expected_units(base, day, disc)

    lamp, sheet, table = (next(p for p in PRODUCTS if p.sku == s)
                          for s in ("brass-floor-lamp", "blockprint-bedsheet", "sheesham-side-table"))
    shift_day = CUTOFF - timedelta(days=35)
    stop_day = CUTOFF - timedelta(days=60)
    products = {
        # sku: (demand(day, discount), first day, what it tests)
        "brass-floor-lamp": (like("brass-floor-lamp", lamp), SHOP_START, "regular"),
        "blockprint-bedsheet": (like("blockprint-bedsheet", sheet), SHOP_START, "regular"),
        "sheesham-side-table": (like("sheesham-side-table", table), SHOP_START, "regular"),
        "new-3w": (like("new-3w", lamp, 0.6), CUTOFF - timedelta(days=20), "launched 3 weeks ago"),
        "new-3m": (like("new-3m", sheet, 0.8), CUTOFF - timedelta(days=89), "launched 3 months ago"),
        "new-9m": (like("new-9m", lamp, 0.5), CUTOFF - timedelta(days=269), "launched 9 months ago"),
        "slow-mover": (lambda day, disc: 0.12 * WEEKLY[day.weekday()] * (1 - disc) ** -2, SHOP_START,
                       "sporadic sales (about 1 day in 8)"),
        "stockouts": (like("stockouts", sheet, 0.7), SHOP_START, "out of stock ~25% of days, incl. the last 9"),
        "bulk-orders": (like("bulk-orders", table, 3.0), SHOP_START, "three one-off 500-unit orders"),
        "feed-gaps": (like("feed-gaps", sheet, 0.6), SHOP_START, "8% of days missing, plus a shop-wide outage"),
        "level-shift": (lambda day, disc: expected_units(table, day, disc) * (2.0 if day >= shift_day else 1.0),
                        SHOP_START, "sales doubled 5 weeks ago"),
        "discontinued": (lambda day, disc: 0.0 if day >= stop_day else expected_units(table, day, disc),
                         SHOP_START, "stopped selling 60 days ago (zero rows)"),
        "left-feed": (lambda day, disc: 0.0 if day >= stop_day else expected_units(sheet, day, disc) * 0.5,
                      SHOP_START, "stopped 60 days ago and left the feed"),
        "promo-heavy": (lambda day, disc: 12 * WEEKLY[day.weekday()] * (1 - disc) ** -3.0, SHOP_START,
                        "a 20-40% sale every 3 weeks"),
    }
    outage = {SHOP_START + timedelta(days=d) for d in range(670, 680)}  # Jul 2026: the feed was down
    stockout_days = set()
    for _ in range(14):
        start = SHOP_START + timedelta(days=int(rng.integers(0, (CUTOFF - SHOP_START).days - 20)))
        stockout_days |= {start + timedelta(days=i) for i in range(int(rng.integers(5, 16)))}
    stockout_days |= {CUTOFF - timedelta(days=i) for i in range(9)}
    bulk_days = {SHOP_START + timedelta(days=200), SHOP_START + timedelta(days=500), CUTOFF - timedelta(days=12)}

    rows, truth = [], {}
    for sku, (mean_of, first, _) in products.items():
        truth[sku] = {}
        day = first
        while day <= end:
            disc = max(discount_on([], sku, day), max((p["discount"] for p in promo_rows
                                                       if p["starts"] <= day <= p["ends"] and p["sku"] in (None, sku)),
                                                      default=0.0))
            demand = _draw(rng, mean_of(day, disc))
            if day > CUTOFF:
                truth[sku][day] = float(demand)
            else:
                units, in_stock = demand, True
                if sku == "stockouts" and day in stockout_days:
                    units, in_stock = 0, False
                if sku == "bulk-orders" and day in bulk_days:
                    units += 500
                skip = (day in outage
                        or (sku == "feed-gaps" and rng.random() < 0.08)
                        or (sku == "left-feed" and day >= stop_day))
                if not skip:
                    rows.append({"sku": sku, "day": day, "units": units, "discount": disc, "in_stock": in_stock})
            day += timedelta(days=1)
    return rows, promo_rows, truth, {sku: v[2] for sku, v in products.items()}


# ---------- reporting ----------

def report(title: str, results: list, truth: dict[str, dict[date, float]], tests: dict[str, str] | None = None) -> dict:
    print(f"\n## {title}\n")
    print(f"{'product':<15}{'chosen':<19}{'grade':<6}{'model':<19}{'WAPE':>7}{'bias':>8}{'cov80':>7}{'cov95':>7}"
          f"{'  bt cov80':>10}")
    out = {}
    for r in results:
        _check(r.points)
        chosen = _score(r.points, truth[r.sku])
        naive = r.candidates.get("seasonal_naive")
        out[r.sku] = {"result": r, "chosen": chosen,
                      "naive": _score(naive["points"], truth[r.sku]) if naive else chosen}
        rows = [(r.model, chosen)] + [(name, _score(c["points"], truth[r.sku]))
                                      for name, c in r.candidates.items() if name != r.model]
        for i, (name, s) in enumerate(rows):
            lead = f"{r.sku:<15}{r.model:<19}{r.quality['grade']:<6}" if i == 0 else " " * 40
            bt = r.quality.get("coverage80") if i == 0 else None
            print(f"{lead}{name:<19}{_pct(s['wape'])}{_pct(s['bias'], signed=True)}{_pct(s['cov80'])}{_pct(s['cov95'])}"
                  f"{_pct(bt) if bt is not None else '':>10}")
        if tests:
            print(f"{'':<15}tests: {tests[r.sku]}; {r.quality['summary']}")
            for w in r.quality["warnings"]:
                print(f"{'':<15}  - {w}")
    return out


def _pct(v: float | None, signed: bool = False) -> str:
    if v is None or not math.isfinite(v):
        return f"{'-':>7}"
    return f"{v:>+7.0%}" if signed else f"{v:>7.0%}"


def pooled(rows: list[dict]) -> str:
    keys = ("wape", "bias", "cov80", "cov95")
    return ", ".join(f"{k} {np.nanmean([r[k] for r in rows]):.1%}" for k in keys)


def main() -> int:
    failures = []

    def check(ok: bool, what: str) -> None:
        print(f"  [{'ok' if ok else 'FAIL'}] {what}")
        if not ok:
            failures.append(what)

    desi = {}
    demo, longer = generate(), generate(end=date(2027, 3, 5))
    for (rows, promos), cutoff, horizon in ((longer, date(2026, 11, 5), 120), (demo, date(2026, 7, 6), 120),
                                            (demo, date(2025, 10, 15), 56)):
        promo_rows = [vars(p) for p in promos]
        history = [r for r in rows if r["day"] <= cutoff]
        truth = {}
        for r in rows:
            if cutoff < r["day"] <= cutoff + timedelta(days=horizon) and r["in_stock"]:
                truth.setdefault(r["sku"], {})[r["day"]] = float(r["units"])
        start = time.perf_counter()
        results = backtest.run(history, promo_rows, horizon, candidates=True)
        took = time.perf_counter() - start
        desi[cutoff] = report(f"Desi Living, cut {cutoff}, next {horizon} days ({took:.0f} s)", results, truth)

    start = time.perf_counter()
    shop_rows, shop_promos, shop_truth, tests = messy_shop()
    results = backtest.run(shop_rows, shop_promos, HORIZON, candidates=True)
    took = time.perf_counter() - start
    shop = report(f"Messy shop, cut {CUTOFF}, next {HORIZON} days ({took:.0f} s)", results, shop_truth, tests)

    print("\n## Summary (chosen model, averaged over products)\n")
    for cutoff, res in desi.items():
        print(f"  Desi Living cut {cutoff}: {pooled([v['chosen'] for v in res.values()])}")
        print(f"    median WAPE per product: chosen {np.median([v['chosen']['wape'] for v in res.values()]):.1%}, "
              f"seasonal naive {np.median([v['naive']['wape'] for v in res.values()]):.1%}")
    regular = [v["chosen"] for k, v in shop.items() if shop[k]["result"].model not in ("discontinued", "no_sales")]
    print(f"  Messy shop (all forecast products): {pooled(regular)}")

    print("\n## Sanity checks\n")
    every = [v for res in desi.values() for v in res.values()] + list(shop.values())
    check(all(math.isfinite(p["forecast"]) for v in every for p in v["result"].points),
          "no NaN/inf/negative points, ranges ordered lo95<=lo80<=forecast<=hi80<=hi95")
    demo_res = desi[date(2026, 11, 5)]
    cov80 = np.mean([v["chosen"]["cov80"] for v in demo_res.values()])
    cov95 = np.mean([v["chosen"]["cov95"] for v in demo_res.values()])
    check(0.7 <= cov80 <= 0.9, f"Desi Living (demo setting) 80% ranges held {cov80:.0%} of days out of sample (target 80%)")
    check(0.88 <= cov95 <= 0.99, f"Desi Living (demo setting) 95% ranges held {cov95:.0%} of days out of sample (target 95%)")
    wape = np.mean([v["chosen"]["wape"] for v in demo_res.values()])
    check(wape < 0.4, f"Desi Living (demo setting) average WAPE {wape:.1%} out of sample")
    s = {k: v["chosen"] for k, v in shop.items()}
    r = {k: v["result"] for k, v in shop.items()}
    check(r["new-3w"].model == "cold_start" and r["new-3w"].quality["grade"] == "weak",
          f"3-week-old product: cold start, graded weak ({r['new-3w'].model}, {r['new-3w'].quality['grade']})")
    check(all("year_over_year" not in r[k].scores for k in ("new-3w", "new-3m", "new-9m")),
          "no year-over-year model for products without a last year")
    check(0.5 <= s["new-3w"]["forecast"] / max(s["new-3w"]["actual"], 1e-9) <= 2,
          f"3-week-old product forecast within 2x of demand ({s['new-3w']['forecast']:.1f} vs {s['new-3w']['actual']:.1f}/day)")
    check(s["stockouts"]["bias"] > -0.25, f"stockouts don't drag the forecast down (bias {s['stockouts']['bias']:+.0%})")
    check(s["bulk-orders"]["forecast"] <= 1.5 * s["bulk-orders"]["actual"],
          f"bulk orders don't inflate the forecast ({s['bulk-orders']['forecast']:.1f} vs {s['bulk-orders']['actual']:.1f}/day)")
    check(abs(s["feed-gaps"]["bias"]) < 0.3, f"feed gaps don't bias the forecast (bias {s['feed-gaps']['bias']:+.0%})")
    check(abs(s["level-shift"]["bias"]) < 0.3, f"a level shift is followed (bias {s['level-shift']['bias']:+.0%})")
    for k in ("discontinued", "left-feed"):
        total = sum(p["forecast"] for p in r[k].points)
        check(r[k].model == "discontinued" and total == 0, f"{k}: forecast zero ({r[k].model}, total {total:.0f})")
    ratio = s["slow-mover"]["forecast"] / max(s["slow-mover"]["actual"], 1e-9)
    check(0.5 <= ratio <= 2 and r["slow-mover"].quality["grade"] != "good",
          f"slow mover: average rate within 2x ({ratio:.2f}x) and not graded good ({r['slow-mover'].quality['grade']})")
    check(abs(s["promo-heavy"]["bias"]) < 0.3, f"promotion-heavy product unbiased (bias {s['promo-heavy']['bias']:+.0%})")

    print(f"\n{len(failures)} sanity check(s) failed." if failures else "\nAll sanity checks passed.")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
