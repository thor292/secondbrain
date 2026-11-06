"""Plain-language reasons behind a forecast, measured from the product's own history.

The chosen model (e.g. gradient boosting) doesn't hand back a neat breakdown,
so instead of pretending to decompose it, this measures the same patterns the
models use, directly from past sales: the recent level, growth, the weekday
rhythm, last year's same weeks, how promotions moved sales, and the Diwali
effect. Each driver says what was measured and what it means.

Only days that reflect demand count: out-of-stock days, gaps in the feed and
one-off bulk orders are left out (the same cleaning the models get, see
``prepare``), and a comparison with last year is only made when last year's
days are actually there.
"""

from datetime import date, timedelta
from statistics import mean

from secondbrain.forecasting import calendar, prepare

WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
COVERED = 0.8  # share of a period's days that must have real data to compare it


def _avg(values: list[float]) -> float:
    return mean(values) if values else 0.0


def _fmt(d: date) -> str:
    return f"{d:%b} {d.day}"


def drivers(sales: list[dict], promotions: list[dict], cutoff: date, points: list[dict]) -> list[dict]:
    """sales: [{day, units, in_stock}] for about a year and a half up to the cutoff;
    promotions: [{sku, starts, ends, discount, name}] that apply to this product."""
    units = _demand_days(sales)
    window = lambda start, end: [units[d] for d in units if start <= d <= end]
    covered = lambda start, end: len(window(start, end)) >= COVERED * ((end - start).days + 1)
    out = []

    recent = _avg(window(cutoff - timedelta(days=27), cutoff))
    out.append({"name": "Recent level", "value": f"{recent:.1f} a day",
                "explanation": f"Average daily sales over the last 4 weeks (to {_fmt(cutoff)}): the starting point."})

    now = sum(window(cutoff - timedelta(days=90), cutoff))
    before = sum(window(cutoff - timedelta(days=90 + 364), cutoff - timedelta(days=364)))
    if before and covered(cutoff - timedelta(days=90 + 364), cutoff - timedelta(days=364)) \
            and covered(cutoff - timedelta(days=90), cutoff):
        growth = now / before - 1
        out.append({"name": "Growth", "value": f"{growth:+.0%} year over year",
                    "explanation": "The last 13 weeks compared with the same 13 weeks a year earlier."})

    last12 = {d: u for d, u in units.items() if d > cutoff - timedelta(days=84)}
    overall = _avg(list(last12.values()))
    if overall:
        by_day = {i: _avg([u for d, u in last12.items() if d.weekday() == i]) / overall for i in range(7)}
        best, worst = max(by_day, key=by_day.get), min(by_day, key=by_day.get)
        out.append({"name": "Weekday rhythm", "value": f"{WEEKDAYS[best]}s {by_day[best] - 1:+.0%}, {WEEKDAYS[worst]}s {by_day[worst] - 1:+.0%}",
                    "explanation": "How each weekday compares with an average day, over the last 12 weeks."})

    if points:
        first, last = date.fromisoformat(points[0]["day"]), date.fromisoformat(points[-1]["day"])
        ahead = _avg([p["forecast"] for p in points])
        last_year = _avg(window(first - timedelta(days=364), last - timedelta(days=364)))
        if last_year and covered(first - timedelta(days=364), min(last - timedelta(days=364), cutoff)):
            out.append({"name": "Same weeks last year", "value": f"{last_year:.1f} a day",
                        "explanation": f"What sold from {_fmt(first - timedelta(days=364))} to {_fmt(last - timedelta(days=364))} last year. "
                                       f"The forecast for the coming {len(points)} days averages {ahead:.1f} a day."})

    lifts = []
    for p in promotions:
        if p["ends"] > cutoff:
            continue
        during = window(p["starts"], p["ends"])
        around = window(p["starts"] - timedelta(days=21), p["starts"] - timedelta(days=1))
        if during and around and _avg(around):
            lifts.append(_avg(during) / _avg(around) - 1)
    if lifts:
        upcoming = [p for p in promotions if p["starts"] > cutoff and points
                    and p["starts"] <= date.fromisoformat(points[-1]["day"])]
        planned = "; planned: " + ", ".join(f"{p['name']} ({_fmt(p['starts'])}, {p['discount']:.0%} off)" for p in upcoming) if upcoming else ""
        out.append({"name": "Promotions", "value": f"{_avg(lifts):+.0%} during past promotions",
                    "explanation": f"Sales during {len(lifts)} past promotions compared with the 3 weeks before each{planned}."})

    # The last Diwali the data covers (this year's once it is over, else last year's).
    year = calendar.last_diwali(cutoff).year
    dhanteras, diwali = calendar.dhanteras(year), calendar.diwali(year)
    peak = window(dhanteras, diwali)
    base = window(dhanteras - timedelta(days=35), dhanteras - timedelta(days=8))
    if len(peak) >= 2 and len(base) >= 14 and _avg(base):
        coming = calendar.next_diwali(cutoff + timedelta(days=1))
        out.append({"name": "Diwali effect", "value": f"{_avg(peak) / _avg(base):.1f}x normal sales",
                    "explanation": f"Dhanteras to Diwali {year} ({_fmt(dhanteras)}–{_fmt(diwali)}) against the "
                                   f"weeks before. {'This' if coming.year == cutoff.year else 'Next'} year Dhanteras is "
                                   f"{_fmt(calendar.dhanteras(coming.year))} "
                                   f"and Diwali {_fmt(coming)}."})
    return out


def _demand_days(sales: list[dict]) -> dict[date, float]:
    """Units per day on days that reflect demand: in stock, in the feed, not a one-off spike."""
    rows = [{"sku": "_", "day": s["day"], "units": s["units"], "in_stock": s["in_stock"]} for s in sales]
    if not rows:
        return {}
    df = prepare.panel(rows)
    df = df[df["known"]]
    return {d.date(): float(u) for d, u in zip(df["day"], df["units"])}
