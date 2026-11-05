"""Turn a raw sales feed into a clean daily panel, and describe each product's history.

Real feeds are messy, and every model would otherwise have to cope on its own:

- **Gaps.** A day with no rows at all for the whole brain is a feed outage:
  demand that day is unknown. A day missing for one product only is a day
  without sales when the feed leaves out zero rows (no zero rows, and over
  10% of days missing), and unknown otherwise. Days after a product's last
  row are days it did not sell (it left the feed), which is how a
  discontinued product shows up.
- **Out of stock** (``in_stock`` false). Sales that day are a floor on demand,
  not demand: the day is censored, never a zero.
- **Outliers.** A one-off bulk order (500 units of a lamp that sells 30 a
  day) is real revenue but not repeatable demand. Unless a promotion or Black
  Friday explains it, it is set aside like a censored day.

Unknown days are filled from the same weekday in the weeks around them (the
same idea as before), so models see an unbroken series but never learn a
slump or spike that wasn't demand. Scoring only ever uses ``known`` days.
"""

from dataclasses import dataclass, field
from datetime import date, timedelta

import numpy as np
import pandas as pd

from secondbrain.forecasting import calendar

INTERMITTENT_ADI = 1.32  # Syntetos-Boylan: average gap between sales above this is intermittent
OUTLIER_WINDOW = 57  # days, centred
RECENT = 28


def panel(sales: list[dict]) -> pd.DataFrame:
    """Daily rows per product from its first row to the brain's last day.

    Columns: sku, day, units (NaN when unobserved), discount, in_stock,
    observed, outlier, known (usable as demand), demand (known units, the rest
    filled in).
    """
    df = pd.DataFrame(list(sales))
    if df.empty:
        return pd.DataFrame(columns=["sku", "day", "units", "discount", "in_stock", "observed",
                                     "outlier", "known", "demand"])
    df = df.dropna(subset=["sku", "day"])
    df["sku"] = df["sku"].astype(str)
    df["day"] = pd.to_datetime(df["day"]).dt.normalize()
    df["units"] = pd.to_numeric(df["units"], errors="coerce").astype(float)
    df.loc[~np.isfinite(df["units"]), "units"] = np.nan
    df["units"] = df["units"].clip(lower=0)  # returns netted below zero are not negative demand
    if "discount" not in df:
        df["discount"] = 0.0
    df["discount"] = pd.to_numeric(df["discount"], errors="coerce").fillna(0.0).clip(0, 0.95)
    in_stock = df["in_stock"] if "in_stock" in df else pd.Series(True, index=df.index)
    df["in_stock"] = in_stock.astype("boolean").fillna(True).astype(bool)
    # Duplicate rows for a day (two feed batches): add the units up.
    df = (df.groupby(["sku", "day"], as_index=False)
            .agg(units=("units", lambda s: s.sum(min_count=1)), discount=("discount", "max"),
                 in_stock=("in_stock", "all")))

    cutoff = df["day"].max()
    feed_days = pd.DatetimeIndex(df["day"].unique())
    frames = []
    for sku, rows in df.groupby("sku", sort=True):
        days = pd.date_range(rows["day"].min(), cutoff, freq="D")
        full = rows.set_index("day").reindex(days)
        present = full["sku"].notna().to_numpy()
        last_row = rows["day"].max()
        in_feed = days.isin(feed_days)
        after_last = days > last_row
        # A feed that leaves out zero rows shows it: no zero rows, and many days missing.
        # A dense feed (zeros reported, or nearly every day there) has gaps instead.
        reports_zeros = bool(((rows["units"] == 0) & rows["in_stock"]).any())
        span = in_feed & ~after_last
        omits_zeros = not reports_zeros and (~present & span).sum() > 0.1 * max(span.sum(), 1)
        # Missing and not sold: the product left the feed, or the feed omits zero rows.
        zero_sale = ~present & in_feed & (after_last | omits_zeros)
        full["sku"] = sku
        full["units"] = np.where(zero_sale, 0.0, full["units"].to_numpy())
        full["discount"] = full["discount"].fillna(0.0)
        full["in_stock"] = full["in_stock"].astype("boolean").fillna(True).astype(bool)
        full["observed"] = full["units"].notna()
        frames.append(full.rename_axis("day").reset_index())
    out = pd.concat(frames, ignore_index=True)
    out["outlier"] = _outliers(out)
    out["known"] = out["observed"] & out["in_stock"] & ~out["outlier"]
    out["demand"] = _fill(out)
    return out


def _outliers(df: pd.DataFrame) -> pd.Series:
    """One-off spikes far above anything normal for the weeks around them.

    The bar is four times the 90th percentile of the surrounding eight weeks
    (and at least 10 units above it), so a busy weekend or a steady climb never
    counts. Promotion days and Black Friday to Cyber Monday are never outliers:
    those spikes are demand the models should learn.
    """
    flagged = pd.Series(False, index=df.index)
    usable = df["observed"] & df["in_stock"]
    for _, rows in df.groupby("sku", sort=False):
        units = rows["units"].where(usable.loc[rows.index])
        q90 = units.rolling(OUTLIER_WINDOW, center=True, min_periods=14).quantile(0.9)
        bar = np.maximum(4 * q90, q90 + 10)
        spike = units > bar
        if not spike.any():
            continue
        days = rows["day"].dt.date
        bf = days.map(lambda d: 0 <= calendar.days_to_black_friday(d) <= 4)
        flagged.loc[rows.index] = (spike & (rows["discount"] == 0) & ~bf).to_numpy()
    return flagged


def _fill(df: pd.DataFrame) -> pd.Series:
    """Known units, with unknown days filled from the same weekday nearby.

    An out-of-stock day that still sold some units sold at least that many,
    so the filled value never goes below what was sold.
    """
    demand = df["units"].where(df["known"])
    weekday = df["day"].dt.day_of_week
    filled = demand.groupby([df["sku"], weekday]).transform(lambda s: s.ffill().bfill())
    filled = filled.fillna(demand.groupby(df["sku"]).transform("mean")).fillna(0.0)
    floor = df["units"].where(~df["in_stock"]).fillna(0.0)
    return np.maximum(filled, floor)


@dataclass
class Profile:
    """What a product's history looks like at the cutoff, in numbers and in words."""

    sku: str
    start: date
    history_days: int
    known_days: int
    status: str  # active | discontinued | no_sales
    intermittent: bool = False
    sale_share: float = 1.0  # share of known days with a sale (last year)
    recent_level: float = 0.0  # average known demand, last 4 weeks
    stockout_days: int = 0  # last year
    stockout_tail: int = 0  # out of stock up to the cutoff
    gap_days: int = 0  # last year, unknown because the feed had no data
    outliers: list[tuple[date, float]] = field(default_factory=list)
    zero_tail: int = 0  # days without a sale up to the cutoff
    level_shift: float | None = None  # recent level / level before, when it moved a lot
    warnings: list[str] = field(default_factory=list)


def profile(rows: pd.DataFrame, cutoff: pd.Timestamp) -> Profile:
    """Describe one product's panel rows (up to and including the cutoff)."""
    rows = rows[rows["day"] <= cutoff]
    start = rows["day"].min()
    known = rows[rows["known"]]
    year = rows[rows["day"] > cutoff - pd.Timedelta(days=365)]
    year_known = year[year["known"]]
    sku = str(rows["sku"].iloc[0])
    p = Profile(sku=sku, start=start.date(), history_days=int((cutoff - start).days) + 1,
                known_days=int(len(known)), status="active")
    p.stockout_days = int((year["observed"] & ~year["in_stock"]).sum())
    p.gap_days = int((~year["observed"]).sum())
    p.outliers = [(d.date(), float(u)) for d, u in rows.loc[rows["outlier"], ["day", "units"]].itertuples(index=False)]
    p.stockout_tail = _tail_run(~rows["in_stock"].to_numpy())
    recent = known[known["day"] > cutoff - pd.Timedelta(days=RECENT)]
    p.recent_level = float(recent["units"].mean()) if len(recent) else 0.0
    if len(year_known):
        p.sale_share = float((year_known["units"] > 0).mean())
    if len(year_known) >= 28 and p.sale_share > 0:
        p.intermittent = 1 / p.sale_share >= INTERMITTENT_ADI

    if not (known["units"] > 0).any():
        p.status = "no_sales"
    else:
        sold = (rows["known"] & (rows["units"] > 0)).to_numpy()
        # Days in stock without a sale (or without data) up to the cutoff.
        quiet = (rows["known"].to_numpy() & ~sold) | ~rows["observed"].to_numpy()
        p.zero_tail = _tail_run(quiet & rows["in_stock"].to_numpy())
        before = known[(known["day"] <= cutoff - pd.Timedelta(days=p.zero_tail))
                       & (known["day"] > cutoff - pd.Timedelta(days=p.zero_tail + 180))]
        zero_chance = float((before["units"] == 0).mean()) if len(before) else 1.0
        # Discontinued: a run of zero days this long would almost never happen by chance.
        if p.zero_tail >= 90 or (p.zero_tail >= 28 and zero_chance ** p.zero_tail < 1e-3):
            p.status = "discontinued"
    p.level_shift = _level_shift(known, cutoff)
    p.warnings = _warnings(p)
    return p


def _tail_run(flags: np.ndarray) -> int:
    """Length of the run of True values at the end."""
    if not len(flags) or not flags[-1]:
        return 0
    falses = np.flatnonzero(~flags)
    return int(len(flags) - (falses[-1] + 1 if len(falses) else 0))


def _level_shift(known: pd.DataFrame, cutoff: pd.Timestamp) -> float | None:
    """Recent 4 weeks against the 8 before, when sales moved far more than seasons do.

    With a year of history, the same comparison a year earlier is divided out,
    so an autumn ramp-up that happens every year isn't a shift. Both halves of
    the recent 4 weeks must agree, so one big week isn't one either.
    """
    def ratio(end: pd.Timestamp) -> tuple[float, list[float]] | None:
        last = known[(known["day"] > end - pd.Timedelta(days=RECENT)) & (known["day"] <= end)]
        before = known[(known["day"] <= end - pd.Timedelta(days=RECENT))
                       & (known["day"] > end - pd.Timedelta(days=3 * RECENT))]
        if len(last) < 14 or len(before) < 28 or before["units"].mean() <= 0:
            return None
        base = before["units"].mean()
        mid = end - pd.Timedelta(days=14)
        halves = [h["units"].mean() / base for h in (last[last["day"] <= mid], last[last["day"] > mid]) if len(h)]
        return last["units"].mean() / base, halves

    now = ratio(cutoff)
    if now is None or len(now[1]) < 2:
        return None
    seasonal = ratio(cutoff - pd.Timedelta(days=364))
    if seasonal is not None and seasonal[0] > 0:
        r, halves = now[0] / seasonal[0], [h / seasonal[0] for h in now[1]]
        up, down = (1.5, 1.3), (0.67, 0.77)
    else:
        r, halves = now
        up, down = (1.8, 1.4), (0.55, 0.7)
    if (r >= up[0] and min(halves) >= up[1]) or (r <= down[0] and max(halves) <= down[1]):
        return float(r)
    return None


def _fmt(d: date) -> str:
    return f"{d:%b} {d.day}"


def _warnings(p: Profile) -> list[str]:
    """Plain-language notes about the data behind a forecast."""
    out = []
    if p.status == "no_sales":
        out.append("No sales recorded for this product yet, so the forecast is zero.")
    if p.status == "discontinued":
        out.append(f"No sales for the last {p.zero_tail} days although it used to sell regularly: "
                   "treated as discontinued, so the forecast is zero.")
    if p.history_days < 365 and p.status == "active":
        span = f"{p.history_days} days" if p.history_days < 70 else f"{p.history_days // 7} weeks"
        out.append(f"Only {span} of sales history: no same weeks last year to learn the season from.")
    if p.stockout_tail:
        out.append(f"Out of stock for the last {p.stockout_tail} days: demand is estimated from before the stockout.")
    elif p.stockout_days:
        out.append(f"Out of stock on {p.stockout_days} days in the last year; those days count as unknown demand, not zero.")
    if p.gap_days:
        out.append(f"{p.gap_days} days in the last year are missing from the sales feed and count as unknown.")
    if p.outliers:
        biggest = max(p.outliers, key=lambda o: o[1])
        out.append(f"{len(p.outliers)} one-off spike{'s' if len(p.outliers) > 1 else ''} "
                   f"(largest {biggest[1]:,.0f} units on {_fmt(biggest[0])}) left out as unusual orders.")
    if p.intermittent and p.status == "active":
        out.append(f"Sells on only {p.sale_share:.0%} of days: daily numbers are averages, plan on weekly totals.")
    if p.level_shift is not None and p.status == "active":
        out.append(f"Sales level changed recently: the last 4 weeks ran {p.level_shift - 1:+.0%} against the 8 weeks "
                   "before (allowing for the season); the forecast follows the new level only partly at first.")
    return out
