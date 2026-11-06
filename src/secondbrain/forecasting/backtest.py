"""Backtest every model on past data, keep the best per product, forecast with it.

Each backtest window pretends it is an earlier day: models see only the
history up to then and forecast the next ``horizon`` days, which we already
know. Windows are the most recent periods plus the same weeks last year, so a
model that misses the festive season can't win on quiet monsoon weeks alone.

Real catalogues are uneven, so every product gets the test its history allows:

- a full history gets full-length windows;
- a shorter one (a product launched this spring) gets shorter windows, and
  only the models that can work on it compete (no year-over-year without a
  last year);
- a product too new for any test gets a **cold start**: its recent average,
  shaped by the rest of the shop's forecast (weekdays, promotions, Diwali), with wide ranges;
- a product that stopped selling (or never sold) is forecast at zero rather
  than at its old average forever.

The model is chosen by WAPE, except for products that sell only now and then:
WAPE rewards forecasting zero for those, so RMSSE (squared error, which is
smallest at the average rate) chooses, among Croston-type models built for
them. Days that say nothing about demand (out of stock, missing from the
feed, one-off bulk orders) never count in a score.

Prediction intervals come from the chosen model's own backtest errors,
calibrated by cross-validation over the windows (see ``intervals``), and every
forecast carries a ``quality`` grade (see ``quality``).
"""

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date, timedelta

import numpy as np
import pandas as pd

from secondbrain.forecasting import calendar, intervals, metrics, prepare, quality
from secondbrain.forecasting.models import YEAR, Model, default_models

DEFAULT_HORIZON = 56  # eight weeks: from the festive sale it reaches past Diwali into the wedding season
RECENT_WINDOWS = 3
LEVELS = intervals.LEVELS
MIN_TRAIN = 28  # days of history a model needs at the start of a window
SHORT_WINDOWS = (56, 28, 14, 7)  # window lengths to fall back to for short histories
ASSUMED_WAPE = 0.6  # reported for a forecast that could not be tested: assume it is rough


@dataclass(frozen=True)
class Window:
    origin: date  # last day the models see
    horizon: int
    label: str


@dataclass
class ProductForecast:
    sku: str
    model: str
    reason: str
    scores: dict  # model -> {wape, mase, smape, bias, rmsse, windows: {label: wape}}
    points: list[dict] = field(default_factory=list)
    quality: dict = field(default_factory=dict)
    # Only when asked for: every eligible model's points and held-out coverage.
    candidates: dict = field(default_factory=dict)


def windows(cutoff: date, start: date, horizon: int) -> list[Window]:
    """The most recent periods, and the same weeks a year ago when history allows.

    "The same weeks a year ago" are aligned on Diwali when the cutoff is in the
    festive season (``calendar.same_day_last_year``): a forecast made three days
    before Dhanteras is tested on the days from three days before last year's
    Dhanteras, so the models are judged on the peak they are about to forecast.
    A long horizon makes that window overlap the oldest recent one; the year-ago
    window then takes its place.
    """
    found = [Window(cutoff - timedelta(days=k * horizon), horizon, f"recent {k}")
             for k in range(1, RECENT_WINDOWS + 1)]
    last_year = calendar.same_day_last_year(cutoff, YEAR)
    # Models need at least a few months to learn from.
    enough = lambda origin: (origin - start).days >= 3 * horizon
    if enough(last_year):
        found = [w for w in found if abs((last_year - w.origin).days) >= horizon]
        found.append(Window(last_year, horizon, "same weeks last year"))
    return [w for w in found if enough(w.origin)]


def plan_for(cutoff: date, start: date, horizon: int) -> list[Window]:
    """Full-length windows when the history allows, else the longest shorter ones that fit."""
    full = windows(cutoff, start, horizon)
    if full:
        return full
    # Two or more windows if possible, so the ranges can be checked on a window they didn't see.
    options = []
    for length in SHORT_WINDOWS:
        if length >= horizon:
            continue
        found = [Window(cutoff - timedelta(days=k * length), length, f"recent {k}")
                 for k in range(1, RECENT_WINDOWS + 1)]
        options.append([w for w in found if (w.origin - start).days + 1 >= max(MIN_TRAIN, 2 * length)])
    return next((o for o in options if len(o) >= 2), next((o for o in options if o), []))


def future_frame(skus: list[str], first: date, horizon: int, promotions: list[dict]) -> pd.DataFrame:
    """Days to forecast for each product, with the discount planned for each day."""
    days = pd.date_range(first, periods=horizon, freq="D")
    dates = days.date
    sitewide = np.zeros(horizon)
    own: dict[str, np.ndarray] = defaultdict(lambda: np.zeros(horizon))
    for p in promotions:
        active = (dates >= p["starts"]) & (dates <= p["ends"])
        if not active.any():
            continue
        target = sitewide if p["sku"] is None else own[p["sku"]]
        np.maximum(target, np.where(active, float(p["discount"]), 0.0), out=target)
    frames = [pd.DataFrame({"sku": sku, "day": days, "discount": np.maximum(sitewide, own[sku])}) for sku in skus]
    if not frames:
        return pd.DataFrame(columns=["sku", "day", "discount"])
    return pd.concat(frames, ignore_index=True)


def _can(model: Model, profile: prepare.Profile, history_days: int, horizon: int) -> bool:
    if model.intermittent_only and not profile.intermittent:
        return False
    return history_days >= model.min_history(horizon)


def _span(profile: prepare.Profile, origin: date) -> int:
    return (origin - profile.start).days + 1


def _backtest(data: pd.DataFrame, profiles: dict, plans: dict, models: list[Model]) -> dict:
    """(model, sku) -> {window label: (scored rows, training demand)} for every window a model managed."""
    groups: dict[tuple, list[str]] = defaultdict(list)
    for sku, plan in plans.items():
        if plan:
            groups[tuple(plan)].append(sku)
    out: dict[tuple[str, str], dict] = {}
    for plan, group in groups.items():
        for w in plan:
            origin = pd.Timestamp(w.origin)
            history = data[data["day"] <= origin]
            test = data[(data["day"] > origin) & (data["day"] <= origin + pd.Timedelta(days=w.horizon))
                        & data["sku"].isin(group)].reset_index(drop=True)
            for model in models:
                ok = [s for s in group if _can(model, profiles[s], _span(profiles[s], w.origin), w.horizon)]
                if not ok:
                    continue
                rows = test[test["sku"].isin(ok)].reset_index(drop=True)
                forecast = np.clip(model.forecast(history, rows[["sku", "day", "discount"]]), 0, None)
                rows = rows.assign(forecast=forecast, step=(rows["day"] - origin).dt.days, window=w.label)
                for sku, r in rows.groupby("sku"):
                    if np.isfinite(r["forecast"]).all():
                        train = history.loc[history["sku"] == sku, "demand"].to_numpy(float)
                        out.setdefault((model.name, sku), {})[w.label] = (r, train)
    return out


def _summarize(per_window: dict) -> dict:
    by_window = {}
    for label, (rows, train) in per_window.items():
        sold = rows[rows["known"]]  # out-of-stock, missing and one-off days say nothing about quality
        a, f = sold["units"].to_numpy(float), sold["forecast"].to_numpy(float)
        by_window[label] = {
            "wape": metrics.wape(a, f), "mase": metrics.mase(a, f, train),
            "smape": metrics.smape(a, f) if len(a) else float("nan"), "bias": metrics.bias(a, f),
            "rmsse": metrics.rmsse(a, f, train),
        }
    summary = {}
    for k in ("wape", "mase", "smape", "bias", "rmsse"):
        values = [w[k] for w in by_window.values() if np.isfinite(w[k])]
        summary[k] = round(float(np.mean(values)), 4) if values else None
    summary["windows"] = {label: (round(w["wape"], 4) if np.isfinite(w["wape"]) else None)
                          for label, w in by_window.items()}
    return summary


def _rank_key(intermittent: bool):
    big = float("inf")

    def key(item):
        name, s = item
        wape = s["wape"] if s["wape"] is not None else big
        rmsse = s["rmsse"] if s["rmsse"] is not None else big
        return (rmsse, wape, name) if intermittent else (wape, rmsse, name)
    return key


def _residuals(per_window: dict) -> pd.DataFrame:
    rows = pd.concat([r for r, _ in per_window.values()], ignore_index=True)
    return rows.loc[rows["known"], ["window", "step", "day", "units", "forecast"]]


def run(sales: list[dict], promotions: list[dict], horizon: int = DEFAULT_HORIZON,
        models: list[Model] | None = None, candidates: bool = False) -> list[ProductForecast]:
    """Backtest, choose a model per product and forecast ``horizon`` days after the last sale."""
    models = models or default_models()
    data = prepare.panel(sales)
    if data.empty:
        return []
    cutoff_ts = data["day"].max()
    cutoff = cutoff_ts.date()
    skus = sorted(data["sku"].unique())
    profiles = {sku: prepare.profile(rows, cutoff_ts) for sku, rows in data.groupby("sku")}
    active = [s for s in skus if profiles[s].status == "active"]
    plans = {s: plan_for(cutoff, profiles[s].start, horizon) for s in active}
    scored = _backtest(data, profiles, plans, models)

    # Models that ran on every window of a product's plan compete for it.
    eligible = {s: [m for m in models if plans[s] and len(scored.get((m.name, s), {})) == len(plans[s])
                    and _can(m, profiles[s], profiles[s].history_days, horizon)] for s in active}
    future = future_frame(active, cutoff + timedelta(days=1), horizon, promotions)
    finals: dict[str, dict[str, np.ndarray]] = defaultdict(dict)
    for model in models:
        wanted = [s for s in active if model in eligible[s]]
        if not wanted:
            continue
        rows = future[future["sku"].isin(wanted)].reset_index(drop=True)
        values = np.clip(model.forecast(data, rows), 0, None)
        for sku in wanted:
            v = values[(rows["sku"] == sku).to_numpy()]
            if len(v) == horizon and np.isfinite(v).all():
                finals[sku][model.name] = v

    results: dict[str, ProductForecast] = {}
    describe = {m.name: m.description for m in models}
    days = pd.date_range(cutoff + timedelta(days=1), periods=horizon, freq="D")
    for sku in active:
        p = profiles[sku]
        scores = {name: _summarize(scored[(name, sku)]) for name in finals[sku]}
        if not scores:
            continue
        ranked = [name for name, _ in sorted(scores.items(), key=_rank_key(p.intermittent))]
        best = ranked[0]
        if scores[best]["wape"] is None:  # no sales in any test window: nothing to measure against
            scores[best] = {**scores[best], "wape": ASSUMED_WAPE, "estimated": True}
        cal = intervals.calibrate(_residuals(scored[(best, sku)]))
        plan = plans[sku]
        result = ProductForecast(
            sku, best, _reason(best, ranked, scores, describe, plan, p.intermittent), scores,
            intervals.points(days, finals[sku][best], cal))
        result.quality = quality.assess(p, best, scores[best], len(plan), plan[0].horizon, horizon, cal.coverage)
        if candidates:
            for name in ranked:
                c = intervals.calibrate(_residuals(scored[(name, sku)]))
                result.candidates[name] = {"points": intervals.points(days, finals[sku][name], c),
                                           "coverage": c.coverage}
        results[sku] = result

    shape = _shop_shape(results, profiles, plans, horizon)
    for sku in skus:
        if sku in results:
            continue
        p = profiles[sku]
        if p.status == "active":
            results[sku] = _cold_start(p, data[data["sku"] == sku], days, shape, horizon)
        else:
            results[sku] = _zero(p, days, horizon)
    return [results[s] for s in skus]


def _shop_shape(results: dict, profiles: dict, plans: dict, horizon: int) -> np.ndarray | None:
    """The shop's forecast day by day, relative to its recent level, from fully tested products."""
    mature = [s for s, r in results.items() if plans[s] and plans[s][0].horizon == horizon
              and profiles[s].recent_level > 0 and not profiles[s].intermittent]
    if not mature:
        return None
    total = np.sum([[p["forecast"] for p in results[s].points] for s in mature], axis=0)
    base = sum(profiles[s].recent_level for s in mature)
    return total / base if base > 0 else None


def _cold_start(p: prepare.Profile, rows: pd.DataFrame, days: pd.DatetimeIndex, shape: np.ndarray | None,
                horizon: int) -> ProductForecast:
    """Recent average, shaped like the rest of the shop (or the product's own weekdays), with wide ranges."""
    known = rows[rows["known"]]
    level = p.recent_level or (float(known["units"].mean()) if len(known) else 0.0)
    if shape is not None:
        forecast = level * shape
        how = "shaped by the forecast for the shop's other products (weekdays, promotions, festivals)"
    else:
        forecast = np.full(horizon, level)
        how = "flat"
        if len(known) >= 14 and known["units"].mean() > 0:
            weekday = known.groupby(known["day"].dt.day_of_week)["units"].mean() / known["units"].mean()
            forecast = level * days.day_of_week.map(weekday).fillna(1.0).to_numpy()
            how = "with its own weekday pattern"
    cal = intervals.default(p.known_days)
    reason = (f"cold_start: only {p.history_days} days of history, too few to test models on, so the forecast "
              f"is its recent average of {level:.1f} a day, {how}. Ranges are deliberately wide.")
    scores = {"cold_start": {"wape": ASSUMED_WAPE, "mase": None, "smape": None, "bias": None, "rmsse": None,
                             "windows": {}, "estimated": True}}
    result = ProductForecast(p.sku, "cold_start", reason, scores, intervals.points(days, forecast, cal))
    result.quality = quality.assess(p, "cold_start", scores["cold_start"], 0, 0, horizon, cal.coverage)
    return result


def _zero(p: prepare.Profile, days: pd.DatetimeIndex, horizon: int) -> ProductForecast:
    """A product that stopped selling, or never sold: forecast zero, and say why."""
    reason = (f"discontinued: no sales for the last {p.zero_tail} days although it used to sell, "
              "so it is forecast at zero instead of at its old average." if p.status == "discontinued"
              else "no_sales: nothing has sold yet, so the forecast is zero.")
    scores = {p.status: {"wape": ASSUMED_WAPE, "mase": None, "smape": None, "bias": None, "rmsse": None,
                         "windows": {}, "estimated": True}}
    cal = intervals.Calibration(a=0.0, b=0.0)
    cal.quantiles = {b: {level: (0.0, 0.0) for level in LEVELS} for b in range(len(intervals.BUCKETS) + 1)}
    result = ProductForecast(p.sku, p.status, reason, scores, intervals.points(days, np.zeros(horizon), cal))
    result.quality = quality.assess(p, p.status, scores[p.status], 0, 0, horizon, {80: None, 95: None})
    return result


def _reason(best: str, ranked: list[str], scores: dict, describe: dict, plan: list[Window],
            intermittent: bool) -> str:
    s = scores[best]
    metric = (f"RMSSE {s['rmsse']:.2f}, which suits products that sell now and then"
              if intermittent and s["rmsse"] is not None else f"WAPE {_pct(s['wape'])}")
    mase = f", MASE {s['mase']:.2f}" if s["mase"] is not None else ""
    text = (f"{best} ({describe[best]}) had the lowest average error: {metric}{mase} "
            f"over {len(plan)} backtest window{'s' if len(plan) != 1 else ''} of {plan[0].horizon} days")
    if any(w.label == "same weeks last year" for w in plan):
        text += ", including the same weeks last year"
    if len(ranked) > 1:
        runner = ranked[1]
        r = scores[runner]
        runner_metric = (f"RMSSE {r['rmsse']:.2f}" if intermittent and r["rmsse"] is not None
                         else f"WAPE {_pct(r['wape'])}")
        text += f". Next best: {runner} at {runner_metric}"
    return text + "."


def _pct(value: float | None) -> str:
    return f"{value:.1%}" if value is not None else "n/a"
