"""Backtest every model on past data, keep the best per product, forecast with it.

Each backtest window pretends it is an earlier day: models see only the
history up to then and forecast the next ``horizon`` days, which we already
know. Windows are the most recent periods plus the same weeks last year, so a
model that misses the holiday season can't win on quiet October weeks alone.

Prediction intervals come from the chosen model's own backtest errors
(empirical, like split conformal prediction), not from a model's assumptions.
"""

from dataclasses import dataclass, field
from datetime import date, timedelta

import numpy as np
import pandas as pd

from secondbrain.forecasting import metrics
from secondbrain.forecasting.models import YEAR, Model, NotEnoughHistory, default_models, demand

DEFAULT_HORIZON = 56  # eight weeks: from early November it reaches past Christmas
RECENT_WINDOWS = 3
LEVELS = (80, 95)


@dataclass
class Window:
    origin: date  # last day the models see
    horizon: int
    label: str


@dataclass
class ProductForecast:
    sku: str
    model: str
    reason: str
    scores: dict  # model -> {wape, mase, smape, bias, windows: {label: wape}}
    points: list[dict] = field(default_factory=list)


def windows(cutoff: date, start: date, horizon: int) -> list[Window]:
    """The most recent periods, and the same weeks a year ago when history allows."""
    found = [Window(cutoff - timedelta(days=k * horizon), horizon, f"recent {k}")
             for k in range(1, RECENT_WINDOWS + 1)]
    last_year = cutoff - timedelta(days=YEAR)
    overlaps = any(abs((last_year - w.origin).days) < horizon for w in found)
    if not overlaps:
        found.append(Window(last_year, horizon, "same weeks last year"))
    # Models need at least a few months to learn from.
    return [w for w in found if (w.origin - start).days >= 3 * horizon]


def future_frame(skus: list[str], first: date, horizon: int, promotions: list[dict]) -> pd.DataFrame:
    """Days to forecast for each product, with the discount planned for each day."""
    days = pd.date_range(first, periods=horizon, freq="D")
    rows = []
    for sku in skus:
        for day in days:
            discount = max((p["discount"] for p in promotions
                            if p["starts"] <= day.date() <= p["ends"] and p["sku"] in (None, sku)),
                           default=0.0)
            rows.append({"sku": sku, "day": day, "discount": discount})
    return pd.DataFrame(rows)


def _frame(sales: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(sales)
    df["day"] = pd.to_datetime(df["day"])
    df["discount"] = df["discount"].astype(float)
    df["in_stock"] = df["in_stock"].astype(bool)
    return df


def _score(model: Model, history: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame | None:
    """Forecast a window and return rows (sku, actual, forecast, in_stock) with training scale."""
    future = test[["sku", "day", "discount"]].reset_index(drop=True)
    try:
        forecast = model.forecast(history, future)
    except NotEnoughHistory:
        return None
    if np.isnan(forecast).any():
        return None
    return test.reset_index(drop=True).assign(forecast=forecast)


def run(sales: list[dict], promotions: list[dict], horizon: int = DEFAULT_HORIZON,
        models: list[Model] | None = None) -> list[ProductForecast]:
    """Backtest, choose a model per product and forecast ``horizon`` days after the last sale."""
    models = models or default_models()
    df = _frame(sales)
    cutoff, start = df["day"].max().date(), df["day"].min().date()
    plan = windows(cutoff, start, horizon)
    if not plan:
        raise ValueError(f"Need at least {4 * horizon} days of history to backtest")

    # errors[model][sku] -> [(window label, scored rows, training demand)]
    errors: dict[str, dict[str, list[tuple[str, pd.DataFrame, np.ndarray]]]] = {}
    skipped: set[str] = set()
    for w in plan:
        origin = pd.Timestamp(w.origin)
        history = df[df["day"] <= origin]
        test = df[(df["day"] > origin) & (df["day"] <= origin + pd.Timedelta(days=w.horizon))]
        train_demand = demand(history)
        for model in models:
            scored = _score(model, history, test)
            if scored is None:
                skipped.add(model.name)
                continue
            for sku, rows in scored.groupby("sku"):
                train = train_demand.loc[train_demand["sku"] == sku, "demand"].to_numpy()
                errors.setdefault(model.name, {}).setdefault(sku, []).append((w.label, rows, train))

    # A model only competes if it ran on every window.
    eligible = [m for m in models if m.name not in skipped]
    if not eligible:
        raise ValueError("No model could run on every backtest window")

    skus = sorted(df["sku"].unique())
    future = future_frame(skus, cutoff + timedelta(days=1), horizon, promotions)
    finals = {m.name: m.forecast(df, future) for m in eligible}

    results = []
    for sku in skus:
        scores = {m.name: _summarize(errors[m.name][sku]) for m in eligible}
        ranked = sorted(scores, key=lambda name: scores[name]["wape"])
        best = ranked[0]
        rows = future["sku"] == sku
        points = _with_intervals(future.loc[rows, "day"], finals[best][rows.to_numpy()],
                                 errors[best][sku])
        results.append(ProductForecast(sku, best, _reason(best, ranked, scores, models, plan), scores, points))
    return results


def _summarize(per_window: list[tuple[str, pd.DataFrame, np.ndarray]]) -> dict:
    by_window = {}
    for label, rows, train in per_window:
        sold = rows[rows["in_stock"]]  # out-of-stock days say nothing about forecast quality
        a, f = sold["units"].to_numpy(float), sold["forecast"].to_numpy(float)
        by_window[label] = {
            "wape": metrics.wape(a, f), "mase": metrics.mase(a, f, train),
            "smape": metrics.smape(a, f), "bias": metrics.bias(a, f),
        }
    mean = {k: float(np.nanmean([w[k] for w in by_window.values()])) for k in ("wape", "mase", "smape", "bias")}
    return {**{k: round(v, 4) for k, v in mean.items()},
            "windows": {label: round(w["wape"], 4) for label, w in by_window.items()}}


def _with_intervals(days: pd.Series, forecast: np.ndarray,
                    per_window: list[tuple[str, pd.DataFrame, np.ndarray]]) -> list[dict]:
    """Intervals from the model's relative backtest errors, so they widen with the forecast."""
    rows = pd.concat([r for _, r, _ in per_window])
    rows = rows[rows["in_stock"] & (rows["forecast"] > 0.5)]
    ratio = (rows["units"] - rows["forecast"]) / rows["forecast"]
    points = []
    for day, value in zip(days, forecast):
        point = {"day": day.date().isoformat(), "forecast": round(float(value), 1)}
        for level in LEVELS:
            lo, hi = np.quantile(ratio, [(100 - level) / 200, 1 - (100 - level) / 200])
            point[f"lo{level}"] = round(max(0.0, float(value * (1 + lo))), 1)
            point[f"hi{level}"] = round(float(value * (1 + hi)), 1)
        points.append(point)
    return points


def _reason(best: str, ranked: list[str], scores: dict, models: list[Model], plan: list[Window]) -> str:
    describe = {m.name: m.description for m in models}
    s = scores[best]
    text = (f"{best} ({describe[best]}) had the lowest average error: WAPE {s['wape']:.1%}, "
            f"MASE {s['mase']:.2f} over {len(plan)} backtest windows of {plan[0].horizon} days")
    if any(w.label == "same weeks last year" for w in plan):
        text += ", including the same weeks last year"
    if len(ranked) > 1:
        runner = ranked[1]
        text += f". Next best: {runner} at WAPE {scores[runner]['wape']:.1%}"
    return text + "."
