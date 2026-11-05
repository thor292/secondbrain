"""Forecasting model variants.

Every model gets the same inputs and returns daily unit forecasts:

- ``history``: the prepared daily panel (see ``prepare.panel``) up to the
  cutoff, with ``demand`` already cleaned of stockouts, feed gaps and one-off
  spikes, and ``known`` marking the days whose sales were real demand;
- ``future``: rows (sku, day, discount) for the days to forecast, with the
  planned promotions, since a merchant knows its sale calendar in advance.

A model returns NaN for a product it can't forecast (too little history, say)
instead of failing for every product. ``min_history`` says how many days of a
product's history a model needs, so the backtest only asks for what can work.

The backtest runs every model on past data and keeps the best per product.
"""

from typing import Protocol

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from secondbrain.forecasting import calendar

WEEK = 7
YEAR = 364  # 52 weeks, so a year ago is the same weekday


class NotEnoughHistory(Exception):
    pass


class Model(Protocol):
    name: str
    description: str
    intermittent_only: bool

    def min_history(self, horizon: int) -> int:
        """Days of a product's history needed to forecast ``horizon`` days ahead."""
        ...

    def forecast(self, history: pd.DataFrame, future: pd.DataFrame) -> np.ndarray:
        """Forecasts aligned with the rows of ``future``; NaN where it can't."""
        ...


def demand(history: pd.DataFrame) -> pd.DataFrame:
    """The prepared panel, sorted. Raw sales rows are prepared first.

    A day with nothing to sell records zero sales, not zero demand; see
    ``prepare`` for how such days, feed gaps and one-off spikes are filled.
    """
    if "demand" not in history:
        from secondbrain.forecasting import prepare

        history = prepare.panel(history.to_dict("records"))
    return history.sort_values(["sku", "day"]).reset_index(drop=True)


def _series(history: pd.DataFrame, future: pd.DataFrame) -> pd.DataFrame:
    """Prepared history of just the products being forecast, in statsforecast's shape."""
    df = demand(history)
    df = df[df["sku"].isin(future["sku"].unique())]
    return df.rename(columns={"sku": "unique_id", "day": "ds", "demand": "y"})[["unique_id", "ds", "y"]]


class SeasonalNaive:
    """Same as the same weekday last week. The baseline every other model must beat."""

    name = "seasonal_naive"
    description = "repeats the last week"
    intermittent_only = False

    def min_history(self, horizon: int) -> int:
        return 2 * WEEK

    def forecast(self, history: pd.DataFrame, future: pd.DataFrame) -> np.ndarray:
        df = demand(history)
        out = np.full(len(future), np.nan)
        for sku, rows in future.groupby("sku"):
            last_week = df[df["sku"] == sku].set_index("day")["demand"].iloc[-WEEK:]
            if len(last_week) < WEEK:
                continue
            by_weekday = dict(zip(last_week.index.day_of_week, last_week.to_numpy()))
            out[future.index.get_indexer(rows.index)] = rows["day"].dt.day_of_week.map(by_weekday).to_numpy()
        return out


class YearOverYear:
    """Same weekday last year, scaled by recent growth. How merchants often plan."""

    name = "year_over_year"
    description = "last year's same weeks, scaled by growth over the last 4 weeks"
    intermittent_only = False
    recent = 28

    def min_history(self, horizon: int) -> int:
        # Last year's same weeks must exist, and can't reach past the cutoff.
        return YEAR + self.recent if horizon <= YEAR else 10**6

    def forecast(self, history: pd.DataFrame, future: pd.DataFrame) -> np.ndarray:
        df = demand(history)
        out = np.full(len(future), np.nan)
        for sku, rows in future.groupby("sku"):
            series = df[df["sku"] == sku].set_index("day")["demand"]
            if series.empty:
                continue
            cutoff = series.index.max()
            if (cutoff - series.index.min()).days < YEAR + self.recent or (rows["day"].max() - cutoff).days > YEAR:
                continue
            now = series.iloc[-self.recent:].sum()
            year_ago = series.reindex(series.index[-self.recent:] - pd.Timedelta(days=YEAR)).sum()
            # A product that barely sold a year ago would otherwise get a wild growth factor.
            growth = float(np.clip(now / year_ago, 0.25, 4.0)) if year_ago > 0 else 1.0
            last_year = series.reindex(rows["day"] - pd.Timedelta(days=YEAR)).to_numpy()
            out[future.index.get_indexer(rows.index)] = last_year * growth
        return out


class _StatsForecastModel:
    """A statsforecast model fitted per product, with one failure not spoiling the rest."""

    alias: str

    def build(self):
        raise NotImplementedError

    def forecast(self, history: pd.DataFrame, future: pd.DataFrame) -> np.ndarray:
        from statsforecast import StatsForecast

        df = _series(history, future)
        horizon = int(future.groupby("sku").size().max())
        try:
            result = StatsForecast(models=[self.build()], freq="D").forecast(df=df, h=horizon)
        except Exception:
            parts = []
            for sku, one in df.groupby("unique_id"):
                try:
                    parts.append(StatsForecast(models=[self.build()], freq="D").forecast(df=one, h=horizon))
                except Exception:
                    continue
            if not parts:
                return np.full(len(future), np.nan)
            result = pd.concat(parts)
        result = result.reset_index().rename(columns={"unique_id": "sku", "ds": "day"})
        merged = future[["sku", "day"]].merge(result[["sku", "day", self.alias]], on=["sku", "day"], how="left")
        return merged[self.alias].to_numpy(dtype=float)


class ETS(_StatsForecastModel):
    """Exponential smoothing with weekly seasonality (statsforecast AutoETS)."""

    name = "ets"
    description = "exponential smoothing with a weekly pattern, picked automatically per product"
    intermittent_only = False
    alias = "AutoETS"

    def min_history(self, horizon: int) -> int:
        return 4 * WEEK

    def build(self):
        from statsforecast.models import AutoETS

        return AutoETS(season_length=WEEK)


class Croston(_StatsForecastModel):
    """Croston's method with the Syntetos-Boylan correction, for products that sell now and then.

    It forecasts the size of a sale and the gap between sales separately; the
    forecast is the average rate, which is what stock planning needs.
    """

    name = "croston"
    description = "Croston's method (SBA): average sale size over the average gap between sales"
    intermittent_only = True
    alias = "CrostonSBA"

    def min_history(self, horizon: int) -> int:
        return 4 * WEEK

    def build(self):
        from statsforecast.models import CrostonSBA

        return CrostonSBA()


class TSB(_StatsForecastModel):
    """Teunter-Syntetos-Babai: like Croston, but the chance of a sale updates every day.

    So a product whose sales dry up sees its forecast decay instead of
    staying at the last average forever.
    """

    name = "tsb"
    description = "TSB: chance of a sale on a day times the usual sale size, updated daily"
    intermittent_only = True
    alias = "TSB"

    def min_history(self, horizon: int) -> int:
        return 4 * WEEK

    def build(self):
        from statsforecast.models import TSB as StatsTSB

        return StatsTSB(alpha_d=0.1, alpha_p=0.1)


class ADIDA(_StatsForecastModel):
    """Aggregate to the typical gap between sales, smooth, and spread back to days."""

    name = "adida"
    description = "ADIDA: sales summed over the typical gap between sales, smoothed and spread over days"
    intermittent_only = True
    alias = "ADIDA"

    def min_history(self, horizon: int) -> int:
        return 4 * WEEK

    def build(self):
        from statsforecast.models import ADIDA as StatsADIDA

        return StatsADIDA()


class GradientBoosting:
    """Gradient boosted trees over all products at once, with calendar and promotion features.

    scikit-learn's histogram gradient boosting is the LightGBM algorithm and
    installs anywhere; the lightgbm wheel on macOS needs Homebrew's libomp.
    On Linux servers LightGBM is a drop-in swap if it proves faster.

    One model learns from every product (as in the M5 competition), predicting
    each day relative to the product's recent level so a sofa and a candle can
    share what they learn about Black Friday. It forecasts every day directly,
    from features that are already known that far ahead.

    How far back "recent" is depends on how far ahead the day is: next week
    is predicted from the level a week before it, next month from the level a
    month before, and so on (lag buckets of 7, 14, 28, 56, 91 and 182 days up
    to the horizon). The training rows are repeated once per bucket with the
    lag as a feature, so one model learns them all. With a single lag as long
    as the horizon, a level change in the last few months stayed invisible to
    the whole forecast.
    """

    name = "gradient_boosting"
    description = "gradient boosted trees using promotions, holidays and last year's sales"
    intermittent_only = False
    level_days = 28
    lag_buckets = (7, 14, 28, 56, 91, 182)

    def __init__(self, seed: int = 7, single_lag: bool = False) -> None:
        self.seed = seed
        self.single_lag = single_lag  # the original design, kept for comparison

    def min_history(self, horizon: int) -> int:
        # The last day's level looks back a full horizon, and needs a week of it.
        return horizon + WEEK

    def lags(self, horizon: int) -> list[int]:
        if self.single_lag:
            return [horizon]
        return sorted({lag for lag in self.lag_buckets if lag < horizon} | {horizon})

    def features(self, frame: pd.DataFrame, lag: int) -> pd.DataFrame:
        g = frame.groupby("sku")["demand"]
        day = frame["day"]
        dates = day.dt.date
        level = g.transform(lambda s: s.shift(lag).rolling(self.level_days, min_periods=7).mean())
        level = level.clip(lower=0.5)
        feats = pd.DataFrame({
            "sku": frame["sku"].astype("category"),
            "weekday": day.dt.day_of_week,
            "day_of_year": day.dt.day_of_year,
            "to_black_friday": dates.map(calendar.days_to_black_friday).clip(-30, 40),
            "to_christmas": dates.map(calendar.days_to_christmas).clip(-40, 10),
            "discount": frame["discount"],
            "lag": lag,
            "recent_week": g.transform(lambda s: s.shift(lag).rolling(WEEK, min_periods=3).mean()) / level,
            "last_year": g.shift(YEAR) / level,
            "last_year_week": g.transform(lambda s: s.shift(YEAR - 3).rolling(WEEK, min_periods=3).mean()) / level,
        }, index=frame.index)
        return feats.assign(level=level)

    def forecast(self, history: pd.DataFrame, future: pd.DataFrame) -> np.ndarray:
        from sklearn.ensemble import HistGradientBoostingRegressor

        hist = demand(history)
        horizon = int(future.groupby("sku").size().max())
        origin = hist["day"].max()
        frame = pd.concat([hist.assign(future=False),
                           future.assign(units=np.nan, demand=np.nan, in_stock=True, known=False, future=True)],
                          ignore_index=True).sort_values(["sku", "day"], kind="stable").reset_index(drop=True)
        steps = (frame["day"] - origin).dt.days
        lags = self.lags(horizon)
        # Each future day uses the shortest lag that reaches back to the cutoff.
        bucket = np.array(lags)[np.searchsorted(lags, steps.clip(lower=1).to_numpy(), side="left").clip(max=len(lags) - 1)]
        train_parts, test_parts = [], []
        for lag in lags:
            feats = self.features(frame, lag)
            train = (~frame["future"]) & frame["known"].astype(bool) & feats["level"].notna() & feats["recent_week"].notna()
            train_parts.append(feats[train].assign(units=frame.loc[train, "units"]))
            test_parts.append(feats[frame["future"].to_numpy() & (bucket == lag)])
        train_rows = pd.concat(train_parts)
        test_rows = pd.concat(test_parts)
        if len(train_rows) < 200 * len(lags):
            return np.full(len(future), np.nan)
        # A feature with nothing to learn from (no last year yet, say) is left out.
        columns = [c for c in train_rows.columns if c not in ("level", "units") and train_rows[c].nunique() > 1]
        model = HistGradientBoostingRegressor(
            max_iter=400, learning_rate=0.03, max_leaf_nodes=15, min_samples_leaf=20,
            categorical_features="from_dtype", random_state=self.seed,
        )
        # One thread: on small data OpenMP's start-up costs far more than it saves
        # (a fit took 17 s with all cores and 0.2 s with one on a 10-core Mac).
        with threadpool_limits(1):
            # Relative target, weighted by level, so the loss is total unit error like WAPE.
            model.fit(train_rows[columns], train_rows["units"] / train_rows["level"], sample_weight=train_rows["level"])
            predicted = model.predict(test_rows[columns]) * test_rows["level"].to_numpy()
        keyed = frame.loc[test_rows.index, ["sku", "day"]].assign(yhat=predicted)
        merged = future[["sku", "day"]].merge(keyed, on=["sku", "day"], how="left")
        return merged["yhat"].clip(lower=0).to_numpy(dtype=float)


def default_models() -> list[Model]:
    return [SeasonalNaive(), YearOverYear(), ETS(), GradientBoosting(), Croston(), TSB(), ADIDA()]
