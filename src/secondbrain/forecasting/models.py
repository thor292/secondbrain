"""Forecasting model variants.

Every model gets the same inputs and returns daily unit forecasts:

- ``history``: rows (sku, day, units, discount, in_stock) up to the cutoff;
- ``future``: rows (sku, day, discount) for the days to forecast, with the
  planned promotions, since a merchant knows its sale calendar in advance.

The backtest runs every model on past data and keeps the best per product.
"""

from typing import Protocol

import numpy as np
import pandas as pd

from secondbrain.forecasting import calendar

WEEK = 7
YEAR = 364  # 52 weeks, so a year ago is the same weekday


class NotEnoughHistory(Exception):
    pass


class Model(Protocol):
    name: str
    description: str

    def forecast(self, history: pd.DataFrame, future: pd.DataFrame) -> np.ndarray:
        """Forecasts aligned with the rows of ``future``."""
        ...


def demand(history: pd.DataFrame) -> pd.DataFrame:
    """Units per day with out-of-stock days filled in.

    A day with nothing to sell records zero sales, not zero demand. Those days
    take the same weekday's value from the weeks around them, so models don't
    learn a slump that wasn't there.
    """
    df = history.sort_values(["sku", "day"]).copy()
    df["demand"] = df["units"].astype(float).where(df["in_stock"])
    weekday = df["day"].dt.day_of_week
    df["demand"] = df.groupby([df["sku"], weekday])["demand"].transform(lambda s: s.ffill().bfill())
    return df


class SeasonalNaive:
    """Same as the same weekday last week. The baseline every other model must beat."""

    name = "seasonal_naive"
    description = "repeats the last week"

    def forecast(self, history: pd.DataFrame, future: pd.DataFrame) -> np.ndarray:
        df = demand(history)
        out = np.empty(len(future))
        for sku, rows in future.groupby("sku"):
            last_week = df[df["sku"] == sku].set_index("day")["demand"].iloc[-WEEK:]
            by_weekday = dict(zip(last_week.index.day_of_week, last_week.to_numpy()))
            out[future.index.get_indexer(rows.index)] = rows["day"].dt.day_of_week.map(by_weekday).to_numpy()
        return out


class YearOverYear:
    """Same weekday last year, scaled by recent growth. How merchants often plan."""

    name = "year_over_year"
    description = "last year's same weeks, scaled by growth over the last 4 weeks"
    recent = 28

    def forecast(self, history: pd.DataFrame, future: pd.DataFrame) -> np.ndarray:
        df = demand(history)
        out = np.empty(len(future))
        for sku, rows in future.groupby("sku"):
            series = df[df["sku"] == sku].set_index("day")["demand"]
            cutoff = series.index.max()
            if (cutoff - series.index.min()).days < YEAR + self.recent or (rows["day"].max() - cutoff).days > YEAR:
                raise NotEnoughHistory(self.name)
            now = series.iloc[-self.recent:].sum()
            year_ago = series.reindex(series.index[-self.recent:] - pd.Timedelta(days=YEAR)).sum()
            growth = now / year_ago if year_ago > 0 else 1.0
            last_year = series.reindex(rows["day"] - pd.Timedelta(days=YEAR)).to_numpy()
            out[future.index.get_indexer(rows.index)] = last_year * growth
        return out


class ETS:
    """Exponential smoothing with weekly seasonality (statsforecast AutoETS)."""

    name = "ets"
    description = "exponential smoothing with a weekly pattern, picked automatically per product"

    def forecast(self, history: pd.DataFrame, future: pd.DataFrame) -> np.ndarray:
        from statsforecast import StatsForecast
        from statsforecast.models import AutoETS

        df = demand(history).rename(columns={"sku": "unique_id", "day": "ds", "demand": "y"})
        horizon = int(future.groupby("sku").size().max())
        sf = StatsForecast(models=[AutoETS(season_length=WEEK)], freq="D")
        result = sf.forecast(df=df[["unique_id", "ds", "y"]], h=horizon)
        result = result.rename(columns={"unique_id": "sku", "ds": "day"})
        merged = future[["sku", "day"]].merge(result, on=["sku", "day"], how="left")
        return merged["AutoETS"].to_numpy()


class GradientBoosting:
    """Gradient boosted trees over all products at once, with calendar and promotion features.

    scikit-learn's histogram gradient boosting is the LightGBM algorithm and
    installs anywhere; the lightgbm wheel on macOS needs Homebrew's libomp.
    On Linux servers LightGBM is a drop-in swap if it proves faster.

    One model learns from every product (as in the M5 competition), predicting
    each day relative to the product's recent level so a sofa and a candle can
    share what they learn about Black Friday. It forecasts every day directly,
    from features that are already known that far ahead: lags reach back at
    least as far as the forecast horizon.
    """

    name = "gradient_boosting"
    description = "gradient boosted trees using promotions, holidays and last year's sales"
    level_days = 28

    def __init__(self, seed: int = 7) -> None:
        self.seed = seed

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
            "recent_week": g.transform(lambda s: s.shift(lag).rolling(WEEK, min_periods=3).mean()) / level,
            "last_year": g.shift(YEAR) / level,
            "last_year_week": g.transform(lambda s: s.shift(YEAR - 3).rolling(WEEK, min_periods=3).mean()) / level,
        }, index=frame.index)
        return feats.assign(level=level)

    def forecast(self, history: pd.DataFrame, future: pd.DataFrame) -> np.ndarray:
        from sklearn.ensemble import HistGradientBoostingRegressor

        hist = demand(history)
        lag = int(future.groupby("sku").size().max())
        frame = pd.concat([hist.assign(future=False),
                           future.assign(units=np.nan, demand=np.nan, in_stock=True, future=True)],
                          ignore_index=True).sort_values(["sku", "day"], kind="stable")
        feats = self.features(frame, lag)
        train = (~frame["future"]) & frame["in_stock"] & feats["level"].notna() & feats["recent_week"].notna()
        if train.sum() < 200:
            raise NotEnoughHistory(self.name)
        columns = [c for c in feats.columns if c != "level"]
        model = HistGradientBoostingRegressor(
            max_iter=400, learning_rate=0.03, max_leaf_nodes=15, min_samples_leaf=20,
            categorical_features="from_dtype", random_state=self.seed,
        )
        # Relative target, weighted by level, so the loss is total unit error like WAPE.
        model.fit(feats.loc[train, columns], frame.loc[train, "units"] / feats.loc[train, "level"],
                  sample_weight=feats.loc[train, "level"])
        test = frame["future"]
        predicted = pd.Series(model.predict(feats.loc[test, columns]) * feats.loc[test, "level"].to_numpy(),
                              index=frame.index[test])
        keyed = frame.loc[test, ["sku", "day"]].assign(yhat=predicted)
        merged = future[["sku", "day"]].merge(keyed, on=["sku", "day"], how="left")
        return merged["yhat"].clip(lower=0).to_numpy()


def default_models() -> list[Model]:
    return [SeasonalNaive(), YearOverYear(), ETS(), GradientBoosting()]
