"""Forecast error metrics. See docs/FORECASTING.md for why WAPE picks the model.

All take arrays of actuals and forecasts over the same days.
"""

import numpy as np


def wape(actual: np.ndarray, forecast: np.ndarray) -> float:
    """Total absolute error as a share of total sales. Robust to slow days, unlike MAPE."""
    total = np.abs(actual).sum()
    return float(np.abs(actual - forecast).sum() / total) if total else float("nan")


def mase(actual: np.ndarray, forecast: np.ndarray, train: np.ndarray, season: int = 7) -> float:
    """Error relative to a seasonal naive forecast on the training data; below 1 beats it."""
    if len(train) <= season:
        return float("nan")
    scale = np.abs(train[season:] - train[:-season]).mean()
    return float(np.abs(actual - forecast).mean() / scale) if scale else float("nan")


def smape(actual: np.ndarray, forecast: np.ndarray) -> float:
    """Symmetric MAPE in [0, 2]; days where both are zero count as perfect."""
    denom = np.abs(actual) + np.abs(forecast)
    ratio = np.divide(2 * np.abs(actual - forecast), denom, out=np.zeros_like(denom, dtype=float),
                      where=denom > 0)
    return float(ratio.mean())


def bias(actual: np.ndarray, forecast: np.ndarray) -> float:
    """Over (positive) or under (negative) forecasting, as a share of total sales."""
    total = np.abs(actual).sum()
    return float((forecast - actual).sum() / total) if total else float("nan")


def rmsse(actual: np.ndarray, forecast: np.ndarray, train: np.ndarray, season: int = 7) -> float:
    """Root mean squared error scaled like MASE (the M5 metric).

    Squared error is smallest at the average, not the median, so for products
    that sell now and then it doesn't reward forecasting zero the way WAPE and
    MASE do. That is why it chooses the model for intermittent products.
    """
    if len(train) <= season or not len(actual):
        return float("nan")
    scale = np.mean((train[season:] - train[:-season]) ** 2)
    return float(np.sqrt(np.mean((actual - forecast) ** 2) / scale)) if scale else float("nan")
