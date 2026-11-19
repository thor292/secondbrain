"""Prediction intervals from backtest errors, calibrated so they hold as often as they claim.

Split conformal prediction, adapted to daily sales:

- **Normalised errors.** A day's error grows with its forecast, but not in
  proportion: a slow seller's days are mostly Poisson noise (spread ~ the
  square root of the forecast), a busy day's mostly relative error. So each
  backtest error is divided by ``sigma(f) = sqrt(a*f + b*f^2)``, with ``a`` and
  ``b`` fitted to the product's own squared errors (Papadopoulos et al.'s
  normalised conformal scores). Intervals then fit Dhanteras and a quiet
  Tuesday alike.
- **By horizon.** Quantiles of those scores are taken separately for days
  1-7, 8-28, 29-56 and later, when there are enough of them, and never get
  narrower further out.
- **Calibrated by cross-validation.** Leaving one backtest window out at a
  time, the others predict its range, and we count how often it held. If the
  80% range held only 70% of the time, every range is scaled up until it
  holds about 80% (and down, if it was too cautious). That held-out coverage
  is what the forecast's quality reports.
"""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.optimize import nnls

LEVELS = (80, 95)
BUCKETS = (7, 28, 56)  # last day of each horizon bucket; later days form the last one
MIN_PER_BUCKET = 30
SCALES = np.round(np.arange(0.6, 3.01, 0.05), 2)
F_FLOOR = 0.25  # a forecast of zero still gets a range


@dataclass
class Calibration:
    a: float = 1.0
    b: float = 0.25
    # bucket index -> level -> (low score, high score)
    quantiles: dict = field(default_factory=dict)
    scale: dict = field(default_factory=lambda: {level: 1.0 for level in LEVELS})
    max_step: int = 1
    coverage: dict = field(default_factory=lambda: {level: None for level in LEVELS})

    def sigma(self, f: np.ndarray) -> np.ndarray:
        f = np.asarray(f, dtype=float)
        return np.sqrt(self.a * np.maximum(f, F_FLOOR) + self.b * f * f)

    def bounds(self, steps: np.ndarray, forecast: np.ndarray, level: int) -> tuple[np.ndarray, np.ndarray]:
        steps, forecast = np.asarray(steps), np.asarray(forecast, dtype=float)
        table = np.array([self.quantiles[b][level] for b in range(len(BUCKETS) + 1)])
        lo_q, hi_q = table[_buckets(np.minimum(steps, self.max_step))].T
        # Beyond the horizon the backtest reached, uncertainty keeps growing.
        growth = np.sqrt(np.maximum(steps / self.max_step, 1.0))
        spread = self.sigma(forecast) * self.scale[level] * growth
        return forecast + lo_q * spread, forecast + hi_q * spread


def _buckets(steps: np.ndarray) -> np.ndarray:
    """Horizon bucket of each step (days ahead, from 1)."""
    return np.searchsorted(np.array(BUCKETS), np.asarray(steps), side="left")


def _variance_model(f: np.ndarray, err: np.ndarray) -> tuple[float, float]:
    """a, b >= 0 with squared error ~ a*f + b*f^2 (least squares)."""
    f = np.maximum(f, F_FLOOR)
    try:
        (a, b), _ = nnls(np.column_stack([f, f * f]), err ** 2)
    except Exception:
        a, b = 1.0, 0.0
    if a + b <= 0:
        a = 1.0
    # A Poisson floor: sales can't be less noisy than counting.
    return max(float(a), 0.5), float(b)


def _quantiles(scores: np.ndarray, steps: np.ndarray, level: int) -> dict[int, tuple[float, float]]:
    tail = (100 - level) / 200
    overall = (float(np.quantile(scores, tail)), float(np.quantile(scores, 1 - tail)))
    out, prev = {}, None
    buckets = _buckets(steps)
    for b in range(len(BUCKETS) + 1):
        mask = buckets == b
        lo, hi = (float(np.quantile(scores[mask], tail)), float(np.quantile(scores[mask], 1 - tail))) \
            if mask.sum() >= MIN_PER_BUCKET else overall
        if prev is not None:  # never narrower further out
            lo, hi = min(lo, prev[0]), max(hi, prev[1])
        out[b] = prev = (lo, hi)
    return out


def _fit(rows: pd.DataFrame) -> Calibration:
    f, y, steps = rows["forecast"].to_numpy(float), rows["units"].to_numpy(float), rows["step"].to_numpy()
    cal = Calibration()
    cal.a, cal.b = _variance_model(f, y - f)
    scores = (y - f) / cal.sigma(f)
    cal.quantiles = {}
    per_level = {level: _quantiles(scores, steps, level) for level in LEVELS}
    for b in range(len(BUCKETS) + 1):
        cal.quantiles[b] = {level: per_level[level][b] for level in LEVELS}
    cal.max_step = int(steps.max())
    return cal


def _covered(cal: Calibration, rows: pd.DataFrame, level: int, scale: float) -> np.ndarray:
    saved = cal.scale[level]
    cal.scale[level] = scale
    lo, hi = cal.bounds(rows["step"].to_numpy(), rows["forecast"].to_numpy(float), level)
    cal.scale[level] = saved
    y = rows["units"].to_numpy(float)
    return (y >= np.minimum(lo, rows["forecast"])) & (y <= np.maximum(hi, rows["forecast"]))


def calibrate(rows: pd.DataFrame) -> Calibration:
    """Fit intervals to backtest rows (window, step, units, forecast) on known days."""
    rows = rows[np.isfinite(rows["forecast"]) & np.isfinite(rows["units"])]
    if len(rows) < 14:
        return default(len(rows))
    cal = _fit(rows)
    folds = rows["window"].unique()
    if len(folds) < 2:
        cal.scale = {level: 1.2 for level in LEVELS}  # can't check: lean wide
        return cal
    # Each window's coverage when the other windows set the range, for every candidate scale.
    held_out = {level: np.zeros((len(SCALES), 0), dtype=bool) for level in LEVELS}
    for w in folds:
        train, test = rows[rows["window"] != w], rows[rows["window"] == w]
        if len(train) < 14:
            continue
        fold = _fit(train)
        for level in LEVELS:
            hits = np.array([_covered(fold, test, level, s) for s in SCALES])
            held_out[level] = np.concatenate([held_out[level], hits], axis=1)
    for level in LEVELS:
        if held_out[level].shape[1] == 0:
            continue
        rates = held_out[level].mean(axis=1)
        # The smallest scale whose held-out coverage reaches the nominal level.
        ok = np.flatnonzero(rates >= level / 100)
        i = int(ok[0]) if len(ok) else len(SCALES) - 1
        cal.scale[level] = float(SCALES[i])
        cal.coverage[level] = round(float(rates[i]), 3)
    # The 95% range must stay outside the 80% one.
    cal.scale[95] = max(cal.scale[95], cal.scale[80])
    return cal


def default(observations: int = 0) -> Calibration:
    """Wide ranges for a forecast with no backtest: about +-50% on top of counting noise."""
    from statistics import NormalDist

    cal = Calibration(a=1.0, b=0.25)
    widen = 1.5 if observations < 14 else 1.25
    cal.quantiles = {b: {level: (-NormalDist().inv_cdf(0.5 + level / 200) * widen,
                                 NormalDist().inv_cdf(0.5 + level / 200) * widen) for level in LEVELS}
                     for b in range(len(BUCKETS) + 1)}
    cal.max_step = 28
    return cal


def points(days: pd.Series, forecast: np.ndarray, cal: Calibration) -> list[dict]:
    """Forecast points with ordered, non-negative, finite ranges."""
    forecast = np.nan_to_num(np.asarray(forecast, dtype=float), nan=0.0, posinf=0.0, neginf=0.0).clip(min=0)
    steps = np.arange(1, len(forecast) + 1)
    lo80, hi80 = cal.bounds(steps, forecast, 80)
    lo95, hi95 = cal.bounds(steps, forecast, 95)
    out = []
    for i, day in enumerate(days):
        f = float(forecast[i])
        l80 = min(f, max(0.0, _finite(lo80[i], f)))
        h80 = max(f, _finite(hi80[i], f))
        l95 = min(l80, max(0.0, _finite(lo95[i], l80)))
        h95 = max(h80, _finite(hi95[i], h80))
        out.append({"day": pd.Timestamp(day).date().isoformat(), "forecast": round(f, 1),
                    "lo80": round(l80, 1), "hi80": round(h80, 1), "lo95": round(l95, 1), "hi95": round(h95, 1)})
    return out


def _finite(value: float, fallback: float) -> float:
    return float(value) if np.isfinite(value) else fallback


def coverage(rows: pd.DataFrame, pts: list[dict]) -> dict[int, float | None]:
    """Share of actual days inside each range: rows (day, units) against points."""
    by_day = {p["day"]: p for p in pts}
    hits = {level: [] for level in LEVELS}
    for day, units in zip(rows["day"], rows["units"]):
        p = by_day.get(pd.Timestamp(day).date().isoformat())
        if p is None or not np.isfinite(units):
            continue
        for level in LEVELS:
            hits[level].append(p[f"lo{level}"] <= units <= p[f"hi{level}"])
    return {level: (float(np.mean(hits[level])) if hits[level] else None) for level in LEVELS}
