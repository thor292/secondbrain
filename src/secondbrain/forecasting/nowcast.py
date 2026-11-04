"""Adjust a stored forecast with the sales that arrived after it was made.

Models retrain on a schedule, but sales keep coming in. If the days since the
cutoff ran above or below the forecast, the rest of the forecast moves the same
way, partly: a day or two of surprise moves it a little, a full week more.
This is the realistic part of "evolving every second".
"""

from dataclasses import dataclass
from datetime import date

# Days of evidence at which an observed surprise counts half.
HALF_WEIGHT_DAYS = 7
MAX_FACTOR, MIN_FACTOR = 1.5, 0.67


@dataclass
class Adjustment:
    days_observed: int
    actual: float
    forecast: float
    factor: float


def adjust(points: list[dict], actuals: dict[date, float]) -> tuple[list[dict], Adjustment]:
    """Points for the days still ahead, scaled by how the observed days compared."""
    seen = [p for p in points if date.fromisoformat(p["day"]) in actuals]
    actual = sum(actuals[date.fromisoformat(p["day"])] for p in seen)
    expected = sum(p["forecast"] for p in seen)
    factor = 1.0
    if seen and expected > 0:
        weight = len(seen) / (len(seen) + HALF_WEIGHT_DAYS)
        factor = min(MAX_FACTOR, max(MIN_FACTOR, 1 + weight * (actual / expected - 1)))
    ahead = [
        {k: (round(v * factor, 1) if k != "day" else v) for k, v in p.items()}
        for p in points if date.fromisoformat(p["day"]) not in actuals
    ]
    return ahead, Adjustment(len(seen), actual, round(expected, 1), round(factor, 3))
