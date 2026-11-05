"""How far a forecast can be trusted, in a grade, numbers and plain words.

Stored with every forecast (``forecasts.quality``) so the UI and decision
signals can lean on a good forecast and discount a weak one:

    {"grade": "good" | "fair" | "weak", "summary": "...", "pattern": "...",
     "history_days": 793, "wape": 0.29, "bias": 0.03, "coverage80": 0.81,
     "coverage95": 0.95, "backtest_windows": 3, "backtest_days": 120,
     "warnings": ["..."]}

Grades:

- **good**: at least a year of history, a full-length backtest on 2+ windows,
  typical error (WAPE) up to 35%, no recent break in the sales level, and an
  80% range that held 70-92% of the time when it was checked;
- **weak**: no backtest at all (new, discontinued or never sold), under 8
  weeks of history, or a typical error above 60% (intermittent products:
  RMSSE above 1.2), or an 80% range that held under 60% of the time;
- **fair**: everything in between. Intermittent products are at best fair:
  their daily numbers are averages of an uneven trickle.
"""

from secondbrain.forecasting.prepare import Profile

GOOD_WAPE, WEAK_WAPE = 0.35, 0.60
WEAK_RMSSE = 1.2


def assess(profile: Profile, model: str, score: dict | None, windows: int, window_days: int,
           horizon: int, coverage: dict) -> dict:
    cov80, cov95 = coverage.get(80), coverage.get(95)
    wape = score.get("wape") if score and not score.get("estimated") else None
    rmsse = score.get("rmsse") if score else None
    bias = score.get("bias") if score and not score.get("estimated") else None
    warnings = list(profile.warnings)
    if windows and window_days < horizon:
        warnings.append(f"Only tested on {windows} past {window_days}-day periods; days further ahead are less certain.")

    reasons_weak = []
    if not windows:
        reasons_weak.append({"cold_start": "too new to test against its own past sales",
                             "discontinued": "it looks discontinued",
                             "no_sales": "it has no sales yet"}.get(model, "it could not be tested on past sales"))
    if profile.history_days < 56 and profile.status == "active":
        reasons_weak.append(f"only {profile.history_days} days of history")
    if wape is not None and not profile.intermittent and wape > WEAK_WAPE:
        reasons_weak.append(f"past forecasts were off by {wape:.0%} on a typical day")
    if profile.intermittent and rmsse is not None and rmsse > WEAK_RMSSE:
        reasons_weak.append("past forecasts missed its sporadic sales by more than a naive guess")
    if cov80 is not None and cov80 < 0.6:
        reasons_weak.append(f"its 80% range held only {cov80:.0%} of the time in tests")

    tested = f"tested on {windows} past {window_days}-day period{'s' if windows != 1 else ''}"
    if reasons_weak:
        grade, summary = "weak", "Weak: " + "; ".join(reasons_weak) + ". Treat as a rough guide."
    elif (profile.history_days >= 365 and not profile.intermittent and windows >= 2 and window_days >= horizon
          and profile.level_shift is None
          and wape is not None and wape <= GOOD_WAPE and (cov80 is None or 0.7 <= cov80 <= 0.92)):
        held = f"; the 80% range held {cov80:.0%} of the time" if cov80 is not None else ""
        grade, summary = "good", f"Good: {tested}, off by {wape:.0%} on a typical day{held}."
    else:
        why = []
        if profile.intermittent:
            why.append(f"it sells on only {profile.sale_share:.0%} of days")
        if profile.history_days < 365:
            why.append("less than a year of history")
        if window_days < horizon:
            why.append(f"{tested}")
        if wape is not None and wape > GOOD_WAPE and not profile.intermittent:
            why.append(f"past forecasts were off by {wape:.0%} on a typical day")
        if cov80 is not None and not 0.7 <= cov80 <= 0.92:
            why.append(f"its 80% range held {cov80:.0%} of the time in tests")
        if profile.level_shift is not None:
            why.append("its sales level changed recently, so past tests say less about the coming weeks")
        if windows < 2 and windows:
            why.append("only one test period")
        grade, summary = "fair", "Fair: " + ("; ".join(why) or tested) + "."
    return {
        "grade": grade, "summary": summary,
        "pattern": profile.status if profile.status != "active" else
                   ("intermittent" if profile.intermittent else "new" if profile.history_days < 56 else "regular"),
        "history_days": profile.history_days,
        "wape": wape, "bias": bias,
        "coverage80": cov80, "coverage95": cov95,
        "backtest_windows": windows, "backtest_days": window_days if windows else 0,
        "warnings": warnings,
    }
