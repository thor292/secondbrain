"""US retail calendar: the dates that move home decor demand.

Black Friday moves by up to a week from year to year, so models can't learn
it from the day of the year alone; they get the distance to it instead.
"""

from datetime import date, timedelta


def thanksgiving(year: int) -> date:
    """Fourth Thursday of November."""
    first = date(year, 11, 1)
    return first + timedelta(days=(3 - first.weekday()) % 7 + 21)


def black_friday(year: int) -> date:
    return thanksgiving(year) + timedelta(days=1)


def cyber_monday(year: int) -> date:
    return thanksgiving(year) + timedelta(days=4)


def memorial_day(year: int) -> date:
    """Last Monday of May."""
    last = date(year, 5, 31)
    return last - timedelta(days=last.weekday())


def labor_day(year: int) -> date:
    """First Monday of September."""
    first = date(year, 9, 1)
    return first + timedelta(days=(7 - first.weekday()) % 7)


def days_to_black_friday(day: date) -> int:
    """Signed days from this year's Black Friday (negative before it)."""
    return (day - black_friday(day.year)).days


def days_to_christmas(day: date) -> int:
    return (day - date(day.year, 12, 25)).days
