"""Indian retail calendar: the dates that move home decor demand.

Dhanteras and Diwali follow the lunar calendar, so they move by up to three
weeks from year to year (1 Nov 2024, 20 Oct 2025, 8 Nov 2026). Models can't
learn them from the day of the year; they get the distance to them instead.
Holi and Raksha Bandhan move the same way. Independence Day (15 Aug) and
Republic Day (26 Jan) are fixed, and so are the wedding seasons, roughly.

The lunar dates are a table, not a formula: the years the demo's data and
forecasts reach (2023-2028). Outside it a year gets a typical date, which is
close enough for a feature but should be added to the table before real use.
"""

from datetime import date, timedelta

DIWALI = {2023: date(2023, 11, 12), 2024: date(2024, 11, 1), 2025: date(2025, 10, 20),
          2026: date(2026, 11, 8), 2027: date(2027, 10, 29), 2028: date(2028, 10, 17)}
DHANTERAS = {2023: date(2023, 11, 10), 2024: date(2024, 10, 29), 2025: date(2025, 10, 18),
             2026: date(2026, 11, 6), 2027: date(2027, 10, 27), 2028: date(2028, 10, 15)}
HOLI = {2023: date(2023, 3, 8), 2024: date(2024, 3, 25), 2025: date(2025, 3, 14),
        2026: date(2026, 3, 4), 2027: date(2027, 3, 22), 2028: date(2028, 3, 11)}
RAKSHA_BANDHAN = {2023: date(2023, 8, 30), 2024: date(2024, 8, 19), 2025: date(2025, 8, 9),
                  2026: date(2026, 8, 28), 2027: date(2027, 8, 17), 2028: date(2028, 8, 5)}
# The marketplaces' festive sale (Shopify / JioMart "Festive Sale") opens 4-5 weeks before Diwali.
FESTIVE_SALE_START = {2023: date(2023, 10, 8), 2024: date(2024, 9, 27), 2025: date(2025, 9, 22),
                      2026: date(2026, 10, 3), 2027: date(2027, 9, 25), 2028: date(2028, 9, 14)}

# Parcels ordered by Dhanteras minus 2 days arrive before Diwali.
SHIPPING_CUTOFF_DAYS = 2
# Lunar dates move: inside this window around Diwali (days before, days after),
# "the same day last year" means the same distance from Diwali, not the same date.
FESTIVE_WINDOW = (-45, 15)


def diwali(year: int) -> date:
    return DIWALI.get(year) or date(year, 11, 1)


def dhanteras(year: int) -> date:
    return DHANTERAS.get(year) or diwali(year) - timedelta(days=2)


def holi(year: int) -> date:
    return HOLI.get(year) or date(year, 3, 15)


def raksha_bandhan(year: int) -> date:
    return RAKSHA_BANDHAN.get(year) or date(year, 8, 15)


def festive_sale_start(year: int) -> date:
    return FESTIVE_SALE_START.get(year) or diwali(year) - timedelta(days=33)


def independence_day(year: int) -> date:
    return date(year, 8, 15)


def republic_day(year: int) -> date:
    return date(year, 1, 26)


def shipping_cutoff(year: int) -> date:
    """Last day an online parcel order still arrives before Diwali."""
    return dhanteras(year) - timedelta(days=SHIPPING_CUTOFF_DAYS)


def is_wedding_season(day: date) -> bool:
    """Mid-November to mid-December and mid-January to the end of February."""
    md = (day.month, day.day)
    return (11, 15) <= md <= (12, 15) or (1, 15) <= md <= (2, 29)


def _nearest(day: date, when) -> int:
    """Signed days from the nearest occurrence (negative before it)."""
    return min(((day - when(y)).days for y in (day.year - 1, day.year, day.year + 1)), key=abs)


def days_to_diwali(day: date) -> int:
    """Signed days from the nearest Diwali (negative before it)."""
    return _nearest(day, diwali)


def days_to_dhanteras(day: date) -> int:
    return _nearest(day, dhanteras)


def days_to_holi(day: date) -> int:
    return _nearest(day, holi)


def days_to_raksha_bandhan(day: date) -> int:
    return _nearest(day, raksha_bandhan)


def days_since_festive_sale(day: date) -> int:
    """Days since the nearest festive sale opened (negative before it)."""
    return _nearest(day, festive_sale_start)


def days_to_national_sale(day: date) -> int:
    """Signed days from the nearest Independence Day or Republic Day."""
    return min(_nearest(day, independence_day), _nearest(day, republic_day), key=abs)


def in_diwali_peak(day: date) -> bool:
    """The week before Dhanteras through the day after Diwali: the year's spike."""
    to = days_to_diwali(day)
    year = (day - timedelta(days=to)).year
    return dhanteras(year) - timedelta(days=7) <= day <= diwali(year) + timedelta(days=1)


def next_diwali(day: date) -> date:
    """This year's Diwali, or next year's once it has passed."""
    return diwali(day.year) if diwali(day.year) >= day else diwali(day.year + 1)


def last_diwali(day: date) -> date:
    """The most recent Diwali before ``day``."""
    return diwali(day.year) if diwali(day.year) < day else diwali(day.year - 1)


def same_day_last_year(day: date, year_days: int = 364) -> date:
    """The comparable day a year earlier: the same weekday 52 weeks back, except
    around Diwali, where it is the same distance from last year's Diwali."""
    to = days_to_diwali(day)
    if FESTIVE_WINDOW[0] <= to <= FESTIVE_WINDOW[1]:
        this = day - timedelta(days=to)
        return diwali(this.year - 1) + timedelta(days=to)
    return day - timedelta(days=year_days)
