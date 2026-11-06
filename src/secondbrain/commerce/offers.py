"""Decision layer for Live Commerce: forecast deviations become priced offers, and offers teach it.

Every recommendation is an expected-profit choice, not a rule of thumb: for a discount the
grid (5-20%) is scored with the forecast, the stock, the cost and the price elasticity, and
the best one wins only if it beats doing nothing and keeps the margin floor. Leftover stock
counts at its realistic value: units needed before a reorder can land are full-price sales, normal
cycle stock only costs carrying, excess beyond the cycle loses value (perishables spoil). So
elastic, high-margin products and overstock get offers, while a thin-margin staple that is
merely slow does not: a discount there only gives margin away. After a drop, a volume-defence
offer (``choose_defence``) may also count the customers it keeps, valued by their measured baskets.

Every offer carries a promise: the 80% lower bound of its uplift (demand noise, forecast error,
elasticity uncertainty). An offer "worked" if it met that bound, so a calibrated engine hits ~80%.

The learning loop: when an applied offer's window closes, actual units are compared with the
no-offer forecast frozen when it was applied. That measured uplift gives an observed
elasticity, which moves the category's and the merchant's estimate (Bayesian shrinkage toward
the category prior), and feeds the hit rate. Priors are textbook values by category family.
"""

import hashlib
import math
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np

GRID = np.array([0.05, 0.08, 0.10, 0.12, 0.15, 0.20])
MARGIN_FLOOR = 0.12        # never price below this margin (or half the normal margin, if lower)
LATER_VALUE = 0.95         # a full-price sale lost later (stock gone before the reorder lands), discounted
HOLD_COST = 0.01           # carrying cost of normal stock over an offer window, share of cost
EXCESS_COST = 0.15         # share of cost lost on stock beyond what the cycle needs (carrying, markdowns)
CYCLE_DAYS = 21            # stock beyond lead time + three weeks of demand counts as excess
SPOIL_COST = 0.8           # perishables: most of it is thrown away
SAFETY_Z = 1.65            # 95% service level
REVIEW_DAYS = 7            # an order covers lead time + a week
WINBACK_RESPONSE = 0.15    # assumed share of lapsed customers a coupon brings back
RECOVER_SHARE = 0.5        # a volume-defence offer aims to win back at least half of the lost units...
MIN_RECOVER = 0.25         # ...or, if the margin floor allows no more, at least a quarter
KEEP_DAYS = 28             # customers kept by defending a price are worth their next four weeks of baskets...
KEEP_PROB = 0.5            # ...counted at half, since some would have come back anyway
Z_ASSURED = 0.84           # the promised uplift is the 80% lower bound: four offers in five should meet it
COOLDOWN_HOURS = {"discount": 48, "restock": 72, "surge": 24, "festival": 72, "clearance": 20, "winback": 168}
AUTO_KINDS = {"discount", "clearance", "festival"}

PRIORS = {"staples": -1.0, "fashion": -2.5, "electronics": -1.8, "beauty": -1.5, "home": -1.6}
PRIOR_STRENGTH = 3.0       # the prior counts as this many well-measured offers
PRIOR_SD = 0.6             # uncertainty of a category prior
MERCHANT_STRENGTH = 4.0    # a merchant's own offers start to matter after a handful


def prior_group(category: str) -> str:
    c = category.lower()
    for key, words in (("fashion", ("fashion", "apparel", "cloth", "footwear")),
                       ("electronics", ("electronic", "mobile", "appliance", "gadget")),
                       ("beauty", ("beauty", "personal", "cosmetic")),
                       ("home", ("home", "household", "furnish", "kitchen"))):
        if any(w in c for w in words):
            return key
    return "staples"


def format_inr(x: float) -> str:
    """Rupees the Indian way: ₹45,600, ₹3.4 lakh, ₹2.15 crore."""
    sign = "-" if x < 0 else ""
    x = abs(x)
    if x >= 1e7:
        return f"{sign}₹{x / 1e7:.2f} crore"
    if x >= 1e5:
        return f"{sign}₹{x / 1e5:.2f} lakh"
    return f"{sign}₹{round(x):,}"


def autopilot_merchant(merchant_id: str, fraction: float) -> bool:
    """Deterministic per merchant, so the same shops are on autopilot in every run."""
    h = int.from_bytes(hashlib.md5(merchant_id.encode()).digest()[:4], "big") / 2 ** 32
    return h < fraction


def margin_floor(price: np.ndarray, cost: np.ndarray, floor: float = MARGIN_FLOOR) -> np.ndarray:
    """The lower of the floor and half the normal margin (staples and electronics run on ~10%)."""
    return np.minimum(floor, 0.5 * (price - cost) / price)


def choose_discount(price, cost, stock, window_units, lead_units, cycle_units, elasticity, excess_cost, floor,
                    grid: np.ndarray = GRID) -> dict:
    """Best discount per candidate (arrays of equal length); 0 when nothing beats no offer.

    window_units: no-offer demand during the offer window. lead_units: demand after it until a reorder
    could land; units sold now that leave those customers short are full-price sales lost. cycle_units:
    the demand the stock is meant for; anything beyond it is excess that loses ``excess_cost`` of its cost
    (carrying and markdowns, or spoilage). Value = margin in the window - lost later margin - holding.
    """
    price, cost, stock = (np.asarray(a, float) for a in (price, cost, stock))
    d = np.concatenate([[0.0], grid])[None, :]
    p = price[:, None] * (1 - d)
    lift = (1 - d) ** np.asarray(elasticity, float)[:, None]
    units = np.minimum(stock[:, None], np.asarray(window_units, float)[:, None] * lift)
    left = stock[:, None] - units
    short = np.maximum(np.asarray(lead_units, float)[:, None] - left, 0)
    cycle = np.asarray(cycle_units, float)[:, None]
    hold = HOLD_COST * np.minimum(left, cycle) + np.asarray(excess_cost, float)[:, None] * np.maximum(left - cycle, 0)
    value = (units * (p - cost[:, None]) - short * ((price - cost) * LATER_VALUE)[:, None] - hold * cost[:, None])
    ok = p - cost[:, None] >= np.asarray(floor, float)[:, None] * p
    ok[:, 0] = True
    value = np.where(ok, value, -np.inf)
    best = value.argmax(axis=1)
    r = np.arange(len(price))
    pb = p[r, best]
    return {"discount": d[0, best], "units": units[r, best], "uplift": units[r, best] - units[:, 0],
            "revenue": units[r, best] * pb, "margin": (pb - cost) / pb, "gain": value[r, best] - value[:, 0],
            "price": pb}


def choose_defence(price, cost, stock, cf_units, gap_units, elasticity, floor, keep_value,
                   grid: np.ndarray = GRID) -> dict:
    """Volume defence after a drop (a competitor's price cut, a local dip): the smallest discount expected to win
    back RECOVER_SHARE of the lost units, else the largest the floor allows if it wins back MIN_RECOVER.

    cf_units: expected units without an offer (the drop continuing); gap_units: units lost vs normal.
    keep_value: rupees per recovered unit from the customers it keeps (their future baskets). It is chosen only
    if margin on recovered units + keep value pays for the discount given on the units that sell anyway.
    """
    price, cost, stock, cf, gap = (np.asarray(a, float) for a in (price, cost, stock, cf_units, gap_units))
    d = grid[None, :]
    p = price[:, None] * (1 - d)
    lift = (1 - d) ** np.asarray(elasticity, float)[:, None]
    rec = np.minimum(cf[:, None] * (lift - 1), np.maximum(stock - cf, 0)[:, None])
    ok = p - cost[:, None] >= np.asarray(floor, float)[:, None] * p
    share = rec / np.maximum(gap, 1e-9)[:, None]
    full = ok & (share >= RECOVER_SHARE)
    part = ok & (share >= MIN_RECOVER)
    last_ok = len(grid) - 1 - np.argmax(part[:, ::-1], axis=1)
    pick = np.where(full.any(axis=1), np.argmax(full, axis=1), np.where(part.any(axis=1), last_ok, -1))
    r = np.arange(len(price))
    k = np.maximum(pick, 0)
    gain = (rec * (p - cost[:, None]) - cf[:, None] * price[:, None] * d
            + rec * np.asarray(keep_value, float)[:, None])[r, k]
    valid = (pick >= 0) & (gain >= 0)
    pb = p[r, k]
    return {"discount": np.where(valid, grid[k], 0.0), "units": cf + rec[r, k], "uplift": rec[r, k],
            "recovered_share": share[r, k], "revenue": (cf + rec[r, k]) * pb, "margin": (pb - cost) / pb,
            "gain": gain, "price": pb, "margin_cost": cf * price * grid[k]}


def restock_plan(stock, mean_daily, sd_daily, lead_days, perishable=False) -> dict:
    """Cover in days, the reorder threshold (lead time + safety) and units to order (perishables: two days' worth)."""
    stock, mean, sd, lead = (np.asarray(a, float) for a in (stock, mean_daily, sd_daily, lead_days))
    review = np.where(np.asarray(perishable, bool), 2, REVIEW_DAYS)
    safety = SAFETY_Z * sd * np.sqrt(np.maximum(lead, 1))
    m = np.maximum(mean, 1e-6)
    cover = stock / m
    threshold = lead + np.maximum(1.0, safety / m)
    units = np.ceil(np.maximum(mean * (lead + review) + safety - stock, 0))
    units = np.where(units > 20, np.ceil(units / 5) * 5, units)
    return {"cover": cover, "threshold": threshold, "units": units, "safety": safety, "need": cover < threshold}


class Cooldowns:
    """Per (merchant, sku, kind) quiet period in simulated hours, so one problem is one signal."""

    def __init__(self):
        self.until: dict[tuple, float] = {}

    def ready(self, key: tuple, now_h: float) -> bool:
        return self.until.get(key, -1e18) <= now_h

    def hit(self, key: tuple, now_h: float, hours: float | None = None) -> None:
        self.until[key] = now_h + (hours if hours is not None else COOLDOWN_HOURS[key[-1]])

    def clear(self, key: tuple) -> None:
        self.until.pop(key, None)


class Elasticity:
    """Price elasticity per category and per merchant x category, shrunk toward the category prior."""

    def __init__(self):
        self.cat: dict[str, list[float]] = {}     # category -> [sum w*e, sum w, n]
        self.merch: dict[tuple[str, str], list[float]] = {}
        self.backfill: dict[str, tuple[float, float, int]] = {}  # category -> (elasticity, weight, promo days)

    @staticmethod
    def prior(category: str) -> float:
        return PRIORS[prior_group(category)]

    def category_value(self, category: str) -> float:
        s = self.cat.get(category, [0.0, 0.0, 0])
        e, w, _ = self.backfill.get(category, (0.0, 0.0, 0))
        return (PRIOR_STRENGTH * self.prior(category) + w * e + s[0]) / (PRIOR_STRENGTH + w + s[1])

    def observe_backfill(self, evidence: dict[str, tuple[float, float]]) -> None:
        """Elasticities measured on past promotions in the history, worth up to one prior."""
        for category, (e, days) in evidence.items():
            self.backfill[category] = (float(np.clip(e, -6.0, 0.5)), min(days / 8.0, PRIOR_STRENGTH), int(days))

    def value(self, merchant: str, category: str) -> float:
        base = self.category_value(category)
        s = self.merch.get((merchant, category))
        return base if not s else (MERCHANT_STRENGTH * base + s[0]) / (MERCHANT_STRENGTH + s[1])

    def sd(self, merchant: str, category: str) -> float:
        w = self.cat.get(category, [0, 0.0, 0])[1] + self.backfill.get(category, (0, 0.0, 0))[1]
        w += self.merch.get((merchant, category), [0, 0.0, 0])[1]
        return PRIOR_SD * math.sqrt(PRIOR_STRENGTH / (PRIOR_STRENGTH + w))

    def update(self, merchant: str, category: str, observed: float, weight: float) -> None:
        e = float(np.clip(observed, -6.0, 0.5))
        for table, key in ((self.cat, category), (self.merch, (merchant, category))):
            s = table.setdefault(key, [0.0, 0.0, 0])
            s[0] += weight * e
            s[1] += weight
            s[2] += 1

    def summary(self, categories: list[str]) -> dict:
        return {c: {"prior": self.prior(c), "learned": round(self.category_value(c), 3),
                    "n": int(self.cat.get(c, [0, 0, 0])[2]),
                    "backfill_promo_days": self.backfill.get(c, (0, 0, 0))[2]} for c in categories}


@dataclass
class Offer:
    signal_id: str
    series: int
    platform: str
    merchant: str
    merchant_name: str
    sku: str
    product: str
    category: str
    discount: float
    start: datetime
    until: datetime
    by: str
    list_price: float
    baseline_units: float      # no-offer forecast for the window, frozen at apply time
    expected_uplift: float
    elasticity_before: float
    assured_uplift: float = 0.0   # the 80% lower bound promised with the signal
    kind: str = "discount"
    units: float = 0.0
    revenue: float = 0.0
    stockout: bool = False
    applied_ms: int = 0


@dataclass
class OfferBook:
    """Applied offers: what they sold during their window, and what that taught us."""
    elasticity: Elasticity = field(default_factory=Elasticity)
    active: dict[str, Offer] = field(default_factory=dict)
    by_series: dict[int, str] = field(default_factory=dict)
    history: deque = field(default_factory=lambda: deque(maxlen=5000))
    outcomes: deque = field(default_factory=lambda: deque(maxlen=500))
    applied: int = 0
    n_outcomes: int = 0
    n_worked: int = 0
    n_judged: int = 0
    uplift_revenue: float = 0.0

    def apply(self, offer: Offer) -> None:
        old = self.by_series.get(offer.series)
        if old in self.active:          # a newer offer replaces the running one; judge the old one now
            self.active[old].until = offer.start
        self.active[offer.signal_id] = offer
        self.by_series[offer.series] = offer.signal_id
        self.history.append(offer)
        self.applied += 1

    def record(self, series: int, units: float, revenue: float) -> None:
        sid = self.by_series.get(series)
        if sid is not None:
            o = self.active[sid]
            o.units += units
            o.revenue += revenue

    def due(self, now: datetime) -> list[Offer]:
        done = [o for o in self.active.values() if o.until <= now]
        for o in done:
            del self.active[o.signal_id]
            if self.by_series.get(o.series) == o.signal_id:
                del self.by_series[o.series]
        return done

    def evaluate(self, o: Offer) -> dict:
        """Measured uplift vs the frozen no-offer forecast; updates elasticity unless stock ran out."""
        base = max(o.baseline_units, 0.0)
        uplift = o.units - base
        before = self.elasticity.value(o.merchant, o.category)
        if not o.stockout and base >= 1 and o.discount > 0:
            observed = math.log(max(o.units, 0.5) / base) / math.log(1 - o.discount)
            self.elasticity.update(o.merchant, o.category, observed, base / (base + 5.0))
        after = self.elasticity.value(o.merchant, o.category)
        gain = o.revenue - base * o.list_price
        worked = bool(uplift > 0 and uplift >= o.assured_uplift)
        self.n_outcomes += 1
        if not o.stockout or worked:   # a window that ran out of stock below its promise can't be judged
            self.n_judged += 1
            self.n_worked += worked
        self.uplift_revenue += gain
        out = {"signal_id": o.signal_id, "sku": o.sku, "product": o.product, "merchant_name": o.merchant_name,
               "discount": o.discount, "expected_uplift_units": round(o.expected_uplift, 1),
               "actual_uplift_units": round(uplift, 1), "uplift_pct": round(uplift / base, 4) if base else None,
               "revenue_gain": round(gain, 2), "worked": worked, "elasticity_before": round(before, 3),
               "elasticity_after": round(after, 3), "category": o.category,
               "merchant": o.merchant, "platform": o.platform, "actual_units": round(o.units, 1),
               "baseline_units": round(base, 1), "assured_uplift_units": round(o.assured_uplift, 1),
               "censored": o.stockout, "by": o.by, "kind": o.kind}
        self.outcomes.append(out)
        return out

    def hit_rate(self) -> float | None:
        return round(self.n_worked / self.n_judged, 4) if self.n_judged else None
