"""Live Commerce simulator: a demand model with hidden truths, streamed over MQTT as if from the platforms.

Every merchant x SKU is a series with a daily rate built from effects a real retailer
sees: a lognormal base by platform and category, the weekday (weekends up for DMart),
payday (1st-5th), festivals (a ramp into Dhanteras and Diwali, a dip right after), slow
trend drift and day-to-day noise. On top of that sit the events the engine's signals are
meant to catch, all hidden from it:

- shocks: a competitor price cut drops one series 25-40% for 3-10 days (discount
  signals), and a few viral surges lift one +60-150% for 2-5 days (surge signals);
- stock: merchants reorder on a fixed rule sized for normal demand, so festival ramps
  and sloppy merchants run out; stockouts cap sales until the restock lands after the
  lead time (restock signals);
- price elasticity per category with merchant variation: an ``apply_offer`` multiplies
  demand by ``(1 - discount) ** elasticity`` until ``until`` (what the engine learns);
- customers who lapse (win-back signals).

The backfill (14 months, last year's festivals included) runs the same model day by day, stock and reorders included, so
history and live are one continuous world and the catalog's stock is where history ended.

Live, a wall tick every 100 ms advances the sim clock by speed x 0.1 s, draws Poisson
order lines for all series at once (numpy) using the exact share of the day's demand that
falls in that slice of the intraday profile, and publishes one ``orders`` message per
merchant that sold something. If generating and publishing take longer than the tick
allows, the sim clock slows instead of the loop drifting, and ``_clock.speed`` reports the
speed actually achieved. ``{"cmd": "run_days", "days": N}`` (the HQ time machine) runs at the
fastest speed until N sim days have passed, then pauses and goes back to the speed it had;
``_clock.run_until`` shows the target while it runs.

    uv run python -m secondbrain.commerce.simulator --autostart
"""

import argparse
import json
import logging
import math
import signal
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Callable

import numpy as np
import paho.mqtt.client as mqtt

from . import contract
from .world import CATEGORIES, PLATFORMS, World, build_world, catalog_payload, customer_id, world_payload

log = logging.getLogger(__name__)

SIM_START = datetime.fromisoformat(contract.SIM_START)
DAY = 86400.0
TICK = 0.1
FASTEST = max(contract.SPEEDS, key=contract.SPEEDS.get)   # the time machine's speed ("1day")
MAX_RUN_DAYS = 60
CATS = list(CATEGORIES)
PIDS = [p.id for p in PLATFORMS]

# Hidden truths. Order lines per SKU per day for a merchant of size 1, before popularity.
CAT_BASE = {"Staples": 2.0, "Snacks": 1.8, "Sweets": 0.8, "Dairy": 3.0, "Fresh": 2.5, "Personal Care": 1.0,
            "Household": 1.0, "Beauty": 0.8, "Fashion": 0.7, "Electronics": 0.25, "Electronics Accessories": 0.8,
            "Home": 0.6}
ELASTICITY = {"Staples": (0.8, 1.3), "Snacks": (1.3, 1.8), "Sweets": (1.5, 2.2), "Dairy": (0.5, 0.9),
              "Fresh": (0.9, 1.4), "Personal Care": (1.2, 1.7), "Household": (1.1, 1.6), "Beauty": (1.8, 2.6),
              "Fashion": (2.0, 3.0), "Electronics": (1.5, 2.2), "Electronics Accessories": (1.8, 2.6),
              "Home": (1.6, 2.4)}
WEEKDAY = {  # Monday .. Sunday
    "shopify": [1.05, 1.02, 1.0, 1.0, 0.98, 0.95, 1.0],
    "reliance": [0.92, 0.90, 0.92, 0.95, 1.05, 1.20, 1.25],
    "dmart": [0.85, 0.82, 0.85, 0.88, 1.05, 1.30, 1.45],
}
HOURLY = {  # relative demand per hour of day; nothing between 2 and 6 am anywhere
    "shopify": [.6, .3, 0, 0, 0, 0, .1, .3, .5, .7, .9, 1, 1.1, 1.1, 1, .9, .9, 1, 1.2, 1.4, 1.7, 1.9, 1.7, 1.2],
    "reliance": [.15, .05, 0, 0, 0, 0, .1, .3, .5, .8, 1, 1.1, 1.1, 1, .9, .9, 1, 1.2, 1.5, 1.8, 1.8, 1.4, .8, .4],
    "dmart": [0, 0, 0, 0, 0, 0, 0, 0, .3, .6, .9, 1, 1, .9, .8, .8, 1, 1.3, 1.7, 2, 1.9, 1.3, .3, 0],
}
PAYDAY = {"Staples": .30, "Snacks": .20, "Household": .25, "Personal Care": .25, "Dairy": .05, "Fresh": .05,
          "Beauty": .20, "Fashion": .25, "Electronics": .25, "Electronics Accessories": .20, "Home": .20, "Sweets": .10}


@dataclass(frozen=True)
class Festival:
    window: int                 # days of ramp before
    tau: float                  # ramp steepness (days)
    boost: dict[str, float]     # extra demand on the day, by category
    after_days: int = 0
    after_tau: float = 2.5
    after: dict[str, float] = field(default_factory=dict)  # negative: dip after; positive: plateau


FESTIVAL_EFFECTS = {
    "Dussehra": Festival(10, 4, {"Fashion": .5, "Electronics": .4, "Electronics Accessories": .3, "Sweets": .5,
                                 "Home": .2}, 2, 2, {"Fashion": -.1, "Electronics": -.1}),
    "Dhanteras": Festival(21, 6, {"Electronics": 1.8, "Electronics Accessories": .8, "Home": 1.0, "Household": .4,
                                  "Sweets": .6, "Fashion": .8, "Beauty": .5, "Staples": .25, "Snacks": .3,
                                  "Personal Care": .2}),
    "Diwali": Festival(21, 5, {"Sweets": 2.0, "Snacks": .6, "Fashion": .7, "Home": .6, "Beauty": .5, "Staples": .2,
                               "Electronics": .5, "Electronics Accessories": .4, "Dairy": .3, "Household": .3},
                       7, 2.5, {"Electronics": -.45, "Fashion": -.4, "Home": -.4, "Sweets": -.5, "Snacks": -.3,
                                "Beauty": -.3, "Electronics Accessories": -.35, "Staples": -.15, "Household": -.25,
                                "Personal Care": -.1}),
    "Bhai Dooj": Festival(4, 2, {"Sweets": .6, "Beauty": .3, "Fashion": .2, "Electronics Accessories": .2}),
    "Wedding season": Festival(25, 12, {"Fashion": .6, "Electronics": .3, "Home": .3, "Sweets": .3, "Beauty": .4},
                               21, 15, {"Fashion": .45, "Electronics": .2, "Home": .2, "Sweets": .2, "Beauty": .3}),
    "Christmas": Festival(10, 4, {"Sweets": .4, "Beauty": .3, "Fashion": .3, "Electronics Accessories": .3,
                                  "Home": .2}, 3, 2, {"Sweets": -.1, "Fashion": -.1}),
}

SHOCK_RATE = 0.0025       # competitor price cuts per series per day (~40 a day at default scale)
SURGE_RATE = 0.0005       # viral surges per series per day
SURGE_CATS = {"Beauty", "Fashion", "Electronics Accessories", "Sweets", "Home"}
SHOCK_HORIZON = 180       # days after SIM_START with scheduled shocks


def _cat_vec(values: dict[str, float]) -> np.ndarray:
    return np.array([values.get(c, 0.0) for c in CATS])


def sim_date(day: int) -> date:
    return SIM_START.date() + timedelta(days=day)


def sim_ts(t: float) -> str:
    return (SIM_START + timedelta(seconds=int(t))).isoformat(timespec="seconds")


def parse_ts(ts: str) -> float:
    return (datetime.fromisoformat(ts) - SIM_START).total_seconds()


def festival_boost(day: date) -> np.ndarray:
    """Extra demand per category from the festival calendar (0 = none, negative = dip)."""
    b = np.zeros(len(CATS))
    for fdate, name in contract.FESTIVALS.items():
        e = FESTIVAL_EFFECTS.get(name)
        if e is None:
            continue
        dt = (fdate - day).days
        if 0 <= dt <= e.window:
            b += _cat_vec(e.boost) * math.exp(-dt / e.tau)
        elif 0 < -dt <= e.after_days:
            b += _cat_vec(e.after) * math.exp((dt + 1) / e.after_tau)
    return b


def payday_factor(day: date) -> np.ndarray:
    amp = _cat_vec(PAYDAY)
    if day.day <= 5:
        return 1 + amp * (6 - day.day) / 5
    if day.day >= 26:
        return 1 - 0.3 * amp
    return np.ones(len(CATS))


def _intraday_cdf() -> np.ndarray:
    """Per platform, the share of a day's demand done by each hour boundary (3 x 25)."""
    h = np.array([HOURLY[p] for p in PIDS], float)
    return np.concatenate([np.zeros((3, 1)), np.cumsum(h, axis=1) / h.sum(axis=1, keepdims=True)], axis=1)


class DemandModel:
    """The hidden truth for every series (merchant x SKU), laid out merchant by merchant."""

    def __init__(self, world: World):
        rng = np.random.default_rng([world.seed, 1])
        self.world = world
        self.seed = world.seed
        sizes = [len(c) for c in world.catalogs]
        self.n_merchants = len(world.merchants)
        self.offsets = np.concatenate([[0], np.cumsum(sizes)]).astype(np.int64)
        self.merchant = np.repeat(np.arange(self.n_merchants), sizes)
        products = [p for c in world.catalogs for p in c]
        self.n = len(products)
        self.sku = [p.sku for p in products]
        self.cat = np.array([CATS.index(p.category) for p in products])
        self.plat = np.array([PIDS.index(m.platform) for m in world.merchants])[self.merchant]
        self.price = np.array([p.price for p in products])
        self.lead = np.array([p.lead_days for p in products])
        self.perishable = np.array([p.perishable for p in products])
        msize = np.array([m.size for m in world.merchants])
        self.base = _cat_vec(CAT_BASE)[self.cat] * msize[self.merchant] * rng.lognormal(0, 0.7, self.n)
        lo, hi = _cat_vec({c: v[0] for c, v in ELASTICITY.items()}), _cat_vec({c: v[1] for c, v in ELASTICITY.items()})
        self.elasticity = -rng.uniform(lo[self.cat], hi[self.cat]) * rng.lognormal(0, 0.1, self.n_merchants)[self.merchant]
        self.qty_extra = _cat_vec({c: v.qty_mean - 1 for c, v in CATEGORIES.items()})[self.cat]
        self.slope = rng.normal(0, 0.04, self.n) + rng.normal(0, 0.03, self.n_merchants)[self.merchant]
        self.fest_sens = rng.lognormal(0, 0.2, self.n_merchants)[self.merchant]
        self.weekday = np.array([WEEKDAY[p] for p in PIDS])
        self.cdf = _intraday_cdf()
        self._shocks(rng)
        self._promos(rng)
        self._reorder_policy(rng)
        self._day_cache: dict[int, np.ndarray] = {}

    def _shocks(self, rng: np.random.Generator) -> None:
        span = contract.HISTORY_DAYS + SHOCK_HORIZON
        w = np.sqrt(self.base)
        neg = rng.poisson(w / w.mean() * SHOCK_RATE * span)
        surge_ok = np.isin(self.cat, [CATS.index(c) for c in SURGE_CATS])
        pos = rng.poisson(surge_ok * SURGE_RATE * span * self.n / max(surge_ok.sum(), 1))
        s_neg, s_pos = np.repeat(np.arange(self.n), neg), np.repeat(np.arange(self.n), pos)
        start = rng.integers(-contract.HISTORY_DAYS, SHOCK_HORIZON, len(s_neg) + len(s_pos))
        dur = np.concatenate([rng.integers(3, 11, len(s_neg)), rng.integers(2, 6, len(s_pos))])
        self.shock_series = np.concatenate([s_neg, s_pos])
        self.shock_start, self.shock_end = start, start + dur
        self.shock_factor = np.concatenate([1 - rng.uniform(0.25, 0.40, len(s_neg)), 1 + rng.uniform(0.6, 1.5, len(s_pos))])

    def _promos(self, rng: np.random.Generator) -> None:
        """Merchants' own past promotions in the backfill, so history carries some price variation."""
        pick = np.flatnonzero(rng.random(self.n) < 0.04)
        self.promo_series = pick
        self.promo_start = rng.integers(-contract.HISTORY_DAYS + 2, -4, len(pick))
        self.promo_end = self.promo_start + rng.integers(3, 6, len(pick))
        self.promo_disc = rng.choice([0.05, 0.10, 0.15], len(pick))

    def _reorder_policy(self, rng: np.random.Generator) -> None:
        """A fixed rule sized for normal demand; sloppy merchants keep thin safety stock."""
        units = self.base * (1 + self.qty_extra)
        safety = rng.uniform(0.5, 1.6, self.n_merchants)[self.merchant]
        cover = np.where(self.perishable, rng.uniform(2, 4, self.n), rng.uniform(10, 20, self.n))
        self.reorder_point = np.ceil(units * (self.lead + 1) * safety)
        self.order_up_to = np.ceil(self.reorder_point + units * cover) + 1

    def shock(self, day: int) -> np.ndarray:
        f = np.ones(self.n)
        on = (self.shock_start <= day) & (day < self.shock_end)
        np.multiply.at(f, self.shock_series[on], self.shock_factor[on])
        return f

    def active_shocks(self, day: int) -> list[tuple[int, float]]:
        on = (self.shock_start <= day) & (day < self.shock_end)
        return list(zip(self.shock_series[on].tolist(), self.shock_factor[on].tolist()))

    def promo_discount(self, day: int) -> np.ndarray:
        d = np.zeros(self.n)
        on = (self.promo_start <= day) & (day < self.promo_end)
        d[self.promo_series[on]] = self.promo_disc[on]
        return d

    def day_factor(self, day: int) -> np.ndarray:
        """Everything that moves a day's demand except noise, as a multiplier on the base."""
        d = sim_date(day)
        fest = np.maximum(0.3, 1 + festival_boost(d)[self.cat] * self.fest_sens)
        return (self.weekday[self.plat, d.weekday()] * payday_factor(d)[self.cat] * fest
                * np.exp(self.slope * day / 30) * self.shock(day))

    def day_rate(self, day: int) -> np.ndarray:
        """Expected order lines per series for a sim day (deterministic, noise included)."""
        if day not in self._day_cache:
            noise = np.random.default_rng([self.seed, 2, day + 10_000]).gamma(12, 1 / 12, self.n)
            if len(self._day_cache) > 8:
                self._day_cache.clear()
            self._day_cache[day] = self.base * self.day_factor(day) * noise
        return self._day_cache[day]

    def intraday_share(self, t0: float, t1: float) -> np.ndarray:
        """Per platform, the share of the day's demand between two times on the same sim day."""
        h0, h1 = (t0 % DAY) / 3600, (t1 - (t0 - t0 % DAY)) / 3600
        return np.array([np.interp(h1, np.arange(25), c) - np.interp(h0, np.arange(25), c) for c in self.cdf])


@dataclass
class Lines:
    """Order lines from one tick, sorted by merchant."""
    series: np.ndarray
    qty: np.ndarray
    price: np.ndarray
    customer: np.ndarray
    offered: np.ndarray

    def __len__(self) -> int:
        return len(self.series)


class State:
    """Everything that changes as the sim runs: clock, stock, open restocks, offers, customers."""

    def __init__(self, model: DemandModel):
        self.model = model
        w, n = model.world, model.n
        self.rng = np.random.default_rng([model.seed, 3])
        self.t = 0.0
        self.day = 0
        self.stock = np.zeros(n, np.int64)
        self.pending_qty = np.zeros(n, np.int64)
        self.pending_at = np.full(n, np.inf)
        self.offer_disc = np.zeros(n)
        self.offer_until = np.full(n, -np.inf)
        self.offer_id: list[str | None] = [None] * n
        self.last = -self.rng.random(w.n_customers) * w.customer_interval
        self.history = backfill(model, self)
        self.rate = model.day_rate(0)

    # -- stock

    def arrivals(self, t: float) -> np.ndarray:
        due = np.flatnonzero(self.pending_at <= t)
        self.stock[due] += self.pending_qty[due]
        self.pending_at[due] = np.inf
        return due

    def reorder(self, day: int) -> None:
        """Merchants' end-of-day reorder rule; deliveries land at 7am after the lead time (sometimes late)."""
        m = self.model
        need = np.flatnonzero(np.isinf(self.pending_at) & (self.stock <= m.reorder_point))
        delay = np.where(self.rng.random(len(need)) < 0.1, self.rng.integers(1, 4, len(need)), 0)
        self.pending_qty[need] = (m.order_up_to[need] - self.stock[need]).astype(np.int64)
        self.pending_at[need] = (day + m.lead[need] + delay) * DAY + 7 * 3600

    # -- offers

    def apply_offer(self, series: int, discount: float, until: float, signal_id: str | None) -> None:
        self.offer_disc[series] = min(max(discount, 0.0), 0.9)
        self.offer_until[series] = until
        self.offer_id[series] = signal_id

    def multiplier(self, t: float) -> np.ndarray:
        on = self.offer_until > t
        return np.where(on, (1 - self.offer_disc) ** self.model.elasticity, 1.0)

    # -- demand

    def expected(self, t0: float, t1: float) -> np.ndarray:
        """Expected order lines per series between t0 and t1 (same sim day), offers included."""
        m = self.model
        return self.rate * self.multiplier(t0) * m.intraday_share(t0, t1)[m.plat]

    def step(self, t1: float) -> Lines:
        """Draw the order lines between self.t and t1 (same sim day), take them from stock, pick customers."""
        m, rng, t0 = self.model, self.rng, self.t
        n = rng.poisson(self.expected(t0, t1))
        self.t = t1
        hit = np.flatnonzero(n)
        series = np.repeat(hit, n[hit])
        if not len(series):
            return Lines(series, series, series.astype(float), series, series.astype(bool))
        qty = 1 + rng.poisson(m.qty_extra[series])
        series, qty = self._take_stock(series, qty)
        order = np.lexsort((rng.random(len(series)), m.merchant[series]))
        series, qty = series[order], qty[order]
        on = self.offer_until[series] > t0
        price = np.round(m.price[series] * np.where(on, 1 - self.offer_disc[series], 1.0), 2)
        return Lines(series, qty, price, self._customers(m.merchant[series], t1 / DAY), on)

    def _take_stock(self, series: np.ndarray, qty: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Stockouts cap sales: lines past what is on the shelf shrink or are lost."""
        if not len(series):  # a quiet tick (real time at night)
            return series, qty
        cs = np.cumsum(qty)
        first = np.r_[True, series[1:] != series[:-1]]
        group_start = np.maximum.accumulate(np.where(first, np.arange(len(series)), 0))
        before = cs - qty - (cs[group_start] - qty[group_start])
        qty = np.minimum(qty, np.maximum(self.stock[series] - before, 0))
        keep = qty > 0
        series, qty = series[keep], qty[keep]
        np.subtract.at(self.stock, series, qty)
        return series, qty

    def _customers(self, merchant: np.ndarray, now: float) -> np.ndarray:
        """Group a merchant's lines into baskets and give each basket a customer, due customers first."""
        w = self.model.world
        n = len(merchant)
        if not n:
            return np.empty(0, dtype=np.int64)
        first = np.r_[True, merchant[1:] != merchant[:-1]]
        pos = np.arange(n) - np.maximum.accumulate(np.where(first, np.arange(n), 0))
        basket_size = np.array([m.basket for m in w.merchants])[merchant]
        slot = np.floor(pos / np.maximum(np.round(basket_size * self.rng.uniform(0.6, 1.4, n)), 1)).astype(np.int64)
        new = first | np.r_[True, slot[1:] != slot[:-1]]
        basket = np.cumsum(new) - 1
        bm = merchant[new]
        cand = w.pool_start[bm][:, None] + (self.rng.random((len(bm), 4)) * w.pool_size[bm][:, None]).astype(np.int64)
        due = np.clip((now - self.last[cand]) / w.customer_interval[cand], 0.02, 4) ** 2
        due = np.where(now >= w.customer_lapse[cand], due * 1e-4, due)
        chosen = cand[np.arange(len(bm)), np.argmax(due / self.rng.exponential(1, cand.shape), axis=1)]
        self.last[chosen] = now
        return chosen[basket]


def backfill(model: DemandModel, st: State) -> dict:
    """Run the model through HISTORY_DAYS before SIM_START, stock and reorders included; leaves st at SIM_START."""
    rng = np.random.default_rng([model.seed, 4])
    units_per_day = model.base * (1 + model.qty_extra)
    st.stock[:] = np.ceil(units_per_day * np.where(model.perishable, rng.uniform(1, 3, model.n),
                                                   rng.uniform(5, 25, model.n))).astype(np.int64)
    days = list(range(-contract.HISTORY_DAYS, 0))
    units = np.zeros((len(days), model.n), np.int64)
    prices = np.zeros((len(days), model.n))
    demand = np.zeros((len(days), model.n), np.int64)
    for i, d in enumerate(days):
        st.arrivals((d + 1) * DAY)
        disc = model.promo_discount(d)
        lines = rng.poisson(model.day_rate(d) * (1 - disc) ** model.elasticity)
        demand[i] = lines + rng.poisson(lines * model.qty_extra)
        units[i] = np.minimum(demand[i], st.stock)
        st.stock -= units[i]
        prices[i] = np.round(model.price * (1 - disc), 2)
        st.reorder(d)
    return {"days": days, "units": units, "prices": prices, "demand": demand}


def history_payload(model: DemandModel, hist: dict, mi: int) -> dict:
    a, b = model.offsets[mi], model.offsets[mi + 1]
    return {"days": [sim_date(d).isoformat() for d in hist["days"]], "skus": model.sku[a:b],
            "units": hist["units"][:, a:b].tolist(), "prices": hist["prices"][:, a:b].tolist(),
            "sold_out": (hist["units"][:, a:b] < hist["demand"][:, a:b]).astype(int).tolist()}


def stock_payload(model: DemandModel, st: State, mi: int, t: float, restocked: np.ndarray | None = None) -> dict:
    a, b = model.offsets[mi], model.offsets[mi + 1]
    stock = dict(zip(model.sku[a:b], st.stock[a:b].tolist()))
    re = {model.sku[s]: int(st_q) for s, st_q in restocked} if restocked is not None else {}
    return {"sim_ts": sim_ts(t), "stock": stock, "restocked": re}


Publish = Callable[[str, str, int, bool], None]


class Simulator:
    """The streaming loop's logic, independent of the broker (``publish`` is injected)."""

    def __init__(self, world: World, publish: Publish, speed: str = contract.DEFAULT_SPEED):
        self.world = world
        self.model = DemandModel(world)
        self.publish = publish
        self.catching_up = True
        self.requested = speed if speed in contract.SPEEDS else contract.DEFAULT_SPEED
        self.factor = 1.0
        self.running = False
        self.run_until: float | None = None   # run_days: sim seconds at which the clock pauses
        self.resume_speed: str | None = None   # the speed to go back to after a run_days
        self.seq = 0
        m = self.model
        self.topics = [contract.merchant_topic(x.platform, x.id, "orders") for x in world.merchants]
        self.stock_topics = [contract.merchant_topic(x.platform, x.id, "stock") for x in world.merchants]
        self.sku_json = [json.dumps(s) for s in m.sku]
        self.cust_json = [f'"{customer_id(i)}"' for i in range(world.n_customers)]
        self.mindex = world.merchant_index()
        self.sku_index = [{p.sku: m.offsets[i] + j for j, p in enumerate(c)} for i, c in enumerate(world.catalogs)]
        self.order_window: deque[tuple[float, int]] = deque()
        self.speed_window: deque[tuple[float, float]] = deque()
        self.reset_state()

    def reset_state(self) -> None:
        self.state = State(self.model)
        self.started_ms = contract.wall_ms()

    # -- setup messages

    def publish_setup(self) -> None:
        w, m, st = self.world, self.model, self.state
        dumps = lambda o: json.dumps(o, separators=(",", ":"))
        self.publish(contract.WORLD, dumps(world_payload(w, self.started_ms)), 1, True)
        for i, x in enumerate(w.merchants):
            a, b = m.offsets[i], m.offsets[i + 1]
            self.publish(contract.merchant_topic(x.platform, x.id, "catalog"),
                         dumps(catalog_payload(w.catalogs[i], st.stock[a:b].tolist())), 1, True)
        for i, x in enumerate(w.merchants):
            self.publish(contract.merchant_topic(x.platform, x.id, "history"),
                         dumps(history_payload(m, st.history, i)), 1, False)

    # -- control

    def handle(self, cmd: dict) -> None:
        kind = cmd.get("cmd")
        if kind == "start":
            self.running = True
        elif kind == "pause":
            self._end_run()
            self.running = False
        elif kind == "speed" and cmd.get("speed") in contract.SPEEDS:
            self._end_run()
            self.requested, self.factor = cmd["speed"], 1.0
        elif kind == "run_days":
            self._run_days(cmd.get("days"))
        elif kind == "apply_offer":
            self._apply_offer(cmd)
        elif kind == "reset":
            self._end_run()
            self.running = False
            self.catching_up = True
            self.reset_state()
            self.publish_setup()
        else:
            log.warning("ignored control %s", cmd)

    def _run_days(self, days) -> None:
        """The HQ time machine: run at the fastest speed for ``days`` sim days, then pause."""
        try:
            days = int(days)
        except (TypeError, ValueError):
            days = 0
        if not 1 <= days <= MAX_RUN_DAYS:
            log.warning("ignored run_days %r (1-%d)", days, MAX_RUN_DAYS)
            return
        if self.run_until is None:
            self.resume_speed = self.requested
        self.run_until = self.state.t + days * DAY
        self.requested, self.factor = FASTEST, 1.0
        self.catching_up = False   # the time machine runs past the real time
        self.running = True

    def _end_run(self) -> None:
        if self.run_until is not None:
            self.run_until = None
            self.requested = self.resume_speed or self.requested
            self.resume_speed = None

    def step(self, sim_dt: float) -> float:
        """This tick's sim seconds: catch-up first, never past a run_days target."""
        sim_dt = self.catch_up_step(sim_dt)
        if self.run_until is not None:
            sim_dt = max(min(sim_dt, self.run_until - self.state.t), 0.0)
        return sim_dt

    def finish_step(self) -> None:
        """After advancing: a run_days that reached its target pauses the clock."""
        if self.run_until is not None and self.state.t >= self.run_until - 1e-6:
            self._end_run()
            self.running = False

    def _apply_offer(self, cmd: dict) -> None:
        mi = self.mindex.get(cmd.get("merchant", ""))
        s = self.sku_index[mi].get(cmd.get("sku", "")) if mi is not None else None
        if s is None or self.world.merchants[mi].platform != cmd.get("platform", self.world.merchants[mi].platform):
            log.warning("offer for unknown merchant/sku %s", cmd)
            return
        try:
            until = parse_ts(cmd["until"])
            discount = float(cmd["discount"])
        except (KeyError, TypeError, ValueError):
            log.warning("bad offer %s", cmd)
            return
        self.state.apply_offer(int(s), discount, until, cmd.get("signal_id"))

    # -- ticking

    def advance(self, sim_seconds: float, sent_ms: int | None = None) -> int:
        """Advance the sim clock, publishing orders, restocks and day closes. Returns order lines published."""
        st = self.state
        end, total = st.t + sim_seconds, 0
        while st.t < end - 1e-9:
            midnight = (st.day + 1) * DAY
            t1 = min(end, midnight)
            total += self._publish_orders(st.step(t1), sent_ms)
            self._publish_arrivals(st.arrivals(t1), t1)
            if t1 >= midnight - 1e-9:
                self._close_day()
        return total

    def _publish_orders(self, lines: Lines, sent_ms: int | None) -> int:
        if not len(lines):
            return 0
        m = self.model
        merchant = m.merchant[lines.series]
        bounds = np.flatnonzero(np.r_[True, merchant[1:] != merchant[:-1], True])
        head = f'{{"sent_ms":{sent_ms or contract.wall_ms()},"sim_ts":"{sim_ts(self.state.t)}","orders":['
        sku, cust, offer_id = self.sku_json, self.cust_json, self.state.offer_id
        seq = self.seq
        rows = [f'{{"id":"o{seq + k}","customer":{cust[c]},"sku":{sku[s]},"qty":{q},"price":{p},"offer":'
                f'{json.dumps(offer_id[s]) if o else "null"}}}'
                for k, (s, q, p, c, o) in enumerate(zip(lines.series.tolist(), lines.qty.tolist(),
                                                        lines.price.tolist(), lines.customer.tolist(),
                                                        lines.offered.tolist()))]
        self.seq += len(rows)
        for a, b in zip(bounds[:-1].tolist(), bounds[1:].tolist()):
            self.publish(self.topics[merchant[a]], head + ",".join(rows[a:b]) + "]}", 0, False)
        return len(rows)

    def _publish_arrivals(self, due: np.ndarray, t: float) -> None:
        if not len(due):
            return
        m, st = self.model, self.state
        for mi in np.unique(m.merchant[due]).tolist():
            mine = due[m.merchant[due] == mi]
            payload = stock_payload(m, st, mi, t, np.c_[mine, st.pending_qty[mine]])
            self.publish(self.stock_topics[mi], json.dumps(payload, separators=(",", ":")), 1, False)

    def _close_day(self) -> None:
        st, m = self.state, self.model
        close = st.day * DAY + DAY - 1
        for mi in range(m.n_merchants):
            self.publish(self.stock_topics[mi], json.dumps(stock_payload(m, st, mi, close), separators=(",", ":")), 1, False)
        st.reorder(st.day)
        st.day += 1
        st.rate = m.day_rate(st.day)

    # -- clock and measurements

    def record(self, wall: float, orders: int, sim_seconds: float) -> None:
        self.order_window.append((wall, orders))
        self.speed_window.append((wall, sim_seconds))
        while self.order_window and wall - self.order_window[0][0] > 1.0:
            self.order_window.popleft()
        while self.speed_window and wall - self.speed_window[0][0] > 2.0:
            self.speed_window.popleft()

    def rates(self) -> tuple[float, float]:
        """Order lines per wall second and sim seconds per wall second, measured over the recent window."""
        def per_sec(win: deque) -> float:
            if len(win) < 2:
                return 0.0
            return sum(v for _, v in list(win)[1:]) / max(win[-1][0] - win[0][0], 1e-3)
        return per_sec(self.order_window), per_sec(self.speed_window)

    def clock_payload(self) -> dict:
        ops, speed = self.rates()
        return {"sent_ms": contract.wall_ms(), "sim_ts": sim_ts(self.state.t),
                "speed": round(speed if self.running else 0.0, 1), "requested": self.requested,
                "requested_speed": contract.SPEEDS[self.requested], "running": self.running,
                "catching_up": self.running and self.catching_up, "orders_per_sec": round(ops, 1),
                "run_until": sim_ts(self.run_until) if self.run_until is not None else None}

    def catch_up_step(self, sim_dt: float) -> float:
        """Until the sim clock reaches the real time, run at catch-up speed (never past now)."""
        if not self.catching_up:
            return sim_dt
        behind = (contract.ist_now() - SIM_START).total_seconds() - self.state.t
        if not 2 < behind < 86400:  # caught up (within a tick or two of now), or a pinned start that isn't today
            self.catching_up = False
            return sim_dt
        return min(max(sim_dt, contract.CATCH_UP_SPEED * self.factor * TICK), behind)

    def adapt(self, work: float) -> None:
        """Slow the sim clock when a tick's work no longer fits the tick budget; recover when it does."""
        if work > 0.75 * TICK:
            self.factor = max(0.01, self.factor * max(0.5, 0.6 * TICK / work))
        elif work < 0.5 * TICK and self.factor < 1.0:
            self.factor = min(1.0, self.factor * 1.05)


class _Broker:
    """paho client driven from the loop's own thread: publishes write straight to the socket."""

    def __init__(self, client_id: str):
        self.commands: deque[dict] = deque()
        self.client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id=client_id)
        self.client.max_inflight_messages_set(0)
        self.client.max_queued_messages_set(0)
        self.client.on_connect = self._on_connect
        self.client.on_message = self._on_message

    def connect(self) -> None:
        self.client.connect(contract.INGEST_HOST, contract.INGEST_PORT, keepalive=30)
        deadline = time.monotonic() + 5
        while not self.client.is_connected() and time.monotonic() < deadline:
            self.client.loop(0.05)
        if not self.client.is_connected():
            raise ConnectionError(f"no MQTT broker at {contract.INGEST_HOST}:{contract.INGEST_PORT}")

    def _on_connect(self, client, userdata, flags, reason, properties) -> None:
        client.subscribe(contract.CONTROL, qos=1)

    def _on_message(self, client, userdata, msg) -> None:
        try:
            self.commands.append(json.loads(msg.payload))
        except ValueError:
            log.warning("bad control payload")

    def publish(self, topic: str, payload: str, qos: int, retain: bool) -> None:
        self.client.publish(topic, payload, qos=qos, retain=retain)

    def pump(self, until: float) -> None:
        """Serve the network (writes, acks, incoming control) until the perf_counter deadline."""
        while True:
            left = until - time.perf_counter()
            rc = self.client.loop(max(min(left, 0.05), 0))
            if rc != mqtt.MQTT_ERR_SUCCESS:
                self._reconnect()
            if left <= 0:
                return

    def _reconnect(self) -> None:
        try:
            self.client.reconnect()
        except OSError:
            time.sleep(0.5)

    def drain(self, timeout: float = 10.0) -> None:
        """Wait until queued QoS>0 messages are acknowledged (after the big setup burst)."""
        deadline = time.perf_counter() + timeout
        while (self.client._out_messages or self.client._out_packet) and time.perf_counter() < deadline:
            self.client.loop(0.02)


def run(merchants: int = 600, customers: int = 120000, seed: int = 7, speed: str = contract.DEFAULT_SPEED,
        autostart: bool = False) -> None:
    """Build the world, publish it, and stream until interrupted (Ctrl-C or SIGTERM)."""
    t0 = time.perf_counter()
    world = build_world(merchants, customers, seed)
    broker = _Broker(f"sb-commerce-sim-{seed}-{contract.wall_ms() % 100000}")
    broker.connect()
    sim = Simulator(world, broker.publish, speed)
    log.info("world: %d merchants, %d series, %d customers (%.1fs)", merchants, sim.model.n, customers,
             time.perf_counter() - t0)
    sim.publish_setup()
    broker.drain()
    sim.running = autostart
    stop = False

    def _stop(*_):
        nonlocal stop
        stop = True
    signal.signal(signal.SIGTERM, _stop)
    signal.signal(signal.SIGINT, _stop)

    next_tick = time.perf_counter()
    last_log = next_tick
    try:
        while not stop:
            while broker.commands:
                sim.handle(broker.commands.popleft())
            start = time.perf_counter()
            orders, sim_dt = 0, 0.0
            if sim.running:
                sim_dt = sim.step(contract.SPEEDS[sim.requested] * sim.factor * TICK)
                orders = sim.advance(sim_dt)
                broker.client.loop_write()
                sim.adapt(time.perf_counter() - start)
                sim.finish_step()
            sim.record(start, orders, sim_dt)
            broker.publish(contract.CLOCK, json.dumps(sim.clock_payload(), separators=(",", ":")), 1, False)
            if start - last_log > 10:
                last_log = start
                log.info("%s", sim.clock_payload())
            next_tick += TICK
            if next_tick < time.perf_counter():
                next_tick = time.perf_counter()  # behind: never burst to catch up
            broker.pump(next_tick)
    finally:
        broker.client.disconnect()


def main() -> None:
    p = argparse.ArgumentParser(description="Live Commerce simulator: streams a synthetic commerce world over MQTT.")
    p.add_argument("--merchants", type=int, default=600)
    p.add_argument("--customers", type=int, default=120000)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--speed", choices=list(contract.SPEEDS), default=contract.DEFAULT_SPEED)
    p.add_argument("--autostart", action="store_true", help="start the clock now instead of waiting for a start command")
    a = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    run(a.merchants, a.customers, a.seed, a.speed, a.autostart)


if __name__ == "__main__":
    main()
