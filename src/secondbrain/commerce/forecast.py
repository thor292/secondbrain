"""Streaming daily demand forecaster for Live Commerce: every merchant x SKU in one numpy step.

Why not ``secondbrain.forecasting``: that package backtests several models per product
(ETS, ARIMA, Theta, LightGBM) and keeps the best, which costs seconds per merchant. The live
wall needs ~20k series updated at every simulated day close (every few wall seconds) and
compared with the clock every simulated hour, so this is a vectorised recursive filter that
updates all series at once. It reuses ``forecasting.metrics.wape`` for scoring.

Model (Holt-Winters style, multiplicative, no trend):

    forecast = level x weekday[wd] x event[category, kind of day] x price effect

- level: exponential smoothing of de-seasonalised units; alpha is picked per category from
  one-step-ahead errors on the backfill.
- weekday: smoothed ratios shrunk towards the category's pooled pattern, so a product that
  sells two a day borrows its weekly shape from its category.
- event: two multipliers. A calendar one (month end, paydays) and a festival one for every day
  from four weeks before a festival to three weeks after it: the build-up, the day itself and the
  slump after. The engine knows the dates (``contract.FESTIVALS``, last year's included); it
  learns each size by pooling every product of a category on those days, shrunk towards 1 until
  seen. The backfill covers a year, so this year's Diwali is forecast from last year's.
- price effect: (price / list price) ** elasticity, so offer days don't inflate the level.

Days that closed out of stock below forecast are censored (lost sales don't pull the level down).
80% intervals come from recent residuals, scaled like Poisson noise.

Nowcast: a learned hour-of-day profile per platform turns the day's forecast into "expected
units by now", so a drop shows at 2 pm, not at midnight.

Accuracy: every closed day is scored against the forecast made at the previous day's close, on
the products that stayed in stock (a sold-out day records the shelf, not demand; the standard
retail measure): WAPE per series-day overall, per platform, at merchant level and against
seasonal naive (same weekday last week).
"""

from dataclasses import dataclass
from datetime import date, timedelta
from functools import lru_cache

import numpy as np

from secondbrain.commerce.contract import FESTIVALS
from secondbrain.forecasting.metrics import wape

EVENTS = ("normal", "month_end", "payday_1", "payday_2", "payday_3", "payday_4_5")
MONTH_END, PAYDAY = 1, 2
# Festival days by distance from the festival: days before it (0 = the day itself), then days after.
BEFORE = ((0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6), (7, 7), (8, 10), (11, 14), (15, 20), (21, 28))
AFTER = ((1, 1), (2, 2), (3, 3), (4, 5), (6, 8), (9, 14), (15, 21))
FEST_NAMES = tuple(dict.fromkeys(sorted(FESTIVALS.values())))
FEST_KINDS = ("none",) + tuple(f"{n}:-{a}" if a else f"{n}:day" for n in FEST_NAMES for a, _ in BEFORE) \
    + tuple(f"{n}:+{a}" for n in FEST_NAMES for a, _ in AFTER)
ALPHAS = np.array([0.05, 0.1, 0.2, 0.35])
GAMMA = 0.15          # weekday smoothing
SEASON_PSEUDO = 8.0   # units of evidence before a series' own weekday shape beats its category's
EVENT_PSEUDO = 30.0   # units of evidence before an event multiplier moves away from 1
RESID_BETA = 0.1      # residual / accuracy smoothing
Z80 = 1.2816
HORIZON = 14
LEVEL_FLOOR = 0.02
SENS_PSEUDO = 40.0    # units of festival-day evidence before a merchant's own festival sensitivity moves from 1
SENS_RANGE = (0.6, 1.6)
FEST_FLOOR = 0.3      # a festival never takes more than 70% off a day (keeps the filter stable after big dips)


@lru_cache(maxsize=4096)
def day_kind(day: date) -> int:
    """Which calendar multiplier applies: the last week of the month (money runs short), paydays, or none."""
    if day.day >= 26:
        return MONTH_END
    if day.day <= 3:
        return PAYDAY + day.day - 1
    if day.day <= 5:
        return PAYDAY + 3
    return 0


def _bucket(buckets: tuple, dist: int) -> int | None:
    return next((k for k, (a, b) in enumerate(buckets) if a <= dist <= b), None)


@lru_cache(maxsize=4096)
def fest_kind(day: date) -> int:
    """Which festival multiplier applies: the nearest festival within four weeks before or three after."""
    best = None
    for fdate, name in FESTIVALS.items():
        dist = (fdate - day).days
        if 0 <= dist <= BEFORE[-1][1]:
            key, k = (dist, 0), FEST_KINDS.index(f"{name}:-{BEFORE[_bucket(BEFORE, dist)][0]}" if dist else f"{name}:day")
        elif 0 < -dist <= AFTER[-1][1]:
            key, k = (-dist, 1), FEST_KINDS.index(f"{name}:+{AFTER[_bucket(AFTER, -dist)][0]}")
        else:
            continue
        if best is None or key < best[0]:
            best = (key, k)
    return best[1] if best else 0


FESTIVAL_DAY = {n: FEST_KINDS.index(f"{n}:day") for n in FEST_NAMES}


class EventPool:
    """Per category x event kind: pooled actual vs baseline units, so the lift is learned once per category."""

    def __init__(self, n_cat: int = 16, n_kinds: int = len(EVENTS)):
        self.num = np.zeros((n_cat, n_kinds))
        self.den = np.zeros((n_cat, n_kinds))
        self.days = np.zeros((n_cat, n_kinds), dtype=int)

    def grow(self, n_cat: int) -> None:
        if n_cat > len(self.num):
            pad, k = n_cat - len(self.num) + 8, self.num.shape[1]
            self.num = np.vstack([self.num, np.zeros((pad, k))])
            self.den = np.vstack([self.den, np.zeros((pad, k))])
            self.days = np.vstack([self.days, np.zeros((pad, k), dtype=int)])

    def mult(self) -> np.ndarray:
        m = (self.num + EVENT_PSEUDO) / (self.den + EVENT_PSEUDO)
        m[:, 0] = 1.0
        return m

    def learn(self, kind: int, cat: np.ndarray, actual: np.ndarray, base: np.ndarray) -> None:
        n = len(self.num)
        self.num[:, kind] += np.bincount(cat, actual, n)
        self.den[:, kind] += np.bincount(cat, base, n)
        self.days[np.unique(cat), kind] += 1

    def copy(self) -> "EventPool":
        p = EventPool(*self.num.shape)
        p.num, p.den, p.days = self.num.copy(), self.den.copy(), self.days.copy()
        return p


class FestSens:
    """Per merchant: how strongly its sales follow the category's festival lift (1 = like the category).
    Least squares of (sales - base) on base x (lift - 1) over festival days, shrunk towards 1."""

    def __init__(self, n: int = 64):
        self.num = np.zeros(n)
        self.den = np.zeros(n)

    def grow(self, n: int) -> None:
        if n > len(self.num):
            pad = n - len(self.num) + 64
            self.num, self.den = np.concatenate([self.num, np.zeros(pad)]), np.concatenate([self.den, np.zeros(pad)])

    def k(self, mer: np.ndarray) -> np.ndarray:
        return np.clip((self.num[mer] + SENS_PSEUDO) / (self.den[mer] + SENS_PSEUDO), *SENS_RANGE)

    def learn(self, mer: np.ndarray, y: np.ndarray, base: np.ndarray, lift: np.ndarray) -> None:
        g = base * (lift - 1)
        n = len(self.num)
        self.num += np.bincount(mer, (y - base) * (lift - 1), n)
        self.den += np.bincount(mer, g * (lift - 1), n)

    def copy(self) -> "FestSens":
        f = FestSens(len(self.num))
        f.num, f.den = self.num.copy(), self.den.copy()
        return f


@dataclass
class State:
    """Filter state for n series."""
    level: np.ndarray
    A: np.ndarray        # smoothed units on each weekday (n, 7)
    B: np.ndarray        # smoothed level on each weekday (n, 7)
    season: np.ndarray   # weekday factors, mean 1 (n, 7)
    alpha: np.ndarray
    var_r: np.ndarray    # variance of (y - f) / sqrt(f + 1)
    ew_abs: np.ndarray   # smoothed |y - f| and y, for a per-series WAPE
    ew_y: np.ndarray

    @classmethod
    def zeros(cls, n: int) -> "State":
        return cls(np.full(n, LEVEL_FLOOR), np.ones((n, 7)), np.ones((n, 7)), np.ones((n, 7)),
                   np.full(n, 0.1), np.ones(n), np.zeros(n), np.zeros(n))

    def take(self, rows: np.ndarray) -> "State":
        return State(*(getattr(self, f)[rows].copy() for f in self.__dataclass_fields__))

    def put(self, rows: np.ndarray, other: "State") -> None:
        for f in self.__dataclass_fields__:
            getattr(self, f)[rows] = getattr(other, f)

    def grow(self, n: int) -> None:
        extra = State.zeros(n - len(self.level))
        for f in self.__dataclass_fields__:
            setattr(self, f, np.concatenate([getattr(self, f), getattr(extra, f)]))


def reseason(st: State, rows: np.ndarray, grp: np.ndarray, pool_A: np.ndarray, pool_B: np.ndarray) -> None:
    """Weekday factors: own evidence shrunk towards the group's pooled ratios (platform x category: a
    hypermarket's weekend is not an online store's), normalised to mean 1."""
    n_grp = len(pool_A)
    A, B = st.A[rows], st.B[rows]
    pA = pool_A + np.stack([np.bincount(grp, A[:, w], n_grp) for w in range(7)], axis=1)
    pB = pool_B + np.stack([np.bincount(grp, B[:, w], n_grp) for w in range(7)], axis=1)
    spool = np.divide(pA, pB, out=np.ones_like(pA), where=pB > 1e-9)
    spool /= spool.mean(axis=1, keepdims=True)
    s = (A / GAMMA + SEASON_PSEUDO * spool[grp]) / (B / GAMMA + SEASON_PSEUDO)
    st.season[rows] = s / s.mean(axis=1, keepdims=True)


def step(st: State, rows: np.ndarray, cat: np.ndarray, day: date, y: np.ndarray, pmult: np.ndarray,
         censored: np.ndarray, pools: tuple, pool_A: np.ndarray, pool_B: np.ndarray,
         grp: np.ndarray | None = None, mer: np.ndarray | None = None) -> np.ndarray:
    """One closed day for ``rows``; returns the one-step-ahead forecast it was judged against.
    pools: (calendar EventPool, festival EventPool, FestSens)."""
    pool, fpool, sens = pools
    wd, kind, fk = day.weekday(), day_kind(day), fest_kind(day)
    level, s = st.level[rows], st.season[rows, wd]
    base = level * s
    cal = pool.mult()[cat, kind]
    lift = fpool.mult()[cat, fk]
    k = sens.k(mer) if mer is not None and fk else 1.0
    fest = np.maximum(1 + k * (lift - 1), FEST_FLOOR)
    pred = base * cal * fest * pmult
    r2 = np.minimum((y - pred) ** 2 / (pred + 1.0), 25.0)
    st.var_r[rows] += RESID_BETA * (r2 - st.var_r[rows])
    st.ew_abs[rows] += RESID_BETA * (np.abs(y - pred) - st.ew_abs[rows])
    st.ew_y[rows] += RESID_BETA * (y - st.ew_y[rows])
    ok = ~(censored | ((y == 0) & (pred >= 3)))  # a zero day against a forecast of 3+ is a stockout, not demand
    if ok.any():
        if fk:
            # the category lift on the day's mean across merchants, then each merchant's sensitivity to it
            fpool.learn(fk, cat[ok], (y / pmult)[ok], (base * cal)[ok])
            if mer is not None:
                sens.learn(mer[ok], (y / pmult)[ok], (base * cal)[ok], fpool.mult()[cat, fk][ok])
        if kind:
            pool.learn(kind, cat[ok], (y / pmult)[ok], (base * fest)[ok])
    k = sens.k(mer) if mer is not None and fk else 1.0
    e = pool.mult()[cat, kind] * np.maximum(1 + k * (fpool.mult()[cat, fk] - 1), FEST_FLOOR) * pmult
    x = y / (s * e)
    st.level[rows] = np.maximum(np.where(ok, level + st.alpha[rows] * (x - level), level), LEVEL_FLOOR)
    st.A[rows, wd] = np.where(ok, (1 - GAMMA) * st.A[rows, wd] + GAMMA * y / e, st.A[rows, wd])
    st.B[rows, wd] = np.where(ok, (1 - GAMMA) * st.B[rows, wd] + GAMMA * level, st.B[rows, wd])
    reseason(st, rows, cat if grp is None else grp, pool_A, pool_B)
    return pred


class Forecaster:
    """All series' state, daily history and next-day forecasts. Day indices count from ``base_day``."""

    def __init__(self, base_day: date, capacity: int = 1024, days: int = 480):
        self.base_day = base_day
        self.n = 0
        self.st = State.zeros(capacity)
        self.cat = np.zeros(capacity, dtype=np.int64)
        self.grp = np.zeros(capacity, dtype=np.int64)            # weekday-shape group: platform x category
        self.mer = np.zeros(capacity, dtype=np.int64)            # merchant, for its festival sensitivity
        self.elast = np.full(capacity, -1.5)
        self.fitted = np.zeros(capacity, dtype=bool)
        self.start = np.full(capacity, -1, dtype=np.int64)      # first day with data
        self.hist = np.zeros((capacity, days), dtype=np.float32)
        self.ratio = np.ones((capacity, days), dtype=np.float32)  # average price paid / list price
        self.cens = np.zeros((capacity, days), dtype=bool)
        self.f_day = np.full(capacity, -1, dtype=np.int64)       # the day f_next is for
        self.f_next = np.zeros(capacity)
        self.lo_next = np.zeros(capacity)
        self.hi_next = np.zeros(capacity)
        self.live_abs = np.zeros(capacity)                      # per-series live scoring
        self.live_y = np.zeros(capacity)
        self.y_lag = np.zeros((capacity, 2))                    # the last two closed days, for multi-day evidence
        self.f_lag = np.zeros((capacity, 2))
        self.disp_sums = np.zeros(5)                            # moments for var(y) = a f + b f^2
        self.categories: list[str] = []
        self.cat_index: dict[str, int] = {}
        self.grp_index: dict[str, int] = {}
        self.mer_index: dict[str, int] = {}
        self.pool = EventPool()
        self.fpool = EventPool(n_kinds=len(FEST_KINDS))
        self.sens = FestSens()
        self.price_ev = {k: np.zeros(0) for k in ("y", "base", "lr", "days", "y0", "base0")}
        self.backtest = {"abs": 0.0, "y": 0.0, "naive_abs": 0.0, "naive_y": 0.0, "days": 0,
                         "m_abs": 0.0, "m_y": 0.0, "m_naive_abs": 0.0}

    # ------------------------------------------------------------------ series

    def day(self, idx: int) -> date:
        return self.base_day + timedelta(days=int(idx))

    def day_index(self, d: date) -> int:
        return (d - self.base_day).days

    def category(self, name: str) -> int:
        if name not in self.cat_index:
            self.cat_index[name] = len(self.categories)
            self.categories.append(name)
            self.pool.grow(len(self.categories))
            self.fpool.grow(len(self.categories))
        return self.cat_index[name]

    def add_series(self, category: str, elasticity: float, group: str | None = None, merchant: str = "") -> int:
        """group: who shares a weekday shape (default the category; the engine passes platform/category);
        merchant: whose festival sensitivity it shares."""
        if self.n == len(self.cat):
            self._grow(2 * self.n)
        i = self.n
        self.n += 1
        self.cat[i] = self.category(category)
        self.grp[i] = self.grp_index.setdefault(group or category, len(self.grp_index))
        self.mer[i] = self.mer_index.setdefault(merchant, len(self.mer_index))
        self.sens.grow(len(self.mer_index))
        self.elast[i] = elasticity
        return i

    def _grow(self, cap: int) -> None:
        self.st.grow(cap)
        for name in ("cat", "grp", "mer", "elast", "fitted", "start", "f_day", "f_next", "lo_next", "hi_next",
                     "live_abs", "live_y"):
            a = getattr(self, name)
            fill = {"elast": -1.5, "start": -1, "f_day": -1}.get(name, 0)
            setattr(self, name, np.concatenate([a, np.full(cap - len(a), fill, dtype=a.dtype)]))
        for name, fill in (("hist", 0), ("ratio", 1), ("cens", False), ("y_lag", 0), ("f_lag", 0)):
            a = getattr(self, name)
            setattr(self, name, np.vstack([a, np.full((cap - len(a), a.shape[1]), fill, dtype=a.dtype)]))

    def ensure_day(self, idx: int) -> None:
        if idx >= self.hist.shape[1]:
            extra = idx - self.hist.shape[1] + 64
            n = len(self.hist)
            self.hist = np.hstack([self.hist, np.zeros((n, extra), dtype=np.float32)])
            self.ratio = np.hstack([self.ratio, np.ones((n, extra), dtype=np.float32)])
            self.cens = np.hstack([self.cens, np.zeros((n, extra), dtype=bool)])

    def set_history(self, rows: np.ndarray, day_idx: np.ndarray, units: np.ndarray, ratio: np.ndarray,
                    sold_out: np.ndarray | None = None) -> None:
        """Backfill: units, price ratios and sold-out days with shape (days, len(rows)), as the history message
        sends them."""
        keep = day_idx >= 0
        if sold_out is None:
            sold_out = np.zeros(units.shape, dtype=bool)
        day_idx, units, ratio, sold_out = day_idx[keep], units[keep], ratio[keep], sold_out[keep]
        if not len(day_idx):
            return
        self.ensure_day(int(day_idx.max()))
        self.hist[np.ix_(rows, day_idx)] = units.T
        self.ratio[np.ix_(rows, day_idx)] = ratio.T
        self.cens[np.ix_(rows, day_idx)] = sold_out.T
        first = int(day_idx.min())
        self.start[rows] = np.where(self.start[rows] < 0, first, np.minimum(self.start[rows], first))

    def pmult(self, rows: np.ndarray, ratio: np.ndarray) -> np.ndarray:
        return np.clip(ratio, 0.3, 2.0) ** self.elast[rows].reshape((-1,) + (1,) * (ratio.ndim - 1))

    # ------------------------------------------------------------------ fitting

    def fit(self, rows: np.ndarray, through: int) -> None:
        """Initialise and filter ``rows`` over their history up to day ``through`` (inclusive)."""
        rows = rows[(self.start[rows] >= 0) & (self.start[rows] <= through)]
        if not len(rows):
            return
        t0 = int(self.start[rows].min())
        Y = self.hist[rows, t0:through + 1].astype(float)
        R = self.ratio[rows, t0:through + 1].astype(float)
        P = self.pmult(rows, R)
        C = self.cens[rows, t0:through + 1]
        off = self.start[rows] - t0
        cat, grp, mer = self.cat[rows], self.grp[rows], self.mer[rows]
        others = np.flatnonzero(self.fitted[:self.n])
        pool_A = self._pool(others, self.st.A)
        pool_B = self._pool(others, self.st.B)
        init = self._init_state(Y, P, off, t0)
        T = Y.shape[1]
        if T >= 21:
            init.alpha[:] = self._choose_alpha(init, Y, P, C, off, cat, grp, mer, t0, pool_A, pool_B)
        st = init
        bt_from = max(T - 14, 0)
        for t in range(T):
            act = np.flatnonzero(off <= t)
            d = self.day(t0 + t)
            pred = step(st, act, cat[act], d, Y[act, t], P[act, t], C[act, t], self.pools, pool_A, pool_B,
                        grp[act], mer[act])
            self._price_evidence(act, cat[act], Y[act, t], pred / P[act, t], R[act, t],
                                 (off[act] <= t - 14) & ~C[act, t])   # a sold-out day understates demand
            warm = off[act] <= t - 14
            self._disp_add(pred[warm], Y[act[warm], t])
            if t >= bt_from and t >= 7:
                ok = (off[act] <= t - 14) & ~C[act, t]   # in-stock days, as live
                if ok.any():
                    y, a, bt = Y[act[ok], t], act[ok], self.backtest
                    bt["abs"] += float(np.abs(y - pred[ok]).sum())
                    bt["y"] += float(y.sum())
                    bt["naive_abs"] += float(np.abs(y - Y[a, t - 7]).sum())
                    bt["naive_y"] += float(y.sum())
                    m, nm = mer[a], len(self.mer_index)
                    my = np.bincount(m, y, nm)
                    bt["m_abs"] += float(np.abs(my - np.bincount(m, pred[ok], nm)).sum())
                    bt["m_naive_abs"] += float(np.abs(my - np.bincount(m, Y[a, t - 7], nm)).sum())
                    bt["m_y"] += float(y.sum())
                    bt["days"] = max(bt["days"], T - bt_from)
        self.st.put(rows, st)
        self.fitted[rows] = True
        self.predict_next(rows, through + 1)

    def _price_evidence(self, act, cat, y, base, ratio, warm) -> None:
        """Days sold at a different price (past promotions) vs the no-price forecast, pooled by category."""
        lr = np.log(np.clip(ratio, 0.3, 2.0))
        on = warm & (np.abs(lr) > 0.02)
        n_cat = len(self.pool.num)
        pe = self.price_ev
        if len(pe["y"]) < n_cat:
            for k in pe:
                pe[k] = np.concatenate([pe[k], np.zeros(n_cat - len(pe[k]))])
        off = warm & ~on & (y > 0)  # the same filter's bias on ordinary days (stockouts in the history)
        pe["y0"] += np.bincount(cat[off], y[off], n_cat)
        pe["base0"] += np.bincount(cat[off], base[off], n_cat)
        if not on.any():
            return
        c = cat[on]
        pe["y"] += np.bincount(c, y[on], n_cat)
        pe["base"] += np.bincount(c, base[on], n_cat)
        pe["lr"] += np.bincount(c, base[on] * lr[on], n_cat)
        pe["days"] += np.bincount(c, None, n_cat)

    def price_elasticity(self) -> dict[str, tuple[float, float]]:
        """Per category: (elasticity, series-days of evidence) from the backfill's price changes."""
        pe, out = self.price_ev, {}
        for k, name in enumerate(self.categories[:len(pe["y"])]):
            if pe["base"][k] >= 30 and pe["y"][k] > 0:
                mean_lr = pe["lr"][k] / pe["base"][k]
                bias = pe["y0"][k] / pe["base0"][k] if pe["base0"][k] > 0 else 1.0
                if abs(mean_lr) > 0.03:
                    out[name] = (float(np.log(pe["y"][k] / pe["base"][k] / bias) / mean_lr), float(pe["days"][k]))
        return out

    def _disp_add(self, f: np.ndarray, y: np.ndarray) -> None:
        w = 1.0 / (f + 1.0) ** 2
        r2 = (y - f) ** 2
        self.disp_sums += [np.sum(w * f * f), np.sum(w * f ** 3), np.sum(w * f ** 4), np.sum(w * f * r2),
                           np.sum(w * f * f * r2)]

    def dispersion(self) -> tuple[float, float]:
        """(a, b) in var(y) = a f + b f^2, by weighted least squares on recent one-step errors.

        a: Poisson-like noise (above 1 when orders carry several units); b: day-to-day swings that scale with
        the volume, which a few hours of a day can't average away.
        """
        s11, s12, s22, t1, t2 = self.disp_sums
        det = s11 * s22 - s12 * s12
        if det <= 1e-9:
            return 1.5, 0.08
        a = (t1 * s22 - t2 * s12) / det
        b = (s11 * t2 - s12 * t1) / det
        return float(np.clip(a, 1.0, 6.0)), float(np.clip(b, 0.005, 0.4))

    def _pool(self, rows: np.ndarray, M: np.ndarray) -> np.ndarray:
        n_grp = len(self.grp_index)
        if not len(rows):
            return np.zeros((n_grp, 7))
        return np.stack([np.bincount(self.grp[rows], M[rows, w], n_grp) for w in range(7)], axis=1)

    def _init_state(self, Y: np.ndarray, P: np.ndarray, off: np.ndarray, t0: int) -> State:
        n, T = Y.shape
        t = np.arange(T)
        adj = Y / P
        w28 = (t >= off[:, None]) & (t < off[:, None] + 28)
        w14 = (t >= off[:, None]) & (t < off[:, None] + 14)
        level = np.maximum((adj * w14).sum(1) / np.maximum(w14.sum(1), 1), LEVEL_FLOOR)
        st = State.zeros(n)
        st.level[:] = level
        wds = np.array([self.day(t0 + k).weekday() for k in range(T)])
        mean28 = (adj * w28).sum(1) / np.maximum(w28.sum(1), 1)
        for w in range(7):
            m = w28 & (wds == w)
            cnt = m.sum(1)
            st.A[:, w] = np.where(cnt > 0, (adj * m).sum(1) / np.maximum(cnt, 1), mean28)
            st.B[:, w] = mean28
        st.A = np.maximum(st.A, 1e-3)
        st.B = np.maximum(st.B, 1e-3)
        st.season = st.A / st.B
        st.season /= st.season.mean(1, keepdims=True)
        st.var_r[:] = 1.0
        return st

    def _choose_alpha(self, init: State, Y, P, C, off, cat, grp, mer, t0, pool_A, pool_B) -> np.ndarray:
        """Run every alpha side by side (rows tiled) on a copy of the pools; best per category by WAPE."""
        k, n = len(ALPHAS), len(off)
        st = State(*(np.concatenate([getattr(init, f)] * k) for f in init.__dataclass_fields__))
        st.alpha = np.repeat(ALPHAS, n)
        cat_k, grp_k, mer_k, off_k = np.tile(cat, k), np.tile(grp, k), np.tile(mer, k), np.tile(off, k)
        Yk, Pk, Ck = np.tile(Y, (k, 1)), np.tile(P, (k, 1)), np.tile(C, (k, 1))
        pools = (self.pool.copy(), self.fpool.copy(), self.sens.copy())
        err = np.zeros(k * n)
        for t in range(Y.shape[1]):
            act = np.flatnonzero(off_k <= t)
            pred = step(st, act, cat_k[act], self.day(t0 + t), Yk[act, t], Pk[act, t], Ck[act, t],
                        pools, pool_A * k, pool_B * k, grp_k[act], mer_k[act])
            scored = off_k[act] <= t - 14
            err[act[scored]] += np.abs(Yk[act[scored], t] - pred[scored])
        n_cat = len(self.pool.num)
        per_cat = np.stack([np.bincount(cat, err[j * n:(j + 1) * n], n_cat) for j in range(k)])
        return ALPHAS[per_cat.argmin(axis=0)][cat]

    # ------------------------------------------------------------------ live

    def close_day(self, day_idx: int, y: np.ndarray, ratio: np.ndarray, censored: np.ndarray) -> dict:
        """A sim day closed: record it, score yesterday's forecast, update every fitted series.

        y, ratio, censored: length n. Returns the scored rows for LiveAccuracy.
        """
        n = self.n
        self.ensure_day(day_idx)
        self.hist[:n, day_idx] = y
        self.ratio[:n, day_idx] = ratio
        self.cens[:n, day_idx] = censored
        self.start[:n] = np.where(self.start[:n] < 0, day_idx, self.start[:n])
        rows = np.flatnonzero(self.fitted[:n])
        scored = rows[self.f_day[rows] == day_idx]
        scored = scored[~censored[scored]]   # scored on in-stock days: a sold-out day shows the shelf, not demand
        out = {"rows": scored, "y": y[scored], "f": self.f_next[scored], "lo": self.lo_next[scored],
               "hi": self.hi_next[scored]}
        out["naive"] = self.hist[scored, day_idx - 7].astype(float) if day_idx >= 7 else np.zeros(len(scored))
        out["naive_ok"] = (day_idx - 7 >= self.start[scored])
        self.live_abs[scored] += np.abs(out["y"] - out["f"])
        self.live_y[scored] += out["y"]
        self.y_lag[:n, 1], self.f_lag[:n, 1] = self.y_lag[:n, 0], self.f_lag[:n, 0]
        self.y_lag[:n, 0] = self.f_lag[:n, 0] = 0
        clean = scored[~censored[scored]]  # a day that ran out of stock is no evidence of weak demand
        self.y_lag[clean, 0], self.f_lag[clean, 0] = y[clean], self.f_next[clean]
        self.disp_sums *= 0.5
        self._disp_add(out["f"], out["y"].astype(float))
        if len(rows):
            under = censored[rows] & (y[rows] < self.f_next[rows])
            step(self.st, rows, self.cat[rows], self.day(day_idx), y[rows].astype(float),
                 self.pmult(rows, ratio[rows].astype(float)), under, self.pools,
                 np.zeros((len(self.grp_index), 7)), np.zeros((len(self.grp_index), 7)), self.grp[rows], self.mer[rows])
        cold = np.flatnonzero(~self.fitted[:n] & (self.start[:n] >= 0))
        if len(cold):
            self.fit(cold, day_idx)
        if len(rows):
            self.predict_next(rows, day_idx + 1)
        return out

    def base(self, rows: np.ndarray, day_idx: int) -> np.ndarray:
        """No-offer forecast of rows for one day."""
        d = self.day(day_idx)
        c, fk = self.cat[rows], fest_kind(d)
        fest = 1.0 if not fk else np.maximum(1 + self.sens.k(self.mer[rows]) * (self.fpool.mult()[c, fk] - 1), FEST_FLOOR)
        return self.st.level[rows] * self.st.season[rows, d.weekday()] * self.pool.mult()[c, day_kind(d)] * fest

    @property
    def pools(self) -> tuple:
        return self.pool, self.fpool, self.sens

    def festival_lift(self, name: str | None = None) -> tuple[np.ndarray, np.ndarray]:
        """Per category: the learned multiplier on a festival's day (all festivals' days pooled when no
        name), and how many category-days it was learned from."""
        ks = [FESTIVAL_DAY[name]] if name in FESTIVAL_DAY else list(FESTIVAL_DAY.values())
        num, den = self.fpool.num[:, ks].sum(1), self.fpool.den[:, ks].sum(1)
        return (num + EVENT_PSEUDO) / (den + EVENT_PSEUDO), self.fpool.days[:, ks].sum(1)

    def interval(self, rows: np.ndarray, f: np.ndarray, h: int = 1) -> tuple[np.ndarray, np.ndarray]:
        sd = np.sqrt(self.st.var_r[rows] * (f + 1.0) * (1 + (h - 1) * self.st.alpha[rows] ** 2))
        return np.maximum(f - Z80 * sd, 0.0), f + Z80 * sd

    def predict_next(self, rows: np.ndarray, day_idx: int, offer_mult: np.ndarray | None = None) -> None:
        f = self.base(rows, day_idx) * (1.0 if offer_mult is None else offer_mult)
        self.f_next[rows] = f
        self.lo_next[rows], self.hi_next[rows] = self.interval(rows, f)
        self.f_day[rows] = day_idx

    def horizon(self, rows: np.ndarray, start: int, h: int = HORIZON) -> np.ndarray:
        """No-offer forecasts, shape (len(rows), h), for days start .. start + h - 1."""
        return np.stack([self.base(rows, start + k) for k in range(h)], axis=1) if len(rows) else np.zeros((0, h))

    def series_wape(self, rows: np.ndarray) -> np.ndarray:
        """Live WAPE where days were scored, else the smoothed one-step WAPE from the backfill."""
        live = np.divide(self.live_abs[rows], self.live_y[rows], out=np.full(len(rows), np.nan),
                         where=self.live_y[rows] >= 5)
        ew = self.st.ew_abs[rows] / np.maximum(self.st.ew_y[rows], 0.1)
        return np.clip(np.where(np.isnan(live), ew, live), 0.0, 2.0)

    def describe(self, i: int) -> str:
        m = self.pool.mult()[self.cat[i]]
        s = self.st.season[i]
        days = "MTWTFSS"
        peak = days[int(s.argmax())]
        parts = [f"Holt-Winters (level a={self.st.alpha[i]:.2f}, weekday x{s.min():.2f}-{s.max():.2f}, peak {peak})"]
        lift, seen = self.festival_lift()
        if seen[self.cat[i]]:
            parts.append(f"festival day x{lift[self.cat[i]]:.2f}")
        parts.append(f"payday x{m[PAYDAY]:.2f}")
        return ", ".join(parts)


# ------------------------------------------------------------------ nowcast

# Prior share of a day's units by hour (Indian retail: late morning and evening peaks), until learned.
_PRIOR = np.array([0.2, 0.1, 0.1, 0.1, 0.1, 0.3, 1, 2.5, 4, 5.5, 6.5, 7, 7, 6, 5, 4.5, 5, 6, 7.5, 8, 7.5, 5.5, 3.5, 1.5])


class HourProfile:
    """Share of a day's units by hour, per platform, learned from closed days."""

    def __init__(self, n_platforms: int = 4, rate: float = 0.3):
        self.prof = np.tile(_PRIOR / _PRIOR.sum(), (n_platforms, 1))
        self.today = np.zeros((n_platforms, 24))
        self.days = np.zeros(n_platforms, dtype=int)
        self.rate = rate

    def grow(self, n: int) -> None:
        if n > len(self.prof):
            k = n - len(self.prof)
            self.prof = np.vstack([self.prof, np.tile(_PRIOR / _PRIOR.sum(), (k, 1))])
            self.today = np.vstack([self.today, np.zeros((k, 24))])
            self.days = np.concatenate([self.days, np.zeros(k, dtype=int)])

    def add(self, platform: int, hour: int, units: float) -> None:
        self.today[platform, hour] += units

    def cum_share(self, platform: np.ndarray, hour: float) -> np.ndarray:
        """Expected share of the day's units sold by ``hour`` (0-24, fractional)."""
        h = int(min(max(hour, 0.0), 23.999))
        frac = min(max(hour, 0.0), 24.0) - h
        cum = np.cumsum(self.prof, axis=1)
        before = cum[:, h - 1] if h else np.zeros(len(self.prof))
        return (before + self.prof[:, h] * frac)[platform]

    def share_between(self, platform: np.ndarray, h0: float, h1: float) -> np.ndarray:
        return self.cum_share(platform, h1) - self.cum_share(platform, h0)

    def close_day(self) -> None:
        total = self.today.sum(axis=1)
        ok = total >= 50
        if ok.any():
            share = self.today[ok] / total[ok, None]
            rate = np.maximum(self.rate, 1.0 / (self.days[ok] + 1))[:, None]  # first days count fully
            self.prof[ok] = (1 - rate) * self.prof[ok] + rate * share
            self.days[ok] += 1
        self.today[:] = 0


# ------------------------------------------------------------------ live accuracy

class LiveAccuracy:
    """Running WAPE of the day-ahead forecast vs seasonal naive, honestly out of sample."""

    def __init__(self):
        self.abs = self.y = self.naive_abs = self.naive_y = 0.0
        self.m_abs = self.m_y = 0.0
        self.m_naive_abs = 0.0
        self.covered = self.n = 0
        self.days = 0
        self.series: set[int] = set()
        self.plat_abs: dict[int, float] = {}
        self.plat_y: dict[int, float] = {}
        self.last: dict = {}
        # per merchant, so any branch of the tree (a platform, one merchant) gets its own accuracy
        self.mer = {k: np.zeros(0) for k in ("abs", "y", "naive_abs", "naive_y", "m_abs")}

    def _mer_add(self, key: str, m: np.ndarray, values: np.ndarray) -> None:
        a = self.mer[key]
        if len(m):
            need = int(m.max()) + 1
            if need > len(a):
                a = self.mer[key] = np.concatenate([a, np.zeros(need - len(a) + 64)])
            a[:need] += np.bincount(m, values, need)

    def scoped(self, merchants: np.ndarray) -> dict:
        """WAPE over the merchants in a boolean mask: per product, per merchant (daily totals), seasonal naive."""
        def total(key):
            a = self.mer[key]
            k = min(len(a), len(merchants))
            return float(a[:k][merchants[:k]].sum())
        y, ny = total("y"), total("naive_y")
        return {"wape": round(total("abs") / y, 4) if y else None,
                "merchant_wape": round(total("m_abs") / y, 4) if y else None,
                "baseline_wape": round(total("naive_abs") / ny, 4) if ny else None}

    def score(self, s: dict, platform: np.ndarray, merchant: np.ndarray) -> None:
        rows, y, f = s["rows"], s["y"].astype(float), s["f"]
        if not len(rows) or y.sum() <= 0:
            return
        self.days += 1
        self.series.update(rows.tolist())
        err = np.abs(y - f)
        self.abs += err.sum()
        self.y += y.sum()
        ok = s["naive_ok"]
        self.naive_abs += np.abs(y[ok] - s["naive"][ok]).sum()
        self.naive_y += y[ok].sum()
        self.covered += int(((y >= np.floor(s["lo"])) & (y <= np.ceil(s["hi"]))).sum())
        self.n += len(rows)
        p, m = platform[rows], merchant[rows]
        for pid in np.unique(p):
            sel = p == pid
            self.plat_abs[int(pid)] = self.plat_abs.get(int(pid), 0.0) + err[sel].sum()
            self.plat_y[int(pid)] = self.plat_y.get(int(pid), 0.0) + y[sel].sum()
        nm = int(m.max()) + 1
        my, mf = np.bincount(m, y, nm), np.bincount(m, f, nm)
        self._mer_add("abs", m, err)
        self._mer_add("y", m, y)
        self._mer_add("naive_abs", m[ok], np.abs(y[ok] - s["naive"][ok]))
        self._mer_add("naive_y", m[ok], y[ok])
        self._mer_add("m_abs", np.arange(nm), np.abs(my - mf))
        self.m_abs += np.abs(my - mf).sum()
        self.m_y += my.sum()
        self.m_naive_abs += np.abs(np.bincount(m[ok], y[ok], nm) - np.bincount(m[ok], s["naive"][ok], nm)).sum()
        my_ok, mn_ok = np.bincount(m[ok], y[ok], nm), np.bincount(m[ok], s["naive"][ok], nm)
        self.last = {"wape": round(wape(y, f), 4),
                     "baseline_wape": round(wape(y[ok], s["naive"][ok]), 4) if ok.any() else None,
                     "merchant_wape": round(wape(my, mf), 4),
                     "merchant_baseline_wape": round(wape(my_ok, mn_ok), 4) if ok.any() else None,
                     "series": len(rows), "units": int(y.sum())}

    def summary(self, platform_ids: list[str], backtest: dict) -> dict:
        def r(a, b):
            return round(a / b, 4) if b else None
        return {
            "wape": r(self.abs, self.y),
            "baseline_wape": r(self.naive_abs, self.naive_y),
            "days_scored": self.days,
            "series_scored": len(self.series),
            "by_platform": {platform_ids[p]: r(self.plat_abs[p], self.plat_y[p]) for p in self.plat_abs
                            if p < len(platform_ids)},
            "merchant_wape": r(self.m_abs, self.m_y),
            "merchant_baseline_wape": r(self.m_naive_abs, self.m_y),
            "coverage_80": r(self.covered, self.n),
            "last_day": self.last or None,
            "backtest": {"wape": r(backtest["abs"], backtest["y"]),
                         "baseline_wape": r(backtest["naive_abs"], backtest["naive_y"]),
                         "merchant_wape": r(backtest.get("m_abs", 0.0), backtest.get("m_y", 0.0)),
                         "merchant_baseline_wape": r(backtest.get("m_naive_abs", 0.0), backtest.get("m_y", 0.0)),
                         "days": backtest["days"]},
        }
