"""Live Commerce engine: learns from the platform feeds, forecasts every merchant x SKU and pushes offer signals.

It only knows what arrives on the ingest topics (world, catalogs, history, orders, stock, clock,
control), never the simulator itself, so real Shopify / Reliance / DMart feeds can replace it.
The paho thread only queues raw messages; one worker drains the queue in batches (robust to
bursts) and does all the work, then publishes to the permission-checked live broker:

    stats every 500 ms wall, a ticker sample every 250 ms, signals and outcomes as they happen.

Cadence in simulated time: an intraday nowcast check every sim hour (drops, surges, evening
clearance), and at each sim day close the forecaster update, live accuracy scoring and the
day-level decisions (restock, price-response and overstock offers, festival, win-back). The command
centre's views (/platforms, /lessons, /attention, /tree) live in ``hq``; the engine keeps the per-merchant
ledgers they read. Budgets keep
the mix to roughly half offers (discount, clearance, coupon), then restock, festival and a few surges. The HTTP API from the contract
runs in the same process (uvicorn in a thread).

    uv run python -m secondbrain.commerce.engine [--autopilot 0.5]
"""

import argparse
import json
import logging
import math
import random
import signal
import threading
import time
from collections import deque
from datetime import date, datetime, timedelta

import numpy as np

from secondbrain.commerce import contract as C
from secondbrain.commerce.forecast import Forecaster, HourProfile, LiveAccuracy
from secondbrain.commerce.offers import (AUTO_KINDS, CYCLE_DAYS, EXCESS_COST, KEEP_DAYS, KEEP_PROB, SPOIL_COST,
                                         Z_ASSURED, choose_defence,
                                         WINBACK_RESPONSE, Cooldowns, Offer, OfferBook, autopilot_merchant,
                                         choose_discount, format_inr, margin_floor, restock_plan)
from secondbrain.commerce.state import Registry

log = logging.getLogger(__name__)

CHECK_EVERY = timedelta(hours=1)    # the intraday signal budget renews every sim hour
CHECK_WALL_SECONDS = 0.05           # how often the nowcast check runs (wall time)
HOURLY_BUDGET = {"discount": 1, "surge": 1, "clearance": 1}   # per sim hour: the best few, not every twitch
CLOSE_BUDGET = {"restock": 6, "discount": 14, "surge": 2, "festival": 5, "winback": 8}  # per sim day close
DAILY_CAP = {"surge": 4}            # surges are many on a noisy day; only the most valuable few reach the wall
MIN_SURGE_VALUE = 1000.0            # rupees of full margin a surge must protect
DEFEND_HOURS = 72                   # a volume-defence offer runs long enough to win the customers back
PERSIST_PRIOR = 0.5                 # share of a flagged drop expected to last, until measured...
PERSIST_STRENGTH = 5.0              # ...worth this many observed drops
DROP_PERSIST_SD = 0.3               # uncertainty of how much of a drop lasts, in the uplift bound
OFFER_HOURS = 48
CLEARANCE_HOURS = 24
FESTIVAL_DAYS = 10                  # festival signals start this many days ahead
FESTIVAL_MIN_LIFT = 1.15
Z_STRONG = -3.0                     # a drop this clear is more often a real, lasting one (persistence learned apart)
Z_BELOW = -2.0                      # 2-3 day evidence for a drop (one day's swing is ~30% at the sim's noise)
Z_SURGE = 4.0                       # one day is enough for a surge, but 16k series are checked every hour
WINBACK_MIN = 5                     # lapsed regulars before a merchant gets a win-back signal
MIN_GAIN = 150.0                    # rupees of expected profit before an offer is worth a merchant's attention
EPOCH = datetime(2026, 1, 1)


def hours(ts: datetime) -> float:
    return (ts - EPOCH).total_seconds() / 3600


def pct(values, q: float) -> float | None:
    return round(float(np.percentile(np.fromiter(values, float), q)), 1) if values else None


def recent(samples, seconds: float = 120.0, at_least: int = 30) -> list[float]:
    """(wall s, ms) samples of the last two minutes, or the last 30 when signals are sparse: a latency that
    moves with the run, not the median of everything since the start."""
    if not samples:
        return []
    cut = time.time() - seconds
    live = [v for t, v in samples if t >= cut]
    return live if len(live) >= at_least else [v for _, v in list(samples)[-at_least:]]


def _above(up: float) -> str:
    """'64% above forecast', or '3.1x its forecast' once a percentage stops reading naturally."""
    return f"{up:.0%} above forecast" if up < 1.5 else f"{up + 1:.1f}x its forecast"


def hour_label(ts: datetime) -> str:
    h = ts.hour % 12 or 12
    return f"{h} {'am' if ts.hour < 12 else 'pm'}"


class Engine:
    """All streaming state and decisions. Feed it messages with ``process``; publishing is injected."""

    def __init__(self, publish_live=None, publish_control=None, autopilot: float = 0.5, on_reset=None):
        self.publish_live = publish_live or (lambda topic, payload, qos=0: None)
        self.publish_control = publish_control or (lambda payload: None)
        self.on_reset = on_reset
        self.autopilot = autopilot
        self.lock = threading.RLock()
        self.started = time.time()
        self.signal_seq = 0
        # Snapshots the HTTP API answers from, so pages never wait behind a busy batch:
        # key -> (computed at, value), key -> when a page last asked, key -> (compute, min seconds between)
        self.views: dict = {}
        self.wanted: dict = {}
        self.view_fns: dict = {}
        self.reset()

    # ------------------------------------------------------------------ state

    def reset(self, sim_start: str = C.SIM_START) -> None:
        self.sim_start = sim_start
        self.views = {}   # a new run: nothing from the last one
        self.reg = Registry()
        self.fc = Forecaster(date.fromisoformat(sim_start[:10]) - timedelta(days=C.HISTORY_DAYS + 7))
        self.profile = HourProfile()
        self.acc = LiveAccuracy()
        self.book = OfferBook()
        self.cool = Cooldowns()
        self.signals: deque = deque(maxlen=500)
        self.signal_by_id: dict[str, dict] = {}
        self.plans: dict[str, dict] = {}
        self.drop_watch: dict[tuple[int, int], tuple[float, np.ndarray]] = {}   # (series, day) -> (dev, forecasts)
        self.drop_obs: list[tuple[float, float, bool]] = []    # (observed persistence, weight, strong evidence)
        self.day_counts: dict[str, int] = {}
        self.applied_signals: set[str] = set()
        self.outbox: list[tuple[dict, int]] = []
        self.outcome_box: list[dict] = []
        self.today: date | None = None
        self.now: datetime | None = None
        self.next_check: datetime | None = None
        self.hour_budget: dict[str, int] = dict(HOURLY_BUDGET)
        self._closing = False
        self.last_check_wall = 0.0
        self.speed, self.running, self.speed_actual = C.DEFAULT_SPEED, False, None
        self.world_started: int | None = None
        self.customers_world = 0
        self.active_today: set[str] = set()
        self.pending_history: dict[int, dict] = {}
        self.unfitted: set[int] = set()
        self.last_history = 0.0
        self.last_clock_sent = 0
        self.elast_dirty = True
        self.totals = {"orders": 0, "units": 0.0, "revenue": 0.0, "msgs": 0}
        self.platform_totals: dict[str, list] = {}
        self.pipeline = {"ingested": 0, "forecasts_updated": 0, "signals_raised": 0, "delivered": 0}
        self.signals_by_kind: dict[str, int] = {}
        self.lat = {k: deque(maxlen=4000) for k in ("broker", "process", "signal", "end_to_end")}
        self.window: deque = deque()       # (wall s, orders, revenue, msgs, busy s, {platform: orders})
        self.run_since = 0.0               # wall time the clock last started or stopped
        self.batch_platform: dict[str, int] = {}
        self.ticker: list = []
        self.ticker_seen = 0
        self.last_stats: dict = {}
        self.last_stats_at = self.last_ticker_at = 0.0
        self.run_until: str | None = None
        # What the command centre (HQ, Graph, Chat) reads, kept per merchant so any branch of the tree is cheap:
        self.live_start: int | None = None          # day index of the first live day
        self.ledger: dict[int, list] = {}           # merchant -> [offers applied, uplift rupees, worked, judged]
        self.sig_count: dict[tuple[int, str, str], int] = {}    # (merchant, kind, category) -> signals raised
        self.outcome_log: list[tuple] = []          # every closed offer, compact (see _log_outcome)
        self.price_checks: dict[tuple[int, str], list] = {}     # (merchant, category) -> [checked, declined,
                                                                #    margin sum of declined, elasticity sum]

    def day_idx(self) -> int:
        return self.fc.day_index(self.today)

    # ------------------------------------------------------------------ ingest

    def process(self, batch: list[tuple[str, bytes | dict, float]]) -> None:
        """Handle a batch of (topic, payload, received wall ms), then decide and publish."""
        t0 = time.perf_counter()
        n_orders = self.totals["orders"]
        revenue = self.totals["revenue"]
        with self.lock:
            order_recv = []
            for topic, payload, recv in batch:
                try:
                    if self._handle(topic, payload, recv):
                        order_recv.append(recv)
                except Exception:
                    log.exception("could not handle %s", topic)
            self._after_batch()
            done = time.time() * 1000
            for r in order_recv[-200:]:
                self.lat["process"].append(done - r)
            self._flush(done)
            busy = time.perf_counter() - t0
            self.window.append((time.time(), self.totals["orders"] - n_orders, self.totals["revenue"] - revenue,
                                len(batch), busy, self.batch_platform))
            if busy > 0.5:
                log.info("slow batch: %d messages in %.2f s", len(batch), busy)
            self.batch_platform = {}

    def _handle(self, topic: str, payload, recv: float) -> bool:
        msg = json.loads(payload) if isinstance(payload, (bytes, str)) else payload
        self.totals["msgs"] += 1
        if topic == C.CLOCK:
            self._on_clock(msg)
        elif topic == C.WORLD:
            self._on_world(msg)
        elif topic == C.CONTROL:
            self._on_control(msg)
        else:
            parsed = C.parse_merchant_topic(topic)
            if not parsed:
                return False
            platform, merchant, kind = parsed
            if kind == "orders":
                self._on_orders(platform, merchant, msg, recv)
                return True
            handler = {"catalog": self._on_catalog, "history": self._on_history, "stock": self._on_stock}.get(kind)
            if handler:
                handler(platform, merchant, msg)
        return False

    def _on_world(self, msg: dict) -> None:
        started = msg.get("started_ms")
        if self.world_started is not None and started != self.world_started:
            self._reset(msg.get("sim_start", C.SIM_START))
        self.world_started = started
        if self.reg.n == 0 and msg.get("sim_start") and msg["sim_start"] != self.sim_start:
            self.reset(msg["sim_start"])
            self.world_started = started
        for p in msg.get("platforms", []):
            self.reg.platform(p["id"], p)
        self.profile.grow(len(self.reg.platform_ids))
        for m in msg.get("merchants", []):
            self.reg.merchant(m["id"], m["platform"], m)
        self.customers_world = msg.get("customers", 0)

    def _on_catalog(self, platform: str, merchant: str, msg: dict) -> None:
        m = self.reg.merchant(merchant, platform)
        self.profile.grow(len(self.reg.platform_ids))
        for p in msg.get("products", []):
            i, created = self.reg.series(m, p)
            if created:
                cat = p.get("category", "Other")
                j = self.fc.add_series(cat, self.book.elasticity.value(merchant, cat), f"{platform}/{cat}", merchant)
                assert i == j
        self.reg.has_catalog.add(m)
        if m in self.pending_history:
            self._apply_history(m, self.pending_history.pop(m))

    def _on_history(self, platform: str, merchant: str, msg: dict) -> None:
        m = self.reg.merchant(merchant, platform)
        if m in self.reg.has_catalog:
            self._apply_history(m, msg)
        else:
            self.pending_history[m] = msg

    def _apply_history(self, m: int, msg: dict) -> None:
        skus = msg.get("skus", [])
        known = [(k, self.reg.series_index.get((m, s))) for k, s in enumerate(skus)]
        known = [(k, i) for k, i in known if i is not None]
        if not known or not msg.get("days"):
            return
        cols = np.array([k for k, _ in known])
        rows = np.array([i for _, i in known])
        days = np.array([self.fc.day_index(date.fromisoformat(d)) for d in msg["days"]])
        units = np.asarray(msg["units"], float)[:, cols]
        prices = np.asarray(msg.get("prices") or np.zeros_like(units), float)
        prices = prices[:, cols] if prices.ndim == 2 and prices.shape[1] == len(skus) else np.zeros_like(units)
        ratio = np.where((units > 0) & (prices > 0), prices / np.maximum(self.reg.price[rows], 1e-9), 1.0)
        out = np.asarray(msg.get("sold_out") or np.zeros_like(units), float)
        out = out[:, cols] > 0 if out.shape == (len(days), len(skus)) else np.zeros_like(units, dtype=bool)
        self.fc.set_history(rows, days, units, ratio, out)
        self.fc.fitted[rows] = False
        self.unfitted.update(rows.tolist())
        self.last_history = time.time()

    def _on_stock(self, platform: str, merchant: str, msg: dict) -> None:
        m = self.reg.merchant_index.get(merchant)
        if m is None:
            return
        for sku, units in (msg.get("stock") or {}).items():
            i = self.reg.series_index.get((m, sku))
            if i is not None:
                self.reg.stock[i] = float(units)
                if units <= 0:
                    self.reg.t_out[i] = True
        for sku in (msg.get("restocked") or {}):
            self.cool.clear((merchant, sku, "restock"))

    def _on_clock(self, msg: dict) -> None:
        self.speed = msg.get("requested", msg.get("speed", self.speed))  # the label the wall shows
        self.speed_actual = msg.get("speed")
        running = bool(msg.get("running", self.running))
        if running != self.running:   # rates measure this run only: 0 the moment it pauses, no ramp after Start
            self.window.clear()
            self.run_since = time.time()
        self.running = running
        self.catching_up = bool(msg.get("catching_up"))
        self.last_clock_sent = msg.get("sent_ms", self.last_clock_sent)
        self.run_until = msg.get("run_until")
        if msg.get("sim_ts"):
            self._advance(datetime.fromisoformat(msg["sim_ts"]))

    def _on_orders(self, platform: str, merchant: str, msg: dict, recv: float) -> None:
        ts = datetime.fromisoformat(msg["sim_ts"])
        self._advance(ts)
        reg = self.reg
        m = reg.merchant(merchant, platform)
        sent = msg.get("sent_ms", 0)
        if sent:
            self.lat["broker"].append(recv - sent)
            reg.m_last_sent[m] = sent
        p = int(reg.m_plat[m])
        day = self.day_idx()
        units_msg = rev_msg = 0.0
        orders = msg.get("orders", [])
        idx = reg.series_index
        offers = self.book.by_series
        for o in orders:
            q = o["qty"]
            amount = q * o["price"]
            units_msg += q
            rev_msg += amount
            i = idx.get((m, o["sku"]))
            if i is not None:
                reg.t_units[i] += q
                reg.t_rev[i] += amount
                reg.t_orders[i] += 1
                left = reg.stock[i] - q
                if left > 0:
                    reg.stock[i] = left
                else:
                    reg.stock[i] = 0.0
                    reg.t_out[i] = True
                if i in offers:
                    self.book.record(i, q, amount)
                    if left <= 0:
                        self.book.active[offers[i]].stockout = True
            cust = o.get("customer")
            if cust:
                reg.visit(m, cust, day)
                self.active_today.add(cust)
            self.ticker_seen += 1
            if len(self.ticker) < 5:
                self.ticker.append((m, i, o, amount))
            else:
                j = random.randrange(self.ticker_seen)
                if j < 5:
                    self.ticker[j] = (m, i, o, amount)
        n = len(orders)
        reg.m_orders_today[m] += n
        reg.m_rev_today[m] += rev_msg
        reg.m_orders_total[m] += n
        reg.m_rev_total[m] += rev_msg
        self.profile.add(p, ts.hour, units_msg)
        self.totals["orders"] += n
        self.totals["units"] += units_msg
        self.totals["revenue"] += rev_msg
        self.pipeline["ingested"] += n
        pid = reg.platform_ids[p]
        self.batch_platform[pid] = self.batch_platform.get(pid, 0) + n
        pt = self.platform_totals.setdefault(pid, [0, 0.0])
        pt[0] += n
        pt[1] += rev_msg

    def _on_control(self, msg: dict) -> None:
        cmd = msg.get("cmd")
        if cmd == "reset":
            self._reset(self.sim_start)
        elif cmd == "apply_offer":
            self._register_offer(msg)

    def _reset(self, sim_start: str) -> None:
        log.info("reset: clearing engine state")
        self.reset(sim_start)
        if self.on_reset:
            self.on_reset()

    # ------------------------------------------------------------------ time

    def _advance(self, ts: datetime) -> None:
        d = ts.date()
        if self.today is None:
            self.today = d
            self.live_start = self.day_idx()
            self._fit_pending(force=True)
        elif d < self.today - timedelta(days=1):
            self._reset(self.sim_start)   # the clock went back: a restarted simulator
            self.today = d
            self.live_start = self.day_idx()
        while d > self.today:
            self._close_day()
        if self.now is None or ts > self.now:
            self.now = ts
        if self.next_check is None:
            self.next_check = ts.replace(minute=0, second=0, microsecond=0) + CHECK_EVERY

    def _fit_pending(self, force: bool = False) -> None:
        if not self.unfitted or (not force and time.time() - self.last_history < 0.3):
            return
        if self.today is not None:
            through = self.day_idx() - 1
        else:
            through = int(self.fc.start[:self.fc.n].max()) + C.HISTORY_DAYS
            through = min(through, self.fc.hist.shape[1] - 1)
            filled = np.flatnonzero(self.fc.hist[:self.fc.n].any(axis=0))
            through = int(filled.max()) if len(filled) else through
        self._refresh_elasticity()
        rows = np.array(sorted(self.unfitted))
        self.fc.fit(rows, through)
        self.unfitted.clear()
        self.pipeline["forecasts_updated"] += len(rows)
        self.book.elasticity.observe_backfill(self.fc.price_elasticity())
        self.elast_dirty = True
        if self.today is not None:
            self._offer_mults(self.day_idx())

    def _refresh_elasticity(self) -> None:
        if not self.elast_dirty:
            return
        e = self.book.elasticity
        reg = self.reg
        cache: dict = {}
        for i in range(reg.n):
            key = (reg.merchants[reg.s_merch[i]]["id"], reg.category[i])
            if key not in cache:
                cache[key] = e.value(*key)
            self.fc.elast[i] = cache[key]
        self.elast_dirty = False

    def _close_day(self) -> None:
        t0 = time.perf_counter()
        try:
            self._close_day_inner()
        finally:
            log.info("day close %s: %.2f s", self.today, time.perf_counter() - t0)

    def _close_day_inner(self) -> None:
        """The sim day ``self.today`` closed: score, learn, update forecasts, then decide for the new day."""
        self._fit_pending(force=True)
        self._refresh_elasticity()
        reg, n, di = self.reg, self.reg.n, self.day_idx()
        y = reg.t_units[:n].copy()
        ratio = np.where(y > 0, reg.t_rev[:n] / np.maximum(y, 1e-9) / np.maximum(reg.price[:n], 1e-9), 1.0)
        scored = self.fc.close_day(di, y, ratio, (reg.stock[:n] <= 0) | reg.t_out[:n])
        self.acc.score(scored, reg.s_plat[:n], reg.s_merch[:n])
        self.profile.close_day()
        self.pipeline["forecasts_updated"] += int(self.fc.fitted[:n].sum())
        self.today += timedelta(days=1)
        self.day_counts = {}
        reg.new_day()
        self.active_today = set()
        self._offer_mults(di + 1)
        midnight = datetime.combine(self.today, datetime.min.time())
        if self.now is None or self.now < midnight:
            self.now = midnight
        self._score_drops()
        self._day_decisions(scored)

    def _offer_mults(self, day: int) -> None:
        """Tomorrow's forecast includes the offers that will still be running."""
        if not self.book.active:
            return
        start = datetime.combine(self.fc.day(day), datetime.min.time())
        rows, mult = [], []
        for o in self.book.active.values():
            if o.until <= start or not self.fc.fitted[o.series]:
                continue
            hrs = min((o.until - start).total_seconds() / 3600, 24)
            frac = float(self.profile.cum_share(np.array([self.reg.s_plat[o.series]]), hrs)[0])
            e = self.fc.elast[o.series]
            rows.append(o.series)
            mult.append(1 + frac * ((1 - o.discount) ** e - 1))
        if rows:
            self.fc.predict_next(np.array(rows), day, np.array(mult))

    def _after_batch(self) -> None:
        self._fit_pending()
        if self.now is None:
            return
        for o in self.book.due(self.now):
            out = self.book.evaluate(o)
            self._log_outcome(o, out)
            self.outcome_box.append(out)
            self.elast_dirty = True
        if self.next_check is None:
            return
        new_hour = self.now >= self.next_check
        if new_hour:  # a fresh budget, and always one check per sim hour however fast the clock runs
            self.next_check = self.now.replace(minute=0, second=0, microsecond=0) + CHECK_EVERY
            self.hour_budget = dict(HOURLY_BUDGET)
        wall = time.monotonic()
        if (new_hour or wall - self.last_check_wall >= CHECK_WALL_SECONDS) and any(self.hour_budget.values()):
            # Checked continuously, not once an hour, so a signal follows its order within milliseconds;
            # the hourly budget still keeps it to the best few.
            self.last_check_wall = wall
            self._nowcast_check(self.hour_budget)

    # ------------------------------------------------------------------ forecasts for decisions

    def _rows_today(self) -> np.ndarray:
        n = self.reg.n
        return np.flatnonzero(self.fc.fitted[:n] & (self.fc.f_day[:n] == self.day_idx()))

    def hour_now(self) -> float:
        return self.now.hour + self.now.minute / 60 + self.now.second / 3600

    def window_units(self, rows: np.ndarray, start: datetime, end: datetime) -> np.ndarray:
        """No-offer forecast units between two sim times, spread over the day with the hour profile."""
        total = np.zeros(len(rows))
        plat = self.reg.s_plat[rows]
        d = start.date()
        while d <= end.date() and len(rows):
            midnight = datetime.combine(d, datetime.min.time())
            h0 = max((start - midnight).total_seconds() / 3600, 0.0)
            h1 = min((end - midnight).total_seconds() / 3600, 24.0)
            if h1 > h0:
                total += self.fc.base(rows, self.fc.day_index(d)) * self.profile.share_between(plat, h0, h1)
            d += timedelta(days=1)
        return total

    def _sd(self, rows: np.ndarray, f: np.ndarray) -> np.ndarray:
        return np.sqrt(self.fc.st.var_r[rows] * (f + 1.0))

    def _confidence(self, rows: np.ndarray, strength: np.ndarray) -> np.ndarray:
        """Forecast quality for the series times how decisive the evidence is."""
        acc = np.clip(1 - self.fc.series_wape(rows), 0.1, 0.95)
        return np.clip(np.sqrt(acc * np.clip(strength, 0, 1)), 0.05, 0.99)

    # ------------------------------------------------------------------ decisions

    def _nowcast_check(self, budget: dict[str, int]) -> None:
        rows = self._rows_today()
        if not len(rows):
            return
        reg, fc = self.reg, self.fc
        hour = self.hour_now()
        cs = self.profile.cum_share(reg.s_plat[rows], hour)
        f = fc.f_next[rows]
        exp = f * cs
        act = reg.t_units[rows]
        z, ratio = self._z(act, exp, np.zeros((len(rows), 0)))
        A, F = act + fc.y_lag[rows].sum(1), exp + fc.f_lag[rows].sum(1)
        z3, ratio3 = self._z(A, F, fc.f_lag[rows], exp)
        self.pipeline["forecasts_updated"] += len(rows)
        no_offer = np.array([i not in self.book.by_series for i in rows], dtype=bool)
        rest = f * (1 - cs)
        multi = (z3 <= Z_BELOW) & (ratio3 <= 0.8) & (F >= 6) & (F > exp)
        today = (z <= Z_BELOW) & (ratio <= 0.8) & (exp >= 6)
        below = (multi | today) & (cs >= 0.3) & no_offer & (reg.stock[rows] > rest) & ~reg.t_out[rows]
        above = (z >= Z_SURGE) & (ratio >= 1.5) & (exp >= 4) & (act - exp >= 6) & (cs >= 0.25)
        cands = []
        if below.any():
            m = multi[below]
            hl = hour_label(self.now)
            cands += self._discount_candidates(
                rows[below], np.where(m, A[below], act[below]), np.where(m, F[below], exp[below]),
                np.where(m, z3[below], z[below]),
                [f"over the last 2 days and today by {hl}" if k else f"by {hl}" for k in m])
        if above.any():
            cands += [c for c in self._surge_candidates(rows[above], act[above], exp[above], z[above], cs[above])
                      if c["score"] >= MIN_SURGE_VALUE]
        if hour >= 17:
            per = reg.perishable[rows] & no_offer & (reg.stock[rows] > 0)
            if per.any():
                cands += self._clearance_candidates(rows[per], rest[per])
        self._select(cands, budget)

    def _z(self, actual, expected, lag_f, today=None) -> tuple[np.ndarray, np.ndarray]:
        """Deviation in standard deviations under var = a f + b f^2 (b per day, days independent)."""
        a, b = self.fc.dispersion()
        today = expected if today is None else today
        var = a * (expected + 0.5) + b * (today ** 2 + (lag_f ** 2).sum(axis=1))
        return (actual - expected) / np.sqrt(var), actual / np.maximum(expected, 1e-9)

    def persistence(self, strong: bool = False) -> float:
        """Share of a flagged drop that still shows over the next three days, measured on flagged products that
        took no offer, separately for weak and strong evidence (shrunk toward 0.5). Most weakly flagged dips
        are noise that reverts, so this keeps the counterfactual of a drop offer honest."""
        obs = [o for o in self.drop_obs if o[2] == strong]
        w = sum(o[1] for o in obs)
        v = sum(o[0] * o[1] for o in obs)
        return float(np.clip((PERSIST_PRIOR * PERSIST_STRENGTH + v) / (PERSIST_STRENGTH + w), 0.0, 1.0))

    def _watch_drops(self, rows: np.ndarray, dev: np.ndarray, strong: np.ndarray) -> None:
        day = self.day_idx()
        H = self.fc.horizon(rows, day + 1, 3)
        for j, i in enumerate(rows.tolist()):
            if (i, day) not in self.drop_watch and dev[j] > 0:
                self.drop_watch[(i, day)] = (float(dev[j]), H[j], bool(strong[j]))

    def _score_drops(self) -> None:
        """At a day close: flagged drops whose three following days are now closed."""
        closed = self.day_idx() - 1
        offered = {o.series for o in self.book.history if o.until > datetime.combine(self.fc.day(closed - 3), datetime.min.time())}
        for (i, day), (dev, f, strong) in list(self.drop_watch.items()):
            if closed < day + 3:
                continue
            del self.drop_watch[(i, day)]
            span = slice(day + 1, day + 4)
            if i in offered or self.fc.cens[i, span].any() or f.sum() < 3:
                continue
            actual = float(self.fc.hist[i, span].sum())
            obs = float(np.clip((1 - actual / f.sum()) / dev, -1.0, 1.5))
            self.drop_obs.append((obs, min(f.sum() / 20, 1.0), strong))
        self.drop_obs = self.drop_obs[-2000:]

    def _keep_value(self, rows: np.ndarray) -> np.ndarray:
        """Rupees of future margin per recovered unit: the kept customer's baskets over KEEP_DAYS, at KEEP_PROB."""
        reg = self.reg
        m = reg.s_merch[rows]
        basket = reg.m_rev_total[m] / np.maximum(reg.m_visits[m], 1)
        margin = np.array([self._merchant_margin(int(k)) for k in m])
        visits = KEEP_DAYS / np.maximum(np.where(reg.m_interval[m] > 0, reg.m_interval[m], 7.0), 1.0)
        per_visit = np.where(reg.t_orders[rows] > 0, reg.t_units[rows] / np.maximum(reg.t_orders[rows], 1), 1.3)
        return basket * margin * visits * KEEP_PROB / np.maximum(per_visit, 1.0)

    def _merchant_margin(self, m: int) -> float:
        rows = self.reg.series_of[m]
        if not rows:
            return 0.0
        p, c = self.reg.price[rows], self.reg.cost[rows]
        return float(np.mean((p - c) / np.maximum(p, 1e-9)))

    def _uplift_bounds(self, rows, cf, lift, d, days, dev) -> tuple[np.ndarray, np.ndarray]:
        """Expected uplift and its 80% lower bound: demand noise, forecast error, elasticity uncertainty and,
        after a drop, how long it lasts."""
        a, b = self.fc.dispersion()
        mu = cf * lift
        e_sd = np.array([self.book.elasticity.sd(self.reg.merchants[self.reg.s_merch[i]]["id"], self.reg.category[i])
                         for i in rows])
        var = (a * (mu + cf) + b * cf ** 2 * (0.5 + 0.5 / np.maximum(days, 1))
               + (mu * np.log(1 - d) * e_sd) ** 2 + (cf * dev * DROP_PERSIST_SD) ** 2)
        up = cf * (lift - 1)
        return up, up - Z_ASSURED * np.sqrt(var)

    def _discount_candidates(self, rows, act, exp, z, when: list[str], mode: str = "drop") -> list[dict]:
        """mode drop, overstock or price (customers here answer price strongly enough that an offer adds profit).

        The profit-maximising discount when one pays; after a drop, otherwise the volume-defence offer (smallest
        discount that wins back the lost units, paid for by the customers kept). The no-offer counterfactual
        after a drop keeps only the share of drops that historically lasted (learned). Only offers whose 80%
        lower bound of uplift is positive go out, so every promise can be measured."""
        reg, fc = self.reg, self.fc
        n = len(rows)
        dev = np.clip(1 - act / np.maximum(exp, 1e-9), 0, 0.9) if mode == "drop" else np.zeros(n)
        strong = z <= Z_STRONG
        if mode == "drop":
            self._watch_drops(rows, dev, strong)
        cf_factor = 1 - np.where(strong, self.persistence(True), self.persistence(False)) * dev
        mean = fc.horizon(rows, self.day_idx(), 14).mean(axis=1)
        e = fc.elast[rows]
        per = reg.perishable[rows]
        price, cost, stock = reg.price[rows], reg.cost[rows], reg.stock[rows]
        floor = margin_floor(price, cost)
        cf = self.window_units(rows, self.now, self.now + timedelta(hours=OFFER_HOURS)) * cf_factor
        best = choose_discount(price, cost, stock, cf, mean * reg.lead[rows],
                               mean * np.where(per, 2, reg.lead[rows] + CYCLE_DAYS), e,
                               np.where(per, SPOIL_COST, EXCESS_COST), floor)
        profit = (best["discount"] > 0) & (best["gain"] >= MIN_GAIN)
        if mode in ("price", "overstock"):   # no problem to fix: does any discount beat full price?
            self._note_price_checks(rows, profit, (price - cost) / np.maximum(price, 1e-9), e)
        full_d = self.window_units(rows, self.now, self.now + timedelta(hours=DEFEND_HOURS))
        cf_d = full_d * cf_factor
        keep = self._keep_value(rows)
        dfn = choose_defence(price, cost, stock, cf_d, full_d - cf_d, e, floor, keep)
        defend = ~profit & (mode == "drop") & (dfn["discount"] > 0) & (dev > 0)
        conf = self._confidence(rows, 1 - np.exp(-np.abs(z) / 3))
        out = []
        for j in np.flatnonzero(profit | defend):
            i = int(rows[j])
            plan = best if profit[j] else dfn
            hrs = OFFER_HOURS if profit[j] else DEFEND_HOURS
            base = cf[j] if profit[j] else cf_d[j]
            d = float(plan["discount"][j])
            up, assured = self._uplift_bounds(np.array([i]), np.array([base]), np.array([(1 - d) ** e[j]]), d,
                                              hrs / 24, np.array([dev[j]]))
            up, assured = min(float(up[0]), max(stock[j] - base, 0)), float(assured[0])
            if assured < 0.5:  # a measurable promise: at least one extra unit (rounded) at the 80% bound
                continue
            name, product = self._names(i)
            cover = stock[j] / max(mean[j], 1e-6)
            sd = abs(act[j] - exp[j]) / max(abs(z[j]), 1e-6)
            lo, hi = max(exp[j] - 1.2816 * sd, 0), exp[j] + 1.2816 * sd
            pd = format_inr(plan["price"][j])
            note = f"Elasticity {e[j]:.1f} ({self._elasticity_note(i)})."
            promise = f"+{up:.0f} units expected, at least +{assured:.0f} (80% bound)"
            if mode == "overstock":
                title = f"{product} is overstocked at {name}: {d:.0%} off for {hrs}h"
                why = f"{cover:.0f} days of cover: stock {stock[j]:.0f} units, selling ~{mean[j]:.1f} a day. "
            elif mode == "price":
                title = f"{d:.0%} off {product} at {name} for {hrs}h adds {format_inr(plan['gain'][j])} profit"
                why = (f"Shoppers here answer price strongly (elasticity {e[j]:.1f}) and the margin is "
                       f"{(price[j] - cost[j]) / price[j]:.0%}, so a short offer sells enough extra to pay for itself. ")
            elif profit[j]:
                title = f"Sales of {product} are {dev[j]:.0%} below forecast at {name}: {d:.0%} off for {hrs}h"
                why = f"{act[j]:.0f} sold {when[j]} vs {exp[j]:.0f} expected (80% range {lo:.0f}-{hi:.0f}). "
            else:
                title = f"{product} sales down {dev[j]:.0%} at {name}: {d:.0%} off for {hrs}h wins back volume"
                why = (f"Sales down {dev[j]:.0%} ({act[j]:.0f} sold {when[j]} vs {exp[j]:.0f} forecast, 80% range "
                       f"{lo:.0f}-{hi:.0f}), likely a competitor's price cut or a local dip. ")
            if profit[j]:
                detail = (why + f"{d:.0%} off ({pd}) for {hrs}h: {promise}, {format_inr(plan['revenue'][j])} revenue, "
                          f"margin {plan['margin'][j]:.0%} (floor {floor[j]:.0%}), {format_inr(plan['gain'][j])} more "
                          f"profit than no offer. {note}")
            else:
                detail = (why + f"{d:.0%} off ({pd}) for {hrs}h wins back ~{up / (hrs / 24):.0f} units a day "
                          f"({plan['recovered_share'][j]:.0%} of the lost volume; {promise}); margin stays at "
                          f"{plan['margin'][j]:.0%} (floor {floor[j]:.0%}). It gives away {format_inr(plan['margin_cost'][j])} "
                          f"of margin on units that sell anyway, repaid by the customers it keeps: half of their next "
                          f"{KEEP_DAYS} days of baskets is worth {format_inr(up * keep[j])}. {note}")
            gain = float(plan["gain"][j])
            urgency = "high" if gain >= 5000 or z[j] <= -4 else "medium" if gain >= 1000 else "low"
            out.append({"kind": "discount", "i": i, "score": gain, "urgency": urgency, "title": title,
                        "detail": detail, "confidence": float(conf[j]),
                        "action": {"type": "discount", "discount": d, "hours": hrs},
                        "expected": self._expected(base + up, (base + up) * plan["price"][j], plan["margin"][j], up),
                        "evidence": self._evidence(i, act[j], exp[j], dev[j], cover, e[j]),
                        "plan": {"cf_factor": float(cf_factor[j]), "dev": float(dev[j])}})
        return out

    def _surge_candidates(self, rows, act, exp, z, cs, when: str = "") -> list[dict]:
        reg, fc = self.reg, self.fc
        mean = fc.horizon(rows, self.day_idx(), 7).mean(axis=1)
        pace = np.maximum(act / np.maximum(cs, 0.05), mean)
        cover = reg.stock[rows] / np.maximum(pace, 1e-6)
        rate = (pace + mean) / 2  # order for a lasting part of the surge, not a whole lead time at today's peak
        plan = restock_plan(reg.stock[rows], rate, self._sd(rows, rate), reg.lead[rows], reg.perishable[rows])
        plan["cover"], plan["need"] = cover, cover < plan["threshold"]
        conf = self._confidence(rows, 1 - np.exp(-np.abs(z) / 3))
        out = []
        for j in range(len(rows)):
            i = int(rows[j])
            name, product = self._names(i)
            up = act[j] / max(exp[j], 1e-9) - 1
            extra = (pace[j] - fc.f_next[i])
            protected = max(extra, 0) * (reg.price[i] - reg.cost[i])
            sid = self.book.by_series.get(i)
            hold = (f"Pause the running {self.book.active[sid].discount:.0%} offer and hold price"
                    if sid else "Hold price, no discounts")
            restock = ""
            action = {"type": "hold_price", "hours": 24}
            if plan["need"][j]:
                restock = f" At this pace stock lasts {plan['cover'][j]:.1f} days (lead time {reg.lead[i]:.0f}): restock {plan['units'][j]:.0f} units."
                action["units"] = int(plan["units"][j])
            detail = (f"{act[j]:.0f} sold {when or 'by ' + hour_label(self.now)} vs {exp[j]:.0f} expected. {hold}: "
                      f"the extra ~{max(extra, 0):.0f} units keep {format_inr(protected)} of full margin.{restock}")
            out.append({"kind": "surge", "i": i, "score": float(protected),
                        "urgency": "high" if plan["cover"][j] < reg.lead[i] else "medium",
                        "title": f"{product} is selling {_above(up)} at {name}: hold price",
                        "detail": detail, "confidence": float(conf[j]), "action": action,
                        "expected": self._expected(pace[j], pace[j] * reg.price[i],
                                                   (reg.price[i] - reg.cost[i]) / reg.price[i], max(extra, 0)),
                        "evidence": self._evidence(i, act[j], exp[j], -up, plan["cover"][j], fc.elast[i])})
        return out

    def _clearance_candidates(self, rows, rest_today) -> list[dict]:
        reg, fc = self.reg, self.fc
        tomorrow = fc.base(rows, self.day_idx() + 1)
        window = rest_today + tomorrow
        excess_stock = reg.stock[rows] - window - fc.base(rows, self.day_idx() + 2)
        keep = excess_stock >= np.maximum(5, 0.25 * reg.stock[rows])
        rows, window, rest_today = rows[keep], window[keep], rest_today[keep]
        if not len(rows):
            return []
        e = fc.elast[rows]
        best = choose_discount(reg.price[rows], reg.cost[rows], reg.stock[rows], window, np.zeros(len(rows)),
                               fc.base(rows, self.day_idx() + 2), e, np.full(len(rows), SPOIL_COST),
                               np.zeros(len(rows)))
        conf = self._confidence(rows, np.full(len(rows), 0.8))
        out = []
        for j in np.flatnonzero((best["discount"] > 0) & (best["gain"] >= MIN_GAIN / 2)):
            i = int(rows[j])
            name, product = self._names(i)
            d = float(best["discount"][j])
            up, assured = self._uplift_bounds(np.array([i]), window[j:j + 1], np.array([(1 - d) ** e[j]]), d,
                                              CLEARANCE_HOURS / 24, np.zeros(1))
            if assured[0] <= 0:
                continue
            spare = reg.stock[i] - window[j]
            detail = (f"{reg.stock[i]:.0f} units in stock, ~{window[j]:.0f} expected to sell before it spoils. "
                      f"{d:.0%} off for {CLEARANCE_HOURS}h ({format_inr(best['price'][j])}) sells ~{best['units'][j]:.0f} "
                      f"(at least +{assured[0]:.0f} more than without, 80% bound), saving {format_inr(best['gain'][j])} "
                      f"vs writing off ~{spare:.0f} units; never below cost.")
            out.append({"kind": "clearance", "i": i, "score": float(best["gain"][j]),
                        "urgency": "high" if spare > 0.5 * reg.stock[i] else "medium",
                        "title": f"Clear {product} at {name} before it spoils: {d:.0%} off for {CLEARANCE_HOURS}h",
                        "detail": detail, "confidence": float(conf[j]),
                        "action": {"type": "discount", "discount": d, "hours": CLEARANCE_HOURS},
                        "expected": self._expected(best["units"][j], best["revenue"][j], best["margin"][j], best["uplift"][j]),
                        "evidence": self._evidence(i, reg.t_units[i], fc.f_next[i], 0.0,
                                                   reg.stock[i] / max(tomorrow[keep][j], 1e-6), e[j])})
        return out

    def _day_decisions(self, scored: dict) -> None:
        reg, fc = self.reg, self.fc
        n = reg.n
        rows = np.flatnonzero(fc.fitted[:n] & (fc.f_day[:n] == self.day_idx()))
        if not len(rows):
            return
        H = fc.horizon(rows, self.day_idx(), 14)
        mean = H[:, :7].mean(axis=1)
        cands: list[dict] = []
        # restock
        plan = restock_plan(reg.stock[rows], mean, self._sd(rows, mean), reg.lead[rows], reg.perishable[rows])
        need = plan["need"] & (mean >= 0.3)
        cands += self._restock_candidates(rows[need], mean[need], {k: v[need] for k, v in plan.items()})
        # yesterday below the interval, or overstock: discount
        r, y, f, lo = scored["rows"], scored["y"], scored["f"], scored["lo"]
        if len(r):
            ok = fc.f_day[r] == self.day_idx()
            A, F = fc.y_lag[r].sum(1), fc.f_lag[r].sum(1)
            z3, ratio3 = self._z(A, F, fc.f_lag[r, 1:], fc.f_lag[r, 0])
            low = ok & (y < lo) & (z3 <= Z_BELOW) & (ratio3 <= 0.8) & (F >= 6)
            low &= np.array([i not in self.book.by_series for i in r], dtype=bool) & (reg.stock[r] > 0)
            low &= ~fc.cens[r, self.day_idx() - 1]
            if low.any():
                cands += self._discount_candidates(r[low], A[low], F[low], z3[low], ["over the last 2 days"] * int(low.sum()))
            z, ratio = self._z(y, f, np.zeros((len(r), 0)))
            high = ok & (z >= Z_SURGE) & (ratio >= 1.5) & (y - f >= 6) & (f >= 4)
            if high.any():
                cands += self._surge_candidates(r[high], y[high], f[high], z[high], np.ones(high.sum()), "yesterday")
        hm = H.mean(axis=1)
        over = (reg.stock[rows] > hm * (reg.lead[rows] + CYCLE_DAYS + 7)) & (hm >= 0.3)  # well beyond the cycle
        over &= reg.stock[rows] - hm * (reg.lead[rows] + CYCLE_DAYS) >= 5
        over &= np.array([i not in self.book.by_series for i in rows], dtype=bool)
        if over.any():
            o = rows[over]
            cands += self._discount_candidates(o, fc.base(o, self.day_idx()), fc.base(o, self.day_idx()),
                                               np.full(len(o), 2.0), [""] * len(o), "overstock")
        # price opportunities: elastic, high-margin products where an offer adds profit without any problem to fix
        idle = rows[np.array([i not in self.book.by_series for i in rows], dtype=bool) & ~reg.perishable[rows]
                    & (reg.stock[rows] > 0) & ~over]
        if len(idle):
            b = fc.base(idle, self.day_idx())
            cands += self._discount_candidates(idle, b, b, np.full(len(idle), 3.0), [""] * len(idle), "price")
        cands += self._festival_candidates(rows, H)
        cands += self._winback_candidates()
        self._closing = True  # day-close plans answer the whole day's data, not one order: no order-to-signal latency
        try:
            self._select(cands, dict(CLOSE_BUDGET))
        finally:
            self._closing = False

    def _restock_candidates(self, rows, mean, plan) -> list[dict]:
        reg = self.reg
        out = []
        conf = self._confidence(rows, 1 - plan["cover"] / np.maximum(plan["threshold"], 1e-9) + 0.3)
        for j in range(len(rows)):
            i = int(rows[j])
            name, product = self._names(i)
            lead, cover = reg.lead[i], plan["cover"][j]
            at_risk = max(mean[j] * lead - reg.stock[i], 0)
            units = int(plan["units"][j])
            detail = (f"Stock {reg.stock[i]:.0f} units, forecast {mean[j]:.1f} a day. Order {units} units "
                      f"({format_inr(units * reg.cost[i])} at cost) to cover the {lead:.0f}-day lead time, a week of "
                      f"sales and {plan['safety'][j]:.0f} units of safety stock."
                      + (f" Without it ~{at_risk:.0f} units ({format_inr(at_risk * reg.price[i])}) of sales are lost before "
                         f"the delivery." if at_risk >= 1 else ""))
            out.append({"kind": "restock", "i": i, "score": float(at_risk * reg.price[i] + units * 0.01),
                        "urgency": "high" if cover < lead else "medium" if cover < plan["threshold"][j] else "low",
                        "title": f"Restock {product} at {name}: {cover:.1f} days of cover, lead time {lead:.0f} days",
                        "detail": detail, "confidence": float(conf[j]),
                        "action": {"type": "restock", "units": units},
                        "expected": self._expected(units, max(at_risk, 0) * reg.price[i],
                                                   (reg.price[i] - reg.cost[i]) / reg.price[i], at_risk),
                        "evidence": self._evidence(i, reg.stock[i], mean[j], 0.0, cover, self.fc.elast[i])})
        return out

    def _festival_candidates(self, rows: np.ndarray, H: np.ndarray) -> list[dict]:
        nxt = C.next_festival(self.today)
        if not nxt:
            return []
        fday, fname = nxt
        days_to = (fday - self.today).days
        if days_to > FESTIVAL_DAYS:
            return []
        reg, fc = self.reg, self.fc
        lift, seen = fc.festival_lift(fname)
        mult = lift[fc.cat[rows]]
        learned = (seen[fc.cat[rows]] > 0) & (mult >= FESTIVAL_MIN_LIFT)
        if not learned.any():
            return []
        rows, H, mult = rows[learned], H[learned], mult[learned]
        k = min(days_to + 1, H.shape[1])
        demand = H[:, :k].sum(axis=1)
        mean = H.mean(axis=1)
        sd = self._sd(rows, demand)
        stock = reg.stock[rows]
        short = demand + 1.65 * sd - stock
        out = []
        cat_names = fc.categories
        conf = self._confidence(rows, np.clip(mult - 1, 0, 1) + 0.3)
        for j in np.flatnonzero(short > np.maximum(3, 0.1 * demand)):
            i = int(rows[j])
            name, product = self._names(i)
            units = int(math.ceil(short[j] / 5) * 5)
            late = reg.lead[i] >= days_to
            detail = (f"Learned festival lift for {cat_names[fc.cat[i]]}: x{mult[j]:.2f} (from "
                      f"{fname} last year, {seen[fc.cat[i]]} category-day(s) of data). Forecast {demand[j]:.0f} units "
                      f"through {fname} vs {stock[j]:.0f} in stock: order {units} units now "
                      f"({format_inr(units * reg.cost[i])} at cost, {format_inr(short[j] * reg.price[i])} of festival sales)."
                      + (f" Lead time is {reg.lead[i]:.0f} days, so expedite." if late else ""))
            out.append({"kind": "festival", "i": i, "score": float(short[j] * (reg.price[i] - reg.cost[i])),
                        "urgency": "high" if late or days_to <= 3 else "medium",
                        "title": f"{fname} in {days_to} days: stock up {product} at {name}",
                        "detail": detail, "confidence": float(conf[j]),
                        "action": {"type": "restock", "units": units},
                        "expected": self._expected(demand[j], demand[j] * reg.price[i],
                                                   (reg.price[i] - reg.cost[i]) / reg.price[i], short[j]),
                        "evidence": self._evidence(i, stock[j], demand[j], 0.0, stock[j] / max(mean[j], 1e-6), fc.elast[i])})
        # plenty of stock even for the spike: a pre-festival offer turns the surplus into festival sales
        surplus = (stock > 1.5 * demand + mean * 14) & (days_to >= 1)
        surplus &= np.array([i not in self.book.by_series for i in rows], dtype=bool)
        if surplus.any():
            s = rows[surplus]
            end = datetime.combine(fday, datetime.min.time()) + timedelta(days=1)
            window = self.window_units(s, self.now, min(end, self.now + timedelta(hours=72)))
            hours_ = int(min((end - self.now).total_seconds() / 3600, 72))
            best = choose_discount(reg.price[s], reg.cost[s], reg.stock[s], window, mean[surplus] * reg.lead[s],
                                   mean[surplus] * (reg.lead[s] + CYCLE_DAYS), fc.elast[s], np.where(reg.perishable[s], SPOIL_COST, EXCESS_COST),
                                   margin_floor(reg.price[s], reg.cost[s]))
            for j in np.flatnonzero((best["discount"] > 0) & (best["gain"] >= MIN_GAIN)):
                i = int(s[j])
                name, product = self._names(i)
                d = float(best["discount"][j])
                _, assured = self._uplift_bounds(np.array([i]), window[j:j + 1], np.array([(1 - d) ** fc.elast[i]]),
                                                 d, hours_ / 24, np.zeros(1))
                if assured[0] <= 0:
                    continue
                detail = (f"Festival lift x{mult[surplus][j]:.2f} learned for {cat_names[fc.cat[i]]}, but "
                          f"{reg.stock[i]:.0f} in stock is well above the {window[j]:.0f} units expected before {fname}. "
                          f"{d:.0%} off for {hours_}h: +{best['uplift'][j]:.0f} units, margin {best['margin'][j]:.0%}, "
                          f"{format_inr(best['gain'][j])} better than no offer.")
                out.append({"kind": "festival", "i": i, "score": float(best["gain"][j]), "urgency": "medium",
                            "title": f"{fname} in {days_to} days: pre-festival offer on {product} at {name}",
                            "detail": detail, "confidence": float(conf[surplus][j]),
                            "action": {"type": "discount", "discount": d, "hours": hours_},
                            "expected": self._expected(best["units"][j], best["revenue"][j], best["margin"][j],
                                                       best["uplift"][j]),
                            "evidence": self._evidence(i, reg.stock[i], window[j], 0.0,
                                                       reg.stock[i] / max(mean[surplus][j], 1e-6), fc.elast[i])})
        return out

    def _winback_candidates(self) -> list[dict]:
        reg = self.reg
        today = self.day_idx()
        out = []
        for m in range(reg.nm):
            count, interval = reg.lapsed(m, today)
            if count < WINBACK_MIN or reg.m_orders_total[m] == 0:
                continue
            aov = reg.m_rev_total[m] / reg.m_orders_total[m]
            coupon = int(min(max(round(aov * 0.1 / 10) * 10, 25), 300))
            back = count * WINBACK_RESPONSE
            info = reg.merchants[m]
            out.append({"kind": "winback", "m": m, "score": float(back * aov),
                        "urgency": "medium" if count >= 25 else "low",
                        "title": f"{count} regular customers have not come back to {info['name']}: send a {format_inr(coupon)} coupon",
                        "detail": (f"They usually buy every ~{interval:.0f} days and have been away more than twice as long. "
                                   f"At a {WINBACK_RESPONSE:.0%} response rate, expect ~{back:.0f} returning orders, "
                                   f"{format_inr(back * aov)} revenue for {format_inr(back * coupon)} of coupons."),
                        "confidence": float(np.clip(0.35 + 0.01 * count, 0.35, 0.7)),
                        "action": {"type": "coupon", "coupon": coupon, "customers": count, "hours": 72},
                        "expected": {"units": round(float(back), 1), "revenue": round(float(back * aov), 2),
                                     "margin": None, "uplift_units": round(back, 1)},
                        "evidence": {"actual": count, "forecast": None, "deviation": None, "stock": None,
                                     "cover_days": None, "elasticity": None, "usual_interval_days": round(interval, 1)}})
        return out

    def _select(self, cands: list[dict], budget: dict[str, int]) -> None:
        """Best first within each kind; one signal per merchant per round; cooldowns; a budget per kind (spent in place)."""
        used: set[int] = set()
        now_h = hours(self.now)
        for c in sorted(cands, key=lambda c: -c["score"]):
            m = c["m"] if "m" in c else int(self.reg.s_merch[c["i"]])
            key = (self.reg.merchants[m]["id"], self.reg.sku[c["i"]] if "i" in c else "*", c["kind"])
            if (m in used or budget.get(c["kind"], 0) <= 0 or not self.cool.ready(key, now_h)
                    or self.day_counts.get(c["kind"], 0) >= DAILY_CAP.get(c["kind"], 10 ** 9)):
                continue
            self.day_counts[c["kind"]] = self.day_counts.get(c["kind"], 0) + 1
            self.cool.hit(key, now_h, c["action"].get("hours") if c["kind"] in ("discount", "clearance") else None)
            used.add(m)
            budget[c["kind"]] -= 1
            self._queue(c, m)

    # ------------------------------------------------------------------ signals

    def _names(self, i: int) -> tuple[str, str]:
        return self.reg.merchants[self.reg.s_merch[i]]["name"], self.reg.name[i]

    def _elasticity_note(self, i: int) -> str:
        cat = self.reg.category[i]
        n = int(self.book.elasticity.cat.get(cat, [0, 0, 0])[2])
        return f"{cat}, learned from {n} offer{'s' if n != 1 else ''}" if n else f"{cat} prior, no offers measured yet"

    @staticmethod
    def _expected(units, revenue, margin, uplift) -> dict:
        return {"units": round(float(units), 1), "revenue": round(float(revenue), 2),
                "margin": None if margin is None else round(float(margin), 4), "uplift_units": round(float(uplift), 1)}

    @staticmethod
    def _evidence(i, actual, forecast, deviation, cover, elasticity) -> dict:
        return {"actual": round(float(actual), 1), "forecast": round(float(forecast), 1),
                "deviation": round(float(deviation), 4), "stock": None, "cover_days": round(float(min(cover, 999)), 1),
                "elasticity": round(float(elasticity), 3)}

    def _queue(self, c: dict, m: int) -> None:
        reg = self.reg
        info = reg.merchants[m]
        i = c.get("i")
        self.signal_seq += 1
        auto = (c["kind"] in AUTO_KINDS and c["action"]["type"] == "discount"
                and autopilot_merchant(info["id"], self.autopilot))
        evidence = c["evidence"]
        if i is not None:
            evidence["stock"] = round(float(reg.stock[i]), 1)
        sig = {"id": f"sig-{self.signal_seq:07d}", "sent_ms": 0, "sim_ts": self.now.isoformat(timespec="seconds"),
               "platform": info["platform"], "merchant": info["id"], "merchant_name": info["name"],
               "city": info["city"], "sku": reg.sku[i] if i is not None else None,
               "product": reg.name[i] if i is not None else None,
               "category": reg.category[i] if i is not None else info["category"],
               "kind": c["kind"], "urgency": c["urgency"], "title": c["title"], "detail": c["detail"],
               "action": c["action"], "expected": c["expected"], "confidence": round(c["confidence"], 3),
               "evidence": evidence, "latency_ms": 0, "autopilot": auto}
        # Latency is measured from the newest data the decision saw: this merchant's latest orders, or the
        # clock tick that moved "expected by now" (a drop is noticed from orders that did not come).
        if c.get("plan"):
            self.plans[sig["id"]] = c["plan"]
        trigger = 0 if self._closing else max(int(reg.m_last_sent[m]), int(self.last_clock_sent or 0))
        self.outbox.append((sig, trigger))

    def _flush(self, processed_ms: float) -> None:
        for sig, trigger in self.outbox:
            now = time.time() * 1000
            sig["sent_ms"] = int(now)
            sig["latency_ms"] = int(now - trigger) if trigger else None
            self.publish_live(C.signal_topic(sig["platform"], sig["merchant"]), sig, 1)
            self.lat["signal"].append((now / 1000, now - processed_ms))
            if trigger:
                self.lat["end_to_end"].append((now / 1000, now - trigger))
            self.signals.append(sig)
            self.signal_by_id[sig["id"]] = sig
            self.pipeline["signals_raised"] += 1
            self.signals_by_kind[sig["kind"]] = self.signals_by_kind.get(sig["kind"], 0) + 1
            key = (self.reg.merchant_index[sig["merchant"]], sig["kind"], sig["category"] or "")
            self.sig_count[key] = self.sig_count.get(key, 0) + 1
            if sig["autopilot"]:
                self.publish_control(self.apply_command(sig, "autopilot"))
        self.outbox = []
        if len(self.signal_by_id) > 5000:
            keep = {s["id"] for s in self.signals}
            self.signal_by_id = {k: v for k, v in self.signal_by_id.items() if k in keep}
        for out in self.outcome_box:
            self.publish_live(C.outcome_topic(out["platform"], out["merchant"]), out, 1)
        self.outcome_box = []

    def apply_command(self, sig: dict, by: str) -> dict:
        until = datetime.fromisoformat(sig["sim_ts"]) if self.now is None else self.now
        until += timedelta(hours=sig["action"].get("hours", OFFER_HOURS))
        return {"cmd": "apply_offer", "signal_id": sig["id"], "platform": sig["platform"], "merchant": sig["merchant"],
                "sku": sig["sku"], "discount": sig["action"]["discount"], "until": until.isoformat(timespec="seconds"),
                "by": by}

    def _register_offer(self, msg: dict) -> None:
        reg = self.reg
        m = reg.merchant_index.get(msg.get("merchant", ""))
        i = reg.series_index.get((m, msg.get("sku"))) if m is not None else None
        sid = msg.get("signal_id")
        if i is None or not sid or sid in self.book.active or self.now is None:
            return
        until = datetime.fromisoformat(msg["until"])
        d = float(msg.get("discount", 0))
        cat = reg.category[i]
        e = self.book.elasticity.value(reg.merchants[m]["id"], cat)
        plan = self.plans.get(sid, {})
        base, up, assured = 0.0, 0.0, 0.0
        if self.fc.fitted[i]:   # the no-offer counterfactual: forecast, minus the part of a drop that lasts
            base = float(self.window_units(np.array([i]), self.now, until)[0]) * plan.get("cf_factor", 1.0)
            days = max((until - self.now).total_seconds() / 86400, 0.1)
            u, a = self._uplift_bounds(np.array([i]), np.array([base]), np.array([(1 - d) ** e]), d, days,
                                       np.array([plan.get("dev", 0.0)]))
            up, assured = min(float(u[0]), max(reg.stock[i] - base, 0)), float(a[0])
        self.book.apply(Offer(signal_id=sid, series=i, platform=reg.merchants[m]["platform"],
                              merchant=reg.merchants[m]["id"], merchant_name=reg.merchants[m]["name"],
                              sku=reg.sku[i], product=reg.name[i], category=cat, discount=d, start=self.now,
                              until=until, by=msg.get("by", "merchant"), list_price=float(reg.price[i]),
                              baseline_units=base, expected_uplift=up, elasticity_before=e,
                              assured_uplift=max(assured, 0.0),
                              kind=self.signal_by_id.get(sid, {}).get("kind", "discount"),
                              applied_ms=int(time.time() * 1000)))
        self.applied_signals.add(sid)
        self._ledger(m)[0] += 1

    # ------------------------------------------------------------------ command-centre bookkeeping

    def _ledger(self, m: int) -> list:
        return self.ledger.setdefault(m, [0, 0.0, 0, 0])

    def _log_outcome(self, o: Offer, out: dict) -> None:
        """Per merchant totals for HQ, and a compact record the lessons are learned from."""
        m = self.reg.merchant_index.get(o.merchant)
        if m is None:
            return
        led = self._ledger(m)
        led[1] += out["revenue_gain"]
        judged = not o.stockout or out["worked"]
        led[2] += int(judged and out["worked"])
        led[3] += int(judged)
        hrs = max((o.until - o.start).total_seconds() / 3600, 0.0)
        # (merchant, category, discount, hours, baseline units, actual units, worked, judged, censored, rupees gained)
        self.outcome_log.append((m, o.category, o.discount, hrs, max(o.baseline_units, 0.0), o.units,
                                 bool(out["worked"]), bool(judged), bool(o.stockout), float(out["revenue_gain"])))

    def _note_price_checks(self, rows: np.ndarray, offered: np.ndarray, margin: np.ndarray, e: np.ndarray) -> None:
        """The daily price check on idle products: where no discount beats full price, the brain holds price."""
        if not len(rows):
            return
        k = len(self.fc.categories)
        keys, inv = np.unique(self.reg.s_merch[rows] * k + self.fc.cat[rows], return_inverse=True)
        declined = ~np.asarray(offered, bool)
        sums = (np.bincount(inv, minlength=len(keys)), np.bincount(inv, declined, len(keys)),
                np.bincount(inv, np.where(declined, margin, 0.0), len(keys)), np.bincount(inv, e, len(keys)))
        for u, key in enumerate(keys.tolist()):
            rec = self.price_checks.setdefault((key // k, self.fc.categories[key % k]), [0, 0, 0.0, 0.0])
            for f in range(4):
                rec[f] += sums[f][u]

    # ------------------------------------------------------------------ live outputs

    def view(self, key: tuple, compute, every: float = 1.0):
        """A page's read: computed now when the engine is free; while a batch holds it, the latest snapshot at
        once (the first time, waiting up to 10 s). The worker refreshes snapshots pages are still reading
        between batches (``refresh_views``)."""
        self.wanted[key] = time.time()
        self.view_fns[key] = (compute, every)
        hit = self.views.get(key)
        if not self.lock.acquire(blocking=hit is None, timeout=10 if hit is None else -1):
            if hit is not None:
                return hit[1]
            raise TimeoutError("engine busy")
        try:
            value = compute()
            self.views[key] = (time.time(), value)
            return value
        finally:
            self.lock.release()

    def refresh_views(self, now: float) -> None:
        for key in list(self.wanted):
            if now - self.wanted[key] > 15:   # nobody is reading it any more
                for d in (self.wanted, self.views, self.view_fns):
                    d.pop(key, None)
                continue
            compute, every = self.view_fns[key]
            hit = self.views.get(key)
            if hit is None or now - hit[0] >= every:
                try:
                    self.views[key] = (now, compute())
                except Exception:   # e.g. a merchant that a reset removed
                    for d in (self.wanted, self.views, self.view_fns):
                        d.pop(key, None)

    def tick(self) -> None:
        """Wall-clock cadence: stats every 500 ms, ticker every 250 ms, page snapshots about every second."""
        now = time.time()
        with self.lock:
            self.refresh_views(now)
            if now - self.last_ticker_at >= 0.25:
                self.last_ticker_at = now
                if self.ticker:
                    self.publish_live(C.TICKER, self._ticker_payload(), 0)
                self.ticker, self.ticker_seen = [], 0
            if now - self.last_stats_at >= 0.5:
                self.last_stats_at = now
                self.last_stats = self.stats()
                self.publish_live(C.STATS, self.last_stats, 0)

    def _ticker_payload(self) -> dict:
        reg = self.reg
        orders = []
        for m, i, o, amount in self.ticker:
            info = reg.merchants[m]
            orders.append({"platform": info["platform"], "merchant": info["id"], "merchant_name": info["name"],
                           "city": info["city"], "product": reg.name[i] if i is not None else o["sku"],
                           "qty": o["qty"], "amount": round(amount, 2), "offer": bool(o.get("offer"))})
        return {"sent_ms": C.wall_ms(), "orders": orders}

    def _rates(self) -> dict:
        now = time.time()
        while self.window and self.window[0][0] < now - 3:
            self.window.popleft()
        if not self.window or not self.running:   # paused: nothing is flowing, whatever was in flight
            return {"orders": 0.0, "revenue": 0.0, "msgs": 0.0, "capacity": None, "platforms": {}}
        span = max(min(3.0, now - max(self.started, self.run_since)), 0.25)
        orders = sum(w[1] for w in self.window)
        busy = sum(w[4] for w in self.window)
        plat: dict[str, float] = {}
        for w in self.window:
            for k, v in w[5].items():
                plat[k] = plat.get(k, 0) + v
        return {"orders": orders / span, "revenue": sum(w[2] for w in self.window) / span,
                "msgs": sum(w[3] for w in self.window) / span,
                "capacity": round(orders / busy) if busy > 0 and orders >= 100 else None,
                "busy": busy / span, "platforms": {k: v / span for k, v in plat.items()}}

    def stats(self) -> dict:
        r = self._rates()
        nxt = C.next_festival(self.today) if self.today else None
        L = self.lat
        sig_ms, e2e_ms = recent(L["signal"]), recent(L["end_to_end"])
        return {
            "sent_ms": C.wall_ms(), "sim_ts": self.now.isoformat(timespec="seconds") if self.now else None,
            "speed": self.speed, "speed_actual": self.speed_actual, "running": self.running,
            "catching_up": getattr(self, "catching_up", False), "merchants": self.reg.nm, "customers": self.customers_world,
            "series": self.reg.n, "orders_total": self.totals["orders"], "units_total": round(self.totals["units"]),
            "revenue_total": round(self.totals["revenue"], 2), "orders_per_sec": round(r["orders"], 1),
            "revenue_per_sec": round(r["revenue"], 2), "msgs_per_sec": round(r["msgs"], 1),
            "active_customers_today": len(self.active_today), "signals_total": self.pipeline["signals_raised"],
            "signals_by_kind": dict(self.signals_by_kind), "offers_applied": self.book.applied,
            "uplift_revenue": round(self.book.uplift_revenue, 2),
            "platforms": {pid: {"orders_total": t[0], "revenue_total": round(t[1], 2),
                                "orders_per_sec": round(r["platforms"].get(pid, 0.0), 1)}
                          for pid, t in self.platform_totals.items()},
            "latency_ms": {"broker_p50": pct(L["broker"], 50), "broker_p95": pct(L["broker"], 95),
                           "process_p50": pct(L["process"], 50), "process_p95": pct(L["process"], 95),
                           "signal_p50": pct(sig_ms, 50), "signal_p95": pct(sig_ms, 95),
                           "end_to_end_p50": pct(e2e_ms, 50), "end_to_end_p95": pct(e2e_ms, 95)},
            "accuracy": self.acc.summary(self.reg.platform_ids, self.fc.backtest),
            "learning": {"outcomes": self.book.n_outcomes, "hit_rate": self.book.hit_rate(),
                         "judged": self.book.n_judged,
                         "drop_persistence": {"weak": round(self.persistence(False), 3), "strong": round(self.persistence(True), 3),
                                              "n": len(self.drop_obs)},
                         "elasticity": self.book.elasticity.summary(self.fc.categories),
                         "festival_lift": self._festival_lifts()},
            "pipeline": {**self.pipeline, "capacity_orders_per_sec": r["capacity"],
                         "busy_share": round(r.get("busy", 0.0), 3), "active_offers": len(self.book.active)},
            "festival": {"name": nxt[1], "date": nxt[0].isoformat(), "days_to": (nxt[0] - self.today).days} if nxt else None,
            "run_until": self.run_until,
        }

    def _festival_lifts(self) -> dict:
        m, seen = self.fc.festival_lift()
        return {c: round(float(m[k]), 3) for k, c in enumerate(self.fc.categories) if seen[k]}

    # ------------------------------------------------------------------ API views

    def merchants_view(self) -> list[dict]:
        open_by = self._open_signals()
        reg = self.reg
        return [{**{k: info[k] for k in ("id", "platform", "name", "city", "category", "x", "y")},
                 "orders_today": int(reg.m_orders_today[m]), "revenue_today": round(float(reg.m_rev_today[m]), 2),
                 "open_signals": len(open_by.get(info["id"], []))} for m, info in enumerate(reg.merchants)]

    def _open_signals(self) -> dict[str, list[dict]]:
        out: dict[str, list[dict]] = {}
        if self.now is None:
            return out
        for s in self.signals:
            ends = datetime.fromisoformat(s["sim_ts"]) + timedelta(hours=s["action"].get("hours", 24))
            if ends > self.now and s["id"] not in self.applied_signals:
                out.setdefault(s["merchant"], []).append(s)
        return out

    def merchant_view(self, mid: str) -> dict | None:
        reg, fc = self.reg, self.fc
        m = reg.merchant_index.get(mid)
        if m is None:
            return None
        rows = np.array(reg.series_of[m], dtype=int)
        products = []
        if len(rows) and self.today is not None:
            today = self.day_idx()
            mean = np.where(fc.fitted[rows], fc.horizon(rows, today, 7).mean(axis=1), np.nan)
            cs = self.profile.cum_share(reg.s_plat[rows], self.hour_now()) if self.now else np.zeros(len(rows))
            f = np.where(fc.f_day[rows] == today, fc.f_next[rows], np.nan)
            for j, i in enumerate(rows):
                products.append({"sku": reg.sku[i], "name": reg.name[i], "category": reg.category[i],
                                 "price": float(reg.price[i]), "cost": float(reg.cost[i]), "stock": float(reg.stock[i]),
                                 "lead_days": float(reg.lead[i]), "perishable": bool(reg.perishable[i]),
                                 "cover_days": _num(reg.stock[i] / mean[j]) if mean[j] > 0 else None,
                                 "today": {"actual": float(reg.t_units[i]), "expected_so_far": _num(f[j] * cs[j]),
                                           "forecast": _num(f[j])},
                                 "offer": i in self.book.by_series})
        return {**reg.merchants[m], "orders_today": int(reg.m_orders_today[m]),
                "revenue_today": round(float(reg.m_rev_today[m]), 2), "orders_total": int(reg.m_orders_total[m]),
                "revenue_total": round(float(reg.m_rev_total[m]), 2), "autopilot": autopilot_merchant(mid, self.autopilot),
                "products": products, "open_signals": self._open_signals().get(mid, []),
                "outcomes": [o for o in self.book.outcomes if o["merchant"] == mid][-20:]}

    def series_view(self, mid: str, sku: str) -> dict | None:
        reg, fc = self.reg, self.fc
        m = reg.merchant_index.get(mid)
        i = reg.series_index.get((m, sku)) if m is not None else None
        if i is None or self.today is None:
            return None
        today = self.day_idx()
        lo_day = max(today - 42, 0)
        history = [{"day": fc.day(d).isoformat(), "units": float(fc.hist[i, d])} for d in range(lo_day, today)
                   if fc.start[i] >= 0 and d >= fc.start[i]]
        forecast = []
        if fc.fitted[i]:
            r = np.array([i])
            H = fc.horizon(r, today, 14)[0]
            if fc.f_day[i] == today:
                H[0] = fc.f_next[i]
            for h in range(14):
                lo, hi = fc.interval(r, np.array([H[h]]), h + 1)
                forecast.append({"day": fc.day(today + h).isoformat(), "f": round(float(H[h]), 2),
                                 "lo": round(float(lo[0]), 2), "hi": round(float(hi[0]), 2)})
        f_today = float(fc.f_next[i]) if fc.f_day[i] == today else None
        cs = float(self.profile.cum_share(np.array([reg.s_plat[i]]), self.hour_now())[0]) if self.now else 0.0
        return {"sku": sku, "product": reg.name[i], "history": history, "forecast": forecast,
                "today": {"day": self.today.isoformat(), "actual": float(reg.t_units[i]),
                          "expected_so_far": round(f_today * cs, 2) if f_today is not None else None,
                          "forecast": round(f_today, 2) if f_today is not None else None},
                "offers": [{"from": o.start.isoformat(timespec="seconds"), "until": o.until.isoformat(timespec="seconds"),
                            "discount": o.discount} for o in self.book.history if o.series == i],
                "model": fc.describe(i) if fc.fitted[i] else "not fitted yet",
                "wape": _num(fc.series_wape(np.array([i]))[0]) if fc.fitted[i] else None}


def _num(x) -> float | None:
    x = float(x)
    return None if math.isnan(x) or math.isinf(x) else round(x, 2)


# ------------------------------------------------------------------ HTTP API

CONTROL_CMDS = {"start", "pause", "speed", "reset", "apply_offer", "run_days"}
MAX_RUN_DAYS = 60


def create_app(engine: Engine):
    from fastapi import FastAPI, HTTPException

    from secondbrain.commerce import hq

    app = FastAPI(title="SecondBrain Live Commerce engine")

    def snapshot(key: tuple, compute, every: float = 1.0):
        try:
            return engine.view(key, compute, every)
        except TimeoutError:
            raise HTTPException(504, "The engine is busy; try again in a moment")

    def scoped(view, platform: str | None, merchant: str | None, *args, every: float = 1.0):
        """A command-centre view on one branch of the tree (HQ, a platform, a merchant)."""
        try:
            return snapshot((view.__name__, platform, merchant, *args),
                            lambda: view(engine, hq.resolve(engine, platform or None, merchant or None), *args), every)
        except hq.ScopeError as exc:
            raise HTTPException(exc.status, exc.detail)

    @app.get("/platforms")
    def platforms(platform: str | None = None, merchant: str | None = None):
        return scoped(hq.platforms, platform, merchant)

    @app.get("/lessons")
    def lessons(platform: str | None = None, merchant: str | None = None):
        return scoped(hq.lessons, platform, merchant, every=3.0)

    @app.get("/attention")
    def attention(limit: int = 10, platform: str | None = None, merchant: str | None = None):
        return scoped(hq.attention, platform, merchant, max(0, min(limit, 200)))

    @app.get("/tree")
    def tree(platform: str | None = None, merchant: str | None = None):
        return scoped(hq.tree, platform, merchant, every=2.0)

    @app.get("/status")
    def status():
        if engine.last_stats:   # replaced whole every 500 ms by the worker: safe to hand out as it is
            return engine.last_stats
        return snapshot(("status",), engine.stats)

    @app.get("/merchants")
    def merchants():
        return snapshot(("merchants",), engine.merchants_view)

    @app.get("/merchants/{mid}")
    def merchant(mid: str):
        view = snapshot(("merchant", mid), lambda: engine.merchant_view(mid))
        if view is None:
            raise HTTPException(404, "unknown merchant")
        return view

    @app.get("/merchants/{mid}/series/{sku}")
    def series(mid: str, sku: str):
        view = snapshot(("series", mid, sku), lambda: engine.series_view(mid, sku))
        if view is None:
            raise HTTPException(404, "unknown series (or the clock has not started)")
        return view

    @app.get("/signals")
    def signals(limit: int = 50):
        n = max(0, min(limit, 500))
        return snapshot(("signals", n), lambda: list(reversed(engine.signals))[:n])

    @app.post("/control")
    def control(body: dict):
        cmd = body.get("cmd")
        if cmd not in CONTROL_CMDS:
            raise HTTPException(400, f"cmd must be one of {sorted(CONTROL_CMDS)}")
        if cmd == "speed" and body.get("speed") not in C.SPEEDS:
            raise HTTPException(400, f"speed must be one of {list(C.SPEEDS)}")
        if cmd == "run_days":
            days = body.get("days")
            if not isinstance(days, int) or isinstance(days, bool) or not 1 <= days <= MAX_RUN_DAYS:
                raise HTTPException(400, f"days must be a whole number from 1 to {MAX_RUN_DAYS}")
            body = {"cmd": "run_days", "days": days}
        if cmd == "apply_offer":
            missing = {"signal_id", "platform", "merchant", "sku", "discount", "until"} - set(body)
            if missing:
                raise HTTPException(400, f"missing {sorted(missing)}")
            body = {**body, "by": "merchant"}
        engine.publish_control(body)
        return {"ok": True, "published": body}

    @app.post("/offers/{signal_id}/apply")
    def apply(signal_id: str):
        with engine.lock:
            sig = engine.signal_by_id.get(signal_id)
            if sig is None:
                raise HTTPException(404, "unknown signal")
            if sig["action"].get("type") != "discount":
                raise HTTPException(400, f"a {sig['kind']} signal has no offer to apply")
            if signal_id in engine.applied_signals:
                raise HTTPException(409, "already applied")
            cmd = engine.apply_command(sig, "merchant")
        engine.publish_control(cmd)
        return {"ok": True, "published": cmd}

    return app


# ------------------------------------------------------------------ process

def run(autopilot: float = 0.5, port: int = C.ENGINE_PORT) -> None:
    import paho.mqtt.client as mqtt
    import uvicorn

    from secondbrain.config import get_settings
    from secondbrain.live import MqttSettings
    from secondbrain.live.realtime import PUBLISHER

    settings = MqttSettings.from_app(get_settings())
    queue: deque = deque()
    wake = threading.Event()
    signal_mids: set[int] = set()
    engine: Engine

    live = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id=f"sb-commerce-engine-{time.time_ns() % 10**8}")
    live.username_pw_set(PUBLISHER, settings.publisher_password)
    live.max_queued_messages_set(20000)

    def on_publish(client, userdata, mid, *args):
        if mid in signal_mids:
            signal_mids.discard(mid)
            engine.pipeline["delivered"] += 1

    live.on_publish = on_publish
    live.connect_async(settings.host, settings.port)
    live.loop_start()

    def publish_live(topic: str, payload: dict, qos: int = 0) -> None:
        info = live.publish(topic, json.dumps(payload, separators=(",", ":"), default=str), qos=qos)
        if qos and "/signals/" in topic:
            signal_mids.add(info.mid)

    ingest = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id=f"sb-commerce-ingest-{time.time_ns() % 10**8}")
    ingest.max_inflight_messages_set(1000)

    def subscribe(client=ingest, *args):
        client.subscribe(f"{C.PREFIX}/#", qos=1)

    def resubscribe():  # retained world and catalogs come again after a reset
        ingest.unsubscribe(f"{C.PREFIX}/#")
        ingest.subscribe(f"{C.PREFIX}/#", qos=1)

    def on_message(client, userdata, msg):
        queue.append((msg.topic, msg.payload, time.time() * 1000))
        wake.set()

    ingest.on_connect = lambda c, *a: subscribe(c)
    ingest.on_message = on_message
    engine = Engine(publish_live=publish_live,
                    publish_control=lambda p: ingest.publish(C.CONTROL, json.dumps(p), qos=1),
                    autopilot=autopilot, on_reset=resubscribe)
    ingest.connect_async(C.INGEST_HOST, C.INGEST_PORT)
    ingest.loop_start()

    server = uvicorn.Server(uvicorn.Config(create_app(engine), host="127.0.0.1", port=port, log_level="warning"))
    threading.Thread(target=server.run, daemon=True, name="commerce-api").start()
    stop = threading.Event()
    signal.signal(signal.SIGTERM, lambda *_: stop.set())
    signal.signal(signal.SIGINT, lambda *_: stop.set())
    log.info("commerce engine: ingest %s:%s %s/#, live %s:%s %s, API :%s, autopilot %.0f%%", C.INGEST_HOST,
             C.INGEST_PORT, C.PREFIX, settings.host, settings.port, C.LIVE_PREFIX, port, autopilot * 100)
    try:
        while not stop.is_set():
            wake.wait(0.05)
            wake.clear()
            batch = []
            while queue and len(batch) < 50000:
                batch.append(queue.popleft())
            if batch:
                engine.process(batch)
            engine.tick()
    finally:
        server.should_exit = True
        ingest.loop_stop()
        ingest.disconnect()
        live.loop_stop()
        live.disconnect()


def main() -> None:
    parser = argparse.ArgumentParser(description="SecondBrain Live Commerce engine")
    parser.add_argument("--autopilot", type=float, default=0.5, help="share of merchants that auto-apply offers")
    parser.add_argument("--port", type=int, default=C.ENGINE_PORT)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    run(args.autopilot, args.port)


if __name__ == "__main__":
    main()
