"""The command centre over the platforms: HQ, Graph and Chat read these views of the engine's state.

Every view takes a scope (``contract``: HQ, one platform, or one merchant) and works on the engine's
per-merchant arrays through a boolean merchant mask, so narrowing to a branch of the tree is cheap
and nothing outside the branch is ever counted. Call with ``engine.lock`` held.

Lessons are what the engine has actually learned, written as plain sentences. Each one needs real
evidence (thresholds below) and says where it came from; with no evidence there is no lesson.

    price_response  closed offers in a category: measured uplift vs the frozen no-offer forecast, and
                    the learned elasticity. Before any offer has closed: promotions in the sales history.
    no_discount     the daily price check on idle products: where no discount ever beat full price.
    festival_lift   the festival multiplier per category, learned from festival days in the data.
    restock         stockout product-days on live days, by category.
    winback         measured repurchase intervals and lapsed regulars (the coupon response is assumed,
                    not measured: redemptions are not in the feeds, and the lesson says so).

A merchant sees lessons the network learned (only when at least ``ANON_MIN`` merchants taught them,
so no single other merchant can be read off) plus lessons from its own data.
"""

from dataclasses import dataclass
from statistics import median

import numpy as np

from secondbrain.commerce import contract as C
from secondbrain.commerce.forecast import FESTIVAL
from secondbrain.commerce.offers import WINBACK_RESPONSE
from secondbrain.money import group, inr_short

MIN_OFFERS = 3            # closed, uncensored offers before a category's price response is a lesson
MIN_PROMO_DAYS = 20       # history promotion product-days before the backfill estimate is a lesson
MIN_CHECKS = 20           # daily price checks before "discounts don't pay" is a lesson
NO_DISCOUNT_SHARE = 0.95  # ... and at least this share of them found no offer worth making
NO_DISCOUNT_MARGIN = 0.2  # ... on thin margins
MIN_FEST_LIFT = 0.08      # a festival lift smaller than +-8% is not worth saying
MIN_STOCK_DAYS = 30       # product-days observed before a stockout pattern is a lesson
MIN_STOCKOUT = 0.08       # ... and at least this share of them ran out
MIN_REGULARS = 20         # regulars measured before a repurchase pattern is a lesson
MIN_WINBACK_DAYS = 7      # ... over at least a week of live days, and twice the interval (else it reads too short)
MAX_BACKFILL_E = -0.2     # a history estimate flatter than this (or the wrong sign) is noise, not a lesson
ANON_MIN = 3              # a merchant sees a network lesson only if this many merchants taught it
URGENCY = {"high": 0, "medium": 1, "low": 2}


class ScopeError(Exception):
    def __init__(self, status: int, detail: str):
        super().__init__(detail)
        self.status, self.detail = status, detail


@dataclass
class Scope:
    level: str              # hq | platform | merchant
    id: str | None
    mask: np.ndarray        # merchants in the branch
    platform: str | None    # the platform of the branch (None for HQ)

    def public(self) -> dict:
        return {"level": self.level, "id": self.id}


def resolve(eng, platform: str | None = None, merchant: str | None = None) -> Scope:
    reg = eng.reg
    nm = reg.nm
    if merchant:
        m = reg.merchant_index.get(merchant)
        if m is None:
            raise ScopeError(404, f"unknown merchant {merchant!r}")
        pid = reg.merchants[m]["platform"]
        if platform and platform != pid:
            raise ScopeError(404, f"merchant {merchant!r} is not on {platform!r}")
        mask = np.zeros(nm, bool)
        mask[m] = True
        return Scope("merchant", merchant, mask, pid)
    if platform:
        if platform not in reg.platform_index:
            raise ScopeError(404, f"unknown platform {platform!r}")
        return Scope("platform", platform, reg.m_plat[:nm] == reg.platform_index[platform], platform)
    return Scope("hq", None, np.ones(nm, bool), None)


# ------------------------------------------------------------------ helpers

def _open_by_merchant(eng) -> dict[str, list[dict]]:
    return eng._open_signals()


def _ledger(eng, mask: np.ndarray) -> tuple[int, float, int, int]:
    applied = gain = worked = judged = 0
    for m, (a, g, w, j) in eng.ledger.items():
        if m < len(mask) and mask[m]:
            applied, gain, worked, judged = applied + a, gain + g, worked + w, judged + j
    return applied, gain, worked, judged


def _signal_counts(eng, mask: np.ndarray) -> dict[str, int]:
    out: dict[str, int] = {}
    for (m, kind, _), n in eng.sig_count.items():
        if m < len(mask) and mask[m]:
            out[kind] = out.get(kind, 0) + n
    return out


def _sold_by(eng) -> np.ndarray:
    """(categories x merchants): which merchants sell products in each category."""
    reg, fc = eng.reg, eng.fc
    out = np.zeros((len(fc.categories), reg.nm), bool)
    if reg.n:
        out[fc.cat[:reg.n], reg.s_merch[:reg.n]] = True
    return out


def _now(eng) -> str | None:
    return eng.now.isoformat(timespec="seconds") if eng.now else None


def _slug(text: str) -> str:
    return "".join(c if c.isalnum() else "-" for c in text.lower()).strip("-")


def _pct(x: float) -> str:
    return f"{x:+.0%}"


def _n(x: int) -> str:
    return group(int(x))


def _merchants_word(n: int, platform: str | None = None) -> str:
    noun = "stores" if platform == "dmart" else "merchants"
    return f"{_n(n)} {noun if n != 1 else noun[:-1]}"


# ------------------------------------------------------------------ platforms

def platforms(eng, sc: Scope) -> list[dict]:
    reg = eng.reg
    nm = reg.nm
    open_by = _open_by_merchant(eng)
    rates = eng._rates()["platforms"]
    out = []
    for pid in reg.platform_ids:
        p = reg.platform_index[pid]
        mask = sc.mask & (reg.m_plat[:nm] == p)
        if not mask.any():
            continue
        ms = np.flatnonzero(mask)
        applied, gain, worked, judged = _ledger(eng, mask)
        by_kind = _signal_counts(eng, mask)
        series = sum(len(reg.series_of[m]) for m in ms)
        cats: dict[str, list] = {}
        for m in ms.tolist():
            c = cats.setdefault(reg.merchants[m]["category"], [0, 0.0])
            c[0] += 1
            c[1] += float(reg.m_rev_today[m])
        meta = reg.platform_meta.get(pid, {})
        out.append({
            "id": pid, "name": meta.get("name", pid), "kind": meta.get("kind", ""), "color": meta.get("color"),
            "merchants": int(len(ms)), "customers": int(sum(len(reg.customers[m]) for m in ms)),
            "series": int(series),
            "orders_today": int(reg.m_orders_today[ms].sum()), "revenue_today": round(float(reg.m_rev_today[ms].sum()), 2),
            # the live rate is measured per platform only; a merchant's own rate is not tracked
            "orders_per_sec": round(rates.get(pid, 0.0), 1) if sc.level != "merchant" else None,
            "active_customers_today": int(reg.m_active_today[ms].sum()),
            "accuracy": eng.acc.scoped(mask),
            "open_signals": sum(len(open_by.get(reg.merchants[m]["id"], [])) for m in ms),
            "signals_total": sum(by_kind.values()), "signals_by_kind": by_kind,
            "offers_applied": applied, "uplift_revenue": round(float(gain), 2),
            "hit_rate": round(worked / judged, 4) if judged else None,
            "categories": [{"name": k, "merchants": v[0], "revenue_today": round(v[1], 2)}
                           for k, v in sorted(cats.items(), key=lambda kv: -kv[1][1])],
        })
    return out


# ------------------------------------------------------------------ attention

def attention(eng, sc: Scope, limit: int = 10) -> list[dict]:
    reg = eng.reg
    sigs = [s for mid, ss in _open_by_merchant(eng).items()
            if (m := reg.merchant_index.get(mid)) is not None and sc.mask[m] for s in ss]
    sigs.sort(key=lambda s: (URGENCY.get(s["urgency"], 3), -((s.get("expected") or {}).get("revenue") or 0.0),
                             s["id"]))
    names = {pid: reg.platform_meta.get(pid, {}).get("name", pid) for pid in reg.platform_ids}
    return [{**s, "platform_name": names.get(s["platform"], s["platform"])} for s in sigs[:max(0, limit)]]


# ------------------------------------------------------------------ tree

def tree(eng, sc: Scope) -> dict:
    reg = eng.reg
    open_by = _open_by_merchant(eng)

    def merchant_node(m: int) -> dict:
        info = reg.merchants[m]
        sigs = open_by.get(info["id"], [])
        status = "urgent" if any(s["urgency"] == "high" for s in sigs) else "signal" if sigs else "ok"
        return {"id": info["id"], "name": info["name"], "city": info["city"], "category": info["category"],
                "status": status, "open_signals": len(sigs), "orders_today": int(reg.m_orders_today[m]),
                "revenue_today": round(float(reg.m_rev_today[m]), 2),
                "offers_applied": eng.ledger.get(m, [0])[0]}

    plats = []
    for pid in reg.platform_ids:
        ms = np.flatnonzero(sc.mask & (reg.m_plat[:reg.nm] == reg.platform_index[pid]))
        if not len(ms):
            continue
        cats: dict[str, list] = {}
        for m in ms.tolist():
            cats.setdefault(reg.merchants[m]["category"], []).append(merchant_node(m))
        nodes = [n for c in cats.values() for n in c]
        meta = reg.platform_meta.get(pid, {})
        plats.append({"id": pid, "name": meta.get("name", pid), "color": meta.get("color"),
                      "orders_today": sum(n["orders_today"] for n in nodes),
                      "revenue_today": round(sum(n["revenue_today"] for n in nodes), 2),
                      "open_signals": sum(n["open_signals"] for n in nodes),
                      "categories": [{"name": k, "merchants": sorted(v, key=lambda n: n["id"])}
                                     for k, v in sorted(cats.items())]})
    return {"root": {"id": "root", "name": "SecondBrain",
                     "orders_today": sum(p["orders_today"] for p in plats),
                     "revenue_today": round(sum(p["revenue_today"] for p in plats), 2),
                     "open_signals": sum(p["open_signals"] for p in plats)},
            "platforms": plats, "scope": sc.public()}


# ------------------------------------------------------------------ lessons

def lessons(eng, sc: Scope) -> list[dict]:
    """What the brain learned that matters to this branch, strongest evidence first."""
    reg = eng.reg
    if not reg.nm:
        return []
    everyone = np.ones(reg.nm, bool)
    sold = _sold_by(eng)
    if sc.level == "hq":
        out = _learn(eng, everyone, everyone, None, sold)
    elif sc.level == "platform":
        out = _learn(eng, sc.mask, sc.mask, sc.platform, sold)
    else:  # a merchant: what the network learned (never fewer than ANON_MIN teachers), and its own patterns
        out = [l for l in _learn(eng, everyone, sc.mask, None, sold, kinds=("price_response", "no_discount",
                                                                              "festival_lift"))
               if l["learned_from"] >= ANON_MIN]
        out += _learn(eng, sc.mask, sc.mask, sc.platform, sold, kinds=("restock", "winback"), own=True)
    out.sort(key=lambda l: (-l["_tier"], -l["learned_from"], -l["evidence_n"], l["id"]))
    for l in out:
        del l["_tier"]
    return out


def _learn(eng, evidence: np.ndarray, scope: np.ndarray, platform: str | None, sold: np.ndarray,
           kinds=("price_response", "no_discount", "festival_lift", "restock", "winback"), own: bool = False) -> list:
    """Lessons from the evidence of merchants in ``evidence`` that apply to merchants in ``scope``."""
    fc = eng.fc
    ts = _now(eng)
    applies = {c: int((sold[k] & scope).sum()) for k, c in enumerate(fc.categories)}
    out: list[dict] = []

    def lesson(kind, category, title, detail, value, prior, n, learned_from, applies_to, tier, plat=platform):
        out.append({"id": f"{kind}:{plat or 'all'}:{_slug(category or (plat or 'all'))}"
                          + (f":{_slug(eng.reg.merchants[int(np.flatnonzero(scope)[0])]['id'])}" if own else ""),
                    "kind": kind, "category": category, "platform": plat, "title": title, "detail": detail,
                    "value": None if value is None else round(float(value), 3),
                    "prior": None if prior is None else round(float(prior), 3), "evidence_n": int(n),
                    "learned_from": int(learned_from), "applies_to": int(applies_to), "updated_sim_ts": ts,
                    "_tier": tier})

    if "price_response" in kinds:
        _price_lessons(eng, evidence, platform, applies, sold, lesson)
    if "no_discount" in kinds:
        _no_discount_lessons(eng, evidence, applies, lesson)
    if "festival_lift" in kinds:
        _festival_lessons(eng, applies, sold, lesson)
    if "restock" in kinds:
        _restock_lessons(eng, evidence, applies, lesson, own)
    if "winback" in kinds:
        _winback_lessons(eng, evidence, scope, lesson, own)
    return out


def _response_word(e: float) -> str:
    return "strongly" if e <= -2.0 else "clearly" if e <= -1.2 else "mildly" if e <= -0.5 else "barely"


def _price_lessons(eng, evidence, platform, applies, sold, lesson) -> None:
    el = eng.book.elasticity
    by_cat: dict[str, list[tuple]] = {}
    for rec in eng.outcome_log:
        if rec[0] < len(evidence) and evidence[rec[0]]:
            by_cat.setdefault(rec[1], []).append(rec)
    promo_merchants = _history_promo_merchants(eng)
    for k, cat in enumerate(eng.fc.categories):
        if not applies.get(cat):
            continue
        recs = by_cat.get(cat, [])
        clean = [r for r in recs if not r[8] and r[4] >= 1 and r[2] > 0]
        learned, prior = el.category_value(cat), el.prior(cat)
        if len(clean) >= MIN_OFFERS:
            d = median(r[2] for r in clean)
            hrs = round(median(r[3] for r in clean))
            uplift = float(np.mean([r[5] / r[4] - 1 for r in clean]))
            base, units = sum(r[4] for r in clean), sum(r[5] for r in clean)
            mean_d = float(np.mean([r[2] for r in clean]))
            e_obs = np.log(max(units, 0.5) / base) / np.log(1 - mean_d)
            judged = [r for r in recs if r[7]]
            worked = sum(r[6] for r in judged)
            gain = sum(r[9] for r in recs)
            teachers = {r[0] for r in clean}
            if uplift > 0:
                title = (f"{cat} shoppers respond {_response_word(e_obs)} to price: a {d:.0%} offer for {hrs}h sold "
                         f"{_pct(uplift)} on average across {_n(len(clean))} offers")
            else:
                title = (f"{cat} offers did not lift sales: a {d:.0%} offer for {hrs}h sold {_pct(uplift)} vs the "
                         f"no-offer forecast on average across {_n(len(clean))} offers")
            detail = (f"Each offer is measured against the no-offer forecast frozen when it started. "
                      f"{_n(worked)} of {_n(len(judged))} judged offers met their promised minimum uplift"
                      + (f"; {len(recs) - len(clean)} more ran out of stock or had too little baseline to measure"
                         if len(recs) > len(clean) else "")
                      + f". Revenue vs no offer: {inr_short(gain)}. The learned price elasticity for {cat} is "
                      f"{learned:.1f} (textbook prior {prior:.1f}), now used to price offers for "
                      f"{_merchants_word(applies[cat])} selling {cat}.")
            lesson("price_response", cat, title, detail, learned, prior, len(clean), len(teachers), applies[cat], 3)
        elif cat in el.backfill and el.backfill[cat][2] >= MIN_PROMO_DAYS and el.backfill[cat][0] <= MAX_BACKFILL_E:
            e_b, _, days = el.backfill[cat]
            lift10 = 0.9 ** e_b - 1
            teachers = promo_merchants.get(k, set())
            title = (f"{cat} shoppers respond {_response_word(e_b)} to price: in past promotions a 10% price cut "
                     f"lifted sales by about {lift10:.0%}")
            detail = (f"Measured on {_n(days)} promotion product-days in the sales history (elasticity {e_b:.1f}, "
                      f"textbook prior {prior:.1f}; the brain uses {learned:.1f}). "
                      + (f"Only {len(clean)} measurable live {cat} offer{'s have' if len(clean) != 1 else ' has'} closed "
                         f"so far ({MIN_OFFERS} needed for a lesson)" if clean else f"No measurable live {cat} offer has closed yet")
                      + "; live outcomes will confirm or correct it.")
            lesson("price_response", cat, title, detail, learned, prior, days, len(teachers), applies[cat], 1,
                   plat=None)


def _history_promo_merchants(eng) -> dict[int, set]:
    """Category index -> merchants whose sales history contains promotion days (price paid != list price)."""
    fc, reg = eng.fc, eng.reg
    n = reg.n
    end = eng.live_start if eng.live_start is not None else fc.hist.shape[1]
    if not n or end <= 0:
        return {}
    promo = ((np.abs(np.log(np.clip(fc.ratio[:n, :end], 0.3, 2.0))) > 0.02) & (fc.hist[:n, :end] > 0)).any(axis=1)
    out: dict[int, set] = {}
    for i in np.flatnonzero(promo).tolist():
        out.setdefault(int(fc.cat[i]), set()).add(int(reg.s_merch[i]))
    return out


def _no_discount_lessons(eng, evidence, applies, lesson) -> None:
    agg: dict[str, list] = {}
    for (m, cat), (checked, declined, margin, e) in eng.price_checks.items():
        if m < len(evidence) and evidence[m]:
            a = agg.setdefault(cat, [0, 0, 0.0, 0.0, set()])
            a[0] += checked
            a[1] += declined
            a[2] += margin
            a[3] += e
            if declined:
                a[4].add(m)
    for cat, (checked, declined, margin, e, teachers) in agg.items():
        if not applies.get(cat) or checked < MIN_CHECKS or not declined:
            continue
        share, avg_margin, avg_e = declined / checked, margin / declined, e / checked
        if share < NO_DISCOUNT_SHARE or avg_margin > NO_DISCOUNT_MARGIN:
            continue
        lift5 = 0.95 ** avg_e - 1
        title = (f"Discounts don't pay on {cat}: at a ~{avg_margin:.0%} margin, {_n(declined)} of {_n(checked)} "
                 f"daily price checks found no offer that beats full price, so the brain holds price")
        if avg_margin > 0.05:
            need = avg_margin / (avg_margin - 0.05) - 1
            why = (f"Even a 5% offer would need {need:.0%} more sales just to break even, and {cat} shoppers buy only "
                   f"~{lift5:.0%} more at 5% off (elasticity {avg_e:.1f}). ")
        else:
            why = "At that margin even a 5% offer sells below cost. "
        detail = (why + f"Checked on idle products every day at "
                  f"{_merchants_word(len(teachers))}; offers still go out for overstock, spoilage or a real drop in sales.")
        lesson("no_discount", cat, title, detail, share, None, checked, len(teachers), applies[cat], 2)


def _festival_lessons(eng, applies, sold, lesson) -> None:
    fc, reg = eng.fc, eng.reg
    if eng.today is None or not reg.n:
        return
    seen = [(d, name) for d, name in sorted(C.FESTIVALS.items()) if fc.base_day <= d < eng.today]
    if not seen:
        return
    idx = np.array([fc.day_index(d) for d, _ in seen])
    n = reg.n
    start = fc.start[:n]
    mult = fc.pool.mult()
    signals = {}
    for (m, kind, cat), c in eng.sig_count.items():
        if kind == "festival":
            signals[cat] = signals.get(cat, 0) + c
    for k, cat in enumerate(fc.categories):
        if not applies.get(cat) or not fc.pool.days[k, FESTIVAL]:
            continue
        lift = float(mult[k, FESTIVAL])
        if abs(lift - 1) < MIN_FEST_LIFT:
            continue
        rows = np.flatnonzero((fc.cat[:n] == k) & (start >= 0))
        had = (start[rows][:, None] <= idx[None, :])           # series with data on each festival day
        days_seen = int(had.any(axis=0).sum())
        teachers = set(reg.s_merch[rows[had.any(axis=1)]].tolist())
        if not days_seen or not teachers:
            continue
        names = ", ".join(name for (d, name), ok in zip(seen, had.any(axis=0)) if ok)
        more = "more" if lift > 1 else "less"
        title = (f"{cat} sells {abs(lift - 1):.0%} {more} on festival days, learned from {names} across "
                 f"{_merchants_word(len(teachers))}")
        detail = (f"Festival multiplier x{lift:.2f} (x1.00 = no effect), learned from {days_seen} festival "
                  f"day{'s' if days_seen != 1 else ''} in the sales data. Every {cat} forecast now includes it, for "
                  f"{_merchants_word(applies[cat])}" + (f"; {_n(signals[cat])} festival stock-up or offer signals raised so far."
                                                    if signals.get(cat) else "."))
        lesson("festival_lift", cat, title, detail, lift, 1.0, days_seen, len(teachers), applies[cat], 2, plat=None)


def _restock_lessons(eng, evidence, applies, lesson, own) -> None:
    fc, reg = eng.fc, eng.reg
    if eng.live_start is None or eng.today is None or not reg.n:
        return
    lo, hi = eng.live_start, fc.day_index(eng.today)     # closed live days
    if hi <= lo:
        return
    n = reg.n
    rows = np.flatnonzero(evidence[reg.s_merch[:n]] & (fc.start[:n] >= 0))
    if not len(rows):
        return
    cens = fc.cens[rows, lo:hi]
    restocks: dict[str, int] = {}
    for (m, kind, cat), c in eng.sig_count.items():
        if kind in ("restock", "festival") and m < len(evidence) and evidence[m]:
            restocks[cat] = restocks.get(cat, 0) + c
    days = hi - lo
    for k, cat in enumerate(fc.categories):
        sel = fc.cat[rows] == k
        if not applies.get(cat) or not sel.any():
            continue
        product_days = int(sel.sum()) * days
        out_days = int(cens[sel].sum())
        share = out_days / product_days if product_days else 0.0
        if product_days < (1 if own else MIN_STOCK_DAYS) or share < MIN_STOCKOUT:
            continue
        teachers = set(reg.s_merch[rows[sel][cens[sel].any(axis=1)]].tolist())
        lead = float(np.mean(reg.lead[rows[sel]]))
        where = "" if own else f" at {_merchants_word(len(teachers))}"
        title = (f"{cat} runs out often: {share:.0%} of product-days had a stockout over the last {days} "
                 f"day{'s' if days != 1 else ''}{where}")
        detail = (f"{_n(out_days)} of {_n(product_days)} product-days ran out of stock (sales on those days are "
                  f"capped, and forecasts treat them as censored); supplier lead time averages {lead:.0f} "
                  f"day{'s' if round(lead) != 1 else ''}. "
                  + (f"{_n(restocks[cat])} restock and festival stock-up signals raised for {cat}."
                     if restocks.get(cat) else "No restock signal has been raised for it yet."))
        lesson("restock", cat, title, detail, share, None, out_days, len(teachers), applies[cat], 2)


def _winback_lessons(eng, evidence, scope, lesson, own) -> None:
    reg = eng.reg
    nm = reg.nm
    if eng.live_start is None or eng.today is None:
        return
    live_days = eng.fc.day_index(eng.today) - eng.live_start
    sent: dict[int, int] = {}
    for (m, kind, _), c in eng.sig_count.items():
        if kind == "winback":
            sent[m] = sent.get(m, 0) + c
    for pid in reg.platform_ids:
        p = reg.platform_index[pid]
        ms = np.flatnonzero(evidence & (reg.m_plat[:nm] == p) & (reg.m_interval[:nm] > 0) & (reg.m_regulars[:nm] > 0))
        if not len(ms):
            continue
        regulars, lapsed = int(reg.m_regulars[ms].sum()), int(reg.m_lapsed[ms].sum())
        if regulars < MIN_REGULARS:
            continue
        interval = float(np.median(reg.m_interval[ms]))
        if live_days < max(MIN_WINBACK_DAYS, 2 * interval):
            continue
        name = reg.platform_meta.get(pid, {}).get("name", pid)
        who = reg.merchants[int(ms[0])]["name"] if own else name
        where = "" if own else f" at {_merchants_word(int((reg.m_lapsed[ms] > 0).sum()), pid)}"
        every = "every day" if interval < 1.5 else f"every ~{interval:.0f} days"
        title = (f"{who} regulars come back {every}; "
                 + (f"{_n(lapsed)} of them{where} are now away more than twice that long" if lapsed
                    else "none has been away twice that long yet"))
        coupons = sum(sent.get(int(m), 0) for m in ms)
        detail = (f"Repurchase intervals measured on {_n(regulars)} regular customers (two or more visit days) over "
                  f"{live_days} live days. "
                  + (f"{_n(coupons)} win-back coupon signals raised. " if coupons else "")
                  + f"The coupon response is assumed at {WINBACK_RESPONSE:.0%}, not yet learned: redemptions are not "
                  f"in the platform feeds.")
        applies_to = int((scope & (reg.m_plat[:nm] == p) & (reg.m_interval[:nm] > 0)).sum())
        lesson("winback", None, title, detail, interval, None, regulars, len(ms), applies_to, 1, plat=pid)
