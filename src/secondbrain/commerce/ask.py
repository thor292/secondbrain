"""Chat over the platforms: answer a question from the engine's live data, within one scope.

The app gathers the scope's data from the engine's HTTP API (platforms, attention, lessons, and any
merchant, platform or product named in the question), filters it to the scope, and only then
writes it into the prompt. The permission boundary is that filter, never the model: a merchant
scope can't see another merchant's signals, products or numbers, because they are dropped here
(and the engine already scopes its answers; this is the second, independent check).

    answer(llm, "Which DMart stores need restocking today?", "platform", "dmart")
    -> {"answer", "sources": [{"kind": "signal" | "merchant" | "platform" | "lesson", "id", "title"}], "scope"}
"""

import json
import re
from collections.abc import Callable

import httpx

from secondbrain.commerce import contract as C
from secondbrain.money import inr, inr_short

LEVELS = ("hq", "platform", "merchant")
ATTENTION = 10          # signals most worth acting on, always included
TOPIC_SIGNALS = 10      # more signals of the kind the question is about
MAX_MERCHANTS = 4       # merchant details fetched (named in the question, or the scope's own)
MAX_PRODUCTS = 40
OFFER_MERCHANTS = 5     # merchants with the most applied offers, for "which offers worked" questions

SYSTEM = """You are SecondBrain, the command centre over the commerce platforms (Shopify, Reliance Retail, DMart) \
and their merchants. You answer questions for one scope: {scope}.

Rules:
- Answer only from the DATA below. Never invent numbers, merchants, products or events.
- Quote rupee figures exactly as they are written in the DATA (₹ with Indian digit grouping, lakh, crore).
- If the DATA does not cover part of the question, say plainly what it doesn't cover instead of guessing.
- Keep to the scope. Never speculate about merchants or platforms outside it.
- Cite the items you used with their tags, like [S1] or [L2], right after the sentence they support.
- Be brief and practical: lead with the answer in one or two sentences, then up to six short "- " bullets with the specifics (who, what, how much, by when).
- Plain text only: **bold** is allowed, but no headings, tables, horizontal rules or emoji.
- Times are simulated time on the platforms (IST)."""

TOPICS = {   # question words -> the signal kinds it is about
    "restock": (r"restock|stock|run(ning)? out|reorder|inventory|shortage", ("restock", "festival", "surge")),
    "offers": (r"offer|discount|promo|price|deal|sale|clearance|work(ed)?|uplift", ("discount", "clearance", "festival")),
    "winback": (r"win.?back|lapsed|coupon|lost customers|regulars|churn", ("winback",)),
    "festival": (r"festival|diwali|dhanteras|bhai dooj|wedding|christmas|dussehra", ("festival",)),
    "surge": (r"surge|spike|viral|selling fast", ("surge",)),
}


class EngineDown(Exception):
    pass


class BadScope(ValueError):
    def __init__(self, status: int, detail: str):
        super().__init__(detail)
        self.status, self.detail = status, detail


Fetch = Callable[[str, dict | None], object]


def fetch_json(path: str, params: dict | None = None) -> object:
    """GET from the engine's API. EngineDown if it isn't running, BadScope for an unknown scope."""
    try:
        r = httpx.get(f"http://127.0.0.1:{C.ENGINE_PORT}{path}", params=params or {}, timeout=5)
    except httpx.HTTPError as exc:
        raise EngineDown(str(exc)) from exc
    if r.status_code == 404:
        raise BadScope(404, r.json().get("detail", "not found") if r.content else "not found")
    if r.status_code >= 400:
        raise EngineDown(f"engine answered {r.status_code} for {path}")
    return r.json()


def _mentions(text: str, name: str) -> bool:
    return bool(name) and re.search(r"(?<!\w)" + re.escape(name.lower()) + r"(?!\w)", text) is not None


def _pct(x) -> str:
    return "n/a" if x is None else f"{x:.0%}"


# ------------------------------------------------------------------ gathering, filtered to the scope

def gather(question: str, level: str, scope_id: str | None, fetch: Fetch | None = None) -> dict:
    """The scope's data as tagged context items, with everything outside the scope removed."""
    fetch = fetch or fetch_json
    if level not in LEVELS:
        raise BadScope(400, f"scope level must be one of {LEVELS}")
    if level != "hq" and not scope_id:
        raise BadScope(400, f"a {level} scope needs an id")
    q = question.lower()
    params = {} if level == "hq" else {level: scope_id}

    everyone = fetch("/merchants", None) or []
    if level == "merchant":
        in_scope = lambda m: m.get("id") == scope_id
    elif level == "platform":
        in_scope = lambda m: m.get("platform") == scope_id
    else:
        in_scope = lambda m: True
    mine = [m for m in everyone if in_scope(m)]
    if level != "hq" and not mine:
        raise BadScope(404, f"unknown {level} {scope_id!r}")
    allowed = {m["id"] for m in mine}
    # Fail closed: any item that names a merchant outside the scope is dropped, whatever its shape.
    outside = [s for m in everyone if m["id"] not in allowed for s in (m["id"], m.get("name") or "") if len(s) >= 4]
    # in-scope names that contain an outside one ("DMart Powai 2" vs "DMart Powai") are blanked before the check
    shadows = [s for m in mine for s in (m["id"], m.get("name") or "") if any(o in s for o in outside)]

    def clean(item) -> bool:
        if isinstance(item, dict) and item.get("merchant") and item["merchant"] not in allowed:
            return False
        text = json.dumps(item, ensure_ascii=False)
        for s in shadows:
            text = text.replace(s, "")
        return not any(s in text for s in outside)

    raw = fetch("/platforms", params) or []
    names = {p["id"]: p["name"] for p in raw if p.get("id") and p.get("name")}   # platform names are public
    platforms = [p for p in raw if clean(p)]
    if level == "merchant":   # the engine's per-merchant view of its platform: exactly one merchant
        platforms = [p for p in platforms if p.get("merchants") == 1 and p.get("id") == mine[0]["platform"]]
    elif level == "platform":
        platforms = [p for p in platforms if p.get("id") == scope_id]

    status = fetch("/status", None) or {}
    signals: dict[str, dict] = {}
    for s in fetch("/attention", {**params, "limit": ATTENTION}) or []:
        signals.setdefault(s["id"], s)
    kinds = {k for pat, ks in TOPICS.values() if re.search(pat, q) for k in ks}
    named_platforms = [p for p in platforms if _mentions(q, p["name"]) or _mentions(q, p["id"])]
    if kinds:
        extra_scopes = [{**params}] + ([{"platform": p["id"]} for p in named_platforms] if level == "hq" else [])
        for sp in extra_scopes:
            more = [s for s in fetch("/attention", {**sp, "limit": 200}) or [] if s["kind"] in kinds]
            for s in more[:TOPIC_SIGNALS]:
                signals.setdefault(s["id"], s)

    # merchants: the scope's own, or those named in the question; for offer questions, the busiest with offers
    wanted = [m for m in mine if level == "merchant" or _mentions(q, m.get("name", ""))][:MAX_MERCHANTS]
    if re.search(TOPICS["offers"][0], q) and level != "merchant":
        tree = fetch("/tree", params) or {}
        busiest = sorted((m for p in tree.get("platforms", []) for c in p.get("categories", []) for m in c["merchants"]
                          if m.get("offers_applied") and m["id"] in allowed),
                         key=lambda m: -m["offers_applied"])
        wanted += [m for m in busiest if m["id"] not in {w["id"] for w in wanted}][:OFFER_MERCHANTS]
    details = []
    for m in wanted:
        d = fetch(f"/merchants/{m['id']}", None)
        if d and d.get("id") in allowed:
            details.append(d)

    # products named in the question: their signals, wherever in the scope they are
    if level != "merchant" and len(q) > 3:
        for s in fetch("/signals", {"limit": 500}) or []:
            if s.get("product") and _mentions(q, s["product"]) and s.get("merchant") in allowed:
                signals.setdefault(s["id"], s)

    lessons = [l for l in fetch("/lessons", params) or [] if clean(l)]
    signals = {k: s for k, s in signals.items() if clean(s)}
    details = [d for d in details if d["id"] in allowed and clean(d)]
    return {"scope": _scope_label(level, scope_id, mine, names), "status": status, "platforms": platforms,
            "signals": list(signals.values()), "merchants": details, "lessons": lessons, "names": names}


def _scope_label(level, scope_id, mine, names) -> str:
    if level == "hq":
        return "SecondBrain HQ, all platforms and all their merchants"
    if level == "platform":
        return f"the {names.get(scope_id, scope_id)} platform and its {len(mine)} merchants"
    m = mine[0]
    return f"the merchant {m['name']} ({names.get(m['platform'], m['platform'])}, {m.get('city', '')}) only"


# ------------------------------------------------------------------ the prompt

def _platform_line(p: dict) -> str:
    acc = p.get("accuracy") or {}
    kinds = ", ".join(f"{k} {v}" for k, v in sorted((p.get("signals_by_kind") or {}).items(), key=lambda kv: -kv[1]))
    cats = "; ".join(f"{c['name']} ({c['merchants']} merchants, {inr(c['revenue_today'])} today)"
                     for c in (p.get("categories") or [])[:6])
    return (f"{p['name']}: {p['merchants']} merchant(s), {p['customers']:,} customers seen, {p['series']:,} products. "
            f"Today: {p['orders_today']:,} orders, {inr(p['revenue_today'])} revenue, {p['active_customers_today']:,} "
            f"active customers. Open signals {p['open_signals']}; signals raised so far {p['signals_total']}"
            + (f" ({kinds})" if kinds else "") + f". Offers applied {p['offers_applied']}, revenue vs no offer "
            f"{inr_short(p['uplift_revenue'])}, offers meeting their promise {_pct(p.get('hit_rate'))}. Forecast error "
            f"(WAPE): {_pct(acc.get('wape'))} per product-day, {_pct(acc.get('merchant_wape'))} per merchant-day, vs "
            f"{_pct(acc.get('baseline_wape'))} for a same-day-last-week guess. Merchant categories: {cats}.")


def _signal_line(s: dict, names: dict) -> str:
    exp = s.get("expected") or {}
    where = f"{s.get('merchant_name')} ({s.get('platform_name') or names.get(s['platform'], s['platform'])}, {s.get('city')})"
    return (f"{s['urgency']} {s['kind']} signal at {where}, raised {s['sim_ts']}"
            + (f", product {s['product']} ({s.get('category')})" if s.get("product") else "")
            + f": {s['title']}. {s['detail']}"
            + (f" Expected revenue {inr(exp['revenue'])}." if exp.get("revenue") else "")
            + (" Applied automatically (autopilot)." if s.get("autopilot") else ""))


def _merchant_lines(d: dict, names: dict) -> str:
    out = [f"{d['name']} ({names.get(d['platform'], d['platform'])}, {d.get('city')}, {d.get('category')}): today "
           f"{d['orders_today']:,} orders, {inr(d['revenue_today'])}; since the feed started {d['orders_total']:,} orders, "
           f"{inr_short(d['revenue_total'])}. "
           + ("Autopilot on: the brain applies its own discount offers here automatically; restocks and coupons stay "
              "the merchant's call." if d.get("autopilot") else "Autopilot off: every offer waits for the merchant to apply it.")]
    prods = sorted(d.get("products") or [], key=lambda p: (p.get("cover_days") is None, p.get("cover_days") or 0))
    for p in prods[:MAX_PRODUCTS]:
        t = p.get("today") or {}
        out.append(f"  - product {p['name']} ({p['category']}): price {inr(p['price'])}, stock {p['stock']:.0f}, "
                   + (f"cover {p['cover_days']:.1f} days, " if p.get("cover_days") is not None else "")
                   + f"lead time {p['lead_days']:.0f} days; today sold {t.get('actual', 0):.0f}"
                   + (f" of {t['forecast']:.0f} forecast" if t.get("forecast") is not None else "")
                   + ("; an offer is running" if p.get("offer") else ""))
    for o in (d.get("outcomes") or [])[-10:]:
        out.append(f"  - closed offer: {o['discount']:.0%} off {o['product']} ({o.get('category')}): "
                   f"{o['actual_uplift_units']:+.0f} units vs the no-offer forecast (promised at least "
                   f"+{o.get('assured_uplift_units', 0):.0f}), {_pct(o.get('uplift_pct'))} uplift, revenue vs no offer "
                   f"{inr(o['revenue_gain'])}, {'met its promise' if o['worked'] else 'missed its promise'}")
    return "\n".join(out)


def build_prompt(question: str, data: dict) -> tuple[str, list[dict]]:
    """The DATA block with a tag per item, and the sources those tags stand for."""
    lines, sources = ["DATA"], []
    status = data["status"] or {}
    if status.get("sim_ts"):
        fest = status.get("festival") or {}
        lines.append(f"Simulated time now: {status['sim_ts']}"
                     + (f". Next festival: {fest['name']} on {fest['date']} (in {fest['days_to']} days)" if fest else ""))

    def item(prefix, kind, sid, title, text):
        tag = f"{prefix}{sum(1 for s in sources if s['tag'].startswith(prefix)) + 1}"
        sources.append({"tag": tag, "kind": kind, "id": sid, "title": title})
        lines.append(f"[{tag}] {text}")

    lines.append("\nPlatforms (numbers for this scope only):")
    for p in data["platforms"]:
        item("P", "platform", p["id"], p["name"], _platform_line(p))
    lines.append("\nMerchants:")
    for d in data["merchants"]:
        item("M", "merchant", d["id"], d["name"], _merchant_lines(d, data["names"]))
    lines.append("\nOpen signals (decisions waiting, most urgent first):")
    for s in data["signals"]:
        item("S", "signal", s["id"], s["title"], _signal_line(s, data["names"]))
    lines.append("\nWhat the brain has learned:")
    for l in data["lessons"]:
        item("L", "lesson", l["id"], l["title"],
             f"{l['title']}. {l['detail']} (learned from {l['learned_from']} merchant(s), applies to {l['applies_to']})")
    lines += ["", f"Question: {question}"]
    return "\n".join(lines), sources


def answer(llm, question: str, level: str, scope_id: str | None, fetch: Fetch | None = None) -> dict:
    data = gather(question, level, scope_id, fetch)
    prompt, sources = build_prompt(question, data)
    reply = llm.complete(SYSTEM.format(scope=data["scope"]), prompt).strip()
    if not reply or reply == "NOT_FOUND":
        reply = "The live data for this scope doesn't answer that."
    cited = [s for s in sources if re.search(r"\[" + s["tag"] + r"\]", reply)]
    return {"answer": reply, "scope": {"level": level, "id": scope_id if level != "hq" else None, "label": data["scope"]},
            "sources": [{k: s[k] for k in ("kind", "id", "title")} | {"tag": s["tag"]} for s in (cited or sources)]}
