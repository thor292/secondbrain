"""The Live Commerce engine's HTTP API, answered from the database instead of the simulator.

Where the simulator isn't running (Cloud Run), the Graph, HQ and Live Commerce pages would
otherwise make up numbers in the browser. This module answers the same endpoints
(see contract.py, "Engine HTTP API" and "HQ, Graph and Chat") from what SecondBrain really
holds: daily sales, stock, stored forecasts with their backtest accuracy, and decision signals.

"Today" is the latest day with sales data. Things there is no data for (live orders per
second, customers, latency, simulated offers) are reported as zero or empty, never invented.
"""

import csv
from collections import Counter, defaultdict
from datetime import date, datetime, time, timedelta
from pathlib import Path

import psycopg
from psycopg.rows import dict_row

from secondbrain.commerce import contract

PLATFORM = {"id": "shoppi", "name": "Shoppi", "kind": "Marketplace merchants", "color": "#E4572E"}
MERCHANTS_CSV = Path(__file__).resolve().parents[3] / "examples" / "shoppi" / "merchants.csv"

# Product category -> the pages' merchant groups (same as static/shoppi_world.js).
GROUP = {
    "Namkeen": "Gourmet", "Ready Meals": "Gourmet", "Frozen": "Gourmet", "Sweets": "Gourmet", "Gifting": "Gourmet",
    "Staples": "Grocery", "Beverages": "Grocery", "Dairy": "Grocery", "Fruits & Vegetables": "Grocery",
    "Household": "Home", "Clothing": "Fashion", "Outerwear": "Fashion", "Accessories": "Fashion", "Footwear": "Fashion",
}
# Signal kinds in the database -> the pages' signal kinds (icons, filters).
KIND = {"stockout": "restock", "backorder": "restock", "excess_stock": "clearance", "business": "surge"}
URGENCY_RANK = {"high": 0, "medium": 1, "low": 2}


def _places() -> dict[str, dict]:
    try:
        with open(MERCHANTS_CSV, encoding="utf-8") as f:
            return {r["slug"]: r for r in csv.DictReader(f)}
    except OSError:
        return {}


def _xy(lat: float, lng: float) -> tuple[float, float]:
    return (round(min(max((lng - 68) / 29.5, 0.0), 1.0), 4), round(min(max((37.5 - lat) / 31, 0.0), 1.0), 4))


def _ms(ts: datetime | None) -> int | None:
    return int(ts.timestamp() * 1000) if ts else None


class Data:
    """One consistent read of everything the pages need, for one request."""

    def __init__(self, conn: psycopg.Connection) -> None:
        conn.row_factory = dict_row
        places = _places()
        companies = conn.execute(
            "SELECT path::text AS path, name FROM brains WHERE kind = 'company' ORDER BY path").fetchall()
        self.merchants: dict[str, dict] = {}
        for c in companies:
            slug = c["path"].split(".", 1)[1]
            place = places.get(slug, {})
            x, y = _xy(float(place.get("lat") or 22.5), float(place.get("lng") or 79.0))
            self.merchants[f"shoppi-{slug}"] = {
                "id": f"shoppi-{slug}", "platform": PLATFORM["id"], "company": c["path"], "name": c["name"],
                "city": place.get("city", ""), "x": x, "y": y, "products": [], "signals": []}
        by_company = {m["company"]: m for m in self.merchants.values()}

        self.history: dict[tuple[str, str], list[dict]] = {}
        self.today = conn.execute("SELECT max(day) AS d FROM sales_daily").fetchone()["d"] or date.today()
        products = conn.execute(
            """SELECT p.brain_path::text AS brain, p.sku, p.name, p.category, p.base_price,
                      i.on_hand, i.unit_cost, i.lead_time_days,
                      f.model, f.quality, f.backtest, f.points,
                      coalesce(t.units, 0) AS units_today, coalesce(t.revenue, 0) AS revenue_today,
                      coalesce(w.units, 0) AS units_28d
               FROM products p
               LEFT JOIN inventory i ON i.brain_path = p.brain_path AND i.sku = p.sku
               LEFT JOIN LATERAL (SELECT model, quality, backtest, points FROM forecasts
                                  WHERE brain_path = p.brain_path AND sku = p.sku
                                  ORDER BY created_at DESC LIMIT 1) f ON true
               LEFT JOIN sales_daily t ON t.brain_path = p.brain_path AND t.sku = p.sku AND t.day = %s
               LEFT JOIN LATERAL (SELECT sum(units) AS units FROM sales_daily
                                  WHERE brain_path = p.brain_path AND sku = p.sku AND day > %s - 28) w ON true
               ORDER BY p.brain_path, p.sku""",
            (self.today, self.today),
        ).fetchall()
        for p in products:
            m = by_company.get(".".join(p["brain"].split(".")[:2]))
            if m is not None:
                m["products"].append(p)

        self.signal_rows = conn.execute(
            """SELECT s.*, o.result AS outcome_result, o.value AS outcome_value
               FROM signals s LEFT JOIN signal_outcomes o ON o.signal_id = s.id
               ORDER BY s.created_at DESC"""
        ).fetchall()
        for s in self.signal_rows:
            m = by_company.get(str(s["company"]))
            if m is not None and s["status"] == "open":
                m["signals"].append(s)
        for m in self.merchants.values():
            groups = Counter(GROUP.get(p["category"], "Home") for p in m["products"])
            m["category"] = groups.most_common(1)[0][0] if groups else "Home"

    # ---------------------------------------------------------------- pieces
    def scoped(self, platform: str | None = None, merchant: str | None = None) -> list[dict]:
        ms = list(self.merchants.values())
        if merchant:
            ms = [m for m in ms if m["id"] == merchant]
        if platform:
            ms = [m for m in ms if m["platform"] == platform]
        return ms

    @staticmethod
    def revenue(ms: list[dict]) -> float:
        return round(sum(float(p["revenue_today"]) for m in ms for p in m["products"]), 2)

    @staticmethod
    def units(ms: list[dict]) -> int:
        return int(sum(p["units_today"] for m in ms for p in m["products"]))

    @staticmethod
    def accuracy(ms: list[dict]) -> dict:
        """Volume-weighted backtest error of the chosen models, and of a seasonal-naive baseline."""
        num = base = weight = 0.0
        for m in ms:
            for p in m["products"]:
                q, bt, w = p["quality"] or {}, p["backtest"] or {}, float(p["units_28d"] or 0) or 1.0
                if q.get("wape") is None:
                    continue
                num += q["wape"] * w
                base += (bt.get("seasonal_naive", {}).get("wape") or q["wape"]) * w
                weight += w
        if not weight:
            return {"wape": None, "baseline_wape": None, "merchant_wape": None}
        return {"wape": round(num / weight, 4), "baseline_wape": round(base / weight, 4),
                "merchant_wape": round(num / weight, 4)}

    def signal(self, s: dict) -> dict:
        m = next((x for x in self.merchants.values() if x["company"] == str(s["company"])), None)
        d = s["details"] or {}
        sku = d.get("sku")
        p = next((x for x in (m["products"] if m else []) if x["sku"] == sku), None)
        units = d.get("quantity")
        price = float(d.get("price") or (p["base_price"] if p else 0) or 0)
        cost = float(p["unit_cost"]) if p and p["unit_cost"] is not None else None
        forecast_days = (p["points"] or [])[:7] if p else []
        daily = sum(x.get("forecast", 0) for x in forecast_days) / max(1, len(forecast_days)) if forecast_days else None
        kind = KIND.get(s["kind"], "restock")
        action = {"type": "restock", "units": units} if kind == "restock" else \
            {"type": "discount", "discount": 0.1, "hours": 72} if kind == "clearance" else {"type": "hold_price"}
        detail = " ".join(x for x in (s["action"], s["reason"]) if x)
        return {
            "id": str(s["id"]), "sent_ms": _ms(s["created_at"]),
            "sim_ts": (s["created_at"] or datetime.now()).strftime("%Y-%m-%dT%H:%M:%S"),
            "platform": PLATFORM["id"], "merchant": m["id"] if m else None, "merchant_name": m["name"] if m else "",
            "city": m["city"] if m else "", "sku": sku, "product": d.get("name") or (p["name"] if p else ""),
            "category": GROUP.get(p["category"], "Home") if p else (m["category"] if m else ""),
            "kind": kind, "urgency": s["urgency"], "title": s["title"], "detail": detail, "action": action,
            "expected": {"units": units, "revenue": round(float(s["impact_value"] or 0)),
                         "margin": round((price - cost) / price, 4) if price and cost is not None else None,
                         "uplift_units": None},
            "confidence": float(s["confidence"]) if s["confidence"] is not None else None,
            "evidence": {"actual": int(p["units_today"]) if p else None,
                         "forecast": round(daily, 1) if daily is not None else None, "deviation": None,
                         "stock": p["on_hand"] if p else None,
                         "cover_days": round(p["on_hand"] / daily, 1) if p and p["on_hand"] is not None and daily else None,
                         "elasticity": None, "facts": (s["evidence"] or {}).get("facts", [])},
            "impact": s["impact"], "due": s["due"].isoformat() if s["due"] else None, "status": s["status"],
            "latency_ms": None, "autopilot": False,
        }

    def open_signals(self, ms: list[dict]) -> list[dict]:
        return [s for m in ms for s in m["signals"]]

    def by_kind(self, rows: list[dict]) -> dict:
        return dict(Counter(KIND.get(s["kind"], "restock") for s in rows))

    def outcomes(self, ms: list[dict]) -> dict:
        companies = {m["company"] for m in ms}
        rows = [s for s in self.signal_rows if str(s["company"]) in companies]
        applied = [s for s in rows if s["status"] == "acted"]
        judged = [s for s in rows if s["outcome_result"] is not None]
        worked = [s for s in judged if s["outcome_result"] in ("worked", "hit", "good", "success")]
        uplift = sum(float(s["outcome_value"] or 0) for s in worked)
        return {"signals_total": len(rows), "offers_applied": len(applied), "uplift_revenue": round(uplift),
                "outcomes": len(judged), "hit_rate": (len(worked) / len(judged)) if judged else None}

    def merchant_row(self, m: dict) -> dict:
        return {"id": m["id"], "platform": m["platform"], "name": m["name"], "city": m["city"],
                "category": m["category"], "x": m["x"], "y": m["y"], "orders_today": self.units([m]),
                "revenue_today": self.revenue([m]), "open_signals": len(m["signals"])}

    def sim_ts(self) -> str:
        return datetime.combine(self.today, time(21, 0)).strftime("%Y-%m-%dT%H:%M:%S")


# ---------------------------------------------------------------- endpoints
def status(d: Data) -> dict:
    ms = d.scoped()
    acc, out = d.accuracy(ms), d.outcomes(ms)
    series = sum(len(m["products"]) for m in ms)
    fest = contract.next_festival(date.today())
    return {
        "sent_ms": _ms(datetime.now()), "sim_ts": d.sim_ts(), "speed": "realtime", "running": False, "source": "database",
        "merchants": len(ms), "customers": 0, "series": series,
        "orders_total": d.units(ms), "units_total": d.units(ms), "revenue_total": d.revenue(ms),
        "orders_per_sec": 0, "revenue_per_sec": 0, "msgs_per_sec": 0, "active_customers_today": 0,
        "signals_total": out["signals_total"], "signals_by_kind": d.by_kind(d.open_signals(ms)),
        "offers_applied": out["offers_applied"], "uplift_revenue": out["uplift_revenue"],
        "platforms": {PLATFORM["id"]: {"orders_total": d.units(ms), "revenue_total": d.revenue(ms), "orders_per_sec": 0}},
        "latency_ms": {"broker_p50": None, "broker_p95": None, "process_p50": None, "signal_p50": None,
                       "end_to_end_p50": None, "end_to_end_p95": None},
        "accuracy": {"wape": acc["wape"], "baseline_wape": acc["baseline_wape"], "days_scored": 168,
                     "series_scored": series, "by_platform": {PLATFORM["id"]: acc["wape"]}},
        "learning": {"outcomes": out["outcomes"], "hit_rate": out["hit_rate"], "elasticity": {}},
        "pipeline": {"ingested": 0, "forecasts_updated": series, "signals_raised": out["signals_total"], "delivered": 0},
        "festival": ({"name": fest[1], "date": fest[0].isoformat(), "days_to": (fest[0] - date.today()).days} if fest else None),
    }


def merchants(d: Data) -> list[dict]:
    return [d.merchant_row(m) for m in d.scoped()]


def series(d: Data, merchant_id: str, sku: str) -> dict | None:
    m = d.merchants.get(merchant_id)
    p = next((x for x in (m["products"] if m else []) if x["sku"] == sku), None)
    if p is None:
        return None
    return {"product": p["name"], "history": d.history.get((p["brain"], sku), []),
            "forecast": [{"day": x["day"], "f": x.get("forecast"), "lo": x.get("lo80"), "hi": x.get("hi80")}
                         for x in (p["points"] or [])[:14]],
            # The stored forecast starts the day after the latest sales day: there is no forecast for that day.
            "today": {"day": d.today.isoformat(), "actual": int(p["units_today"]),
                      "expected_so_far": None, "forecast": None},
            "offers": [], "model": p["model"], "wape": (p["quality"] or {}).get("wape")}


def merchant_detail(d: Data, merchant_id: str) -> dict | None:
    m = d.merchants.get(merchant_id)
    if m is None:
        return None
    products = []
    for p in m["products"]:
        pts = (p["points"] or [])[:14]
        daily = sum(x.get("forecast", 0) for x in pts) / len(pts) if pts else 0
        products.append({
            "sku": p["sku"], "name": p["name"], "category": GROUP.get(p["category"], "Home"),
            "price": float(p["base_price"] or 0), "cost": float(p["unit_cost"]) if p["unit_cost"] is not None else None,
            "stock": p["on_hand"], "lead_days": p["lead_time_days"], "perishable": p["category"] in ("Dairy", "Fruits & Vegetables"),
            "cover_days": round(p["on_hand"] / daily, 1) if daily and p["on_hand"] is not None else None,
            "today": {"day": d.today.isoformat(), "actual": int(p["units_today"]),
                      "expected_so_far": None, "forecast": None}})
    return {**d.merchant_row(m), "products": products,
            "open_signals": [d.signal(s) for s in m["signals"]], "outcomes": []}


def signals(d: Data, limit: int = 50, platform: str | None = None, merchant: str | None = None) -> list[dict]:
    rows = d.open_signals(d.scoped(platform, merchant))
    rows.sort(key=lambda s: s["created_at"], reverse=True)
    return [d.signal(s) for s in rows[:limit]]


def platforms(d: Data, platform: str | None = None, merchant: str | None = None) -> list[dict]:
    ms = d.scoped(platform, merchant)
    if not ms:
        return []
    acc, out = d.accuracy(ms), d.outcomes(ms)
    cats: dict[str, list[dict]] = defaultdict(list)
    for m in ms:
        cats[m["category"]].append(m)
    return [{**PLATFORM, "merchants": len(ms), "customers": 0, "series": sum(len(m["products"]) for m in ms),
             "orders_today": d.units(ms), "revenue_today": d.revenue(ms), "orders_per_sec": 0,
             "active_customers_today": 0, "accuracy": acc,
             "open_signals": len(d.open_signals(ms)), "signals_total": out["signals_total"],
             "signals_by_kind": d.by_kind(d.open_signals(ms)), "offers_applied": out["offers_applied"],
             "uplift_revenue": out["uplift_revenue"], "hit_rate": out["hit_rate"],
             "categories": sorted(({"name": c, "merchants": len(v), "revenue_today": d.revenue(v)} for c, v in cats.items()),
                                  key=lambda c: -c["revenue_today"])}]


def attention(d: Data, limit: int = 10, platform: str | None = None, merchant: str | None = None) -> list[dict]:
    rows = d.open_signals(d.scoped(platform, merchant))
    rows.sort(key=lambda s: (URGENCY_RANK.get(s["urgency"], 3), -float(s["impact_value"] or 0)))
    return [{**d.signal(s), "platform_name": PLATFORM["name"]} for s in rows[:limit]]


def tree(d: Data, platform: str | None = None, merchant: str | None = None) -> dict:
    ms = d.scoped(platform, merchant)
    cats: dict[str, list[dict]] = defaultdict(list)
    for m in ms:
        cats[m["category"]].append(m)

    def status_of(m: dict) -> str:
        if any(s["urgency"] == "high" for s in m["signals"]):
            return "urgent"
        return "signal" if m["signals"] else "ok"

    level = "merchant" if merchant else "platform" if platform else "hq"
    return {
        "root": {"id": "root", "name": "SecondBrain", "orders_today": d.units(ms), "revenue_today": d.revenue(ms),
                 "open_signals": len(d.open_signals(ms))},
        "platforms": [{"id": PLATFORM["id"], "name": PLATFORM["name"], "color": PLATFORM["color"],
                       "orders_today": d.units(ms), "revenue_today": d.revenue(ms),
                       "open_signals": len(d.open_signals(ms)),
                       "categories": [{"name": c, "merchants": [
                           {"id": m["id"], "name": m["name"], "city": m["city"], "category": m["category"],
                            "status": status_of(m), "open_signals": len(m["signals"]), "orders_today": d.units([m]),
                            "revenue_today": d.revenue([m]), "offers_applied": 0} for m in v]}
                           for c, v in sorted(cats.items())]}] if ms else [],
        "scope": {"level": level, "id": merchant or platform},
    }


def lessons(d: Data, conn: psycopg.Connection) -> list[dict]:
    """Lessons the brain has really learned (shared patterns at the root), newest first; none is an empty list."""
    rows = conn.execute(
        """SELECT id::text, content, source, created_at FROM memories
           WHERE brain_path = 'root' AND status = 'active' ORDER BY created_at DESC LIMIT 20""").fetchall()
    return [{"id": r["id"], "kind": "restock", "category": None, "platform": None,
             "title": r["content"].split("\n")[0][:160], "detail": r["content"], "value": None, "prior": None,
             "evidence_n": 1, "learned_from": [], "applies_to": [], "updated_sim_ts": r["created_at"].isoformat()}
            for r in rows]


def answer(conn: psycopg.Connection, path: str, params: dict) -> tuple[int, object]:
    """(status code, body) for a GET on the engine API path."""
    seg = [s for s in path.split("/") if s]
    d = Data(conn)
    if seg[:2] and seg[0] == "merchants" and len(seg) >= 2:
        if len(seg) >= 4 and seg[2] == "series":
            m = d.merchants.get(seg[1])
            if m:
                for r in conn.execute(
                        """SELECT brain_path::text AS brain, sku, day, units FROM sales_daily
                           WHERE brain_path <@ %s::ltree AND sku = %s AND day > %s - 42 ORDER BY day""",
                        (m["company"], seg[3], d.today)).fetchall():
                    d.history.setdefault((r["brain"], r["sku"]), []).append({"day": r["day"].isoformat(), "units": r["units"]})
            body = series(d, seg[1], seg[3])
        else:
            body = merchant_detail(d, seg[1])
        return (200, body) if body is not None else (404, {"detail": "Unknown merchant or product"})
    scope = {"platform": params.get("platform"), "merchant": params.get("merchant")}
    limit = int(params.get("limit") or 50)
    if seg == ["status"]:
        return 200, status(d)
    if seg == ["merchants"]:
        return 200, merchants(d)
    if seg == ["signals"]:
        return 200, signals(d, limit, **scope)
    if seg == ["platforms"]:
        return 200, platforms(d, **scope)
    if seg == ["attention"]:
        return 200, attention(d, int(params.get("limit") or 10), **scope)
    if seg == ["tree"]:
        return 200, tree(d, **scope)
    if seg == ["lessons"]:
        return 200, lessons(d, conn)
    return 404, {"detail": "Not found"}
