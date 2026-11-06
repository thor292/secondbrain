"""Streaming state of Live Commerce: who is connected, what they sell, and what happened today.

Arrays indexed by series (merchant x SKU) and by merchant, so the forecaster and the decision
layer work on all of them at once; dictionaries only map platform ids to those indexes.
Customers are kept per merchant (first and last day seen, visit days) for win-back.
"""

import numpy as np

SERIES_FIELDS = {"s_merch": np.int64, "s_plat": np.int64, "price": float, "cost": float, "stock": float,
                 "lead": float, "perishable": bool, "t_units": float, "t_rev": float, "t_orders": np.int64,
                 "t_out": bool}
MERCHANT_FIELDS = {"m_plat": np.int64, "m_orders_today": np.int64, "m_rev_today": float,
                   "m_orders_total": np.int64, "m_rev_total": float, "m_last_sent": np.int64,
                   "m_visits": np.int64, "m_interval": float}


def _grow(obj, fields: dict, n: int, cap: int) -> int:
    if n < cap:
        return cap
    new = max(2 * cap, 64)
    for name, dtype in fields.items():
        a = getattr(obj, name)
        setattr(obj, name, np.concatenate([a, np.zeros(new - len(a), dtype=dtype)]))
    return new


class Registry:
    def __init__(self):
        self.platform_ids: list[str] = []
        self.platform_index: dict[str, int] = {}
        self.platform_meta: dict[str, dict] = {}
        self.merchants: list[dict] = []
        self.merchant_index: dict[str, int] = {}
        self.series_index: dict[tuple[int, str], int] = {}
        self.series_of: list[list[int]] = []
        self.has_catalog: set[int] = set()
        self.sku: list[str] = []
        self.name: list[str] = []
        self.category: list[str] = []
        self.customers: list[dict[str, list[int]]] = []
        self.n = self.nm = 0
        self._cap = self._mcap = 0
        for name, dtype in {**SERIES_FIELDS, **MERCHANT_FIELDS}.items():
            setattr(self, name, np.zeros(0, dtype=dtype))

    def platform(self, pid: str, meta: dict | None = None) -> int:
        if pid not in self.platform_index:
            self.platform_index[pid] = len(self.platform_ids)
            self.platform_ids.append(pid)
            self.platform_meta[pid] = {"id": pid, "name": pid}
        if meta:
            self.platform_meta[pid] = meta
        return self.platform_index[pid]

    def merchant(self, mid: str, platform: str, meta: dict | None = None) -> int:
        m = self.merchant_index.get(mid)
        if m is None:
            self._mcap = _grow(self, MERCHANT_FIELDS, self.nm, self._mcap)
            m = self.merchant_index[mid] = self.nm
            self.nm += 1
            self.merchants.append({"id": mid, "platform": platform, "name": mid, "city": "", "category": "",
                                   "x": 0.5, "y": 0.5})
            self.series_of.append([])
            self.customers.append({})
            self.m_plat[m] = self.platform(platform)
        if meta:
            self.merchants[m].update({k: meta[k] for k in ("name", "city", "category", "x", "y") if k in meta})
        return m

    def series(self, m: int, product: dict) -> tuple[int, bool]:
        """Register or update a merchant's product. Returns (index, created)."""
        key = (m, product["sku"])
        i = self.series_index.get(key)
        created = i is None
        if created:
            self._cap = _grow(self, SERIES_FIELDS, self.n, self._cap)
            i = self.series_index[key] = self.n
            self.n += 1
            self.sku.append(product["sku"])
            self.name.append(product.get("name", product["sku"]))
            self.category.append(product.get("category", "Other"))
            self.series_of[m].append(i)
            self.s_merch[i] = m
            self.s_plat[i] = self.m_plat[m]
        else:
            self.name[i] = product.get("name", self.name[i])
        self.price[i] = float(product.get("price", self.price[i]))
        self.cost[i] = float(product.get("cost", self.price[i] * 0.75))
        if "stock" in product:
            self.stock[i] = float(product["stock"])
        self.lead[i] = float(product.get("lead_days", self.lead[i] or 3))
        self.perishable[i] = bool(product.get("perishable", False))
        return i, created

    def visit(self, m: int, customer: str, day: int) -> None:
        rec = self.customers[m].get(customer)
        if rec is None:
            self.customers[m][customer] = [day, day, 1]
            self.m_visits[m] += 1
        elif rec[1] != day:
            rec[1] = day
            rec[2] += 1
            self.m_visits[m] += 1

    def lapsed(self, m: int, today: int) -> tuple[int, float]:
        """Regulars now away for more than twice their usual interval (but not gone for good).

        A customer's usual interval is their own average gap shrunk toward the merchant's typical one,
        so a regular seen twice already counts (live data starts empty; waiting for a third visit would
        keep win-back silent for weeks). Also refreshes the merchant's typical interval.
        """
        recs = np.array(list(self.customers[m].values()), dtype=float).reshape(-1, 3)
        recs = recs[recs[:, 2] >= 2]
        if not len(recs):
            return 0, 0.0
        span, gaps = recs[:, 1] - recs[:, 0], recs[:, 2] - 1
        typical = float(np.median(span / gaps))
        self.m_interval[m] = typical
        usual = (span + typical) / (gaps + 1)
        away = today - recs[:, 1]
        lapsed = (usual >= 1.5) & (away > 2 * usual) & (away <= 6 * usual)
        return int(lapsed.sum()), float(usual[lapsed].mean()) if lapsed.any() else 0.0

    def new_day(self) -> None:
        n, nm = self.n, self.nm
        self.t_units[:n] = 0
        self.t_rev[:n] = 0
        self.t_orders[:n] = 0
        self.t_out[:n] = self.stock[:n] <= 0   # out of stock at some point today: no evidence of weak demand
        self.m_orders_today[:nm] = 0
        self.m_rev_today[:nm] = 0
