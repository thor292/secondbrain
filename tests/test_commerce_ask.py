"""Chat over the platforms (POST /commerce/ask): the scope filter removes other merchants' data before the
model sees it. The engine is faked with a feed that ignores the scope on purpose, so only the app's own
filter stands between the model and the other merchants; the model is the stub, wrapped to record its prompt."""
import pytest
from fastapi.testclient import TestClient

from secondbrain.api import create_app
from secondbrain.commerce import ask as commerce_chat
from secondbrain.commerce import contract

MERCHANTS = [
    {"id": "shopify-0001", "platform": "shopify", "name": "Desi Living", "city": "Bengaluru", "category": "Home Decor"},
    {"id": "shopify-0002", "platform": "shopify", "name": "Kurta Co", "city": "Jaipur", "category": "Fashion"},
    {"id": "dmart-0001", "platform": "dmart", "name": "DMart Powai", "city": "Mumbai", "category": "Hypermarket"},
]
NAMES = {"shopify": "Shopify", "dmart": "DMart"}


def signal(m, kind="restock", urgency="high", product="Brass Diya Set"):
    return {"id": f"sig-{m['id']}-{kind}", "sim_ts": "2026-11-06T10:00:00", "platform": m["platform"],
            "platform_name": NAMES[m["platform"]], "merchant": m["id"], "merchant_name": m["name"], "city": m["city"],
            "sku": "X1", "product": f"{product} {m['name']}", "category": "Home", "kind": kind, "urgency": urgency,
            "title": f"Restock {product} at {m['name']}", "detail": f"{m['name']} has 2 days of cover.",
            "action": {"type": "restock", "units": 40}, "expected": {"revenue": 45600.0}, "autopilot": False}


def platform(pid, n):
    return {"id": pid, "name": NAMES[pid], "kind": "", "color": "#000", "merchants": n, "customers": 900 * n,
            "series": 30 * n, "orders_today": 120 * n, "revenue_today": 345600.0 * n, "orders_per_sec": 2.0,
            "active_customers_today": 80 * n, "accuracy": {"wape": 0.3, "merchant_wape": 0.12, "baseline_wape": 0.45},
            "open_signals": n, "signals_total": 9 * n, "signals_by_kind": {"restock": 9 * n}, "offers_applied": 2 * n,
            "uplift_revenue": 12000.0 * n, "hit_rate": 0.8, "categories": [{"name": "Fashion", "merchants": n,
                                                                             "revenue_today": 1000.0}]}


def detail(m):
    return {**m, "orders_today": 12, "revenue_today": 45600.0, "orders_total": 300, "revenue_total": 1234567.0,
            "autopilot": False, "open_signals": [signal(m)],
            "products": [{"sku": "X1", "name": f"Brass Diya Set {m['name']}", "category": "Home", "price": 1299.0,
                          "cost": 700.0, "stock": 8.0, "lead_days": 5.0, "perishable": False, "cover_days": 2.0,
                          "today": {"actual": 3.0, "expected_so_far": 2.5, "forecast": 4.0}, "offer": False}],
            "outcomes": [{"discount": 0.1, "product": f"Cushion {m['name']}", "category": "Home",
                          "actual_uplift_units": 6.0, "assured_uplift_units": 2.0, "uplift_pct": 0.3,
                          "revenue_gain": 5400.0, "worked": True}]}


class LeakyEngine:
    """Answers every call with everyone's data, whatever the scope asked for."""

    def __init__(self):
        self.calls = []

    def __call__(self, path, params=None):
        self.calls.append((path, params))
        if path == "/merchants":
            return MERCHANTS
        if path.startswith("/merchants/"):
            return detail(next(m for m in MERCHANTS if m["id"] == path.rsplit("/", 1)[1]))
        if path == "/platforms":
            return [platform("shopify", 2), platform("dmart", 1)]
        if path in ("/attention", "/signals"):
            return [signal(m) for m in MERCHANTS]
        if path == "/lessons":
            return [{"id": "festival_lift:all:home", "kind": "festival_lift", "category": "Home", "platform": None,
                     "title": "Home sells 17% more on festival days", "detail": "Learned from Dussehra.", "value": 1.17,
                     "prior": 1.0, "evidence_n": 1, "learned_from": 39, "applies_to": 1, "updated_sim_ts": None},
                    {"id": "leak", "kind": "restock", "category": "Home", "platform": "shopify",
                     "title": "Kurta Co runs out often", "detail": "", "value": 0.2, "prior": None, "evidence_n": 4,
                     "learned_from": 1, "applies_to": 1, "updated_sim_ts": None}]
        if path == "/tree":
            return {"platforms": [{"categories": [{"merchants": [{**m, "offers_applied": 3} for m in MERCHANTS]}]}]}
        if path == "/status":
            return {"sim_ts": "2026-11-06T10:00:00", "festival": {"name": "Diwali", "date": "2026-11-08", "days_to": 2},
                    "revenue_total": 99999999.0}
        raise AssertionError(path)


@pytest.fixture()
def app_client(settings, monkeypatch):
    engine = LeakyEngine()
    monkeypatch.setattr(commerce_chat, "fetch_json", engine)
    app = create_app(settings)
    with TestClient(app) as client:
        llm = app.state.brain.llm
        prompts = []
        stub = llm.complete

        def spy(system, prompt):
            prompts.append((system, prompt))
            return stub(system, prompt)
        monkeypatch.setattr(llm, "complete", spy)
        client.prompts, client.engine = prompts, engine
        yield client


def ask(client, question, level="hq", sid=None):
    return client.post("/commerce/ask", json={"question": question, "scope": {"level": level, "id": sid}})


def test_merchant_scope_never_shows_the_model_another_merchant(app_client):
    r = ask(app_client, "What should Desi Living do for Diwali? Compare with Kurta Co and DMart Powai.",
            "merchant", "shopify-0001")
    assert r.status_code == 200, r.text
    system, prompt = app_client.prompts[-1]
    data, question = prompt.split("\nQuestion: ")
    assert "Kurta Co" in question                 # the user may type a name; the data about it never comes in
    seen = system + data
    for other in ("Kurta Co", "shopify-0002", "DMart Powai", "dmart-0001", "Jaipur", "99999999"):
        assert other not in seen, other
    assert "Desi Living" in prompt and "Brass Diya Set Desi Living" in prompt and "₹45,600" in prompt
    assert "Home sells 17% more on festival days" in prompt and "Kurta Co runs out often" not in prompt
    assert "answer only from the data" in system.lower() and "the merchant Desi Living" in system
    body = r.json()
    assert body["answer"] and body["scope"] == {"level": "merchant", "id": "shopify-0001",
                                                 "label": "the merchant Desi Living (Shopify, Bengaluru) only"}
    kinds = {s["kind"] for s in body["sources"]}
    assert kinds <= {"signal", "merchant", "platform", "lesson"} and "merchant" in kinds
    assert all(s["id"] in ("shopify-0001", "shopify", "sig-shopify-0001-restock", "festival_lift:all:home")
               for s in body["sources"])
    # the merchant's own detail is fetched; nobody else's ever is
    fetched = [p for p, _ in app_client.engine.calls if p.startswith("/merchants/")]
    assert fetched == ["/merchants/shopify-0001"]


def test_platform_scope_keeps_its_own_merchants_only(app_client):
    r = ask(app_client, "Which offers worked best on Shopify this week?", "platform", "shopify")
    assert r.status_code == 200, r.text
    prompt = app_client.prompts[-1][1].split("\nQuestion: ")[0]
    assert "Desi Living" in prompt and "Kurta Co" in prompt and "Cushion Kurta Co" in prompt
    assert "DMart Powai" not in prompt and "dmart-0001" not in prompt and "DMart:" not in prompt
    ids = {p for p, _ in app_client.engine.calls if p.startswith("/merchants/")}
    assert ids == {"/merchants/shopify-0001", "/merchants/shopify-0002"}


def test_hq_sees_everything_and_bad_scopes_are_refused(app_client):
    r = ask(app_client, "Which DMart stores need restocking today?")
    assert r.status_code == 200
    _, prompt = app_client.prompts[-1]
    assert all(m["name"] in prompt for m in MERCHANTS)
    assert ask(app_client, "hi", "merchant", "nobody").status_code == 404
    assert ask(app_client, "hi", "platform", None).status_code == 400
    assert ask(app_client, "   ").status_code == 400


def test_engine_down_is_503(settings, monkeypatch):
    monkeypatch.setattr(contract, "ENGINE_PORT", 9)   # nothing listens on the discard port
    with TestClient(create_app(settings)) as client:
        r = ask(client, "How are the platforms doing?")
    assert r.status_code == 503 and "not running" in r.json()["detail"]
