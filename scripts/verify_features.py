"""Run the feature checks from the testing guide against the running demo server.

Every question is asked through the real API and models, and the answer is
checked for the fact it must contain (or, for "none", that nothing was found).
Results go to a JSON file that scripts/build_feature_report.py turns into a PDF.

Changes the demo data (adds notes), so reset afterwards:
    uv run python -m secondbrain.schema --reset && ./scripts/load_examples.sh

    uv run python scripts/verify_features.py [--out verification.json]
"""

import argparse
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

import httpx
import psycopg

from secondbrain.config import get_settings

BASE = "http://localhost:8765"
NONE = None  # expect "not found"

# (feature id, person, question, expected fragment or NONE, note for the reader)
# Facts are from examples/desiliving and examples/bharatfleet (docs/INDIA_WORLD.md).
ASK = [
    # F1 Department answers
    ("F1", "neha", "Can I give a customer a ₹2,000 credit without asking anyone?", "500", "Support Playbook"),
    ("F1", "neha", "What is the first response target for email tickets?", "6 business hours", "Support Playbook"),
    ("F1", "arjun", "Who counts as a VIP customer?", "50,000", "Support Playbook"),
    ("F1", "arjun", "Which issues go straight to the Head of Support?", "legal", "Support Playbook"),
    ("F1", "rohan", "What is the theme of this year's Diwali campaign?", "ghar ki roshni", "Diwali Campaign Plan (Word)"),
    ("F1", "sana", "Which words should we avoid in our copy?", "cheap", "Brand Voice Guide"),
    ("F1", "vikram", "What time is the Blue Dart express pickup?", "6", "Fulfillment SOP (PDF)"),
    ("F1", "priya", "What do we do with damaged inbound stock?", "bay q", "Fulfillment SOP (PDF)"),
    ("F1", "tanvi", "Up to what amount can a manager approve purchases?", "2 lakh", "Spending and Approval Policy"),
    ("F1", "farah", "How much is the referral bonus?", "40,000", "Hiring Process"),
    ("F1", "kabir", "What is the maximum discount for a promotion?", "20", "Pricing and Discount Policy"),
    ("F1", "kabir", "Who supplies our brass lamps?", "noor metalcraft", "Assortment Notes"),
    # F2 Fallback to the company brain
    ("F2", "neha", "How many days of earned leave do I get?", "18", "Employee Handbook (Word)"),
    ("F2", "sana", "What is the hotel cap for a manager?", "6,500", "Employee Handbook table (Word)"),
    ("F2", "tanvi", "How many sick days do we get?", "10", "Employee Handbook (Word)"),
    ("F2", "priya", "How long is maternity leave?", "26 weeks", "Employee Handbook (Word)"),
    ("F2", "kabir", "Which days are we in the Indiranagar office?", "tuesday", "Employee Handbook (Word)"),
    ("F2", "farah", "How long do customers have to return an item?", "10 days", "Returns Policy (PDF)"),
    ("F2", "vikram", "What is the home office allowance for new employees?", "25,000", "Employee Handbook (Word)"),
    ("F2", "rohan", "What are the company's values?", "customers first", "Company Overview"),
    ("F2", "tanvi", "What is our FY2026-27 revenue goal?", "51", "Company Overview"),
    ("F2", "arjun", "Which tool do we use for expenses?", "zoho expense", "Employee Handbook (Word)"),
    # F3 Restricted information: denied vs allowed
    ("F3", "tanvi", "What was gross margin in Q2 FY27?", NONE, "Finance member: restricted"),
    ("F3", "anand", "What was gross margin in Q2 FY27?", "47", "CFO: Q2 FY27 Results (PDF)"),
    ("F3", "sana", "What is the Diwali paid media budget?", NONE, "Marketing member: restricted"),
    ("F3", "rohan", "What is the Diwali paid media budget?", "1.6", "Marketing lead: Marketing Budget"),
    ("F3", "priya", "What does BharatFleet charge per kg in Zone 1?", NONE, "Operations member: restricted"),
    ("F3", "vikram", "What does BharatFleet charge per kg in Zone 1?", "16", "Operations lead: Carrier Contracts"),
    ("F3", "neha", "What is the salary range for a support agent?", NONE, "Support agent: People data, restricted"),
    ("F3", "farah", "What is the salary range for a support agent?", "3.6", "People lead: Compensation Bands"),
    ("F3", "kabir", "What was Q2 FY27 revenue?", NONE, "Merchandiser: Finance restricted"),
    ("F3", "meera", "What was Q2 FY27 revenue?", "11.9", "CEO: Q2 FY27 Results (PDF)"),
    ("F3", "tanvi", "How much cash did we have at the end of September?", NONE, "Finance member: restricted"),
    ("F3", "anand", "How much cash did we have at the end of September?", "14.6", "CFO: Q2 FY27 Results (PDF)"),
    ("F3", "sana", "What was our customer acquisition cost in Q2 FY27?", NONE, "Marketing member: restricted"),
    ("F3", "rohan", "What was our customer acquisition cost in Q2 FY27?", "640", "Marketing lead: Marketing Budget"),
    ("F3", "arjun", "What is the FY2026-27 merit budget?", NONE, "Support lead: People restricted"),
    ("F3", "farah", "What is the FY2026-27 merit budget?", "9", "People lead: Compensation Bands"),
    # F4 Access between departments (Neha may read Operations, Rohan may read Merchandising)
    ("F4", "neha", "Why are Delhivery parcels to Delhi NCR delayed?", "2 days", "Vikram's Operations note"),
    ("F4", "sana", "Why are Delhivery parcels to Delhi NCR delayed?", NONE, "No access to Operations"),
    ("F4", "neha", "What time is the same-day order cutoff?", "1", "Fulfillment SOP (Operations)"),
    ("F4", "tanvi", "What time is the same-day order cutoff?", NONE, "No access to Operations"),
    ("F4", "neha", "Is the Hoskote warehouse keeping the second shift?", "second shift", "Vikram's Operations note"),
    ("F4", "arjun", "Is the Hoskote warehouse keeping the second shift?", NONE, "Support lead has no Operations grant"),
    ("F4", "rohan", "When is the sheesham sofa back in stock?", "december", "Kabir's Merchandising note"),
    ("F4", "sana", "When is the sheesham sofa back in stock?", NONE, "No access to Merchandising"),
    ("F4", "rohan", "What is the margin floor for promotions?", "35", "Pricing and Discount Policy"),
    ("F4", "sana", "What is the margin floor for promotions?", NONE, "No access to Merchandising"),
    ("F4", "rohan", "Who makes our sheesham furniture?", "saharanpur woodcraft", "Assortment Notes"),
    ("F4", "neha", "What happened with the damaged brass floor lamps from Noor Metalcraft?", "36", "Priya's Operations note"),
    # F5 Isolation between companies
    ("F5", "karan", "How many days of earned leave do we get?", "15", "BharatFleet handbook"),
    ("F5", "meera", "How many days of earned leave do we get?", "18", "Desi Living handbook"),
    ("F5", "karan", "Where are our hubs?", "chakan", "BharatFleet notes"),
    ("F5", "vikram", "Where are our warehouses?", "hoskote", "Desi Living SOP"),
    ("F5", "karan", "What is the cutoff for same-day pickups?", "11", "BharatFleet notes"),
    ("F5", "karan", "What is the free shipping threshold?", NONE, "Desi Living data never visible"),
    ("F5", "karan", "Who is Ananya Reddy?", NONE, "Desi Living customer, never visible"),
    ("F5", "karan", "What was Q2 FY27 revenue?", NONE, "Desi Living finance, never visible"),
    ("F5", "meera", "How many client brands does BharatFleet serve?", NONE, "BharatFleet data never visible"),
    ("F5", "karan", "What is the hotel cap for a manager?", NONE, "Desi Living handbook, never visible"),
    # F6 Reading PDF and Word documents
    ("F6", "meera", "What is the hotel cap for department heads?", "9,000", "Table in Employee Handbook (Word)"),
    ("F6", "neha", "Do customers need to send back a damaged item?", "3,000", "Returns Policy (PDF)"),
    ("F6", "neha", "Until when can Diwali festive orders be returned?", "30", "Returns Policy (PDF)"),
    ("F6", "neha", "Is store credit issued faster than a refund?", "instant", "Returns Policy (PDF)"),
    ("F6", "vikram", "How are fragile items packed?", "double", "Fulfillment SOP (PDF)"),
    ("F6", "rohan", "When does the Festive Sale run?", "12", "Diwali Campaign Plan (Word)"),
    ("F6", "rohan", "What is the festive revenue target?", "9.5", "Diwali Campaign Plan (Word)"),
    ("F6", "anand", "What was the return rate in Q2 FY27?", "8.4", "Q2 FY27 Results (PDF)"),
    ("F6", "meera", "Up to what value can we accept gifts from suppliers?", "2,000", "Employee Handbook (Word)"),
    ("F6", "vikram", "Which carrier handles sofas and side tables?", "bharatfleet", "Fulfillment SOP (PDF)"),
]

# F7 conversation: (person, first question, follow-up, expected in follow-up answer)
FOLLOW_UPS = [
    ("neha", "How many days of earned leave do I get?", "and can I carry any over?", "10"),
    ("sana", "What is the hotel cap for a manager?", "and for department heads?", "9,000"),
    ("rohan", "When does the Festive Sale start?", "and when does it end?", "12"),
    ("neha", "How long do customers have to return items?", "what about festive orders?", "30"),
    ("tanvi", "Up to what amount can a manager approve purchases?", "and a department head?", "10 lakh"),
    ("vikram", "What time is the order cutoff for same-day shipping?", "and when is the Blue Dart express pickup?", "6"),
    ("farah", "How much is the referral bonus?", "how long does the new hire have to stay?", "90"),
    ("kabir", "Who supplies our brass lamps?", "where are they based?", "moradabad"),
    ("anand", "What was Q2 FY27 revenue?", "and gross margin?", "47"),
    ("karan", "How many days of earned leave do we get?", "can we carry them over?", "no"),
]

# F8 remember: (note, expected memory type); all saved by Arjun, then recalled
NOTES = [
    ("Met the Kaveri Stores buyer today, they want delivery before 10 December", "episodic"),
    ("The Kaveri Stores buyer prefers WhatsApp over email", "user"),
    ("Our standard payment terms for wholesale orders are net 45", "semantic"),
    ("Called Lotus Hotels yesterday about a bulk order of 200 lamps", "episodic"),
    ("Rahul from Lotus Hotels is the procurement manager and decides by Friday", "user"),
    ("Wholesale orders over 100 units get a 12 percent volume discount", "semantic"),
    ("The team ran a training session on the new returns flow this morning", "episodic"),
    ("Divya in the support team speaks Tamil and Telugu", "user"),
    ("WhatsApp support hours extend to 10pm during the wedding season", "semantic"),
    ("A customer threatened to post a negative review after a late delivery today", "episodic"),
]
RECALL = [
    ("arjun", "When does Kaveri Stores need delivery?", "december"),
    ("arjun", "How does the Kaveri Stores buyer like to be contacted?", "whatsapp"),
    ("arjun", "What are our payment terms for wholesale orders?", "45"),
    ("arjun", "Who decides on the Lotus Hotels order?", "rahul"),
    ("arjun", "What volume discount do wholesale orders get?", "12"),
    ("arjun", "Which languages does Divya speak?", "tamil"),
    ("arjun", "Until what time is WhatsApp support open in the wedding season?", "10"),
    ("rohan", "When does Kaveri Stores need delivery?", NONE),  # Arjun's Support notes stay in Support
]

# F9 maintenance: (who writes the update, note, who asks, question, expected)
UPDATES = [
    ("arjun", "From today support agents can give goodwill credits up to ₹1,000 without approval",
     "neha", "How much goodwill credit can I give without approval?", "1,000"),
    ("farah", "The referral bonus is increased to ₹50,000 from December",
     "farah", "How much is the referral bonus?", "50,000"),
    ("vikram", "The same-day order cutoff moves to 2pm IST from next Monday",
     "vikram", "What time is the same-day order cutoff?", "2"),
    ("tanvi", "Month-end close now happens on the 4th business day instead of the 5th",
     "tanvi", "On which business day are the books closed?", "4"),
    ("kabir", "The maximum promotion discount is now 25 percent, except clearance",
     "kabir", "What is the maximum discount for a promotion?", "25"),
    ("rohan", "The festive revenue target was raised to ₹10.5 crore",
     "rohan", "What is the festive revenue target?", "10.5"),
    ("arjun", "VIP status now starts at ₹40,000 of lifetime spend",
     "arjun", "Who counts as a VIP customer?", "40,000"),
    ("vikram", "Fragile items like lamps are now triple boxed",
     "vikram", "How are fragile items packed?", "triple"),
]
PRELOADED_UPDATES = [
    ("neha", "What's the free shipping threshold right now?", "699", "Vikram's note replaced ₹999 (Neha may read Operations)"),
    ("sana", "What's the free shipping threshold right now?", "999", "Sana can't see the Operations note, keeps the policy"),
    ("rohan", "When does Ghar Club early access to the Wedding Edit start?", "12", "Rohan's note replaced 14 November"),
    ("kabir", "When is the sheesham sofa back in stock?", "december", "Kabir's note replaced 'on backorder'"),
]


LESSON_NAMES = ["desiliving", "desi living", "bharatfleet", "neha", "arjun", "ananya", "reddy", "noor", "metalcraft",
                "saharanpur", "sanganer", "panipat", "khurja", "moradabad", "hoskote", "bhiwandi", "bengaluru",
                "delhivery", "blue dart", "meera", "anand", "vikram", "kaveri", "lotus", "jiomart", "dmart"]
DESI_LIVING_PEOPLE = {"meera", "arjun", "neha", "rohan", "sana", "vikram", "priya", "anand", "tanvi", "farah", "kabir"}


def run_checks(http: httpx.Client) -> tuple[list[dict], list[str]]:
    """Checks that aren't chat questions: learning, graph views, database security,
    relationships and forecasting. Returns (checks, lessons learned at the root)."""
    settings = get_settings()
    token = {"X-Admin-Token": settings.admin_token}
    checks: list[dict] = []

    def add(feature, what, expected, got, ok):
        checks.append({"feature": feature, "check": what, "expected": expected, "got": str(got), "passed": bool(ok)})

    def get(path, user, **params):
        response = http.get(path, params=params, headers={"X-User-Id": user})
        return response.status_code, (response.json() if response.status_code == 200 else None)

    # F10 Learning at the root
    with psycopg.connect(settings.admin_database_url) as conn:
        lessons = [r[0] for r in conn.execute(
            "SELECT content FROM memories WHERE brain_path = 'root' AND source = 'learned pattern' ORDER BY created_at")]
        links = conn.execute("SELECT count(*) FROM memories WHERE brain_path = 'root' AND created_by IS NOT NULL").fetchone()[0]
        sharing = dict(conn.execute("SELECT path::text, sharing FROM brains WHERE kind = 'company'").fetchall())
    add("F10", "Lessons stored at the root", "at least 5", len(lessons), len(lessons) >= 5)
    digits = [l for l in lessons if re.search(r"\d", l)]
    add("F10", "Lessons containing any digit", "0", len(digits), not digits)
    found = [n for n in LESSON_NAMES if any(re.search(rf"\b{n}\b", l.lower()) for l in lessons)]
    add("F10", "Lessons containing a company, person, customer, supplier, product or place name", "none", found or "none", not found)
    add("F10", "Lessons linked to the person or company they came from", "0", links, links == 0)
    add("F10", "BharatFleet sharing setting (contributes nothing)", "none", sharing.get("root.bharatfleet"), sharing.get("root.bharatfleet") == "none")
    add("F10", "Desi Living sharing setting", "patterns", sharing.get("root.desiliving"), sharing.get("root.desiliving") == "patterns")
    http.post("/conversation/reset", headers={"X-User-Id": "neha"})
    answer = http.post("/ask", json={"question": "Should we lower the free-shipping threshold during peak season?"},
                       headers={"X-User-Id": "neha"}).json()
    from_root = [s_ for s_ in answer["sources"] if s_["brain"] == "root"]
    add("F10", "Company user's answer uses any root memory", "no", "yes" if from_root else "no", not from_root)
    status = http.post("/admin/learn").status_code
    add("F10", "Running learning without the admin token", "refused (401)", status, status == 401)

    # F11 Graph views
    def graph(user):
        return http.get("/graph", headers={"X-User-Id": user}).json()

    def memories(g):
        return [n for n in g["nodes"] if n["id"].startswith("memory:")]

    anand, sana, karan, tanvi, rohan = (graph(u) for u in ("anand", "sana", "karan", "tanvi", "rohan"))
    restricted = lambda g: [n for n in memories(g) if n["props"].get("sensitivity") == "restricted"]
    add("F11", "Anand (CFO): restricted memories in his graph", "more than 0", len(restricted(anand)), restricted(anand))
    add("F11", "Sana (content marketer): restricted memories in her graph", "0", len(restricted(sana)), not restricted(sana))
    add("F11", "Sana sees fewer memories than Anand", "fewer", f"{len(memories(sana))} vs {len(memories(anand))}",
        len(memories(sana)) < len(memories(anand)))
    rohan_restricted = sorted({n["props"]["brain"] for n in restricted(rohan)})
    add("F11", "Rohan (lead) sees restricted memories only from Marketing", "root.desiliving.marketing", rohan_restricted,
        rohan_restricted == ["root.desiliving.marketing"])
    add("F11", "Karan's graph scope", "root.bharatfleet", karan["scope"], karan["scope"] == "root.bharatfleet")
    leaked = [n for n in karan["nodes"] if "desiliving" in json.dumps(n).lower()]
    add("F11", "Desi Living nodes in Karan's graph", "0", len(leaked), not leaked)
    tanvi_brains = sorted({n["props"]["brain"] for n in memories(tanvi)})
    add("F11", "Brains whose memories Tanvi sees", "company + Finance only", tanvi_brains,
        set(tanvi_brains) <= {"root.desiliving", "root.desiliving.finance"})
    status = http.get("/admin/graph").status_code
    add("F11", "Whole-SecondBrain view without the token", "refused (401)", status, status == 401)
    full = http.get("/admin/graph", headers=token).json()
    brains = {n["id"] for n in full["nodes"] if n["id"].startswith("brain:")}
    add("F11", "Whole view contains root, both companies and all departments", "10 brains", len(brains),
        {"brain:root", "brain:root.desiliving", "brain:root.bharatfleet"} <= brains and len(brains) == 10)
    links_ = [e for e in full["edges"] if e["label"] in ("REPLACED_BY", "PARTLY_UPDATED_BY", "DUPLICATE_OF")]
    add("F11", "Update links (replaced / partly updated / duplicate) in the whole view", "at least 4", len(links_), len(links_) >= 4)

    # F12 Security enforced by the database
    with psycopg.connect(settings.app_database_url) as conn:
        visible = conn.execute("SELECT count(*) FROM memories").fetchone()[0]
        add("F12", "App database login with no person set: memories visible", "0", visible, visible == 0)
        conn.execute("SELECT set_config('app.user_id', 'sana', true)")
        paths = sorted(r[0] for r in conn.execute("SELECT DISTINCT brain_path::text FROM memories"))
        add("F12", "Same login as Sana: brains visible", "root.desiliving, root.desiliving.marketing", paths,
            paths == ["root.desiliving", "root.desiliving.marketing"])
        hidden = conn.execute("SELECT count(*) FROM memories WHERE sensitivity = 2").fetchone()[0]
        add("F12", "Same login as Sana: restricted memories visible", "0", hidden, hidden == 0)
        try:
            conn.execute("SELECT count(*) FROM grants")
            ok, got = False, "readable"
        except psycopg.errors.InsufficientPrivilege:
            ok, got = True, "permission denied"
        add("F12", "App login reading everyone's permissions table", "permission denied", got, ok)
    with psycopg.connect(settings.app_database_url) as conn:
        conn.execute("SELECT set_config('app.user_id', 'sana', true)")
        try:
            conn.execute(
                "INSERT INTO memories (brain_path, memory_type, tier, sensitivity, content, embedding, embedding_model, created_by)"
                f" VALUES ('root.desiliving.finance', 'semantic', 'long_term', 1, 'x', array_fill(0.1, ARRAY[{settings.embedding_dim}])::vector, 'x', 'sana')")
            ok, got = False, "inserted"
        except psycopg.errors.InsufficientPrivilege:
            ok, got = True, "blocked by row-level security"
        add("F12", "Sana writing a memory into Finance", "blocked", got, ok)
    status = http.post("/ask", json={"question": "hi"}, headers={"X-User-Id": "nobody"}).status_code
    add("F12", "Unknown person asking a question", "refused (401)", status, status == 401)

    # F13 Relationships (FalkorDB)
    people = lambda body: [p["id"] for p in (body or {}).get("people", [])]
    _, body = get("/experts", "neha", topic="Ananya Reddy")
    add("F13", "Neha: who knows about customer Ananya Reddy?", "Neha", people(body), "neha" in people(body))
    _, body = get("/experts", "sana", topic="Ananya Reddy")
    add("F13", "Sana: who knows about Ananya Reddy? (Support is hidden from her)", "nobody", people(body) or "nobody", not people(body))
    _, body = get("/experts", "meera", topic="Diwali")
    add("F13", "Meera (CEO): who knows about Diwali?", "includes Rohan", people(body), "rohan" in people(body))
    _, body = get("/experts", "tanvi", topic="Diwali")
    add("F13", "Tanvi: who knows about Diwali? (the CFO's board note is restricted)", "not Anand",
        people(body) or "nobody", "anand" not in people(body))
    _, body = get("/experts", "karan", topic="Diwali")
    add("F13", "Karan (BharatFleet): Desi Living people who know about Diwali", "nobody",
        sorted(set(people(body)) & DESI_LIVING_PEOPLE) or "nobody", not set(people(body)) & DESI_LIVING_PEOPLE)
    code, body = get("/entities/Noor Metalcraft", "kabir")
    add("F13", "Kabir: what do we know about Noor Metalcraft?", "a supplier", (body or {}).get("kind", code),
        code == 200 and body["kind"] == "supplier")
    code, _ = get("/entities/Noor Metalcraft", "karan")
    add("F13", "Karan: what do we know about Noor Metalcraft? (other company)", "not found (404)", code, code == 404)
    code, body = get("/entities/sheesham sofa", "rohan")
    suppliers = [r["entity"]["name"] for r in (body or {}).get("relations", []) if "suppl" in r["relation"] or "make" in r["relation"]]
    add("F13", "Rohan: who supplies the sheesham sofa?", "Saharanpur Woodcraft", suppliers or code,
        any("saharanpur" in s_.lower() for s_ in suppliers))
    _, meera_bf = get("/entities/BharatFleet Logistics", "meera")
    _, neha_bf = get("/entities/BharatFleet Logistics", "neha")
    counts = ((meera_bf or {}).get("memory_count", 0), (neha_bf or {}).get("memory_count", 0))
    add("F13", "BharatFleet memories: Meera vs Neha (restricted contract hidden from Neha)", "Meera sees more",
        f"{counts[0]} vs {counts[1]}", counts[0] > counts[1] > 0)
    code_a, _ = get("/entities/gross margin", "anand")
    code_t, _ = get("/entities/gross margin", "tanvi")
    add("F13", "Entity 'gross margin': Anand vs Tanvi (restricted Finance)", "200 vs 404", f"{code_a} vs {code_t}",
        code_a == 200 and code_t == 404)
    leaked_text = any(t in json.dumps(neha_bf or {}, ensure_ascii=False).lower() for t in ("carrier contract", "per kg"))
    add("F13", "Restricted contract text in Neha's BharatFleet answer", "absent", "present" if leaked_text else "absent", not leaked_text)
    status = http.get("/experts", params={"topic": "x"}, headers={"X-User-Id": "nobody"}).status_code
    add("F13", "Unknown person asking who knows what", "refused (401)", status, status == 401)

    # F14 Forecasting
    code, products = get("/forecast/products", "kabir")
    skus = [p["sku"] for p in products or []]
    add("F14", "Kabir: products with forecasts", "8", len(skus), len(skus) == 8)
    code, fc = get("/forecast", "kabir", product="brass-floor-lamp", horizon=30)
    days = (fc or {}).get("forecast", [])
    add("F14", "Brass floor lamp forecast for 30 days", "30 days", len(days), len(days) == 30)
    ordered = all(d["lo95"] <= d["lo80"] <= d["forecast"] <= d["hi80"] <= d["hi95"] for d in days)
    add("F14", "Every day has 80% and 95% ranges around the forecast", "yes", "yes" if ordered else "no", days and ordered)
    backtest = (fc or {}).get("backtest", {})
    best = min(backtest, key=lambda m: backtest[m]["wape"]) if backtest else None
    add("F14", "Chosen model is the one with the lowest backtest error", best, (fc or {}).get("model"), best and fc["model"] == best)
    add("F14", "Models compared in the backtest", "4", sorted(backtest), len(backtest) == 4)
    windows = set().union(*(set(m["windows"]) for m in backtest.values())) if backtest else set()
    add("F14", "Backtest includes the same weeks last year", "yes", sorted(windows), "same weeks last year" in windows)
    add("F14", "Forecast explains why the model was chosen", "mentions WAPE", (fc or {}).get("reason", "")[:90],
        "wape" in (fc or {}).get("reason", "").lower())
    # Brass is bought on Dhanteras itself (6 November 2026); the weeks after Diwali are a lull.
    dhanteras = [d["forecast"] for d in days if "2026-11-06" <= d["day"] <= "2026-11-07"]
    after = [d["forecast"] for d in days if "2026-11-15" <= d["day"] <= "2026-11-30"]
    lift = (max(dhanteras) / (sum(after) / len(after))) if dhanteras and after else 0
    add("F14", "Dhanteras peak vs the second half of November (brass floor lamp)", "clearly higher", f"{lift:.1f}x", lift > 1.5)
    for user in ("meera", "rohan"):
        code, _ = get("/forecast", user, product="brass-floor-lamp", horizon=7)
        add("F14", f"{user.capitalize()} can see forecasts (Merchandising access)", "200", code, code == 200)
    for user in ("neha", "sana", "karan"):
        code, _ = get("/forecast", user, product="brass-floor-lamp", horizon=7)
        add("F14", f"{user.capitalize()} asking for a forecast (no access)", "not found (404)", code, code == 404)

    return checks, lessons


def check(answer: dict, expected) -> bool:
    if expected is NONE:
        return answer["answered_from"] == "none"
    text = answer["answer"].lower().replace("**", "")
    return answer["answered_from"] != "none" and expected.lower() in text


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="verification.json")
    args = parser.parse_args()
    http = httpx.Client(base_url=BASE, timeout=180)

    def ask(user, question, fresh=True):
        if fresh:
            http.post("/conversation/reset", headers={"X-User-Id": user})
        return http.post("/ask", json={"question": question}, headers={"X-User-Id": user}).json()

    results = {"verified_at": datetime.now(timezone.utc).isoformat(timespec="minutes"),
               "model": http.get("/health").json(), "ask": [], "follow_ups": [], "notes": [],
               "recall": [], "updates": [], "preloaded_updates": [], "checks": []}

    # Independent questions, one thread per person so conversations don't mix.
    by_user: dict[str, list] = {}
    for item in ASK:
        by_user.setdefault(item[1], []).append(item)

    def run_user(items):
        out = []
        for feature, user, question, expected, note in items:
            answer = ask(user, question)
            out.append({"feature": feature, "user": user, "question": question, "expected": expected,
                        "note": note, "answer": answer["answer"], "from": answer["answered_from"],
                        "passed": check(answer, expected)})
        return out

    with ThreadPoolExecutor(max_workers=8) as pool:
        for chunk in pool.map(run_user, by_user.values()):
            results["ask"] += chunk
    results["ask"].sort(key=lambda r: [i for i, a in enumerate(ASK) if a[2] == r["question"] and a[1] == r["user"]][0])
    print("ask:", sum(r["passed"] for r in results["ask"]), "/", len(results["ask"]))

    for user, first, follow, expected in FOLLOW_UPS:
        a1 = ask(user, first)
        a2 = ask(user, follow, fresh=False)
        results["follow_ups"].append({"user": user, "first": first, "first_answer": a1["answer"],
                                      "follow_up": follow, "understood_as": a2["searched_for"],
                                      "answer": a2["answer"], "expected": expected, "passed": check(a2, expected)})
    print("follow-ups:", sum(r["passed"] for r in results["follow_ups"]), "/", len(FOLLOW_UPS))

    for user, question, expected, note in PRELOADED_UPDATES:
        answer = ask(user, question)
        results["preloaded_updates"].append({"user": user, "question": question, "expected": expected, "note": note,
                                             "answer": answer["answer"], "passed": check(answer, expected)})

    for note, expected_type in NOTES:
        item = http.post("/remember", json={"content": note}, headers={"X-User-Id": "arjun"}).json()
        results["notes"].append({"note": note, "expected": expected_type, "got": item["memory_type"],
                                 "passed": item["memory_type"] == expected_type})
    http.post("/consolidate", headers={"X-User-Id": "arjun"})
    time.sleep(20)  # let background maintenance finish
    for user, question, expected in RECALL:
        answer = ask(user, question)
        results["recall"].append({"user": user, "question": question, "expected": expected,
                                  "answer": answer["answer"], "passed": check(answer, expected)})
    print("notes:", sum(r["passed"] for r in results["notes"]), "/", len(NOTES),
          "recall:", sum(r["passed"] for r in results["recall"]), "/", len(RECALL))

    for writer, note, user, question, expected in UPDATES:
        before = ask(user, question)
        http.post("/remember", json={"content": note}, headers={"X-User-Id": writer})
        http.post("/consolidate", headers={"X-User-Id": writer})
        time.sleep(12)  # background maintenance
        after = ask(user, question)
        results["updates"].append({"writer": writer, "note": note, "user": user, "question": question,
                                   "before": before["answer"], "after": after["answer"], "expected": expected,
                                   "passed": check(after, expected)})
    print("updates:", sum(r["passed"] for r in results["updates"]), "/", len(UPDATES))

    results["checks"], results["lessons"] = run_checks(http)
    print("checks:", sum(c["passed"] for c in results["checks"]), "/", len(results["checks"]))

    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
