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

BASE = "http://localhost:8765"
NONE = None  # expect "not found"

# (feature id, person, question, expected fragment or NONE, note for the reader)
ASK = [
    # F1 Department answers
    ("F1", "nina", "Can I give a customer a 50 dollar credit without asking anyone?", "25", "Support Playbook"),
    ("F1", "nina", "What is the first response target for email tickets?", "4 business hours", "Support Playbook"),
    ("F1", "arjun", "Who counts as a VIP customer?", "2,000", "Support Playbook"),
    ("F1", "arjun", "Which issues go straight to the Head of Support?", "legal", "Support Playbook"),
    ("F1", "leo", "What is the theme of this year's holiday campaign?", "light up your home", "Holiday Campaign Plan (Word)"),
    ("F1", "sofia", "Which words should we avoid in our copy?", "cheap", "Brand Voice Guide"),
    ("F1", "daniel", "What time is the express UPS pickup?", "5", "Fulfillment SOP (PDF)"),
    ("F1", "ivy", "What do we do with damaged inbound stock?", "aisle q", "Fulfillment SOP (PDF)"),
    ("F1", "tom", "Up to what amount can a manager approve purchases?", "2,500", "Spending and Approval Policy"),
    ("F1", "hana", "How much is the referral bonus?", "1,500", "Hiring Process"),
    ("F1", "omar", "What is the maximum discount for a promotion?", "30", "Pricing and Discount Policy"),
    ("F1", "omar", "Who supplies our lighting?", "nordlys", "Assortment Notes"),
    # F2 Fallback to the company brain
    ("F2", "nina", "How many vacation days do I get?", "25", "Employee Handbook (Word)"),
    ("F2", "sofia", "What is the hotel cap for a manager?", "190", "Employee Handbook table (Word)"),
    ("F2", "tom", "How many sick days do we get?", "10", "Employee Handbook (Word)"),
    ("F2", "ivy", "How long is parental leave for primary caregivers?", "16 weeks", "Employee Handbook (Word)"),
    ("F2", "omar", "Which days are we in the Portland office?", "tuesday", "Employee Handbook (Word)"),
    ("F2", "hana", "How long do customers have to return an item?", "30 days", "Returns Policy (PDF)"),
    ("F2", "daniel", "What is the home office allowance for new employees?", "600", "Employee Handbook (Word)"),
    ("F2", "leo", "What are the company's values?", "customers first", "Company Overview"),
    ("F2", "tom", "What is our FY2026 revenue goal?", "61", "Company Overview"),
    ("F2", "arjun", "Which tool do we use for expenses?", "ramp", "Employee Handbook (Word)"),
    # F3 Restricted information: denied vs allowed
    ("F3", "tom", "What was gross margin in Q3?", NONE, "Finance member: restricted"),
    ("F3", "rachel", "What was gross margin in Q3?", "46", "CFO: Q3 Results (PDF)"),
    ("F3", "sofia", "What is the holiday paid media budget?", NONE, "Marketing member: restricted"),
    ("F3", "leo", "What is the holiday paid media budget?", "180,000", "Marketing lead: Marketing Budget"),
    ("F3", "ivy", "What discount do we get on UPS ground shipping?", NONE, "Operations member: restricted"),
    ("F3", "daniel", "What discount do we get on UPS ground shipping?", "42", "Operations lead: Carrier Contracts"),
    ("F3", "nina", "What is the salary range for a support agent?", NONE, "Support agent: People data, restricted"),
    ("F3", "hana", "What is the salary range for a support agent?", "48,000", "People lead: Compensation Bands"),
    ("F3", "omar", "What was Q3 revenue?", NONE, "Merchandiser: Finance restricted"),
    ("F3", "maya", "What was Q3 revenue?", "14.2", "CEO: Q3 Results (PDF)"),
    ("F3", "tom", "How much cash did we have at the end of Q3?", NONE, "Finance member: restricted"),
    ("F3", "rachel", "How much cash did we have at the end of Q3?", "9.8", "CFO: Q3 Results (PDF)"),
    ("F3", "sofia", "What was our customer acquisition cost in Q3?", NONE, "Marketing member: restricted"),
    ("F3", "leo", "What was our customer acquisition cost in Q3?", "38", "Marketing lead: Marketing Budget"),
    ("F3", "arjun", "What is the 2026 merit budget?", NONE, "Support lead: People restricted"),
    ("F3", "hana", "What is the 2026 merit budget?", "3.5", "People lead: Compensation Bands"),
    # F4 Access between departments (Nina may read Operations, Leo may read Merchandising)
    ("F4", "nina", "Why are UPS shipments to Chicago delayed?", "storm", "Daniel's Operations note"),
    ("F4", "sofia", "Why are UPS shipments to Chicago delayed?", NONE, "No access to Operations"),
    ("F4", "nina", "What time is the same-day order cutoff?", "2", "Fulfillment SOP (Operations)"),
    ("F4", "tom", "What time is the same-day order cutoff?", NONE, "No access to Operations"),
    ("F4", "nina", "Is the warehouse adding a second shift?", "second shift", "Daniel's Operations note"),
    ("F4", "arjun", "Is the warehouse adding a second shift?", NONE, "Support lead has no Operations grant"),
    ("F4", "leo", "When is the Aria linen sofa back in stock?", "december 2", "Omar's Merchandising note"),
    ("F4", "sofia", "When is the Aria linen sofa back in stock?", NONE, "No access to Merchandising"),
    ("F4", "leo", "What is the margin floor for promotions?", "35", "Pricing and Discount Policy"),
    ("F4", "sofia", "What is the margin floor for promotions?", NONE, "No access to Merchandising"),
    ("F4", "leo", "Who supplies our upholstery?", "casa forma", "Assortment Notes"),
    ("F4", "nina", "What happened with the damaged Halo lamps from Nordlys?", "40", "Ivy's Operations note"),
    # F5 Isolation between companies
    ("F5", "karim", "How many days of annual leave do we get?", "20", "Brightpath handbook"),
    ("F5", "maya", "How many days of annual leave do we get?", "25", "Lumen handbook"),
    ("F5", "karim", "Where are our warehouses?", "dallas", "Brightpath notes"),
    ("F5", "daniel", "Where are our warehouses?", "reno", "Lumen SOP"),
    ("F5", "karim", "What is the order cutoff for same-day shipping?", "noon", "Brightpath notes"),
    ("F5", "karim", "What is the free shipping threshold?", NONE, "Lumen data never visible"),
    ("F5", "karim", "Who is Priya Shah?", NONE, "Lumen customer, never visible"),
    ("F5", "karim", "What was Q3 revenue?", NONE, "Lumen finance, never visible"),
    ("F5", "maya", "How many client brands does Brightpath serve?", NONE, "Brightpath data never visible"),
    ("F5", "karim", "What is the hotel cap for a manager?", NONE, "Lumen handbook, never visible"),
    # F6 Reading PDF and Word documents
    ("F6", "maya", "What is the hotel cap for directors?", "240", "Table in Employee Handbook (Word)"),
    ("F6", "nina", "Do customers need to send back a damaged item?", "150", "Returns Policy (PDF)"),
    ("F6", "nina", "Can holiday orders be returned in January?", "january 31", "Returns Policy (PDF)"),
    ("F6", "nina", "Is store credit issued faster than a refund?", "instant", "Returns Policy (PDF)"),
    ("F6", "daniel", "How are fragile items packed?", "double", "Fulfillment SOP (PDF)"),
    ("F6", "leo", "When does the Black Friday sale run?", "november 27", "Holiday Campaign Plan (Word)"),
    ("F6", "leo", "What is the holiday revenue target?", "5.6", "Holiday Campaign Plan (Word)"),
    ("F6", "rachel", "What was the return rate in Q3?", "9.5", "Q3 Results (PDF)"),
    ("F6", "maya", "Up to what value can we accept gifts from suppliers?", "75", "Employee Handbook (Word)"),
    ("F6", "daniel", "Which carrier handles sofas and beds?", "hearth freight", "Fulfillment SOP (PDF)"),
]

# F7 conversation: (person, first question, follow-up, expected in follow-up answer)
FOLLOW_UPS = [
    ("nina", "How many vacation days do I get?", "and can I carry any over?", "5"),
    ("sofia", "What is the hotel cap for a manager?", "and for directors?", "240"),
    ("leo", "When does the Black Friday sale start?", "and when does it end?", "december 1"),
    ("nina", "How long do customers have to return items?", "what about holiday orders?", "january 31"),
    ("tom", "Up to what amount can a manager approve purchases?", "and a director?", "10,000"),
    ("daniel", "What time is the order cutoff for same-day shipping?", "and when is the express pickup?", "5"),
    ("hana", "How much is the referral bonus?", "how long does the new hire have to stay?", "90"),
    ("omar", "Who supplies our lighting?", "where are they based?", "denmark"),
    ("rachel", "What was Q3 revenue?", "and gross margin?", "46"),
    ("karim", "How many days of annual leave do we get?", "can we carry them over?", "no"),
]

# F8 remember: (note, expected memory type); all saved by Arjun, then recalled
NOTES = [
    ("Met the Bluebird Stores buyer today, they want delivery before December 10", "episodic"),
    ("The Bluebird buyer prefers WhatsApp over email", "user"),
    ("Our standard payment terms for wholesale orders are net 45", "semantic"),
    ("Called Zenith Hotels yesterday about a bulk order of 200 lamps", "episodic"),
    ("Marco from Zenith Hotels is the procurement manager and decides by Friday", "user"),
    ("Wholesale orders over 100 units get a 12 percent volume discount", "semantic"),
    ("The team ran a training session on the new returns flow this morning", "episodic"),
    ("Elena in the support team speaks Spanish and Portuguese", "user"),
    ("Chat support hours extend to 9pm during December", "semantic"),
    ("A customer threatened to post a negative review after a late delivery today", "episodic"),
]
RECALL = [
    ("arjun", "When does Bluebird need delivery?", "december 10"),
    ("arjun", "How does the Bluebird buyer like to be contacted?", "whatsapp"),
    ("arjun", "What are our payment terms for wholesale orders?", "45"),
    ("arjun", "Who decides on the Zenith Hotels order?", "marco"),
    ("arjun", "What volume discount do wholesale orders get?", "12"),
    ("arjun", "Which languages does Elena speak?", "spanish"),
    ("arjun", "Until what time is chat support open in December?", "9"),
    ("leo", "When does Bluebird need delivery?", NONE),  # Arjun's Support notes stay in Support
]

# F9 maintenance: (who writes the update, note, who asks, question, expected)
UPDATES = [
    ("arjun", "From today support agents can give goodwill credits up to 40 dollars without approval",
     "nina", "How much goodwill credit can I give without approval?", "40"),
    ("hana", "The referral bonus is increased to 2,000 dollars from December",
     "hana", "How much is the referral bonus?", "2,000"),
    ("daniel", "The same-day order cutoff moves to 3pm Pacific from next Monday",
     "daniel", "What time is the same-day order cutoff?", "3"),
    ("tom", "Month-end close now happens on the 3rd business day instead of the 4th",
     "tom", "On which business day are the books closed?", "3"),
    ("omar", "The maximum promotion discount is now 25 percent, except clearance",
     "omar", "What is the maximum discount for a promotion?", "25"),
    ("leo", "The holiday revenue target was raised to 6 million dollars",
     "leo", "What is the holiday revenue target?", "6 million"),
    ("arjun", "VIP status now starts at 1,500 dollars of lifetime spend",
     "arjun", "Who counts as a VIP customer?", "1,500"),
    ("daniel", "Fragile items like lamps are now triple boxed",
     "daniel", "How are fragile items packed?", "triple"),
]
PRELOADED_UPDATES = [
    ("nina", "What's the free shipping threshold right now?", "60", "Daniel's note replaced $75 (Nina may read Operations)"),
    ("sofia", "What's the free shipping threshold right now?", "75", "Sofia can't see the Operations note, keeps the policy"),
    ("leo", "When does loyalty early access for Black Friday start?", "november 23", "Leo's note replaced November 24"),
    ("omar", "When is the Aria linen sofa back in stock?", "december 2", "Omar's note replaced 'on backorder'"),
]


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

    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
