"""Build the features-and-testing PDF from a verification run.

    uv run python scripts/verify_features.py --out verification.json
    uv run python scripts/build_feature_report.py verification.json docs/SecondBrain_Features_and_Testing.pdf
"""

import json
import re
import sys
from datetime import datetime

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

INK, MUTED, ACCENT = colors.HexColor("#1d2330"), colors.HexColor("#667085"), colors.HexColor("#4f46e5")
SOFT, LINE = colors.HexColor("#eef0ff"), colors.HexColor("#d0d5dd")
GREEN, RED = colors.HexColor("#0f7a3d"), colors.HexColor("#b42318")

ss = getSampleStyleSheet()
H1 = ParagraphStyle("H1", parent=ss["Heading1"], fontSize=16, textColor=INK, spaceBefore=4, spaceAfter=6, keepWithNext=1)
H2 = ParagraphStyle("H2", parent=ss["Heading2"], fontSize=11.5, textColor=ACCENT, spaceBefore=8, spaceAfter=3, keepWithNext=1)
BODY = ParagraphStyle("Body", parent=ss["BodyText"], fontSize=9.4, leading=13, textColor=INK, spaceAfter=4)
SMALL = ParagraphStyle("Small", parent=BODY, fontSize=7.8, leading=10, spaceAfter=0)
SMALLB = ParagraphStyle("SmallB", parent=SMALL, fontName="Helvetica-Bold")
NOTE = ParagraphStyle("Note", parent=BODY, backColor=SOFT, borderPadding=6, spaceBefore=3, spaceAfter=8)
BULLET = ParagraphStyle("Bullet", parent=BODY, leftIndent=12, bulletIndent=2, spaceAfter=1)
CODE = ParagraphStyle("Code", parent=BODY, fontName="Courier", fontSize=8, leading=10.5,
                      backColor=colors.HexColor("#f4f5f7"), borderPadding=5, spaceBefore=3, spaceAfter=6)

PEOPLE = {
    "maya": "Maya (CEO)", "rachel": "Rachel (CFO)", "arjun": "Arjun (Head of Support)", "nina": "Nina (Support agent)",
    "leo": "Leo (Marketing lead)", "sofia": "Sofia (Content marketer)", "daniel": "Daniel (Ops manager)",
    "ivy": "Ivy (Warehouse)", "tom": "Tom (Finance analyst)", "hana": "Hana (People partner)",
    "omar": "Omar (Merchandiser)", "karim": "Karim (Brightpath)",
}


def clean(text: str, limit: int = 230) -> str:
    text = re.sub(r"\*\*|\*|__", "", text or "")
    text = re.sub(r"\s*\[\d+\]", "", text)
    text = " ".join(text.split())
    if len(text) > limit:
        text = text[: limit - 1].rsplit(" ", 1)[0] + "..."
    text = text.encode("cp1252", "replace").decode("cp1252")
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def esc(text: str) -> str:
    return clean(text, 10_000)


story = []
def h1(t): story.append(Paragraph(t, H1))
def h2(t): story.append(Paragraph(t, H2))
def p(t): story.append(Paragraph(t, BODY))
def note(t): story.append(Paragraph(t, NOTE))
def bullets(items):
    for i in items:
        story.append(Paragraph(i, BULLET, bulletText="•"))
    story.append(Spacer(1, 3))
def code(t):
    # Keep line breaks and spacing as written (esc() would collapse them).
    text = t.encode("cp1252", "replace").decode("cp1252").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    story.append(Paragraph(text.replace(" ", "&nbsp;").replace("\n", "<br/>"), CODE))


def result_cell(passed: bool):
    color = GREEN if passed else RED
    return Paragraph(f'<font color="{color.hexval()}"><b>{"PASS" if passed else "FAIL"}</b></font>', SMALL)


def table(rows, widths, status_col=None):
    data = []
    for r, row in enumerate(rows):
        cells = []
        for c, value in enumerate(row):
            if r and status_col is not None and c == status_col:
                cells.append(result_cell(value))
            else:
                cells.append(Paragraph(str(value), SMALLB if r == 0 else SMALL))
        data.append(cells)
    t = Table(data, colWidths=[w * mm for w in widths], repeatRows=1)
    t.setStyle(TableStyle([
        ("GRID", (0, 0), (-1, -1), 0.4, LINE), ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("BACKGROUND", (0, 0), (-1, 0), SOFT),
        ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ("LEFTPADDING", (0, 0), (-1, -1), 4), ("RIGHTPADDING", (0, 0), (-1, -1), 4),
    ]))
    story.append(t)
    story.append(Spacer(1, 6))


def expected_text(expected):
    return "Not found (no access)" if expected is None else f'Contains "{esc(expected)}"'


def ask_table(rows):
    table([["#", "Person", "Question", "Expected", "Verified answer (excerpt)", "Result"]] + [
        [i, PEOPLE.get(r["user"], r["user"]), esc(r["question"]), expected_text(r["expected"]) + f'<br/><font color="#667085">{esc(r["note"])}</font>',
         clean(r["answer"], 200), r["passed"]]
        for i, r in enumerate(rows, 1)
    ], [6, 24, 38, 34, 60, 13], status_col=5)


def section(fid, title, what, steps, say):
    h1(f"{fid}. {title}")
    p(f"<b>What it does:</b> {what}")
    h2("How to test it")
    bullets(steps)
    note(f"<b>How to explain it:</b> {say}")


def on_page(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 7.5)
    canvas.setFillColor(MUTED)
    if doc.page > 1:
        canvas.drawString(15 * mm, 10 * mm, "SecondBrain - Features and Testing Guide")
        canvas.drawRightString(A4[0] - 15 * mm, 10 * mm, f"Page {doc.page}")
    canvas.restoreState()


def main() -> None:
    src, out = sys.argv[1], sys.argv[2]
    d = json.load(open(src))
    ask = d["ask"]
    by = lambda f: [r for r in ask if r["feature"] == f]
    groups = {
        "F1": by("F1"), "F2": by("F2"), "F3": by("F3"), "F4": by("F4"), "F5": by("F5"), "F6": by("F6"),
        "F7": d["follow_ups"], "F8": d["notes"] + d["recall"], "F9": d["preloaded_updates"] + d["updates"],
        "F10": [c for c in d["checks"] if c["feature"] == "F10"],
        "F11": [c for c in d["checks"] if c["feature"] == "F11"],
        "F12": [c for c in d["checks"] if c["feature"] == "F12"],
        "F13": [c for c in d["checks"] if c["feature"] == "F13"],
        "F14": [c for c in d["checks"] if c["feature"] == "F14"],
    }
    total = sum(len(v) for v in groups.values())
    passed = sum(r["passed"] for v in groups.values() for r in v)
    when = datetime.fromisoformat(d["verified_at"]).strftime("%d %B %Y, %H:%M UTC")
    model = d["model"]

    # Cover
    story.append(Spacer(1, 45 * mm))
    story.append(Paragraph("SecondBrain", ParagraphStyle("T", parent=H1, fontSize=32, alignment=TA_CENTER, textColor=ACCENT)))
    story.append(Paragraph("Features, how to test them, and what is still pending",
                           ParagraphStyle("S", parent=BODY, fontSize=13, alignment=TA_CENTER)))
    story.append(Spacer(1, 12))
    story.append(Paragraph(f"<b>{passed} of {total} checks passed</b> against the running system",
                           ParagraphStyle("R", parent=BODY, fontSize=15, alignment=TA_CENTER, textColor=GREEN if passed == total else RED)))
    story.append(Paragraph(f"Verified {when} with the real models: {model['llm']} for answers, {model['embedding_model']} for search",
                           ParagraphStyle("D", parent=BODY, alignment=TA_CENTER, textColor=MUTED)))
    story.append(Spacer(1, 20 * mm))
    note("Every answer in this document was produced by the live system and checked automatically for the fact it must "
         "contain. The <b>Verified answer</b> columns show what SecondBrain actually replied. Anyone can repeat every test "
         "by following the steps in each section.")
    story.append(PageBreak())

    # How to use
    h1("How to use this document")
    h2("1. Start SecondBrain")
    bullets([
        "Turn on the <b>VPN</b>. The AI models are only reachable through it.",
        "Open Terminal in the project folder and start the databases and the server:",
    ])
    code("cd ~/Desktop/secondbrain\ndocker compose up -d --wait\nuv run uvicorn secondbrain.api:app --port 8765")
    bullets([
        "Open <b>http://localhost:8765</b> (chat) and <b>http://localhost:8765/graph-view</b> (graph). Press Cmd+Shift+R once if a page looks old.",
        "Admin token for the whole-SecondBrain graph: the SB_ADMIN_TOKEN value in .env (dev-admin-token locally).",
    ])
    h2("2. How to ask a question")
    bullets([
        "In the chat, click a <b>person</b> on the left (each test says who).",
        "Click <b>New conversation</b> first, so earlier questions don't affect the answer.",
        "Type the question, press Enter, and wait 2 to 5 seconds. The coloured badge says where the answer came from: "
        "<b>department</b>, <b>company</b>, or <b>not found</b>. Click <b>sources</b> to see the documents used.",
        "Answers are written by an AI, so the wording differs each time. Check that the <b>expected fact</b> is in the answer.",
    ])
    h2("3. Reset the data when you're done")
    p("Features F8 and F9 add notes. To go back to the original demo data (about a minute, VPN needed):")
    code("uv run python -m secondbrain.schema --reset\n./scripts/load_examples.sh")
    h2("The demo data")
    p("<b>Lumen Home</b> is a fictional online home decor and furniture brand with 6 departments (Customer Support, "
      "Marketing, Operations & Fulfillment, Finance, People & HR, Merchandising), 11 people and 18 documents: an Employee "
      "Handbook and a Holiday Campaign Plan in <b>Word</b>, a Returns Policy, a Fulfillment SOP and Q3 Results in <b>PDF</b>, "
      "and policies and playbooks in Markdown. Four documents are <b>restricted</b>. 20 notes from its people add events "
      "and updates. <b>Brightpath Logistics</b> is a second company that shares nothing, used to show isolation.")
    table([["Person", "Role", "Can see"],
           ["Maya Chen", "CEO", "All of Lumen, including restricted"],
           ["Rachel Kim", "CFO (Finance)", "All of Lumen, including restricted"],
           ["Arjun Mehta", "Head of Support", "Support (incl. restricted) + company"],
           ["Nina Alvarez", "Support agent", "Support + company + Operations (extra grant)"],
           ["Leo Park", "Marketing lead", "Marketing (incl. restricted) + company + Merchandising (extra grant)"],
           ["Sofia Rossi", "Content marketer", "Marketing + company"],
           ["Daniel Okafor", "Operations manager", "Operations (incl. restricted) + company"],
           ["Ivy Thompson", "Warehouse coordinator", "Operations + company"],
           ["Tom Bauer", "Finance analyst", "Finance (not restricted) + company"],
           ["Hana Sato", "People partner", "People & HR (incl. restricted) + company"],
           ["Omar Haddad", "Merchandiser", "Merchandising + company"],
           ["Karim Nasser", "Brightpath Operations director", "Brightpath only"]], [32, 45, 101])
    story.append(PageBreak())

    # Summary
    h1("Summary: features built and verified")
    names = {
        "F1": "Answers from your own department", "F2": "Falling back to the company brain",
        "F3": "Restricted information by role", "F4": "Access between departments (extra grants)",
        "F5": "Isolation between companies", "F6": "Reading PDF and Word documents",
        "F7": "Conversation memory (follow-up questions)", "F8": "Remembering new information, sorted automatically",
        "F9": "Memory keeps itself up to date", "F10": "Learning at the root (anonymous lessons)",
        "F11": "Graph views (per person and whole SecondBrain)", "F12": "Security enforced by the database",
        "F13": "Relationships: who knows what (FalkorDB)", "F14": "Sales forecasting",
    }
    table([["", "Feature", "Checks", "Passed"]] + [
        [fid, names[fid], len(groups[fid]), f"{sum(r['passed'] for r in groups[fid])} / {len(groups[fid])}"]
        for fid in names
    ] + [["", "<b>Total</b>", f"<b>{total}</b>", f"<b>{passed} / {total}</b>"]], [12, 110, 22, 34])
    p("Also built and covered by the automated test suite (64 tests): onboarding a company from one file, document upload "
      "through the admin API, maintenance, learning, relationship extraction, forecasting models, and the permission rules above.")
    story.append(PageBreak())

    # Features
    section("F1", "Answers from your own department",
            "Each department has its own brain. A question is answered first from the asker's department, in plain language, with the sources it used.",
            ["Pick the person in the table, click <b>New conversation</b>, ask the question.",
             "Expect a green badge <b>Answered from your department</b> and the fact in the Expected column."],
            "Nina's question is answered from her own department's knowledge, and she can click through to the exact document section.")
    ask_table(groups["F1"])

    section("F2", "Falling back to the company brain",
            "If the department doesn't know, the company brain answers. It never looks into another company or the root.",
            ["Ask a company-wide question (leave, expenses, returns) as someone in a department.",
             "Expect an orange badge <b>Answered from your company</b>."],
            "Support doesn't know the leave policy, so the company brain answered. It goes up one level, never sideways into other companies.")
    ask_table(groups["F2"])

    section("F3", "Restricted information by role",
            "Documents can be restricted. Members don't see restricted data; leads see their own department's; executives see the whole company. "
            "Each pair below asks the same question as two people.",
            ["Ask the question as the first person: expect <b>Not found</b>.", "Ask the same question as the second person: expect the answer."],
            "Same question, different answers: the analyst isn't cleared for restricted finance data, the CFO is.")
    ask_table(groups["F3"])

    section("F4", "Access between departments (extra grants)",
            "Departments can't read each other, unless a person has an explicit grant. Nina (Support) may read Operations; Leo (Marketing lead) may read Merchandising.",
            ["Ask as the person with the grant: expect the answer.", "Ask the same as someone without it: expect <b>Not found</b>."],
            "Access between departments goes through permissions on a person, not through the departments themselves.")
    ask_table(groups["F4"])

    section("F5", "Isolation between companies",
            "Each company's knowledge is completely separate. Brightpath never sees Lumen, and the reverse.",
            ["Ask as Karim (Brightpath) and as Lumen people."],
            "Two companies ask the same question and get only their own answer. Nothing crosses between companies.")
    ask_table(groups["F5"])

    section("F6", "Reading PDF and Word documents",
            "Imports Markdown, text, PDF and Word files, split along their own sections, with tables kept.",
            ["Ask the questions; click <b>sources</b> to see the document and section, e.g. \"Employee Handbook > Expenses and travel\"."],
            "It reads the documents a company already has, including tables inside Word files.")
    ask_table(groups["F6"])

    h1("F7. Conversation memory (follow-up questions)")
    p("<b>What it does:</b> remembers the conversation, so a short follow-up is understood as a full question.")
    h2("How to test it")
    bullets(["Pick the person, click <b>New conversation</b>, ask the first question and wait.",
             "Then ask the follow-up. Under the answer, grey text shows <b>Understood as: ...</b>."])
    note("<b>How to explain it:</b> you can talk to it like a colleague; it knows what \"and for directors?\" refers to.")
    table([["#", "Person", "First question", "Follow-up", "Understood as", "Verified answer (excerpt)", "Result"]] + [
        [i, PEOPLE[r["user"]], esc(r["first"]), esc(r["follow_up"]), clean(r["understood_as"], 90), clean(r["answer"], 150), r["passed"]]
        for i, r in enumerate(groups["F7"], 1)
    ], [6, 22, 32, 24, 34, 50, 12], status_col=6)

    h1("F8. Remembering new information, sorted automatically")
    p("<b>What it does:</b> people tell it new things in plain language. The AI sorts each note into an <b>event</b> (episodic), "
      "a <b>fact</b> (semantic) or <b>information about a person</b> (user), keeps it in short-term memory, and saves it to long-term memory.")
    h2("How to test it")
    bullets(["Pick <b>Arjun</b>, open the <b>Remember</b> tab, enter each note and click <b>Remember</b>: the reply shows its type.",
             "Click <b>Save to long-term</b>, switch to <b>Ask</b> and ask the recall questions.",
             "Ask the last recall question as Leo: Arjun's Support notes are not visible in Marketing."])
    note("<b>How to explain it:</b> new knowledge is captured in plain words, sorted automatically, and only visible to the right people.")
    table([["#", "Note typed by Arjun", "Expected type", "Got", "Result"]] + [
        [i, esc(r["note"]), r["expected"], r["got"], r["passed"]] for i, r in enumerate(d["notes"], 1)
    ], [6, 110, 22, 22, 15], status_col=4)
    h2("Recall after saving")
    table([["#", "Person", "Question", "Expected", "Verified answer (excerpt)", "Result"]] + [
        [i, PEOPLE[r["user"]], esc(r["question"]), expected_text(r["expected"]), clean(r["answer"], 170), r["passed"]]
        for i, r in enumerate(d["recall"], 1)
    ], [6, 24, 40, 28, 64, 13], status_col=5)

    h1("F9. Memory keeps itself up to date")
    p("<b>What it does:</b> after new memories are saved, the AI compares them with similar older ones. If a note changes a fact, "
      "the newer one wins; if it changes one fact inside a longer document section, that section is marked partly outdated. "
      "Nothing is deleted. A replaced fact is only hidden from people who can see its replacement.")
    h2("Already in the demo data")
    table([["#", "Person", "Question", "Expected", "Why", "Verified answer (excerpt)", "Result"]] + [
        [i, PEOPLE[r["user"]], esc(r["question"]), expected_text(r["expected"]), esc(r["note"]), clean(r["answer"], 130), r["passed"]]
        for i, r in enumerate(d["preloaded_updates"], 1)
    ], [6, 22, 32, 22, 36, 50, 12], status_col=6)
    h2("Live: teach it a change, then ask")
    bullets(["Ask the question first (the old answer).",
             "As the writer, <b>Remember</b> the note and click <b>Save to long-term</b>.",
             "Wait about 10 seconds, click <b>New conversation</b>, and ask again: the new value."])
    table([["#", "Writer: note", "Asked by: question", "Before", "After (verified)", "Result"]] + [
        [i, f"{PEOPLE[r['writer']]}: {esc(r['note'])}", f"{PEOPLE[r['user']]}: {esc(r['question'])}",
         clean(r["before"], 90), clean(r["after"], 120), r["passed"]]
        for i, r in enumerate(d["updates"], 1)
    ], [6, 44, 38, 36, 44, 12], status_col=5)
    note("<b>How to explain it:</b> when something changes, the brain notices that the new note updates an older fact and "
         "uses the newer one, while the full history stays in the graph.")

    h1("F10. Learning at the root (anonymous lessons)")
    p("<b>What it does:</b> companies that allow it (Lumen: sharing = patterns) send <b>general lessons</b> to our root brain. "
      "Restricted data is never used, and code rejects any lesson with a number or a name. Companies that don't share "
      "(Brightpath) contribute nothing, and company users never get answers from the root.")
    h2("How to test it")
    bullets(["Open the graph, choose <b>Whole SecondBrain (admin)</b>, enter the admin token.",
             "Click the small nodes around the gold root to read the lessons."])
    h2(f"The {len(d.get('lessons', []))} lessons the root learned from Lumen in this run")
    for lesson in d.get("lessons", []):
        story.append(Paragraph(esc(lesson), BULLET, bulletText="•"))
    story.append(Spacer(1, 4))
    table([["#", "Check", "Expected", "Got", "Result"]] + [
        [i, esc(c["check"]), esc(c["expected"]), clean(c["got"], 80), c["passed"]] for i, c in enumerate(groups["F10"], 1)
    ], [6, 82, 34, 40, 13], status_col=4)
    note("<b>How to explain it:</b> every company benefits from what others learned, without anyone's data leaving their company.")

    h1("F11. Graph views")
    p("<b>What it does:</b> shows the brains, people and memories as a graph. A person's view is built by the database for "
      "that person, so it can't show more than they may see. The admin view shows the whole SecondBrain.")
    h2("How to test it")
    bullets(["Open http://localhost:8765/graph-view and pick <b>Rachel</b>: all departments filled, restricted memories outlined in red.",
             "Pick <b>Sofia</b>: most department zones are empty.",
             "Pick <b>Whole SecondBrain (admin)</b>: root with lessons in the middle, Lumen and Brightpath around it.",
             "Click any node to see its details and relationships; faded nodes with dashed rings are updated memories."])
    table([["#", "Check", "Expected", "Got", "Result"]] + [
        [i, esc(c["check"]), esc(c["expected"]), clean(c["got"], 80), c["passed"]] for i, c in enumerate(groups["F11"], 1)
    ], [6, 82, 40, 34, 13], status_col=4)
    p('<font color="#667085">Node counts include notes added during this verification run.</font>')

    h1("F12. Security enforced by the database")
    p("<b>What it does:</b> permissions are enforced by PostgreSQL row-level security, not only by the application. "
      "Even with the application's own database login, nothing is visible until a person is set, and then only what they may see.")
    h2("How to test it (for technical audiences)")
    code("docker compose exec postgres psql \"postgresql://secondbrain_app:dev_app_password@localhost/secondbrain\" \\\n"
         "  -c \"SELECT count(*) FROM memories;\"")
    p("Expect 0. Then as Sofia:")
    code("docker compose exec postgres psql \"postgresql://secondbrain_app:dev_app_password@localhost/secondbrain\" \\\n"
         "  -c \"BEGIN; SELECT set_config('app.user_id','sofia',true); SELECT DISTINCT brain_path FROM memories; COMMIT;\"")
    table([["#", "Check", "Expected", "Got", "Result"]] + [
        [i, esc(c["check"]), esc(c["expected"]), clean(c["got"], 80), c["passed"]] for i, c in enumerate(groups["F12"], 1)
    ], [6, 82, 40, 34, 13], status_col=4)
    note("<b>How to explain it:</b> even if someone bypassed our application, the database itself would only return what that person may see.")

    def checks_table(fid):
        table([["#", "Check", "Expected", "Got", "Result"]] + [
            [i, esc(c["check"]), esc(c["expected"]), clean(c["got"], 80), c["passed"]] for i, c in enumerate(groups[fid], 1)
        ], [6, 82, 34, 40, 13], status_col=4)

    h1("F13. Relationships: who knows what (FalkorDB)")
    p("<b>What it does:</b> when notes and documents are saved, the AI picks out the people, customers, suppliers, products, "
      "carriers and places they mention, and how they relate (\"Casa Forma supplies the Aria linen sofa\"). Each company gets its "
      "own graph in FalkorDB. Two new questions can be answered: <b>who knows about X?</b> and <b>what do we know about X?</b> "
      "FalkorDB has no permissions of its own, so every graph query is limited to the memories PostgreSQL says the person may see.")
    h2("How to test it")
    p("There is no screen for this yet; use these commands in Terminal (the person is set with X-User-Id):")
    code('curl -s "http://localhost:8765/experts?topic=Priya%20Shah" -H "X-User-Id: nina"\n'
         'curl -s "http://localhost:8765/experts?topic=Priya%20Shah" -H "X-User-Id: sofia"\n'
         'curl -s "http://localhost:8765/entities/Nordlys" -H "X-User-Id: omar"\n'
         'curl -s "http://localhost:8765/entities/Nordlys" -H "X-User-Id: karim"')
    p("The first returns Nina as the person who knows about Priya Shah; the second returns nobody, because Support's notes are "
      "hidden from Sofia; the third describes Nordlys as a supplier; the fourth is not found, because Karim is in another company.")
    checks_table("F13")
    note("<b>How to explain it:</b> SecondBrain knows who in the company knows about a customer, supplier or product, but only "
         "tells you about the people and notes you're allowed to see.")

    h1("F14. Sales forecasting")
    p("<b>What it does:</b> forecasts daily sales per product with 80% and 95% ranges. Four models are compared on past data "
      "(repeat last week, last year with growth, ETS, and gradient boosting with promotions and holidays), including the same "
      "weeks last year, and the most accurate one is picked per product with the reason. Sales data follows the same "
      "permissions: it lives in Merchandising. The sales history is <b>generated sample data</b> for Lumen (2 years, 8 products), "
      "not real sales.")
    h2("How to test it")
    code('curl -s "http://localhost:8765/forecast/products" -H "X-User-Id: omar"\n'
         'curl -s "http://localhost:8765/forecast?product=halo-lamp&horizon=30" -H "X-User-Id: omar"\n'
         'curl -s "http://localhost:8765/forecast?product=halo-lamp&horizon=30" -H "X-User-Id: nina"')
    p("The second shows the chosen model, why it was chosen, every model's backtest scores and 30 days of forecasts; look for "
      "the Black Friday peak around November 27. The third is not found: Nina has no access to Merchandising.")
    checks_table("F14")
    note("<b>How to explain it:</b> SecondBrain doesn't trust one model: it tests several on the merchant's own history and "
         "uses the one that would have been most accurate, and says so. Caveat: on this sample data the best model still "
         "under-predicts the very top of the Black Friday peak.")

    h1("F15. Onboarding a new company")
    p("<b>What it does:</b> a company is described in one file (departments, people, roles, sharing) and its documents are imported. "
      "Verified by the automated test suite. Steps to show it live:")
    code('~/Desktop/nova_demo/company.json:\n{"id": "nova", "name": "Nova Tech", "sharing": "none",\n'
         ' "departments": [{"id": "eng", "name": "Engineering"}],\n'
         ' "users": [{"id": "priya", "name": "Priya Raman (Engineer)", "department": "eng", "role": "member"}]}')
    code("uv run python -m secondbrain.admin onboard ~/Desktop/nova_demo/company.json\n"
         "uv run python -m secondbrain.admin import root.nova.eng ~/Desktop/nova_demo/oncall.md --sensitivity internal")
    p("Refresh the chat, pick Priya and ask about the on-call document.")
    story.append(PageBreak())

    # Pending
    h1("What is still pending")
    table([["Area", "Status", "What it needs"],
           ["Screens for relationships and forecasts", "Not started (API only today)", "A simple page in the chat or graph view"],
           ["Relationships: merging similar names, using the graph in answers", "Not started", "Name matching; measure whether answers improve"],
           ["Forecasting: foundation models (Chronos, TimesFM) for new products", "Not started", "PyTorch and model downloads; compare in the backtest"],
           ["Forecasting on real merchant data; TimescaleDB in the main setup", "Not started", "A merchant's order history; switch when raw orders arrive"],
           ["Real-time checkout discounts with reasons", "Concept only", "Live event data from a merchant, decision engine, guardrails"],
           ["Real sign-in (SSO)", "Not started", "The company's identity provider (Google, Microsoft, Okta)"],
           ["Connectors: Google Drive, Slack, email", "Not started", "App credentials and approval from IT"],
           ["Pushing root lessons down to companies (variant V1)", "Not started", "A product decision on whether and how"],
           ["Smarter memory: importance scoring, forgetting unused memories", "Not started", "Design of the rules, then build"],
           ["Comparing versions with a scored test set", "Partly (this verification is a start)", "A fixed benchmark and scoring for every variant"],
           ["Running on a company's own servers, MQTT, isolation options (V3, V5)", "Not started", "Deployment work"],
           ["Hosted demo (a URL the client can open)", "Not started", "A server, plus the models reachable without our VPN"]],
          [70, 40, 68])

    h1("What is stopping a client demo version")
    p("The demo works today on this laptop. What stands between that and showing it to a client, or letting them try it:")
    table([["Blocker", "Why it matters", "What would unblock it"],
           ["<b>The AI models need our VPN</b>", "The demo only works from a laptop on the VPN; a client can't use it themselves.",
            "Host the app where it can reach the LiteLLM proxy, or get model access that doesn't need the VPN."],
           ["<b>No hosted version</b>", "It runs on one laptop; there is no link to share.", "A small server (cloud or company) and a deployment."],
           ["<b>Client data and consent</b>", "Real documents would be processed by the models behind our proxy.",
            "Written agreement from the client, or use the industry sample data (Lumen) for the demo."],
           ["<b>No real sign-in</b>", "The demo uses a person picker; fine for a guided demo, not for client self-service.",
            "SSO with the client's identity provider."],
           ["<b>Admin view shows every company</b>", "Must never show another real client's data.", "Only demo with sample companies, or restrict the view per client."],
           ["<b>FalkorDB license</b>", "Believed to be SSPL; matters for a commercial service.", "Confirmation from legal before production use."],
           ["<b>No screens for relationships and forecasts</b>", "They can only be shown with Terminal commands today.", "A simple page for each before showing them to a client."],
           ["<b>Forecasts use sample sales</b>", "The numbers are generated, not a real merchant's history.", "A merchant's real order history (and their consent)."]],
          [42, 66, 70])
    note("<b>Bottom line:</b> a <b>guided demo</b> on this laptop with the Lumen sample data is ready now. A demo the "
         "<b>client can use on their own</b> needs hosting, model access without the VPN, and real sign-in.")

    doc = SimpleDocTemplate(out, pagesize=A4, leftMargin=15 * mm, rightMargin=15 * mm, topMargin=15 * mm,
                            bottomMargin=16 * mm, title="SecondBrain - Features and Testing Guide", author="SecondBrain team")
    doc.build(story, onFirstPage=on_page, onLaterPages=on_page)
    print(f"wrote {out}: {passed}/{total} checks passed")


if __name__ == "__main__":
    main()
