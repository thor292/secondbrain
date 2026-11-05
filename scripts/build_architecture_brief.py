"""Build the platform architecture brief (capabilities, architecture diagram, plain explanation).

    uv run python scripts/build_architecture_brief.py docs/SecondBrain_Platform_Architecture.pdf
"""

import sys
from reportlab.graphics.shapes import Circle, Drawing, Group, Line, Polygon, Rect, String
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.platypus import (KeepTogether, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table,
                                TableStyle)

INK, MUTED, FAINT = colors.HexColor("#0f172a"), colors.HexColor("#5b6477"), colors.HexColor("#94a0b8")
VIOLET, CYAN, GREEN = colors.HexColor("#5b4bdb"), colors.HexColor("#0e7490"), colors.HexColor("#0f7a3d")
LINE = colors.HexColor("#d5dae5")
SOFT_V, SOFT_C, SOFT_G = colors.HexColor("#f0eeff"), colors.HexColor("#e8f7fb"), colors.HexColor("#e9f7ef")
SOFT_S, SOFT_A = colors.HexColor("#f5f7fb"), colors.HexColor("#fff7e8")
AMBER = colors.HexColor("#b45309")

ss = getSampleStyleSheet()
TITLE = ParagraphStyle("Title", parent=ss["Title"], fontName="Helvetica-Bold", fontSize=26, leading=30,
                       textColor=INK, alignment=TA_LEFT, spaceAfter=4)
KICKER = ParagraphStyle("Kicker", fontName="Helvetica-Bold", fontSize=8.5, leading=11, textColor=VIOLET, spaceAfter=6)
LEAD = ParagraphStyle("Lead", parent=ss["BodyText"], fontSize=11.5, leading=16.5, textColor=INK, spaceAfter=10)
H1 = ParagraphStyle("H1", parent=ss["Heading1"], fontName="Helvetica-Bold", fontSize=17, leading=21, textColor=INK,
                    spaceBefore=2, spaceAfter=8, keepWithNext=1)
H2 = ParagraphStyle("H2", parent=ss["Heading2"], fontName="Helvetica-Bold", fontSize=11.5, leading=15, textColor=VIOLET,
                    spaceBefore=10, spaceAfter=4, keepWithNext=1)
BODY = ParagraphStyle("Body", parent=ss["BodyText"], fontSize=9.6, leading=14, textColor=INK, spaceAfter=5)
SMALL = ParagraphStyle("Small", parent=BODY, fontSize=8.4, leading=11.5, spaceAfter=0)
SMALLB = ParagraphStyle("SmallB", parent=SMALL, fontName="Helvetica-Bold")
CAP = ParagraphStyle("Cap", parent=SMALL, textColor=MUTED, fontSize=8, leading=11)
STEP = ParagraphStyle("Step", parent=BODY, leftIndent=26, firstLineIndent=0, spaceAfter=5)


# ---------------------------------------------------------------- drawing helpers

def wrap(text: str, width: float, font: str, size: float) -> list[str]:
    lines, cur = [], ""
    for word in text.split():
        trial = (cur + " " + word).strip()
        if stringWidth(trial, font, size) <= width:
            cur = trial
        else:
            lines.append(cur)
            cur = word
    return lines + ([cur] if cur else [])


def box(d, x, y, w, h, title, sub="", fill=colors.white, stroke=LINE, title_color=INK, size=8.4, r=6):
    d.add(Rect(x, y, w, h, rx=r, ry=r, fillColor=fill, strokeColor=stroke, strokeWidth=0.8))
    sub_lines = wrap(sub, w - 12, "Helvetica", 6.7) if sub else []
    block = size + 2 + len(sub_lines) * 8.2
    ty = y + h / 2 + block / 2 - size + 1
    d.add(String(x + w / 2, ty, title, fontName="Helvetica-Bold", fontSize=size, fillColor=title_color, textAnchor="middle"))
    for i, line in enumerate(sub_lines):
        d.add(String(x + w / 2, ty - 10.5 - i * 8.2, line, fontName="Helvetica", fontSize=6.7, fillColor=MUTED,
                     textAnchor="middle"))


def zone(d, x, y, w, h, label, fill, stroke, label_color):
    d.add(Rect(x, y, w, h, rx=9, ry=9, fillColor=fill, strokeColor=stroke, strokeWidth=0.9))
    d.add(String(x + 10, y + h - 12, label, fontName="Helvetica-Bold", fontSize=7, fillColor=label_color))


def arrow(d, x1, y1, x2, y2, color=MUTED, width=1.2, head=5):
    d.add(Line(x1, y1, x2, y2, strokeColor=color, strokeWidth=width))
    dx, dy = x2 - x1, y2 - y1
    n = (dx * dx + dy * dy) ** 0.5 or 1
    ux, uy = dx / n, dy / n
    px, py = -uy, ux
    d.add(Polygon([x2, y2, x2 - ux * head * 1.6 + px * head, y2 - uy * head * 1.6 + py * head,
                   x2 - ux * head * 1.6 - px * head, y2 - uy * head * 1.6 - py * head],
                  fillColor=color, strokeColor=color, strokeWidth=0.1))


def badge(d, x, y, n, color=VIOLET):
    d.add(Circle(x, y, 7.5, fillColor=color, strokeColor=colors.white, strokeWidth=1.2))
    d.add(String(x, y - 3, str(n), fontName="Helvetica-Bold", fontSize=8, fillColor=colors.white, textAnchor="middle"))


def flow(d, x, y_top, y_bottom, n, label, color=VIOLET):
    arrow(d, x, y_top, x, y_bottom + 1, color, 1.4)
    badge(d, x - 16, (y_top + y_bottom) / 2, n, color)
    d.add(String(x + 9, (y_top + y_bottom) / 2 - 3, label, fontName="Helvetica-Oblique", fontSize=7, fillColor=MUTED))


def architecture() -> Drawing:
    W, H = 523, 640
    d = Drawing(W, H)
    X0, W0 = 26, W - 26

    def row(n, x, y, w, h, items, fill=colors.white, stroke=LINE, gap=8, pad=8, title_color=INK, size=8.4):
        bw = (w - 2 * pad - gap * (n - 1)) / n
        for i, (t, s) in enumerate(items):
            box(d, x + pad + i * (bw + gap), y, bw, h, t, s, fill, stroke, title_color, size)

    # 1 business systems
    zone(d, X0, 568, W0, 72, "BUSINESS SYSTEMS  ·  where the data already lives", SOFT_S, LINE, MUTED)
    row(4, X0, 576, W0, 44, [
        ("Sales & checkout", "POS, e-commerce, daily sales"),
        ("Inventory & ERP", "stock on hand, products"),
        ("Purchasing", "orders, deliveries, suppliers"),
        ("Documents & notes", "policies, PDFs, Word, team notes"),
    ])
    flow(d, X0 + W0 / 2, 568, 540, 1, "live business events  ·  MQTT over TLS / HTTPS")

    # 2 intake
    box(d, X0, 498, W0, 42, "Event gateway",
        "checks every event against a published contract  ·  ignores duplicates  ·  quarantines bad messages for review",
        SOFT_C, colors.HexColor("#9fd8e6"), CYAN, 9)
    flow(d, X0 + W0 / 2, 498, 470, 2, "clean, trusted business data")

    # 3 brain engine + AI layer
    EW = W0 - 140
    zone(d, X0, 300, EW, 170, "BRAIN ENGINE  ·  the core: predicts, advises, learns", SOFT_V,
         colors.HexColor("#c9c1ff"), VIOLET)
    row(3, X0, 394, EW, 54, [
        ("Forecasting", "demand per product, with a likely range; best model chosen by testing on past sales"),
        ("Decision signals", "run-out and overstock warnings, advice from notes, money at stake, deadline"),
        ("Learning & evolution", "grades itself against real outcomes and adjusts what it says next"),
    ], stroke=colors.HexColor("#c9c1ff"), title_color=VIOLET)
    row(3, X0, 330, EW, 54, [
        ("Memory", "short-term notes become long-term knowledge; duplicates merge, old facts retire"),
        ("Ask & answer", "plain-language answers with sources; department first, then company"),
        ("Knowledge map", "people, customers, suppliers and products, and how they connect"),
    ], stroke=colors.HexColor("#c9c1ff"), title_color=VIOLET)
    loop = ["data", "forecast", "signal", "action", "outcome", "lesson"]
    lx, lw = X0 + 14, (EW - 28) / len(loop)
    for i, word in enumerate(loop):
        cx = lx + i * lw + lw / 2
        d.add(String(cx, 309, word, fontName="Helvetica-Bold", fontSize=6.8, fillColor=VIOLET, textAnchor="middle"))
        if i < len(loop) - 1:
            arrow(d, cx + 17, 311.5, cx + lw - 17, 311.5, colors.HexColor("#a99cf5"), 0.8, 2.6)
    d.add(String(X0 + 10, 320, "the learning loop, every day:", fontName="Helvetica-Oblique", fontSize=6.5, fillColor=MUTED))

    AX = X0 + EW + 10
    zone(d, AX, 300, 130, 170, "AI LAYER", SOFT_A, colors.HexColor("#f2d39b"), AMBER)
    for i, (t, s) in enumerate([("Language model", "via a gateway: provider is swappable"),
                                ("Embeddings", "search by meaning, not keywords"),
                                ("Guardrails", "every number checked against the data")]):
        box(d, AX + 8, 404 - i * 48, 114, 40, t, s, colors.white, colors.HexColor("#f2d39b"), AMBER, 8)
    arrow(d, X0 + EW + 1, 385, AX - 1, 385, AMBER, 1.1, 3.5)
    arrow(d, AX - 1, 375, X0 + EW + 1, 375, AMBER, 1.1, 3.5)

    flow(d, X0 + W0 / 2 - 70, 300, 272, 3, "reads and writes, always as the person asking")

    # 4 data layer
    zone(d, X0, 196, W0, 76, "DATA LAYER", SOFT_S, LINE, MUTED)
    row(3, X0, 204, W0, 50, [
        ("PostgreSQL + vectors", "single source of truth; permission checked on every row"),
        ("Redis", "fast short-term memory that expires on its own"),
        ("Graph database", "relationships: who knows what, who works with whom"),
    ])
    flow(d, X0 + W0 / 2 - 70, 196, 168, 4, "only what each person may see")

    # 5 delivery
    zone(d, X0, 104, W0, 64, "DELIVERY", SOFT_G, colors.HexColor("#a7dcb9"), GREEN)
    row(2, X0, 112, W0, 40, [
        ("Secure API", "every request answered as the signed-in person"),
        ("Live notification broker", "pushes alerts the moment they happen, only to people allowed to see them"),
    ], stroke=colors.HexColor("#a7dcb9"), title_color=GREEN)
    flow(d, X0 + W0 / 2 - 70, 104, 76, 5, "instant, on screen", GREEN)

    # 6 people
    zone(d, X0, 0, W0, 76, "PEOPLE  ·  every role sees only its own part of the brain", SOFT_S, LINE, MUTED)
    badge(d, X0 + W0 - 14, 64, 6, GREEN)
    row(4, X0, 8, W0, 48, [
        ("Chat assistant", "ask anything, see where the answer came from"),
        ("Explore", "signals, forecasts, experts, accuracy"),
        ("HQ (root view)", "all companies, the time machine, lessons"),
        ("Knowledge map", "the brain's connections, visual"),
    ])

    # permission ribbon down the left side
    d.add(Rect(0, 104, 16, 366, rx=5, ry=5, fillColor=INK, strokeColor=INK))
    g = Group(String(0, 0, "PERMISSIONS ENFORCED BY THE DATABASE AND THE BROKER, NOT BY THE AI",
                     fontName="Helvetica-Bold", fontSize=6.6, fillColor=colors.white, textAnchor="middle"))
    g.transform = (0, 1, -1, 0, 10.5, 287)
    d.add(g)
    return d


def tree_and_loop() -> Drawing:
    W, H = 523, 205
    d = Drawing(W, H)
    # left: the tree
    d.add(String(0, H - 10, "THE TREE OF BRAINS", fontName="Helvetica-Bold", fontSize=7.5, fillColor=VIOLET))
    box(d, 95, 150, 100, 32, "Root brain", "our platform", SOFT_V, colors.HexColor("#c9c1ff"), VIOLET)
    comps = [(20, "Company A"), (170, "Company B")]
    for x, name in comps:
        box(d, x, 88, 100, 32, name, "company brain", colors.white, LINE)
        arrow(d, x + 50, 121, 145 + (x - 95) * 0.25, 149, VIOLET, 1, 3)
    for i, dept in enumerate(["Sales", "Operations", "Finance"]):
        x = i * 82
        box(d, x, 22, 74, 30, dept, "department", colors.white, LINE, size=7.8)
        arrow(d, x + 37, 53, 45 + i * 25, 87, VIOLET, 1, 3)
    d.add(String(0, 6, "Lessons travel up, anonymized. Questions never leave the company.",
                 fontName="Helvetica-Oblique", fontSize=7, fillColor=MUTED))
    d.add(String(222, 140, "anonymized", fontName="Helvetica-Oblique", fontSize=6.5, fillColor=VIOLET))
    d.add(String(222, 132, "lessons only", fontName="Helvetica-Oblique", fontSize=6.5, fillColor=VIOLET))

    # right: the learning loop, as a racetrack
    d.add(String(290, H - 10, "HOW THE BRAIN GETS SMARTER", fontName="Helvetica-Bold", fontSize=7.5, fillColor=VIOLET))
    bw, bh, gx = 66, 34, 18
    xs = [290 + i * (bw + gx) for i in range(3)]
    top, bottom = 132, 40
    soft = colors.HexColor("#c9c1ff")
    for x, (t, sub) in zip(xs, [("Real data", "sales & stock"), ("Forecast", "what will sell"), ("Signal", "what to do, by when")]):
        box(d, x, top, bw, bh, t, sub, colors.white, soft, VIOLET, 7.8)
    for x, (t, sub) in zip(xs, [("Lesson", "adjust & share up"), ("Outcome", "did it come true?"), ("Action", "team acts or dismisses")]):
        box(d, x, bottom, bw, bh, t, sub, colors.white, soft, VIOLET, 7.8)
    loop_color = colors.HexColor("#8f80f0")
    for i in range(2):
        arrow(d, xs[i] + bw + 2, top + bh / 2, xs[i + 1] - 2, top + bh / 2, loop_color, 1.2, 3)
        arrow(d, xs[i + 1] - 2, bottom + bh / 2, xs[i] + bw + 2, bottom + bh / 2, loop_color, 1.2, 3)
    arrow(d, xs[2] + bw / 2, top - 2, xs[2] + bw / 2, bottom + bh + 2, loop_color, 1.2, 3)
    arrow(d, xs[0] + bw / 2, bottom + bh + 2, xs[0] + bw / 2, top - 2, loop_color, 1.2, 3)
    cx, cy = xs[1] + bw / 2, (top + bottom + bh) / 2
    d.add(Rect(cx - 30, cy - 9, 60, 18, rx=9, ry=9, fillColor=SOFT_V, strokeColor=soft))
    d.add(String(cx, cy - 3, "every day", fontName="Helvetica-Bold", fontSize=7.5, fillColor=VIOLET, textAnchor="middle"))
    d.add(String(290, 6, "Each turn of the loop makes the next forecast and signal better.",
                 fontName="Helvetica-Oblique", fontSize=7, fillColor=MUTED))
    return d


# ---------------------------------------------------------------- content

CAPABILITIES = [
    ("The intelligence core", [
        ("Sales forecasting", "Predicts each product's demand for the coming days and weeks, with a likely range. Several "
         "forecasting models are tested on past sales and the most accurate one is used.", "Live"),
        ("Decision signals", "Warns before a product runs out or piles up, and turns team notes into concrete advice. "
         "Every signal states the action, the deadline, the money at stake and how confident the brain is.", "Live"),
        ("Self-grading", "Checks every forecast against the sales that really happened, and every warning against "
         "what actually followed: prevented, came true, or false alarm.", "Live"),
        ("Learning from feedback", "Learns which signals people act on or dismiss, and why, and adjusts what it raises next.", "Live"),
        ("Learning across companies", "Companies that opt in share anonymized lessons with the root brain. Names, "
         "numbers and raw data never leave the company.", "Live"),
        ("Time machine", "Plays the coming weeks forward to show how following the brain's advice changes outcomes.", "Live"),
    ]),
    ("Knowledge", [
        ("Memory", "New notes start as short-term memory and become long-term knowledge. Duplicates merge and "
         "outdated facts are replaced; nothing is silently deleted.", "Live"),
        ("Ask with sources", "Answers questions in plain language and shows its sources. It looks in the person's "
         "department first, then the company, and never in another company.", "Live"),
        ("Who knows what", "A relationship map of people, customers, suppliers and products.", "Live"),
        ("Document import", "Markdown, text, PDF and Word documents become searchable knowledge.", "Live"),
    ]),
    ("Data in and out", [
        ("Live business data", "Sales, stock levels, purchase orders, deliveries and promotions arrive as events "
         "over MQTT or HTTPS. Each event is validated, duplicates are ignored, bad messages are set aside.", "Live"),
        ("Live notifications", "New signals and new knowledge appear on screen the moment they happen, delivered "
         "only to people allowed to see them.", "Live"),
        ("Onboarding", "A company, its departments, people and permissions are set up from a single file.", "Live"),
    ]),
    ("Next", [
        ("Company sign-in (SSO)", "People sign in with their existing company account.", "Next"),
        ("Encryption everywhere", "TLS on every connection, short-lived logins for the notification broker.", "Next"),
        ("Self-hosted brains", "A company runs its brain on its own servers and still shares lessons with the root. "
         "The secure sync channel is already built.", "Next"),
        ("Ready-made connectors", "Plug-and-play links to common POS, e-commerce and ERP systems.", "Next"),
        ("Alerts beyond the browser", "Signals delivered to email, Teams or Slack, and mobile.", "Next"),
        ("Model choice and scoring", "Pick the production AI model and compare platform versions side by side on "
         "the same test set: accuracy, zero permission leaks, speed and cost.", "Next"),
    ]),
]

STEPS = [
    ("Data arrives on its own.", "The company's existing systems (checkout, inventory, purchasing) send each sale, "
     "stock change and delivery as a small event the moment it happens. Documents and team notes are added too."),
    ("Only clean data gets in.", "The event gateway checks every event against a published format, ignores anything "
     "it has already seen, and sets aside anything malformed so it can be fixed. One bad message never stops the rest."),
    ("The brain thinks.", "The brain engine forecasts demand, compares it with stock and open orders, and raises "
     "decision signals: what to do, by when, and what is at stake. Each day it grades yesterday's forecasts and "
     "warnings against what really happened, and learns from it. An AI model writes the advice; guardrails check "
     "every number in it against the real data."),
    ("Everything is stored safely.", "PostgreSQL is the single source of truth for knowledge, forecasts and "
     "signals, and it checks permissions on every row. Redis keeps short-term memory and a graph database keeps "
     "relationships."),
    ("People are told immediately.", "The secure API answers every request as the signed-in person. The live "
     "notification broker pushes new signals and knowledge to screens instantly, and its permissions are "
     "generated from the same rules, so it cannot deliver anything a person may not see."),
    ("Everyone works from the same brain.", "Staff ask questions in the chat assistant, managers act on signals and "
     "forecasts in Explore, and the root operator watches every company, runs the time machine and sees the "
     "lessons learned across the tree in HQ."),
]

TRUST = [
    ("Permissions live in the database, not the AI.", "Access is enforced by PostgreSQL row-level security and by "
     "the broker's access list. A clever prompt cannot talk the AI into revealing data, because the AI never "
     "receives data the person is not allowed to see."),
    ("Every piece of knowledge is labeled.", "Each memory records which brain owns it and how sensitive it is "
     "(public, internal or restricted). Each person has explicit grants."),
    ("Companies are separated.", "Questions stop at the company boundary. Only anonymized lessons travel to the "
     "root, and only for companies that opt in."),
    ("Notifications carry no content.", "A live alert holds only an id. The screen then asks the API, which checks "
     "permissions again before showing anything."),
    ("Nothing is announced before it is saved.", "Alerts are sent only after the data is committed, so a cancelled "
     "change is never broadcast."),
]


def build(path: str) -> None:
    doc = SimpleDocTemplate(path, pagesize=A4, leftMargin=36, rightMargin=36, topMargin=34, bottomMargin=34,
                            title="SecondBrain: Platform Architecture", author="SecondBrain")
    story = [
        Paragraph("PLATFORM ARCHITECTURE  ·  NOVEMBER 2026", KICKER),
        Paragraph("SecondBrain", TITLE),
        Paragraph("A company brain that forecasts, advises, and learns from what happens next.", H2),
        Spacer(1, 4),
        Paragraph(
            "SecondBrain connects to the systems a business already uses, keeps everything the company knows in one "
            "place, and turns it into <b>decision signals</b>: clear, dated recommendations with the money at stake. "
            "It then checks what actually happened and <b>gets better every day</b>. Each company has its own brain, "
            "each department can have one too, and all of them form a tree that learns together while keeping every "
            "company's data private.", LEAD),
        Paragraph("Capabilities", ParagraphStyle("H1Split", parent=H1, keepWithNext=0)),
    ]
    rows = [[Paragraph("Capability", SMALLB), Paragraph("What it does", SMALLB), Paragraph("Status", SMALLB)]]
    style = [("LINEBELOW", (0, 0), (-1, 0), 0.8, INK), ("VALIGN", (0, 0), (-1, -1), "TOP"),
             ("TOPPADDING", (0, 0), (-1, -1), 4), ("BOTTOMPADDING", (0, 0), (-1, -1), 4)]
    for group, items in CAPABILITIES:
        style += [("SPAN", (0, len(rows)), (-1, len(rows))), ("BACKGROUND", (0, len(rows)), (-1, len(rows)), SOFT_V),
                  ("TOPPADDING", (0, len(rows)), (-1, len(rows)), 5)]
        rows.append([Paragraph(f"<font color='#5b4bdb'><b>{group.upper()}</b></font>", SMALL), "", ""])
        for name, what, status in items:
            color = "#0f7a3d" if status == "Live" else "#5b4bdb"
            rows.append([Paragraph(f"<b>{name}</b>", SMALL), Paragraph(what, SMALL),
                         Paragraph(f"<font color='{color}'><b>{status}</b></font>", SMALL)])
            style.append(("LINEBELOW", (0, len(rows) - 1), (-1, len(rows) - 1), 0.4, LINE))
    table = Table(rows, colWidths=[118, 352, 53], repeatRows=1)
    table.setStyle(TableStyle(style))
    story.append(table)
    story += [
        Paragraph("Where it runs", H2),
        Paragraph("The same platform runs in our cloud or on a company's own servers. A self-hosted brain keeps all of "
                  "its data on site and shares only anonymized lessons with the root, over a secure channel in which "
                  "each company can only use its own topics. The AI model is reached through a gateway, so the "
                  "provider can be changed by configuration without changing the platform.", BODY),
        Paragraph("Technology", H2),
    ]
    tech = [
        ("Backend", "Python, FastAPI"),
        ("Knowledge and permissions", "PostgreSQL with pgvector (search by meaning), ltree (the brain tree), row-level security"),
        ("Short-term memory", "Redis"),
        ("Relationships", "FalkorDB graph database"),
        ("Data intake and live alerts", "MQTT (Eclipse Mosquitto) over TLS, HTTPS"),
        ("Forecasting", "statistical and machine-learning models, chosen per product by backtesting"),
        ("AI", "language model and embeddings through a LiteLLM gateway"),
    ]
    t = Table([[Paragraph(f"<b>{a}</b>", SMALL), Paragraph(b, SMALL)] for a, b in tech], colWidths=[140, 383])
    t.setStyle(TableStyle([("LINEBELOW", (0, 0), (-1, -1), 0.4, LINE), ("VALIGN", (0, 0), (-1, -1), "TOP"),
                           ("TOPPADDING", (0, 0), (-1, -1), 3.5), ("BOTTOMPADDING", (0, 0), (-1, -1), 3.5)]))
    story.append(t)
    story.append(PageBreak())

    story += [
        Paragraph("Architecture", H1),
        Paragraph("Data flows from top to bottom. The numbers match the explanation on the next page.", CAP),
        Spacer(1, 8),
        architecture(),
        PageBreak(),
        Paragraph("How it works", H1),
    ]
    for i, (head, text) in enumerate(STEPS, 1):
        story.append(Paragraph(f"<font color='#5b4bdb'><b>{i}</b></font>&nbsp;&nbsp;&nbsp;<b>{head}</b> {text}", STEP))
    story += [Spacer(1, 6), KeepTogether([Paragraph("The tree of brains and the learning loop", H2), Spacer(1, 4),
                                          tree_and_loop()])]
    story += [Spacer(1, 6), Paragraph("Why it can be trusted", H2)]
    for head, text in TRUST:
        story.append(Paragraph(f"<b>{head}</b> {text}", BODY, bulletText="•"))


    def footer(canvas, doc_):
        canvas.saveState()
        canvas.setFont("Helvetica", 7.5)
        canvas.setFillColor(FAINT)
        canvas.drawString(36, 20, "SecondBrain  ·  Platform architecture")
        canvas.drawRightString(A4[0] - 36, 20, f"{doc_.page}")
        canvas.restoreState()

    doc.build(story, onFirstPage=footer, onLaterPages=footer)


if __name__ == "__main__":
    build(sys.argv[1] if len(sys.argv) > 1 else "docs/SecondBrain_Platform_Architecture.pdf")
