"""Build the demo 2 guide: how to run and present it, and how it works (architecture, flow, stack).

    uv run python scripts/build_demo_guide.py docs/SecondBrain_Demo2_Guide.pdf
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from reportlab.graphics.shapes import Drawing, Line, Rect, String  # noqa: E402
from reportlab.lib import colors  # noqa: E402
from reportlab.lib.pagesizes import A4  # noqa: E402
from reportlab.lib.styles import ParagraphStyle  # noqa: E402
from reportlab.platypus import (KeepTogether, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table,  # noqa: E402
                                TableStyle)

from build_architecture_brief import (AMBER, BODY, CAP, CYAN, FAINT, GREEN, H1, H2, INK, KICKER, LEAD, LINE,  # noqa: E402
                                      MUTED, SMALL, SMALLB, SOFT_A, SOFT_C, SOFT_G, SOFT_S, SOFT_V, TITLE, VIOLET,
                                      arrow, badge, box, zone)
from build_example_docs import pdf_font  # noqa: E402

UNI = pdf_font() or "Helvetica"
PINK, SOFT_P = colors.HexColor("#be185d"), colors.HexColor("#fdf0f6")
CODE = ParagraphStyle("Code", parent=SMALL, fontName="Courier", fontSize=8, leading=10.5, textColor=INK,
                      backColor=SOFT_S, borderColor=LINE, borderWidth=0.5, borderPadding=5, spaceBefore=3, spaceAfter=7)
BUL = ParagraphStyle("Bul", parent=BODY, leftIndent=12, bulletIndent=2, spaceAfter=3)
SAY = ParagraphStyle("Say", parent=BODY, leftIndent=10, textColor=colors.HexColor("#3b3f6b"), fontName="Helvetica-Oblique",
                     borderColor=VIOLET, borderWidth=0, spaceAfter=6)


def rs(text: str) -> str:
    """The rupee sign in a font that has it."""
    return text.replace("₹", f'<font name="{UNI}">₹</font>')


def P(text, style=BODY):
    return Paragraph(rs(text), style)


def bullets(items, style=BUL):
    return [Paragraph(rs(t), style, bulletText="•") for t in items]


def code(text):
    return Paragraph(text.replace("&", "&amp;").replace("<", "&lt;").replace("\n", "<br/>"), CODE)


def table(rows, widths, head=True, zebra=True, size=8.4):
    st = ParagraphStyle("t", parent=SMALL, fontSize=size, leading=size + 3)
    sb = ParagraphStyle("tb", parent=SMALLB, fontSize=size, leading=size + 3)
    data = [[Paragraph(rs(str(c)), sb if (head and r == 0) else st) for c in row] for r, row in enumerate(rows)]
    t = Table(data, colWidths=widths, repeatRows=1 if head else 0)
    style = [("VALIGN", (0, 0), (-1, -1), "TOP"), ("LEFTPADDING", (0, 0), (-1, -1), 5),
             ("RIGHTPADDING", (0, 0), (-1, -1), 5), ("TOPPADDING", (0, 0), (-1, -1), 3.5),
             ("BOTTOMPADDING", (0, 0), (-1, -1), 3.5), ("LINEBELOW", (0, 0), (-1, -1), 0.4, LINE)]
    if head:
        style += [("BACKGROUND", (0, 0), (-1, 0), SOFT_V), ("LINEBELOW", (0, 0), (-1, 0), 0.8, VIOLET)]
    if zebra:
        style += [("BACKGROUND", (0, r), (-1, r), SOFT_S) for r in range(2 if head else 1, len(rows), 2)]
    t.setStyle(TableStyle(style))
    return t


def callout(text, fill=SOFT_A, stroke=AMBER):
    t = Table([[P(text, ParagraphStyle("c", parent=BODY, spaceAfter=0))]], colWidths=[523])
    t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), fill), ("BOX", (0, 0), (-1, -1), 0.6, stroke),
                           ("LEFTPADDING", (0, 0), (-1, -1), 9), ("RIGHTPADDING", (0, 0), (-1, -1), 9),
                           ("TOPPADDING", (0, 0), (-1, -1), 7), ("BOTTOMPADDING", (0, 0), (-1, -1), 7)]))
    return t


# ---------------------------------------------------------------- diagrams

def architecture() -> Drawing:
    """Components and connections of demo 2."""
    W, H = 523, 560
    d = Drawing(W, H)

    # sources
    zone(d, 0, 478, W, 82, "THE PLATFORMS  ·  simulated: 600 merchants, 1.2 lakh customers, 16,295 products", SOFT_S, LINE, MUTED)
    for i, (t, s) in enumerate([("Shopify", "240 D2C online stores"), ("Reliance Retail", "210 Smart, Digital, Trends"),
                                ("DMart", "150 hypermarkets")]):
        box(d, 10 + i * 171, 486, 161, 50, t, s)
    arrow(d, W / 2, 478, W / 2, 452, MUTED, 1.4)
    d.add(String(W / 2 + 8, 462, "orders, catalogues, stock, 14-month history (JSON over MQTT)", fontName="Helvetica-Oblique",
                 fontSize=7, fillColor=MUTED))

    # ingest broker
    box(d, 120, 412, 283, 40, "MQTT ingest broker (Mosquitto)", "commerce2/v1/{platform}/{merchant}/orders · catalog · stock · history",
        SOFT_C, colors.HexColor("#9fd8e6"), CYAN, 8.6)

    # two consumers
    arrow(d, 200, 412, 130, 372, CYAN, 1.3)
    arrow(d, 330, 412, 395, 372, CYAN, 1.3)
    zone(d, 0, 222, 255, 150, "AI ENGINE  ·  in memory, ~0.1 s order to signal", SOFT_V, colors.HexColor("#c9c1ff"), VIOLET)
    box(d, 10, 302, 235, 46, "Forecaster (numpy)", "every merchant x product, festivals from last year, nowcast by the hour")
    box(d, 10, 250, 112, 46, "Decisions", "discount, restock, surge, festival, clearance, win-back")
    box(d, 133, 250, 112, 46, "Learning", "offer outcomes, price response, lessons")
    zone(d, 268, 222, 255, 150, "DATA PLATFORM  ·  TimescaleDB (PostgreSQL 17)", SOFT_G, colors.HexColor("#a7dcbc"), GREEN)
    box(d, 278, 302, 235, 46, "Writer (COPY batches)", "hypertables: every order line; every product every day")
    box(d, 278, 250, 112, 46, "Rollups + columnstore", "hourly, daily totals; old days compressed")
    box(d, 401, 250, 112, 46, "SQL watcher", "shelf losses, out-of-stock streaks, odd store-hours")

    # live broker
    arrow(d, 128, 222, 200, 182, VIOLET, 1.3)
    arrow(d, 395, 222, 323, 182, GREEN, 1.3)
    box(d, 120, 142, 283, 40, "MQTT live broker (Mosquitto + WebSockets)",
        "sb/commerce2/stats · signals · outcomes · tsdb/...  ·  per-user logins and ACLs", SOFT_P, colors.HexColor("#f2b8d2"), PINK, 8.6)
    arrow(d, W / 2, 142, W / 2, 118, PINK, 1.3)

    # app + screens
    zone(d, 0, 0, W, 118, "THE APP  ·  FastAPI (Python) serving the screens; the same login and permission model as the brain",
         SOFT_S, LINE, MUTED)
    for i, (t, s) in enumerate([("Live Commerce", "the live wall"), ("HQ", "command centre"), ("Graph", "network + permissions"),
                                ("Chat", "questions, with sources"), ("TimescaleDB", "the data platform")]):
        box(d, 8 + i * 103, 46, 96, 46, t, s, colors.white, LINE, INK, 7.8)
    d.add(String(10, 26, "Stores behind the app: PostgreSQL + pgvector + ltree (memories, permissions)  ·  Redis (short-term memory)  ·  "
                 "FalkorDB (relationship graph)", fontName="Helvetica", fontSize=6.8, fillColor=MUTED))
    d.add(String(10, 14, "Language models through the company LiteLLM proxy: they explain and answer; the numbers come from the "
                 "statistics, never from the model.", fontName="Helvetica", fontSize=6.8, fillColor=MUTED))
    # side arrows from engine/tsdb HTTP to app
    d.add(Line(4, 250, 4, 92, strokeColor=FAINT, strokeWidth=0.8, strokeDashArray=[2, 2]))
    d.add(Line(519, 250, 519, 92, strokeColor=FAINT, strokeWidth=0.8, strokeDashArray=[2, 2]))
    d.add(String(8, 196, "HTTP API :8776", fontName="Helvetica", fontSize=6.3, fillColor=FAINT))
    d.add(String(515, 196, "SQL", fontName="Helvetica", fontSize=6.3, fillColor=FAINT, textAnchor="end"))
    return d


def order_flow() -> Drawing:
    """The life of one order, left to right."""
    W, H = 523, 150
    d = Drawing(W, H)
    steps = [("A customer orders", "at DMart Mansarovar, 7:42 pm"), ("Ingest broker", "MQTT, ~20 ms"),
             ("Engine", "state + forecast updated, < 1 ms"), ("Decision", "signal checked against the 80% bound"),
             ("Live broker", "only to who may see it"), ("Screen", "about 0.1 s after the order")]
    bw, gap = 78, 11
    for i, (t, s) in enumerate(steps):
        x = i * (bw + gap)
        box(d, x, 74, bw, 54, t, s, SOFT_V if i in (2, 3) else colors.white, LINE, INK, 7.6)
        badge(d, x + 10, 128, i + 1)
        if i < len(steps) - 1:
            arrow(d, x + bw + 1, 101, x + bw + gap - 1, 101, VIOLET, 1.1, 3.5)
    arrow(d, 2 * (bw + gap) - 30, 74, 2 * (bw + gap) - 30, 40, GREEN, 1.1)
    box(d, 2 * (bw + gap) - 110, 4, 250, 36, "Same order, in parallel: TimescaleDB writer",
        "stored within ~0.25 s; rolled up hourly and daily; checked by SQL at each close", SOFT_G, colors.HexColor("#a7dcbc"), GREEN, 7.6)
    return d


def forecast_parts() -> Drawing:
    W, H = 523, 92
    d = Drawing(W, H)
    parts = [("Level", "recent normal sales,\nsmoothed"), ("Weekday", "per platform and\ncategory"), ("Calendar", "paydays,\nmonth end"),
             ("Festival", "4 weeks before to\n3 after, from last year"), ("Price", "offers move demand by\nthe learned elasticity")]
    bw, gap = 92, 15
    for i, (t, s) in enumerate(parts):
        x = i * (bw + gap)
        box(d, x, 22, bw, 52, t, s.replace("\n", " "), SOFT_V if i == 3 else colors.white, LINE, INK, 8.4)
        if i < len(parts) - 1:
            d.add(String(x + bw + gap / 2, 44, "x", fontName="Helvetica-Bold", fontSize=11, fillColor=VIOLET, textAnchor="middle"))
    d.add(String(W / 2, 6, "forecast for tomorrow  =  level  x  weekday  x  calendar  x  festival  x  price effect   (16,295 series at once)",
                 fontName="Helvetica-Oblique", fontSize=7.4, fillColor=MUTED, textAnchor="middle"))
    return d


def tsdb_layout() -> Drawing:
    W, H = 523, 170
    d = Drawing(W, H)
    box(d, 0, 106, 160, 54, "cm_order_lines", "hypertable: every order line, one chunk per simulated day", SOFT_G, colors.HexColor("#a7dcbc"), GREEN)
    box(d, 0, 30, 160, 60, "cm_product_daily", "hypertable: every product every day, 14 months + each closed live day, one chunk a month",
        SOFT_G, colors.HexColor("#a7dcbc"), GREEN)
    box(d, 190, 120, 140, 40, "cm_sales_1h", "orders, units, revenue per store per hour")
    box(d, 190, 70, 140, 40, "cm_sales_1d", "daily, built on the hourly one")
    box(d, 190, 20, 140, 40, "cm_sku_1d", "units per product per day")
    arrow(d, 160, 133, 190, 140, GREEN, 1)
    arrow(d, 160, 128, 190, 40, GREEN, 1)
    arrow(d, 260, 120, 260, 111, MUTED, 1)
    box(d, 360, 106, 163, 54, "SQL watcher", "each sim hour and day: shelf losses, out-of-stock streaks, odd store-hours",
        SOFT_P, colors.HexColor("#f2b8d2"), PINK)
    box(d, 360, 30, 163, 60, "Columnstore", "finished days compressed column by column: about 14x smaller, same SQL", SOFT_C,
        colors.HexColor("#9fd8e6"), CYAN)
    arrow(d, 330, 140, 360, 135, PINK, 1)
    d.add(String(W / 2, 4, "continuous aggregates (middle) are kept current as orders arrive; findings go out over MQTT and into cm_findings",
                 fontName="Helvetica-Oblique", fontSize=7.2, fillColor=MUTED, textAnchor="middle"))
    return d


# ---------------------------------------------------------------- the document

def story() -> list:
    s = []
    s += [Paragraph("SECONDBRAIN  ·  DEMO 2 GUIDE", KICKER),
          Paragraph("Live Commerce on TimescaleDB", TITLE),
          P("How to run and present the demo, and how it works underneath: the architecture, the flow of an order, the "
            "technology and the numbers, so you can explain any part of it with confidence.", LEAD)]
    s.append(callout("<b>In one sentence.</b> SecondBrain is connected to Shopify, Reliance Retail and DMart (simulated: 600 "
                     "merchants, 1.2 lakh customers, 16,295 products). It forecasts every product at every merchant, tells each "
                     "merchant what to do (stock up, discount, hold price, win customers back) about 0.1 s after the order that "
                     "triggered it, measures whether the advice worked, and learns from it. Demo 2 adds a time-series data "
                     "platform (TimescaleDB) that stores every order and the whole history, and raises its own findings in SQL.",
                     SOFT_V, VIOLET))
    s.append(Spacer(1, 10))
    s.append(Paragraph("What is real and what is simulated", H2))
    s.append(table([
        ["Real (production technology)", "Simulated (so we can show it without a live client)"],
        ["The brokers, databases, engine, forecasting, decisions, learning, permissions, every measurement on screen",
         "The platforms' feed: merchants, customers and their orders come from a demand model with weekdays, hours, "
         "paydays, festivals, trends, stock-outs, competitor price cuts and viral surges"],
        ["The engine only learns from the messages it receives, exactly as it would from a real feed",
         "Names are real Indian retail names and places, used internally only"],
    ], [261, 262]))
    s.append(Spacer(1, 6))
    s.append(P("Swapping the simulator for a real platform feed changes the source of the messages, not the system around them."))

    s.append(Paragraph("The links", H2))
    s.append(table([
        ["Page", "Address", "Use it to show"],
        ["Live Commerce", "http://localhost:8775/commerce", "Orders, signals and learning in real time; speed controls"],
        ["HQ", "http://localhost:8775/hq", "The command centre over all three platforms; the time machine"],
        ["TimescaleDB", "http://localhost:8775/timescale", "The data platform: storage, compression, SQL findings"],
        ["Graph", "http://localhost:8775/graph-view", "The network of platforms and merchants; permissions"],
        ["Chat", "http://localhost:8775/", "Asking questions of a platform, a store or a company, with sources"],
        ["Explore", "http://localhost:8775/explore", "The memory and the forecasts of the example company"],
    ], [80, 160, 283]))
    s.append(Spacer(1, 4))
    s.append(P("On this laptop no admin token is asked for. From another machine it is <b>dev-admin-token</b>.", CAP))

    # ------------------------------------------------ running it
    s.append(PageBreak())
    s.append(Paragraph("1. Starting it", H1))
    s.append(P("Everything runs on the laptop: the databases and brokers in Docker, the Python services in two terminals' worth of "
               "processes, the screens in the browser."))
    s.append(Paragraph("Before a meeting", H2))
    s += bullets(["Restart the Mac (frees the memory other apps were holding), then open only Docker and one browser.",
                  "Run one demo at a time. Demo 2 has everything demo 1 has, plus TimescaleDB.",
                  "Start it 2 minutes before you present: the engine learns 14 months of history at start-up (a few seconds) "
                  "and TimescaleDB loads 6.8 million rows (about 30 seconds)."])
    s.append(Paragraph("Commands", H2))
    s.append(P("Databases and brokers (once after a restart):", SMALL))
    s.append(code("cd ~/Desktop/SecondBrain && docker compose up -d --wait\n"
                  "docker compose -f docker-compose.timescale.yml -p secondbrain-ts up -d --wait"))
    s.append(P("Demo 2 (the app on :8775, the engine on :8776, the TimescaleDB writer and watcher):", SMALL))
    s.append(code("cd ~/Desktop/SecondBrain-v2 && scripts/demo2.sh"))
    s.append(P("Leave that terminal open; Ctrl-C stops it. The simulation starts paused at today's date.", SMALL))
    s.append(Paragraph("The clock and the speeds", H2))
    s.append(P("The clock on the screens is the <b>simulated time</b>, the moment the simulated shops are living in. Every order, "
               "forecast and signal is stamped with it."))
    s.append(table([
        ["Button", "One real second is", "Label on screen", "Use it for"],
        ["Real time (1x)", "one simulated second", "live now", "Showing it working live. Quiet by design: 1.5 lakh orders a day "
                                                              "is about 2 orders a second across all 600 merchants."],
        ["1 min/s", "one simulated minute", "fast-forward", "Watching individual signals arrive"],
        ["1 hr/s", "one simulated hour", "fast-forward", "A day in 24 seconds: the daily rhythm, signals and offers"],
        ["1 day/6 s", "four simulated hours", "fast-forward +N days", "What only shows over days: accuracy, offer results, "
                                                                       "festivals, TimescaleDB findings"],
    ], [72, 92, 92, 267]))
    s.append(Spacer(1, 5))
    s += bullets(["On Start in real time, the simulation first races from midnight to the current time (a burst of a few thousand "
                  "orders a second for a few seconds), then settles at the real pace.",
                  "Rates (orders per second, latency, accuracy) show each new measurement; totals (revenue, orders, signals) only "
                  "roll forward. Pause drops orders per second to 0 within about 2 seconds.",
                  "<b>Back to today</b> on HQ, or reset, returns everything to the real date and clears what was simulated."])

    # ------------------------------------------------ demo script
    s.append(PageBreak())
    s.append(Paragraph("2. A 10-minute walk-through", H1))
    s.append(P("One route that tells the whole story. Words in italics are a suggestion of what to say."))
    steps = [
        ("Live Commerce, in real time (1 minute)", "/commerce: press Start, leave it on Real time.",
         ["Point at the connected platforms, the map of India (one dot per merchant, pulsing as orders land) and the live "
          "orders ticker at the bottom.",
          "The pipeline panel shows each hop with its measured latency: order to signal on screen in about 0.1 s."],
         "This is happening now: every order from Shopify, Reliance and DMart arrives over MQTT and the brain updates its "
         "forecast for that product before the next order."),
        ("Speed it up (2 minutes)", "Switch to 1 hr/s, then 1 day/6 s.",
         ["The signals feed fills: restock before a stock-out, a festival stock-up, a pre-festival offer, a win-back coupon, a "
          "demand surge. Click one to see the evidence: actual vs forecast, stock, cover days, price response.",
          "Forecast in focus: the product's actual sales against its forecast and 80% range. It learns: offers that finished, "
          "what was expected and what happened."],
         "Every merchant gets advice for its own products, with the amount, the expected effect and the evidence, and each "
         "offer promises an uplift at its 80% lower bound, so every promise can be measured."),
        ("HQ, the command centre (2 minutes)", "/hq",
         ["Three platform cards; click one to focus the page on it. The tree of brains: SecondBrain, the platforms, their 600 "
          "merchants, lighting up as signals go down and lessons come back up.",
          "What the brain learned across all merchants, and Needs attention now (Apply offer). The time machine runs a day or a "
          "week; Back to today resets."],
         "One brain over all merchants: what it learns from 276 merchants about staples improves the advice for every one of "
         "them."),
        ("TimescaleDB (2 minutes)", "/timescale",
         ["Rows stored (about 68 lakh at start, growing), rows per second, compression (about 14x: 689 MB to 49 MB), chunks, "
          "rollups and query timings in milliseconds.",
          "Findings from TimescaleDB: rupees lost to empty shelves per store, products out of stock three days running. Click "
          "one to see the exact SQL and how long it took. The merchant explorer draws 14 months for any merchant."],
         "Next to the AI engine there is a proper data platform: every order kept, compressed, queryable in milliseconds, and "
         "it raises its own facts in plain SQL."),
        ("Permissions and questions (2 minutes)", "/graph-view, then /",
         ["Graph: the network from SecondBrain to platforms to categories to merchants. See it as: pick DMart or a single store "
          "and everything it may not see fades.",
          "Chat: pick DMart and ask 'Which DMart stores need restocking today?'; pick a single store and ask about another "
          "store: it cannot see it. The answer lists its sources."],
         "A store sees only itself, a platform only its stores, HQ everything. Other merchants' data is removed before the "
         "language model ever sees the question."),
        ("Close (1 minute)", "Back to HQ",
         ["Headline numbers: store-day forecast accuracy around 89-90% against 77% for the naive method, offers that delivered "
          "their promise around 75-85%, about 0.1 s from order to advice."],
         "Forecasts, decisions and learning for every merchant, in real time, on technology that runs the same way in production."),
    ]
    for i, (title, where, points, say) in enumerate(steps, 1):
        block = [Paragraph(f"{i}. {title}", H2), P(f"<b>Where:</b> {where}", SMALL), Spacer(1, 2)]
        block += bullets(points)
        block.append(P(f"“{say}”", SAY))
        s.append(KeepTogether(block))
    s.append(callout("<b>During Dhanteras and Diwali</b> the brain holds price and raises stock-ups instead of discounts: demand "
                     "is already high, so a discount would only give margin away. If someone asks why there are no discounts, "
                     "that is the answer: it knows when not to discount."))
    s.append(Paragraph("The 30-second version", H2))
    s.append(P("“SecondBrain watches every order from Shopify, Reliance and DMart as it happens. For each of 600 merchants and "
               "16,000 products it forecasts demand, including festivals learned from last year, and tells each merchant what to "
               "do: order stock, run an offer, hold price, win customers back, about a tenth of a second after the order that "
               "triggered it. It measures whether each piece of advice worked and learns from it across all merchants. Every "
               "order is also stored in TimescaleDB, compressed about 14 times, where SQL finds money lost to empty shelves. "
               "Everyone sees only what they're allowed to, down to the database.”", SAY))
    s.append(Paragraph("Checklist before you present", H2))
    s.append(table([
        ["", "Step", "Check"],
        ["1", "Restart the Mac; open Docker and one browser only", "Docker shows the containers running"],
        ["2", "Start the services and scripts/demo2.sh (section 1)", "The terminal says Live Commerce running"],
        ["3", "Open Live Commerce, HQ and TimescaleDB in three tabs", "Badges say Live, not Demo feed"],
        ["4", "Wait until TimescaleDB shows about 68 lakh rows", "About 30 seconds after start"],
        ["5", "Leave it paused at today; press Start when you begin", "The clock shows today's date"],
        ["6", "Run a quick day at 1 day/6 s, then Back to today on HQ", "Signals and findings appear; then everything resets"],
    ], [16, 300, 207]))

    # ------------------------------------------------ architecture
    s.append(PageBreak())
    s.append(Paragraph("3. Architecture", H1))
    s.append(P("Two brokers, two consumers of the same stream, one app. Each part does one job and can be replaced or scaled on "
               "its own."))
    s.append(architecture())
    s.append(Spacer(1, 4))
    s.append(P("<b>Why two brokers.</b> The ingest broker carries the platforms' raw data in; the live broker carries decisions "
               "out to people, with a login per person and an access list generated from their permissions, so a store's screen "
               "only receives that store's signals. <b>Why two consumers.</b> The engine keeps everything it needs in memory to "
               "answer in a tenth of a second; TimescaleDB keeps everything forever for history, reporting and SQL. Neither waits "
               "for the other.", BODY))

    s.append(PageBreak())
    s.append(Paragraph("4. The life of one order", H1))
    s.append(order_flow())
    s.append(Spacer(1, 6))
    s.append(table([
        ["Step", "What happens", "Measured"],
        ["1  Order", "The platform publishes the order (customer, product, quantity, price, offer) as a JSON message on its "
                     "merchant's topic.", "~19k orders/s on average at full speed, peaks ~40-55k/s"],
        ["2  Ingest broker", "Mosquitto delivers it to every subscriber: the engine and the TimescaleDB writer.", "~20 ms"],
        ["3  Engine", "Updates the merchant's and product's state, today's sales against \"expected by now\" (the nowcast), "
                      "and stock.", "< 1 ms per batch of orders"],
        ["4  Decision", "If sales are far from expected, stock will run out, a festival is coming or customers have lapsed, it "
                        "sizes an action (discount, units to order, coupon) and keeps it only if its 80% lower bound pays.",
         "at most a few signals per merchant per hour"],
        ["5  Live broker", "Publishes the signal on sb/commerce2/signals/{platform}/{merchant}; the broker's access list decides "
                           "who receives it.", "~5 ms to the screen"],
        ["6  Screen", "The wall, HQ and Graph show it; autopilot merchants apply offers automatically, the others click Apply.",
         "order to screen ~0.1 s"],
        ["Later", "When an offer's window closes, the actual uplift is compared with the promise, the category's price response "
                  "is updated, and HQ shows the lesson.", "offers that delivered: ~75-85%"],
        ["In parallel", "The TimescaleDB writer COPYs the order into cm_order_lines within a quarter second; rollups and the SQL "
                        "watcher work on it at each hour and day close.", "16-22k rows/s live, no backlog"],
    ], [72, 330, 121]))
    s.append(Spacer(1, 6))
    s.append(P("Day close (each simulated midnight): every forecast is scored against what sold, every series is updated, "
               "restock and festival plans are made for the new day, and TimescaleDB writes the closed day for every product. "
               "It takes about 0.3 s for 16,295 series."))

    # ------------------------------------------------ stack
    s.append(PageBreak())
    s.append(Paragraph("5. The technology, and why", H1))
    s.append(table([
        ["Layer", "Technology", "Why this one"],
        ["Messaging", "MQTT, Eclipse Mosquitto 2.0 (two brokers; WebSockets for browsers)",
         "Built for many small devices and senders, light enough for a POS or an app to publish directly, per-topic access "
         "lists, and the browser can subscribe directly. Kafka can be added as a second way in if a client already streams "
         "through it."],
        ["Engine", "Python 3.12, numpy (vectorised), FastAPI for its HTTP API",
         "16,295 forecasts updated in one step; in memory, so a decision takes milliseconds"],
        ["Time-series store", "TimescaleDB 2.31 on PostgreSQL 17",
         "Plain SQL and PostgreSQL tooling, plus automatic time chunks (hypertables), continuously updated rollups and a "
         "compressed columnstore: years of orders stay small and fast"],
        ["Brain store", "PostgreSQL with pgvector and ltree, row-level security",
         "Memories with their meaning (vector embeddings) next to who may see them; the brain tree (company, department) as "
         "ltree paths; permissions enforced by the database itself"],
        ["Short-term memory", "Redis 7", "The current conversation, kept for a day, fast"],
        ["Relationships", "FalkorDB 4.22 (graph database)", "Who and what is connected: people, customers, suppliers, products"],
        ["Language models", "Company LiteLLM proxy (chat model + embedding model)",
         "Explains and answers in plain language; never produces the numbers, which come from the statistics"],
        ["Forecasting (example company)", "ETS, ARIMA, Theta, LightGBM, chosen per product by backtest",
         "Deeper merchandising forecasts on the example company's 2+ years of sales (Explore, forecast tab)"],
        ["Screens", "HTML, vanilla JavaScript, SVG and canvas; mqtt.js",
         "No build step, fast on a laptop; needs the internet only for the fonts and mqtt.js"],
        ["Packaging", "Docker Compose (databases, brokers), uv (Python)", "One command starts the services; the same images run on a server"],
    ], [82, 150, 291]))
    s.append(Spacer(1, 6))
    s.append(Paragraph("Ports", H2))
    s.append(table([
        ["Service", "Port", "Service", "Port"],
        ["App (pages and API)", "8775", "TimescaleDB (demo 2 database)", "55433"],
        ["Engine HTTP API", "8776", "PostgreSQL (demo 1 database)", "55432"],
        ["MQTT ingest broker", "51883", "Redis", "56379"],
        ["MQTT live broker / WebSockets", "51884 / 59001", "FalkorDB (+ its own browser)", "56380 (53000)"],
    ], [150, 90, 180, 103]))

    # ------------------------------------------------ forecasting + signals
    s.append(PageBreak())
    s.append(Paragraph("6. How it forecasts, decides and learns", H1))
    s.append(Paragraph("The forecast", H2))
    s.append(forecast_parts())
    s += bullets([
        "<b>Level</b> follows each product's normal sales, smoothed so one odd day doesn't swing it; how fast it follows is chosen "
        "per category from the history.",
        "<b>Weekday</b>: a hypermarket sells far more on Sunday, an online store doesn't, so the weekly shape is learned per "
        "platform and category, and per product once it has enough sales.",
        "<b>Festival</b>: Diwali moves by up to three weeks a year, so the engine knows the dates and learns, from last year's "
        "history, each day's effect from four weeks before (the build-up) to three weeks after (the slump), per category, and "
        "how festival-sensitive each merchant is.",
        "<b>Price</b>: an offer's effect is (price / list price) raised to the learned elasticity, so offer days don't inflate the "
        "normal level.",
        "<b>Nowcast</b>: a learned hour-of-day profile per platform turns the day's forecast into \"expected by now\", so a "
        "drop shows at 2 pm, not at midnight.",
        "<b>Sold-out days</b> show what was on the shelf, not demand, so they don't pull the forecast down and aren't scored.",
    ])
    s.append(Paragraph("The decisions", H2))
    s.append(table([
        ["Signal", "When", "What it says"],
        ["Restock", "Stock will run out before the next delivery could arrive", "Order N units today (expedite if late)"],
        ["Festival", "A festival is near and stock won't cover the forecast, or there is surplus", "Stock up N units, or a pre-festival offer"],
        ["Discount", "Sales well below forecast for 2-3 days, overstock, or price-sensitive products with margin", "x% off for y hours, with the expected uplift"],
        ["Surge", "Sales far above forecast", "Hold price, check stock"],
        ["Clearance", "Perishables that won't sell before they spoil", "A short markdown"],
        ["Win-back", "A merchant's regular customers have lapsed", "A coupon for N customers"],
    ], [62, 230, 231]))
    s.append(Spacer(1, 5))
    s.append(P("An offer only goes out if, at its <b>80% lower bound</b>, it adds at least one unit and pays for itself: margin "
               "floors, the price response, how long a drop usually lasts and forecast error are all in that bound. Half the "
               "merchants run on autopilot; the others click Apply."))
    s.append(Paragraph("The learning", H2))
    s += bullets(["When an offer's window closes, actual sales are compared with what the forecast said would sell without it; "
                  "the category's price response moves accordingly.",
                  "Lessons on HQ are written from evidence only: for example, discounts on staples at about 11% margin don't pay, "
                  "learned from 276 merchants, so the brain holds price there.",
                  "Accuracy is scored live every simulated day against the forecast made the evening before, and against a naive "
                  "method (same weekday last week)."])

    # ------------------------------------------------ timescale
    s.append(PageBreak())
    s.append(Paragraph("7. The TimescaleDB layer", H1))
    s.append(P("TimescaleDB is PostgreSQL with time-series features. The demo uses the three that matter most:"))
    s += bullets(["<b>Hypertables</b>: a normal SQL table that TimescaleDB splits into time chunks automatically (one per simulated "
                  "day for orders, one per month for daily product sales). Queries read only the chunks they need.",
                  "<b>Continuous aggregates</b>: hourly and daily totals that the database keeps up to date as orders arrive, so a "
                  "question over a year reads hundreds of rows, not millions. The daily one is built on the hourly one.",
                  "<b>Columnstore</b>: finished days are compressed column by column, segmented by merchant: about 14x smaller "
                  "(689 MB to 49 MB) and still answered by the same SQL."])
    s.append(tsdb_layout())
    s.append(Paragraph("What the SQL watcher reports", H2))
    s.append(table([
        ["Finding", "Checked", "Example"],
        ["Sales lost to empty shelves", "each simulated day, per store",
         "Reliance Digital Vaishali Nagar: about ₹13 lakh of sales lost to empty shelves"],
        ["Out of stock three days running", "each simulated day, per product",
         "Apple iPhone 16 at Reliance Digital Main Market: out of stock 3 days, about ₹2.8 lakh lost"],
        ["A store far off its usual this hour", "each simulated hour, per store",
         "Compared with its own platform's other stores at that hour, so festivals and weekdays cancel out; rarely fires"],
    ], [130, 120, 273]))
    s.append(Spacer(1, 5))
    s.append(P("Every finding goes out over MQTT with the exact query it came from and how long it took, and is kept in "
               "cm_findings. Product-level spikes are left to the engine: measured against the simulator's injected events, SQL "
               "rules on raw history got at most 8% of them right, because one product's day-to-day randomness is as large as "
               "most real changes. Only facts that are true by construction are published."))
    s.append(Paragraph("TimescaleDB numbers", H2))
    s.append(table([
        ["What", "Measured on this laptop"],
        ["History loaded at start-up", "6.84 million rows in about 25-40 s (170-270k rows/s, COPY)"],
        ["Live orders written", "16-22k rows/s at a day every 6 s, batches of a quarter second, no backlog"],
        ["Compression", "about 14-16x smaller"],
        ["Engine count vs TimescaleDB count", "identical, order for order"],
        ["Hourly check over all 600 stores", "2-4 ms"],
        ["Daily checks (shelf losses, streaks)", "30-150 ms"],
    ], [200, 323]))

    # ------------------------------------------------ permissions
    s.append(PageBreak())
    s.append(Paragraph("8. Permissions and security", H1))
    s += bullets([
        "<b>The tree of brains</b>: SecondBrain (HQ) above the platforms, the platforms above their merchants; in a company, the "
        "company above its departments. Each brain is a path (ltree) in PostgreSQL.",
        "<b>The database enforces it</b>: memories and sales carry their brain and sensitivity, and row-level security decides "
        "what each person's queries can return; the app's database user cannot bypass it.",
        "<b>The live broker enforces it too</b>: each person has their own broker login, and the access list is generated from "
        "their grants, so a store's screen never receives another store's signals.",
        "<b>The language model never sees what you may not</b>: for a question, the scope's data is filtered first (a store: only "
        "itself), then the model writes the answer from what is left, with sources.",
        "<b>Admin actions</b> (start, speed, apply offer) need the admin token; in the local demo it is handed to pages opened on "
        "the laptop itself, and anyone else must type it.",
        "<b>In production</b>: MQTT over TLS (port 8883) with client certificates, HTTPS, SSO instead of the demo's person picker, "
        "and secrets in a vault rather than a local file.",
    ])

    # ------------------------------------------------ numbers
    s.append(Paragraph("9. The numbers to quote", H1))
    s.append(table([
        ["Measure", "Value", "What it means"],
        ["Order to advice on screen", "about 0.1 s", "From the platform's order to the merchant's signal"],
        ["Throughput", "~19k orders/s average, 40-55k peaks; engine capacity 150-200k/s", "At a simulated day every 6 seconds"],
        ["Store-day forecast accuracy", "about 89-90% (naive 77%)", "100% minus the error of each store's daily sales, in-stock days"],
        ["By platform", "Shopify ~76%, Reliance ~89%, DMart ~90%", "Shopify's online brands sell one or two units of a product a day: harder"],
        ["Product-day accuracy", "about 64% (naive 45%)", "One product in one store on one day: the hardest number in retail"],
        ["Best possible (perfect knowledge)", "Shopify ~83%, Reliance ~94%, DMart ~96%", "The rest is randomness no model can predict"],
        ["80% forecast range held", "about 90%", "The ranges are honest"],
        ["Offers that delivered their promise", "about 75-85%", "Each offer promises its 80% lower bound"],
        ["Diwali week, worst store-day error", "about 16%", "Forecast from last year's festival season"],
    ], [128, 175, 220]))

    # ------------------------------------------------ Q&A
    s.append(PageBreak())
    s.append(Paragraph("10. Questions you may get", H1))
    qa = [
        ("Is this real data?", "The platforms' feed is simulated, with real-world patterns (weekdays, hours, paydays, festivals, "
         "stock-outs, competitor cuts, viral products). Everything else, from the brokers to the forecasts and the measurements, "
         "is the real system. Connecting a real platform means replacing the simulator's messages with theirs."),
        ("How accurate is it?", "About 89-90% for each store's daily sales against 77% for the naive method, about 64% for single "
         "products (naive 45%). For scale: a forecast that knew the simulator's true demand would score 83-96%; the remaining "
         "error is randomness."),
        ("Does the AI make up the numbers?", "No. The forecasts, decisions and measurements are statistics. The language model "
         "only explains them and answers questions, from data the person may see, with sources."),
        ("Why MQTT and not Kafka?", "MQTT is made for many small senders (shops, apps, POS terminals), has access lists per topic, "
         "and browsers can subscribe directly. Kafka is better as a central log between big systems; it can be added as another "
         "way in without changing the engine."),
        ("Why TimescaleDB?", "It is PostgreSQL, so plain SQL, the same tools and the same permissions model, with what time-series "
         "data needs: automatic chunks, continuously updated rollups and about 14x compression."),
        ("Will it scale beyond a laptop?", "On this laptop the engine handles 150-200k orders a second on one core. In production "
         "the brokers, the engine (by platform or region) and the database scale out separately; the messages and code stay the "
         "same."),
        ("What if a merchant ignores the advice?", "Half the merchants run on autopilot, the others click Apply. Ignored offers are "
         "measured too: they tell the brain how long drops last without an offer."),
        ("What does a merchant see?", "Only its own data and signals, enforced by the database and the broker, and also in chat."),
    ]
    for q, a in qa:
        s.append(KeepTogether([P(f"<b>{q}</b>", BODY), P(a, ParagraphStyle("a", parent=BODY, leftIndent=10, spaceAfter=7))]))

    s.append(Paragraph("11. If something looks wrong", H1))
    s.append(table([
        ["You see", "Do this"],
        ["\"Engine busy, connecting…\"", "Wait a few seconds: the engine is loading or in a big batch. The page connects by itself."],
        ["\"Demo feed\" or \"Demo data\" badge", "The engine isn't running: start scripts/demo2.sh, then reload the page."],
        ["Numbers barely move", "Real time is quiet (about 2 orders a second); switch to 1 hr/s or 1 day/6 s."],
        ["TimescaleDB page shows 0 live rows", "Press Start on Live Commerce; history rows are there from the start."],
        ["Stutters, slow pages", "Memory: close other apps and browser tabs, run one demo at a time, restart the Mac before the meeting."],
        ["Want a fresh start", "HQ: Back to today (or restart scripts/demo2.sh)."],
    ], [170, 353]))
    return s


def build(path: str) -> None:
    doc = SimpleDocTemplate(path, pagesize=A4, leftMargin=36, rightMargin=36, topMargin=34, bottomMargin=34,
                            title="SecondBrain demo 2 guide", author="SecondBrain")

    def footer(canvas, doc_):
        canvas.saveState()
        canvas.setFont("Helvetica", 7.5)
        canvas.setFillColor(FAINT)
        canvas.drawString(36, 20, "SecondBrain  ·  Demo 2 guide: Live Commerce on TimescaleDB")
        canvas.drawRightString(A4[0] - 36, 20, f"{doc_.page}")
        canvas.restoreState()

    doc.build(story(), onFirstPage=footer, onLaterPages=footer)


if __name__ == "__main__":
    build(sys.argv[1] if len(sys.argv) > 1 else "docs/SecondBrain_Demo2_Guide.pdf")
    print("written")
