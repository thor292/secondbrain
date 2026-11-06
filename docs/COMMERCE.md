# Live Commerce

A live demonstration of SecondBrain watching three commerce platforms at once:
Shopify (D2C online stores), Reliance Retail (Smart, Digital, Trends, JioMart
sellers) and DMart (hypermarkets). It shows 600 merchants and 120,000
customers, analysed, forecast and advised in real time. The data is simulated,
but the system around it is the real one: the same brokers, the same streaming
engine and the same measurements a production feed would use.

Open **http://localhost:8765/commerce**. It uses the same admin token as HQ.

## Running it

The usual services first (`docker compose up -d`: databases, both brokers),
then the app, then Live Commerce:

```bash
uv run uvicorn secondbrain.api:app --port 8765
uv run python -m secondbrain.commerce            # waits for Start on the wall
```

Options: `--autostart`, `--speed 1x|1min|1h|1day`, `--merchants`, `--customers`,
`--autopilot 0.5` (share of merchants whose offers apply themselves; the others
wait for a click on the wall). The page falls back to a labelled demo feed in
the browser when the engine isn't running.

## What happens

```
simulator --(MQTT ingest broker)--> engine --(live broker, permission-checked)--> wall
    ^                                  |
    +------- apply offer / speed ------+
```

- **Simulator** (`commerce/world.py`, `simulator.py`): the platforms, merchants and
  customers.
  - Demand per merchant and product has weekday and time-of-day rhythms, paydays,
    festivals (Dhanteras, Diwali, the wedding season), slow trends, stock that runs
    out and gets restocked, and price sensitivity per category.
  - It also injects unexpected events: competitor price cuts and viral surges.
  - Orders stream over MQTT, one message per merchant per tick, after a 14-month
    history backfill (sales, prices and sold-out days), so last year's Dussehra, Diwali
    and wedding season are in the data. It runs in its own process.
- **Engine** (`commerce/engine.py`, `forecast.py`, `offers.py`, `state.py`): learns
  only from the messages, so a real platform feed can replace the simulator.
  - **Streaming forecast for every merchant × product** (16k series): level ×
    weekday pattern (per platform: a hypermarket's weekend is not an online store's) ×
    payday and month-end effects × festival effect × price effect, fitted on the
    backfill and updated at every day close. The festival effect covers every day from
    four weeks before a festival to three weeks after (build-up, the day, the slump),
    learned per category from last year, scaled by how festival-sensitive each
    merchant was.
  - **Intraday nowcast:** "expected by now", checked continuously.
  - **Live accuracy:** every day is scored against the forecast made the evening
    before, and against a naive baseline (same weekday last week), on products that
    stayed in stock (a sold-out day shows the shelf, not demand).
  - **Decision signals:** discounts, restocks, demand surges, festival preparation,
    perishable clearance and win-back coupons. Each carries the amount, the expected
    effect, a confidence and the evidence.
  - **Learning loop:** when an offer's window closes, the actual uplift is compared
    with the forecast, and the category's learned price sensitivity moves.
- **Wall** (`static/commerce.html`): the live map of every merchant, the pipeline
  with measured latencies, the signals feed with *Apply offer*, the forecast in
  focus, and what the brain has learned.

The message contract is in `commerce/contract.py`.

## The numbers, and what they mean

Measured on an M1 Pro laptop, 600 merchants, fast-forwarding 21 days from
Dhanteras (6 Nov 2026) through Diwali and into the wedding season:

| | |
|---|---|
| Orders streamed | ~19k/s on average, peaks ~55k/s |
| Engine capacity | ~150-200k orders/s on one core (busy ~20% at demo load) |
| Order → decision signal | ~0.1 s end to end, of which the broker hop ~20 ms, processing < 1 ms |
| Signal → screen | ~5 ms |
| Forecast accuracy, store-day | ~89% (100% − WAPE), naive baseline ~77% |
| By platform, store-day | Shopify ~76%, Reliance Retail ~89%, DMart ~90% |
| Forecast accuracy, product-day | ~64%, naive baseline ~45% |
| Diwali week, worst store-day error | ~16% (naive up to 58%) |
| 80% forecast range held | ~90% of the time |
| Offers that delivered their promised uplift | ~75-85% (each offer states an 80% lower bound) |
| Signal mix | restocks, festival stock-ups and pre-festival offers, win-back coupons, demand surges, clearance of perishables; discounts once Diwali demand has passed |

For scale, a forecast that knew the simulator's true demand would score ~83%
(Shopify), ~94% (Reliance) and ~96% (DMart) store-day: the rest is noise no
model can predict. Shopify is lowest because its D2C brands sell one or two
units of a product a day.

During Dhanteras and Diwali the brain raises stock-ups and holds price: demand
is already high, so a discount would give margin away. Offers pick up after the
festival (clearance, slow movers, price-sensitive high-margin products).

Product-level daily accuracy is always the hardest number in retail. A
single product in a single store sells a handful of units a day, with about
30% random day-to-day swing built into the simulation, and Shopify's D2C items
sell a unit or two a day. What matters is the gap to the naive baseline: the
forecast makes about a quarter fewer errors. For comparison, the merchandising
forecasts in the main demo (`python -m secondbrain.forecasting`, backtested
gradient boosting) reach a median of ~80% product-level accuracy against 62%
naive on 2+ years of history.

Signals raised at a day close (restock plans, festival preparation) answer the
whole day's data, not one order, so they show "from the end-of-day re-plan"
instead of a latency.

Offers are only raised when they pay for themselves. A discount on a
thin-margin staple that is merely slow would give margin away, so discounts come
mostly from price-sensitive, higher-margin products (beauty, fashion, sweets,
accessories) and overstock. Dips that look like a competitor's price cut are
tracked: the engine learns what share of such dips last, and so far in the
simulation almost all of them revert, so it rarely spends a discount on them.

## Before showing it

- Start the engine before the simulator. `python -m secondbrain.commerce` does
  this. The history backfill is not retained, so an engine started late learns
  from the first live day instead. Fitting 14 months for 16k series takes ~10 s
  at startup.
- A day every 6 seconds is the most impressive speed: the simulation starts
  today, so Diwali and the wedding season arrive within a minute or two. Use 1 hr/s to watch
  individual signals and click *Apply offer*.
- The page needs internet for the font and mqtt.js, like the other pages.
