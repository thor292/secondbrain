# Live Commerce on TimescaleDB (demo 2)

Demo 2 is Live Commerce (docs/COMMERCE.md) with a time-series database under it.
Every order from Shopify, Reliance Retail and DMart, and the 14 months of sales
history the merchants bring with them, is stored in TimescaleDB as it streams
in. The database keeps hourly and daily totals current, compresses finished
days, and runs SQL checks whose findings go out over MQTT like the engine's
signals.

The engine (`commerce/engine.py`) is unchanged: it still forecasts in memory,
which is what lets it answer in ~0.1 s. TimescaleDB sits next to it as a second
consumer of the same ingest stream, the way a company's data platform sits next
to its operational systems.

```
simulator --MQTT--> engine ------------------------------MQTT--> screens
              \                                              /
               +--> TimescaleDB writer -> hypertables -> watcher
                                          rollups, columnstore
```

## Running it next to demo 1

Demo 2 lives in its own checkout (branch `demo2-timescale`), with its own
database, ports and MQTT topics, so demo 1 keeps running untouched:

| | Demo 1 | Demo 2 |
|---|---|---|
| App | :8765 | :8775 |
| Engine API | :8766 | :8776 |
| Database | PostgreSQL + pgvector, :55432 | PostgreSQL + TimescaleDB + pgvector, :55433 |
| Ingest topics | `commerce/v1/…` | `commerce2/v1/…` |
| Live topics | `sb/commerce/…` | `sb/commerce2/…` |
| Redis db / graph prefix | 0 / `sb` | 5 / `sb2` (in this checkout's `.env`) |

```bash
docker compose -f docker-compose.timescale.yml -p secondbrain-ts up -d --wait   # once
scripts/demo2.sh                                                                # app + engine + TimescaleDB writer
```

Then open **http://localhost:8775/timescale** (and the usual pages on :8775).
The writer recreates its tables at every start: the simulator sends its history
again each time, so the tables always hold the current run.

## What is stored

| Table | Kind | What |
|---|---|---|
| `cm_order_lines` | hypertable, 1-day chunks, columnstore | every order line: time, platform, merchant, product, customer, qty, price, offer |
| `cm_product_daily` | hypertable, 1-month chunks, columnstore | every product every day: the history, then each live day as it closes (zeros and sold-out days included) |
| `cm_products` | table | the catalogues |
| `cm_stockouts` | table | products that ended a live day with no stock |
| `cm_findings` | hypertable | every finding the watcher published |
| `cm_sales_1h`, `cm_sales_1d` | continuous aggregates (daily built on hourly) | orders, units, revenue, offer revenue per merchant |
| `cm_sku_1d` | continuous aggregate | units and revenue per product per day |

The rollups are refreshed for the window the simulated clock just passed, and
answer in real time beyond it. Time-based refresh and compression policies don't
fit a clock that runs a day every six seconds, so the watcher calls
`refresh_continuous_aggregate` and `convert_to_columnstore` itself. Days older
than two simulated days move to the columnstore, segmented by merchant. Desi
Living's `sales_daily` (the main demo's forecasting table) is a hypertable here
too.

## What the watcher reports

Every finding is a fact worth acting on, with the SQL that found it and how long
the query took:

- **Sales lost to empty shelves** (each simulated day, per store): products that
  sold out, priced at what each sells on an in-stock day.
- **Out of stock three days running** (each simulated day, per product): the
  restock hasn't arrived or wasn't ordered, with the rupees lost so far.
- **A store far off its usual this hour** (each simulated hour): the store's
  same hour on its last seven days, moved the way its platform's other stores
  moved this hour. Festivals and weekdays move every store alike, so they cancel
  out; this rarely fires, as it should when nothing store-wide is wrong.

Product-level spikes and drops are left to the engine's forecasts. Measured
against the simulator's injected events (viral surges, competitor price cuts),
SQL rules on raw history got at most 8% of product surges right: at one product
in one store, a day's random swing (multi-unit baskets on top of ~30% day-to-day
noise) is as large as most real changes. About 1% of product-days land above 2×
their expected demand by chance alone, roughly a hundred a day across 16,000
products.

## The numbers

Measured on an M1 Pro laptop, 600 merchants, alongside demo 1:

| | |
|---|---|
| History backfill | 6.84 million rows in ~40 s (~170k rows/s, COPY) |
| Live orders | ~16-22k rows/s at a day every 6 s, batch p50 ~35 ms, no backlog |
| Rows after a week | ~77 lakh (8.5 lakh order lines + 68 lakh product-days) |
| Columnstore | 13.9-15.7× smaller (689 MB → 49 MB) |
| Engine vs TimescaleDB | order and unit counts match exactly |
| Hourly check, all 600 stores | 2-4 ms |
| Stock-out and streak checks | 30-150 ms |
| Writing a closed day (16,295 rows) | ~75 ms |
| Rollup refresh | 10-200 ms |
| Compressing a day / the whole history | ~0.5 s / ~7 s |

## Reading the data

The page uses read-only endpoints on the app, each returning its SQL and timing:
`GET /commerce/tsdb/overview`, `/commerce/tsdb/merchants/{id}?days=60`,
`/commerce/tsdb/top?hours=24`, `/commerce/tsdb/signals?limit=60`. MQTT topics
(`sb/commerce2/tsdb/stats`, `sb/commerce2/tsdb/signals/{platform}/{merchant}`)
are in `commerce/contract.py`.
