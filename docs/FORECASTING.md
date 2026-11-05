# Forecasting sales and demand

> Status: **first working version** (November 2026). Implements ARCHITECTURE.md
> section 8.2. Built on synthetic history for Lumen Home.

Goal: for each merchant, forecast daily units per product for the next weeks,
with a range and a reason for the model behind it. Uses: stock planning
("will the Halo lamp run out before Christmas?"), promotion planning, and
later the checkout discount engine (8.3), which needs to know whether a
product is scarce or overstocked.

## 1. Approaches compared

| | Statistical (ETS, ARIMA, Theta) | Gradient boosting (LightGBM) | Foundation models (Chronos-2, TimesFM 2.5) |
|---|---|---|---|
| How | One model per product, fitted on its own history | One model across all products, from features | Pretrained on large time series corpora, forecast without training |
| Strengths | Fast, reliable, explainable, good intervals; strong baseline | Uses promotions, prices, holidays and cross-product patterns; won the M5 retail competition | Works with little history (cold start); no training pipeline |
| Weaknesses | Can't see a sale or a moving holiday coming; one seasonality at a time (ETS) | Needs feature engineering, enough history, and care to avoid leaking the future | Large (needs CPU/GPU time and model weights); harder to explain; only Chronos-2 takes covariates natively |
| Library | `statsforecast` (Nixtla) | `lightgbm` (or `mlforecast`) | `chronos-forecasting`, `timesfm` |
| Cost | Milliseconds per series | Seconds per merchant | Seconds per batch on CPU; GPU for scale |

What the evidence says:

- **Statistical models** are the baseline to beat. `statsforecast` fits
  AutoETS / AutoARIMA / Theta quickly enough for millions of series
  ([statsforecast](https://nixtlaverse.nixtla.io/statsforecast/index.html)).
  2026 guides still recommend benchmarking every new model against seasonal
  naive and ETS first.
- **Gradient boosting won retail.** In the M5 competition (Walmart, daily
  sales per product and store), the winning solutions were LightGBM models
  trained across many series ("cross-learning") with lags, rolling statistics,
  calendar and price features, often with Poisson/Tweedie losses
  ([Januschowski et al., Forecasting with trees](https://cdn.amazon.science/17/31/0d99831d4add92e711a56abc75f8/forecasting-with-trees.pdf);
  [M5 summary](https://pmc.ncbi.nlm.nih.gov/articles/PMC9232271)). The M5
  organisers also noted that at the lowest level (single product, single
  store) the margin over simple benchmarks was much smaller.
- **Foundation models** are now strong zero-shot forecasters. On the GIFT-Eval
  benchmark (97 tasks, 55 datasets), Chronos-2 reports the best win rate and
  skill score, ahead of TiRex and TimesFM 2.5
  ([Chronos-2 paper](https://arxiv.org/abs/2510.15821)); an independent check
  that removed datasets leaked into training still has Chronos-2 slightly
  ahead of TimesFM 2.5 on MASE
  ([Cisco TSM report](https://arxiv.org/abs/2511.19841)). Chronos-2 takes
  known future covariates (promotions) natively; Chronos-Bolt and earlier do
  not ([amazon/chronos-2](https://huggingface.co/amazon/chronos-2)). Caveat:
  these are the vendors' benchmarks, mostly not retail product-level demand.

**For SecondBrain:** the three are not rivals but variants, run side by side
and chosen per product by backtest, as the architecture already says.
Foundation models matter most for a **new merchant or a new product** with
weeks of history, where the other two have too little to learn from.

## 2. Storage: TimescaleDB or plain PostgreSQL

| | Plain PostgreSQL | TimescaleDB |
|---|---|---|
| Daily sales per product (this version) | Plenty: a merchant with 5,000 products has ~2M rows a year | Works the same |
| Raw orders and events (minutes, seconds) | Large tables need manual partitioning and rollups | Hypertables (automatic time partitions), continuous aggregates (minute → hour → day, kept up to date), columnar compression |
| Permissions | Row-level security | Row-level security on the hypertable works the same (tested) |
| pgvector, ltree | Yes | Yes: `timescale/timescaledb-ha:pg17` ships TimescaleDB 2.31, pgvector 0.8.7 and ltree |
| Licence | PostgreSQL | Hypertables are Apache 2.0; continuous aggregates and compression are under the Timescale License: free to self-host, but not to offer as a database service ([editions](https://www.tigerdata.com/docs/about/latest/timescaledb-editions)). Some managed Postgres providers only ship the Apache edition |

**Decision:** the schema works on both. `sales_daily` becomes a hypertable
when the TimescaleDB extension is installed and stays a plain table
otherwise; queries don't change. Daily per-product history doesn't need
TimescaleDB yet. It will once we ingest raw orders and checkout events (8.3),
where continuous aggregates replace hand-written rollups. So: keep the
current image for the running demo, and switch the image when event ingestion
starts. Everything was tested on both (`docker-compose.timescale.yml` runs
TimescaleDB on port 55433).

Switching the main database to TimescaleDB:

1. `pg_dump` the `secondbrain` database (or reload it: it's all generated from
   `examples/`).
2. In `docker-compose.yml`, set `image: timescale/timescaledb-ha:pg17` and
   mount the volume at `/home/postgres/pgdata` (that image's data directory).
   The old volume can't be reused as is: new volume, then restore.
3. `uv run python -m secondbrain.schema --reset` and reload, or restore the dump.
4. Self-hosting only: check the Timescale License terms for the features used.

Data model (`src/secondbrain/forecasting/store.py`):

- `products` (brain, SKU, name, category, base price, sensitivity)
- `sales_daily` (brain, SKU, day, units, revenue, price, discount, in stock):
  in stock matters because a sold-out day records zero sales, not zero demand
- `promotions`: past and **planned** promotions, sitewide or per SKU
- `forecasts`: each run's forecast per product, with the chosen model, its
  reason and every model's backtest scores

All four belong to a brain and use the same row-level security as memories:
`sb_can_see(brain_path, sensitivity)`. Lumen's sales live in
`root.lumen.merchandising` at `internal`, so Merchandising, the CEO and CFO,
and Leo (Marketing, who has a grant on Merchandising) see them; Support does
not. The app role can only read; loading history and storing forecasts are
system jobs on the admin connection, like memory maintenance.

## 3. Backtesting and metrics

**Backtest (rolling origin).** Pretend it is an earlier day, give each model
only the history up to then, forecast the next 8 weeks, compare with what
actually sold. We use four windows: the three most recent 8-week periods and
**the same 8 weeks last year**. Without the last one, models would be chosen
on quiet September–October weeks and then trusted with Black Friday.

**Metrics** (from [Hyndman, intermittent demand](https://robjhyndman.com/papers/foresight.pdf);
[Hewamalage et al., forecast evaluation pitfalls](https://arxiv.org/abs/2203.10716)):

| Metric | Meaning | Use |
|---|---|---|
| **WAPE** | Total absolute error ÷ total units sold | **Chooses the model.** Easy to explain ("off by 14% of what we sold"), weights busy days the way revenue does, fine on slow days |
| **MASE** | Error ÷ error of "same as last week" on the training data | Reported. Comparable across products; below 1 beats the naive baseline |
| sMAPE | Symmetric percentage error | Reported for reference; behaves oddly near zero |
| MAPE | Average percentage error | **Not used**: undefined on zero-sale days and explodes on slow ones |
| Bias | Over/under forecasting as a share of sales | Reported: consistent under-forecasting means stockouts |

On products that sell only now and then (intermittent demand), WAPE and MASE
reward forecasting zero (they are smallest at the median, which is zero), so
for those products **RMSSE** (scaled squared error, smallest at the average
rate; used in M5) chooses the model instead.

**Intervals** are split conformal ranges from the chosen model's backtest
errors (`intervals.py`): errors are normalised by `sqrt(a·f + b·f²)` fitted per
product (Poisson-like noise on slow days, relative error on busy ones),
quantiles are taken per horizon bucket (days 1–7, 8–28, 29–56, later), and a
leave-one-window-out check scales them until the 80% and 95% ranges hold about
80% and 95% of the time on a window they didn't see. That held-out coverage is
stored with the forecast.

## 4. How often to retrain, and "evolving every second"

| What | Frequency |
|---|---|
| Model selection (backtest of all variants) | Weekly, or when a product's error drifts |
| Refit and forecast with the chosen model | **Nightly** (a minute for a merchant this size) |
| Nowcast adjustment | **On every request**: if sales since the last run are above or below the forecast, the rest of the forecast moves the same way, partly (shrunk toward 1, capped at ×0.67–×1.5) |
| Live signals (traffic, carts, stock) | Seconds, for the checkout engine (8.3), not for refitting |

Retraining every second wouldn't help: a day's sales carry little new
information about next month, and refitting on noise makes forecasts jumpier,
not better. The realistic "every second" part is that new sales feed straight
into the nowcast and the next nightly run.

## 5. What is built

```
scripts/generate_lumen_sales.py          synthetic history and promotion calendar
src/secondbrain/forecasting/
  calendar.py   Black Friday, Cyber Monday, Christmas, Memorial and Labor Day
  models.py     variants: seasonal_naive, year_over_year, ets, gradient_boosting
  backtest.py   windows, scores, choice per product, intervals
  metrics.py    WAPE, MASE, sMAPE, bias
  nowcast.py    adjusting a forecast with sales since the last run
  store.py      tables, row-level security, TimescaleDB when available
  api.py        GET /forecast, GET /forecast/products
  __main__.py   the nightly job
```

| Variant | What it does | Why it's in |
|---|---|---|
| `seasonal_naive` | Repeats last week | The baseline; anything that can't beat it is dropped |
| `year_over_year` | Last year's same weekdays × growth over the last 4 weeks | How merchants plan by hand; catches Black Friday from last year's spike |
| `ets` | statsforecast AutoETS, weekly season | The standard statistical model |
| `gradient_boosting` | scikit-learn histogram gradient boosting (LightGBM algorithm), one model over all products: weekday, day of year, days to Black Friday and Christmas, planned discount, recent level, last year's sales. The recent level is taken 7, 14, 28, 56, 91 or 182 days back depending on how far ahead the day is (one model, lag as a feature) | The M5 approach; the only one that knows a sale is planned |
| `croston`, `tsb`, `adida` | statsforecast Croston (SBA), TSB and ADIDA; only for products that sell on fewer than ~3 days in 4 | Built for intermittent demand; TSB decays when sales dry up |
| `cold_start` | Recent average × the shop's forecast shape (or the product's weekdays), wide ranges | Products too new to backtest |
| `discontinued`, `no_sales` | Zero | A product that stopped selling isn't forecast at its old average forever |

### Real-world data (`prepare.py`, `quality.py`)

Before any model runs, the feed becomes a clean daily panel per product:

- **Out of stock** days are censored: unknown demand (at least what sold),
  filled from the same weekday nearby, never a zero, and never scored.
- **Gaps**: a day missing for the whole shop is a feed outage (unknown); a day
  missing for one product is a zero sale if its feed omits zero rows (no zero
  rows and >10% of days missing), unknown otherwise; days after a product's
  last row are zero sales (it left the feed).
- **Outliers**: a day above 4× the 90th percentile of the surrounding 8 weeks
  (and 10+ units above it) is set aside as a one-off order, unless a promotion
  or Black Friday–Cyber Monday explains it.
- **Short histories** get shorter backtest windows (56/28/14/7 days) and only
  the models that can work on them; under ~5 weeks, a cold start.
- **Discontinued**: a run of zero days that would almost never happen by
  chance at the product's old rate (or 90+ days) means a zero forecast.
- **Level shifts** (last 4 weeks vs the 8 before, allowing for the season)
  are flagged; the lag-bucketed gradient boosting and ETS follow them.

Every forecast stores `quality` (grade good/fair/weak, a plain-language
summary, history days, WAPE, bias, held-out 80%/95% coverage, backtest windows
and data warnings), returned by `GET /forecast` and shown on the forecast page.
`scripts/forecast_reliability.py` checks all of this out of sample on Lumen and
on a synthetic messy shop.

Not built yet: Chronos-2 / TimesFM (cold start, see 6), AutoARIMA and MSTL
(slower; ETS covers the statistical slot), hierarchical reconciliation
(products → categories → merchant totals), stock levels as an input.

### Synthetic data

Eight Lumen products (the four bestsellers plus a duvet set, table lamp, side
table and candle trio), daily from 1 Sep 2024 to 3 Nov 2026 (26 months),
seed 2026. Weekly pattern (Sunday and Monday busiest), yearly seasons per
category, 18% growth a year (Q3 was +18%), Thanksgiving-week lull, Black
Friday to Cyber Monday spike, December gift season stopping at the Dec 17
shipping cutoff, sitewide and product promotions with price-sensitive demand,
a three-week Aria sofa stockout (Aug 2025), and negative binomial noise.
Planned promotions include Black Friday 2026 (20% sitewide, Nov 27 – Dec 1,
from the Holiday Campaign Plan), so forecasts made on Nov 4 can see it.

These eight products make about $27M of 2025 revenue, 32% of it in November
and December (Lumen's documents say the holiday season is ~34% of annual
revenue).

## 6. Results on Lumen (synthetic)

`uv run python -m secondbrain.forecasting`, cutoff 3 Nov 2026, horizon 56
days, four backtest windows. Average WAPE per model (lower is better;
\* = chosen). The last column is the **noise floor**: the WAPE of a forecast
that knows the generator's true expected demand. No model can beat it; the
gap to it is what is left to win.

| Product | seasonal_naive | year_over_year | ets | gradient_boosting | MASE (chosen) | Noise floor |
|---|---|---|---|---|---|---|
| Aria linen sofa | 51.5% | 46.1% | 41.4% | **39.6%**\* | 0.86 | 32.8% |
| Ember candle trio | 40.4% | 32.2% | 38.3% | **24.6%**\* | 0.60 | 20.7% |
| Cloud waffle throw | 38.8% | 28.7% | 34.8% | **23.0%**\* | 0.69 | 18.8% |
| Halo brass floor lamp | 40.3% | 33.9% | 43.3% | **24.7%**\* | 0.70 | 21.4% |
| Woven jute rug 5x8 | 38.6% | 35.3% | 45.7% | **28.9%**\* | 0.80 | 23.3% |
| Linen duvet set | 35.1% | 31.9% | 29.3% | **26.7%**\* | 0.85 | 21.2% |
| Oak side table | 46.3% | 42.2% | 35.7% | **32.1%**\* | 0.80 | 29.0% |
| Orb table lamp | 38.4% | 34.6% | 30.7% | **23.8%**\* | 0.76 | 20.5% |

What this shows:

- **Gradient boosting wins every product**, 3–7 points above the noise
  floor. It is the only variant that knows Black Friday 2026 is planned
  (from `promotions`) and where it falls; for the Halo lamp it forecasts
  2.6× a normal mid-November day on Black Friday. The true lift is 3.4×, so
  it under-forecasts the peak itself (219 vs ~282 units): with one past Black
  Friday per window to learn from, trees average the spike away. More
  holiday history, or a peak-specific feature, should close that gap.
- **ETS is often worse than repeating last week**, because it can't see the
  holiday coming and smooths the spike away; on the quiet windows it is fine
  (Halo lamp, window 2: 22.9%). This is why the backtest includes last year's
  season.
- **Year over year** is a strong, explainable second for seasonal gift items.
- **Slow sellers** (sofa ~6/day, side table ~10/day) have a high noise floor:
  daily forecasts for them are mostly useful summed over weeks.
- Gradient boosting over-forecasts slightly (bias +2 to +11%); for stock
  planning that is the safer side, but it should be watched.

Caveat: the data is synthetic, generated with the same kind of structure the
features describe, so it flatters feature-based models. Real merchant data
will have messier promotions, returns, new products and stockouts. The value
here is the pipeline (variants, honest backtest, choice per product), not
these numbers.

Runtime: ~20 s for the whole backtest of 8 products at a 120-day horizon on a
laptop (it was ~75–130 s: scikit-learn's OpenMP threads made each gradient
boosting fit ~70× slower on this small data, so fits run single-threaded).

The table above is from the first version. With lag-bucketed gradient
boosting, `scripts/forecast_reliability.py` measures, out of sample, on the
demo's setting (cut 3 Nov, 120 days ahead): WAPE 27.0% (was 30.9%), bias
+1.6%, 80% range held 84% of days, 95% range 97%.

## 7. Recommendation

1. **Keep "variants + backtest + choose per product"** as the core, as built.
   Choose by WAPE over windows that include the same season last year; report
   MASE and bias next to it.
2. **Gradient boosting with promotion and holiday features is the main
   model** for merchants with a year or more of history. Promotions must be
   recorded as a plan (dates, discount, scope) before they happen; that
   single input is what separates it from the statistical models. On Linux
   servers, LightGBM can replace scikit-learn's implementation (same
   algorithm; its macOS wheel needs Homebrew's libomp, so we don't require it).
3. **Keep ETS and year-over-year as fallbacks and sanity checks**: cheap,
   explainable, and they win for some products on real data.
4. **Add Chronos-2 next, for cold start** (new merchant, new product, under
   ~6 months of history), as one more variant. It needs PyTorch and model
   weights (~120M parameters, CPU is fine for nightly runs), so it belongs in
   an optional dependency group and a separate worker, not the API process.
   Not built yet because it adds a heavy dependency and our cold-start case
   needs a second merchant's data to show.
5. **Retrain nightly, re-select weekly, nowcast on every request.** No
   per-second retraining.
6. **Storage: stay on plain PostgreSQL for daily data; move to
   TimescaleDB when raw orders and checkout events arrive** (section 8.3).
   The schema already works on both, and the `timescaledb-ha` image keeps
   pgvector and ltree. Check the Timescale License before offering it hosted.
7. **Next data step: real order history** from a merchant's Shopify export
   (orders, line items, discounts, inventory) into `sales_daily` and
   `promotions`, then rerun this backtest on it before trusting any model.
