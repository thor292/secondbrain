# How reliable is SecondBrain's core? (Nov 5, 2026)

The USP is a brain that **forecasts → gives decision signals → judges what
happened → learns and evolves**. This page says how each part works, what was
measured, and what is still unproven. 103 automated tests pass.

## 1. Real data in

| | |
|---|---|
| How | MQTT/MQTTS topics `sb/v1/{company}/{department}/{kind}`, or `POST /ingest/{kind}` with a per-company key, or CSV upload for past history. Kinds: product, sale, daily_sales, inventory, stockout, purchase_order, goods_receipt, promotion. Contract: `docs/INTEGRATION.md`. |
| Guarantees | Each message is saved in one transaction and acknowledged only after the commit. Replays are ignored (unique `event_id`). Bad messages are dead-lettered with the reason. A key or topic can only write to its own company. Out-of-stock days are recorded, so they aren't read as "no demand". |
| Verified live | 1,179 Lumen events over the MQTT bridge (Nov 4–5). A replay of 551 events was ignored. Forecasts, grades and signals refreshed. |
| Not yet proven | TLS against a real broker; broker ACLs; a dedicated refresh worker (refresh runs inside the API and bridge processes). |

## 2. Forecasting

| | |
|---|---|
| How | For each product, every model that suits its history (seasonal, year over year, ETS, gradient boosting, Croston/TSB/ADIDA for products that sell only now and then, a quick start for new products) is backtested, and the best one is used. Sales data is cleaned first: stockouts, gaps, one-off bulk orders, discontinued products. Ranges are calibrated on data the models didn't see. Every forecast carries a grade (good/fair/weak) with a plain reason. |
| Measured (out of sample) | Lumen: error 30.9% → **27.0%**; 80% range held **84%** of days. Messy 14-product shop: worked (38.6% error), where the old code failed. Runtime 73–130 s → **~20 s**. |
| Weak spots | A shop with only one holiday season so far: holidays under-forecast by ~17%. After a sudden permanent jump in sales, the forecast catches up slowly ("fair" grade). The quick-start range for brand-new products is a heuristic. |

## 3. Decision signals

| | |
|---|---|
| How | Forecast + stock + supplier lead time give: run-out warnings (order-by date at a 10% risk of running out first, quantity sized to the forecast with a safety margin), backorder, excess stock. AI business signals must cite the company's own notes. A signal is dropped if its title or action uses a number that isn't in its evidence. |
| Measured | 18 simulated futures × 3 policies: warnings **93% precise, 100% recall**; orders on time **99%**; ran short after the order arrived **0%** (was 14%); overstock 8%. Confidence calibration (Brier score) 0.139 → **0.055**. Acting on signals beat a standard reorder-point rule. |
| Live AI check | Real model: every kept signal cited real notes; one invented "10%" was caught and dropped; dismissed advice wasn't raised again. |
| Weak spots | These scores come from simulated futures, not a real client. Spotting reworded duplicate advice uses a word-overlap heuristic. |

## 4. Learning and evolving

| | |
|---|---|
| How | When a warning's date has passed and the sales data covers it, the brain judges it from **recorded** sales (real or simulated, same code): prevented / happened / happened anyway / false alarm. It saves each result as a memory and grades every forecast against what really sold. Confidence then blends the forecast, this product's past outcomes and the forecasts' real-sales grade. Repeated dismissals for the same reason ("second warehouse") lower a signal's priority. Anonymous outcome lessons go to the root and feed other businesses. |
| Verified live | After 2 live days: "Forecasts were off by 8% from real sales… 90% of days inside the 80% range". After acting on the candle warning and moving 28 days forward, it was judged **prevented**. The ignored lamp warning waits until its date. |
| Measured (simulation) | 23/23 warnings right; 8/8 ignored ones came true; nothing judged twice or early; stock always adds up. *These numbers flatter the brain: the simulator draws demand from the forecast itself.* |
| Weak spots | A "prevented" outcome is assumed, not proven. Real calibration needs weeks of a real client's data. Inventory tracks one open order per product. |

## Can we rely on it for the client demo?

**Yes, for a demo on Lumen data with live events arriving.** All four loops run
end to end, and each one explains itself. Say clearly that accuracy figures
come from backtests and simulations. Real-world accuracy is what a pilot with
the client's own data will show: the brain grades itself on their data from
week one, and the Brain evolution tab and HQ show that grade.

## Demo: live data arriving

```bash
SB_MQTT_PORT=51883 SB_MQTT_TLS=false uv run python -m secondbrain.ingest.mqtt   # the bridge
uv run python scripts/publish_sample_events.py --delay 0.05                       # one business day of Lumen events
curl -X POST localhost:8765/admin/refresh -H 'X-Admin-Token: dev-admin-token' \
     -H 'Content-Type: application/json' -d '{"company":"root.lumen"}'          # or wait for the auto refresh
```

Then show Explore → Decision signals and Brain evolution, and HQ.
