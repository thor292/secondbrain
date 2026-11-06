# Sending business data to SecondBrain

For the engineer running the MQTT stream and for a client's IT team. Contract
**version 1**. Code: `src/secondbrain/ingest/` (`contract.py` is the source of
truth for fields).

SecondBrain forecasts each product's sales, compares the forecast with stock
and supplier lead times, and raises decision signals (run-outs, backorders,
excess stock). For that it needs, per product: sales, stock, purchase orders
and promotions. Everything below is how to send them.

## 1. Addressing

A business is a **company** (`desiliving`), with **brains** for its departments
(`merchandising`). Data belongs to one brain and is visible only to people
with access to that brain. Products, sales and stock usually go to one
department, e.g. `desiliving/merchandising`.

- `company`: the company id (`desiliving`); `root.desiliving` also works.
- `brain`: a department id (`merchandising`), a deeper path with dots
  (`sales.emea`), `_` for the company brain itself, or a full path
  (`root.desiliving.merchandising`). It must already exist (created at onboarding).

## 2. Two ways in

| | MQTT (live) | HTTP (live or backfill) |
|---|---|---|
| Endpoint | `sb/v1/{company}/{brain}/{kind}` | `POST /ingest/{kind}` |
| Body | one message or a JSON list | one message or a JSON list (max 1000) |
| Auth | broker credentials + topic ACL | `Authorization: Bearer sbk_...` |
| Backfill | no | `POST /ingest/{kind}/csv` |

`{kind}` is one of: `product`, `sale`, `daily_sales`, `inventory`,
`stockout`, `purchase_order`, `goods_receipt`, `promotion`.

### MQTT

- Production: **MQTTS on port 8883** (TLS). Each client gets a username and
  password (or a client certificate) and an ACL allowing publish on
  `sb/v1/{its company}/#` and subscribe on `sb/v1/{its company}/_errors`.
- QoS **1**. SecondBrain acks a message only after it is committed to the
  database (or dead-lettered), so a message is never lost; it may arrive
  twice, which event ids make harmless.
- The topic's company and brain must match the payload's, or the message is
  rejected.
- Errors are published to `sb/v1/{company}/_errors` (see section 6).
- Example: `sb/v1/desiliving/merchandising/sale`.

### HTTP

```bash
curl -X POST https://<secondbrain>/ingest/sale \
  -H 'Authorization: Bearer sbk_...' -H 'Content-Type: application/json' \
  -d @order-lines.json
```

Response: `{"processed": 2, "duplicates": 0, "rejected": [{"index": 3, "event_id": "...", "error": "..."}]}`.
401: missing or revoked key. 403: a message is for another company, or a
brain outside the key's scope; nothing in that request is written. 404:
unknown kind. Invalid messages don't fail the request; they're listed under
`rejected` and dead-lettered.

## 3. The envelope

Every message, on either channel:

```json
{
  "schema_version": 1,
  "event_id": "shopify-line-5531207-1",
  "company": "desiliving",
  "brain": "merchandising",
  "sent_at": "2026-11-05T14:03:12+05:30",
  "data": { ... }
}
```

| Field | Required | Notes |
|---|---|---|
| `schema_version` | yes | `1` |
| `event_id` | yes | Idempotency key, ≤ 200 chars, unique per company. **Must be stable across retries**: derive it from the source record (`order-5531207-line-1`), never a fresh UUID per send. |
| `company`, `brain` | yes | Section 1 |
| `sent_at` | no | For tracing only |
| `data` | yes | The record, by kind (section 4) |

Unknown fields are rejected, so typos don't pass silently.

**Times and days.** Timestamps are ISO 8601 **with a UTC offset**
(`2026-11-05T14:03:12+05:30`); timestamps without one are rejected. Daily
figures belong to the **business's local day**: a record's day is the date of
its timestamp *in the offset you send*. So send local time with its offset
(not UTC `Z`), or set the optional `day` field explicitly. Dates are
`YYYY-MM-DD`.

**Unknown products.** Any per-product record (`sale`, `daily_sales`,
`inventory`, `stockout`, `purchase_order`, `goods_receipt`) may carry
`"product": {"name", "category", "price", "sensitivity"?}`; if the SKU is new,
it's created from that. Without it, a record for an unknown SKU is
dead-lettered. Simplest: send `product` messages first.

## 4. Records

### product

```json
{"sku": "brass-floor-lamp", "name": "Moradabad brass floor lamp", "category": "lighting",
 "price": 7490.00, "sensitivity": "internal"}
```

`price`: list price. `sensitivity` (default `internal`): `public`,
`internal` or `restricted`; restricted sales and stock are only visible to
people with restricted access to the brain. Resending updates the product.

### sale — one order line

```json
{"order_id": "5531207", "line_id": "1", "sku": "brass-floor-lamp", "quantity": 2,
 "unit_price": 7490.00, "discount": 0.1, "ts": "2026-11-05T14:03:12+05:30"}
```

| Field | Required | Notes |
|---|---|---|
| `order_id`, `line_id` | yes | A line is counted once, even if resent under a new event id |
| `sku`, `quantity` | yes | Quantity ≠ 0; negative means a return |
| `unit_price` | yes | List price per unit, before the line's discount |
| `discount` | no | Share off the line, 0 ≤ x < 1 (default 0) |
| `discount_amount` | no | Or: total off the line in rupees (the company's currency); converted to `discount` |
| `ts` | yes | When the order was placed |
| `day` | no | Business day, if not the date of `ts` |
| `kind` | no | `sale` (default) or `return` |

Lines add to the day's units and revenue (net of discount). **Returns** are
recorded but don't reduce the day's units: forecasting needs what customers
ordered. Use `kind: "return"` with the original `order_id`/`line_id`.

### daily_sales — a day's totals, instead of order lines

```json
{"sku": "brass-floor-lamp", "day": "2026-11-05", "units": 41, "revenue": 276381.00,
 "discount": 0.1, "in_stock": true}
```

`units` required; `revenue` (default units × price), `price` (average selling
price after discount; default revenue ÷ units), `discount` (default 0),
`in_stock` (false = out of stock for part of the day) optional. It **sets**
the day's figures, replacing what was there. For a given product, send either
order lines or daily totals, not both: lines arriving after a daily total are
added on top of it.

### inventory — stock snapshot

```json
{"sku": "brass-floor-lamp", "as_of": "2026-11-05T23:45:00+05:30", "on_hand": 120,
 "on_order": 400, "order_eta": "2026-12-03",
 "supplier": "Noor Metalcraft", "lead_time_days": 28, "unit_cost": 3895.00}
```

| Field | Required | Notes |
|---|---|---|
| `sku`, `as_of`, `on_hand` | yes | `as_of`: when counted; a bare date means the end of that day |
| `on_order`, `order_eta` | no | Units ordered and not yet received; earliest arrival |
| `supplier`, `lead_time_days`, `unit_cost` | first time | Required in a product's first snapshot, kept afterwards |

The snapshot **replaces** the stock position, but only if it's newer than
the one already applied; an older snapshot arriving late is ignored (its
out-of-stock day, if any, still counts). Send one per product at least daily,
ideally at close of business or on every change.

### stockout — optional, for exact out-of-stock days

```json
{"sku": "diya-candle-set", "state": "out", "at": "2026-11-05T15:20:00+05:30"}
```

`state`: `out` (default) or `back_in_stock`.

**Out-of-stock days matter**: on those days sales understate demand, so
forecasting ignores them and outcome judging counts them as run-outs. A day
is marked out of stock if stock was zero at any point in it: the day a
snapshot or stockout event shows zero, every day after it until stock is
back, and the day it comes back.

### purchase_order

```json
{"po_id": "PO-88412", "sku": "brass-floor-lamp", "quantity": 400, "supplier": "Noor Metalcraft",
 "ordered_at": "2026-11-05T16:00:00+05:30", "eta": "2026-12-03", "status": "placed"}
```

`status`: `placed` (default) or `cancelled`. Resend with the same `po_id` to
change quantity or ETA; set `updated_at` to when the change happened. Adds
the open quantity to `on_order`.

### goods_receipt

```json
{"po_id": "PO-88412", "sku": "brass-floor-lamp", "quantity": 400,
 "received_at": "2026-12-02T09:30:00+05:30"}
```

Moves units from `on_order` to `on_hand`. `po_id` optional.

Purchase orders and receipts change stock only if they happened after the
last applied snapshot (earlier ones are already in its numbers). Snapshots
are authoritative: the next one corrects any drift.

### promotion

```json
{"promo_id": "wedding-2026", "name": "Wedding season sale", "sku": null,
 "starts": "2026-11-16", "ends": "2026-11-23", "discount": 0.15}
```

`sku: null` = sitewide. Resend with the same `promo_id` to change it;
`"cancelled": true` removes it. Send planned promotions ahead of time: the
forecast uses them.

## 5. Idempotency and ordering

- Every message is applied **at most once** per `(company, event_id)`;
  duplicates are acknowledged and ignored (`duplicates` in HTTP responses).
  A sale line is also unique by `(order_id, line_id, kind)`.
- **No ordering guarantee is needed** across messages: snapshots carry their
  own time and older ones lose; lines add up in any order; purchase orders
  and receipts are compared with the snapshot time. Within one product,
  publish in time order when you can (one MQTT topic per brain and kind
  keeps the broker's order).
- Recommended: send `product` messages before their first sale or snapshot.

## 6. Errors and the dead-letter queue

A message that can't be applied (invalid fields, unknown SKU without
`product`, unknown brain, company mismatch) is rolled back and stored as
**rejected** with the reason and the full payload. It is:

- listed in the HTTP response under `rejected`;
- published on MQTT to `sb/v1/{company}/_errors`:
  `{"topic": "...", "event_id": "...", "error": "unknown sku 'x' in root.desiliving.merchandising: send a product message first, ..."}`;
- visible to SecondBrain admins: `GET /admin/ingest/errors?company=desiliving`
  (`X-Admin-Token`) or `uv run python -m secondbrain.ingest errors desiliving`.

**Fix and resend with the same `event_id`**: a rejected event is processed
normally when it comes again. Database outages are not dead-lettered: over
MQTT the message is not acked and is redelivered; over HTTP you get a 5xx and
should retry.

## 7. Backfill: history at onboarding

Load past data once over HTTP as CSV. Columns are the record fields;
`product_name`, `product_category`, `product_price` fill `product`; an
`event_id` column is optional (default: a hash of the row, so uploading the
same file twice changes nothing and a corrected row is applied).

```bash
KEY=sbk_...
curl -H "Authorization: Bearer $KEY" -F brain=merchandising -F file=@products.csv https://<sb>/ingest/product/csv
curl -H "Authorization: Bearer $KEY" -F brain=merchandising -F file=@daily_sales.csv https://<sb>/ingest/daily_sales/csv
curl -H "Authorization: Bearer $KEY" -F brain=merchandising -F file=@inventory.csv https://<sb>/ingest/inventory/csv
```

```csv
sku,day,units,revenue,discount,in_stock
brass-floor-lamp,2024-09-01,34,254660.00,0,true
```

Any kind works as CSV (order lines too). Rows with times are applied oldest
first. Max 50 MB per file. The same works offline:
`uv run python -m secondbrain.ingest csv desiliving merchandising daily_sales file.csv`.

Forecasts need history: with the default 120-day horizon, **at least 480
days** of daily sales per brain (two years is better: last year's season is a
backtest window). With less, data is stored but forecasts aren't
recalculated yet.

## 8. Ingest keys (HTTP)

A SecondBrain admin creates one key per client system; it's shown once.

```bash
curl -X POST https://<sb>/admin/ingest/keys -H "X-Admin-Token: $SB_ADMIN_TOKEN" \
  -H 'Content-Type: application/json' -d '{"company": "desiliving", "name": "Desi Living ERP"}'
# optional "scope": "merchandising" limits the key to that department's subtree
uv run python -m secondbrain.ingest key create desiliving "Desi Living ERP" [--scope merchandising]
```

List: `GET /admin/ingest/keys?company=desiliving`. Revoke:
`DELETE /admin/ingest/keys/{id}` or `... key revoke <id>`. A key writes only
to its own company (and scope); only a hash is stored.

## 9. Running the MQTT bridge

```bash
uv run python -m secondbrain.ingest.mqtt
```

Settings (environment or `.env`):

| Variable | Default | |
|---|---|---|
| `SB_MQTT_HOST` / `SB_MQTT_PORT` | `localhost` / `8883` | |
| `SB_MQTT_TLS` | `true` | MQTTS; never turn off outside development |
| `SB_MQTT_CA_FILE` | system CAs | CA bundle that signed the broker's certificate |
| `SB_MQTT_CERT_FILE`, `SB_MQTT_KEY_FILE` | — | client certificate (mutual TLS), optional |
| `SB_MQTT_USERNAME`, `SB_MQTT_PASSWORD` | — | |
| `SB_MQTT_CLIENT_ID` | `secondbrain-ingest` | unique per bridge instance |
| `SB_MQTT_SHARE_GROUP` | — | e.g. `secondbrain`: subscribe as `$share/secondbrain/sb/v1/+/+/+` so several bridges split the load |
| `SB_MQTT_PUBLISH_ERRORS` | `true` | publish rejections on `_errors` |
| `SB_MQTT_SESSION_EXPIRY` | `86400` | seconds the broker keeps unacked messages while the bridge is away |

The bridge uses MQTT 5 with a persistent session (`clean_start=false`) and
manual acks; it reconnects with backoff (1 s to 60 s). The bridge account
needs subscribe on `sb/v1/+/+/+` and publish on `sb/v1/+/_errors`.

Development: `docker compose up -d mosquitto` runs a plain broker on
`localhost:51883` (anonymous, no TLS, dev only), then

```bash
SB_MQTT_PORT=51883 SB_MQTT_TLS=false uv run python -m secondbrain.ingest.mqtt
uv run python scripts/publish_sample_events.py      # a day of Desi Living events
```

## 10. After data arrives

When a company's data changes, SecondBrain recalculates its forecasts and
decision signals (`secondbrain.ingest.pipeline.refresh_company`): zero-sale
days are filled in so each product's series has no gaps, every product is
re-forecast, and signals are recalculated against the new stock. This runs
**at most once per 10 minutes per company** (`SB_INGEST_REFRESH_MINUTES`),
in the API and bridge processes, coordinated through Redis; data arriving
during the quiet period is included in the next refresh.
`SB_INGEST_AUTO_REFRESH=false` turns it off. On demand:
`POST /ingest/refresh` with the company's key, or
`uv run python -m secondbrain.ingest refresh desiliving`.
