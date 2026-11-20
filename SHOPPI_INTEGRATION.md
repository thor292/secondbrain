# SecondBrain API for Shoppi

Shoppi reads its merchants' **decision signals** and **sales forecasts** from SecondBrain over HTTPS.

## 1. Connection

| | |
|---|---|
| Base URL | `https://secondbrain-610993990979.us-east4.run.app` |
| Auth | header `Authorization: Bearer <SHOPPI_KEY>` on every request |
| Key | sent to you separately. Keep it in your secret store, never in code or chat. |
| Merchants | Haldiram's, BigBasket, FabIndia, Bata |
| Access | read-only: signals and forecasts, plus feedback on a signal |

A merchant appears in responses as `root.<slug>`. Map it with the `slug` field of Shoppi's merchants (`root.haldirams` = merchant `haldirams`).

## 2. Get the signals

```bash
curl https://secondbrain-610993990979.us-east4.run.app/signals \
  -H "Authorization: Bearer $SHOPPI_KEY"
```

Returns a JSON list of open signals for all your merchants. Add `?include_closed=true` to also get signals that were acted on or dismissed.

One signal:
```json
{
  "id": "82804112-001b-43a3-9220-7367a27cb342",
  "company": "root.fabindia",
  "kind": "stockout",
  "title": "FabIndia Black Skirt runs out on Nov 23",
  "action": "Order 1,060 units from Artisan clusters today and ask them to expedite: ...",
  "reason": "84 units on hand cover the forecast until Nov 23. Demand peaks at about 23 a day on Nov 22.",
  "impact": "About ₹6.5 lakh in lost sales if it runs out before new stock arrives",
  "impact_value": 648777.8,
  "urgency": "high",
  "confidence": 0.97,
  "due": "2026-11-19",
  "evidence": {
    "facts": ["84 on hand", "Artisan clusters lead time: 14 days", "Forecast model: ets (typical error 31%)"],
    "products": [{"sku": "FAB-0033", "brain": "root.fabindia.merchandising"}]
  },
  "status": "open",
  "created_at": "2026-11-20T09:56:11+00:00"
}
```

| Field | Meaning |
|---|---|
| `company` | merchant (`root.<slug>`) |
| `kind` | `stockout`, `backorder`, `excess_stock` or `business` |
| `title` / `action` / `reason` | what's happening, what to do, why (plain sentences, ready to show) |
| `impact_value` | rupees at stake |
| `urgency` | `high`, `medium` or `low` |
| `confidence` | 0 to 1 |
| `due` | act by this date |
| `evidence.products[].sku` | the product (Shoppi SKU) |
| `status` | `open`, `acted` or `dismissed` |

Suggested display: sort by `urgency`, then `impact_value`; show `title`, `action` and `impact`.

## 3. Get forecasts

Products you can see:
```bash
curl https://secondbrain-610993990979.us-east4.run.app/forecast/products -H "Authorization: Bearer $SHOPPI_KEY"
```

Daily forecast for one product (SKU), with 80% and 95% ranges, recent sales and stock:
```bash
curl "https://secondbrain-610993990979.us-east4.run.app/forecast?product=FAB-0033&horizon=28" -H "Authorization: Bearer $SHOPPI_KEY"
```

## 4. Tell SecondBrain what the merchant did (optional, recommended)

When a merchant acts on or dismisses a signal, send it back. SecondBrain learns from it.
```bash
curl -X POST https://secondbrain-610993990979.us-east4.run.app/signals/<signal id>/feedback \
  -H "Authorization: Bearer $SHOPPI_KEY" -H 'Content-Type: application/json' \
  -d '{"status": "acted", "note": "Ordered 1,060 units"}'
```
`status`: `acted`, `dismissed`, or `open` (undo).

## 5. Errors

| Code | Meaning |
|---|---|
| 401 | missing, wrong or revoked key |
| 403 | the key may not do this (it is read-only) |
| 404 | product or signal not found for your merchants |

## 6. How fresh the signals are

Signals are recalculated from the merchants' sales and stock data. Today they are based on data up to 18 Nov 2026. Polling every 15–60 minutes is enough. To keep them current, Shoppi sends new sales and stock data (separate ingest key and endpoints: `POST /ingest/{kind}`; ask for the ingest guide when ready).
