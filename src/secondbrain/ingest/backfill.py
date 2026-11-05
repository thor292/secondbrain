"""CSV files to messages, for loading a new client's history once at onboarding.

Columns are the record's fields (``sku,day,units,revenue`` for daily_sales);
``product_name``, ``product_category``, ``product_price`` fill the optional
``product`` object, and an ``event_id`` column is optional. Without one, the
event id is a hash of the row, so uploading the same file twice changes
nothing, while a corrected row is applied as a new event.
"""

import csv
import hashlib
import io
import json

from secondbrain.ingest.contract import SCHEMA_VERSION

# Rows applied oldest first, so the last row is the current stock position.
TIME_COLUMNS = {"inventory": "as_of", "stockout": "at", "purchase_order": "ordered_at",
                "goods_receipt": "received_at", "sale": "ts"}


def messages(kind: str, text: str, company: str, brain: str) -> list[dict]:
    rows = []
    for row in csv.DictReader(io.StringIO(text.lstrip("﻿"))):
        data = {k.strip(): v.strip() for k, v in row.items() if k and v is not None and v.strip() != ""}
        product = {k.removeprefix("product_"): data.pop(k) for k in list(data) if k.startswith("product_")}
        if product:
            data["product"] = product
        event_id = data.pop("event_id", None) or "csv:" + hashlib.sha256(
            json.dumps([kind, brain, data], sort_keys=True).encode()).hexdigest()[:32]
        rows.append({"schema_version": SCHEMA_VERSION, "event_id": event_id, "company": company,
                     "brain": brain, "data": data})
    if kind in TIME_COLUMNS:
        rows.sort(key=lambda m: str(m["data"].get(TIME_COLUMNS[kind], "")))
    return rows
