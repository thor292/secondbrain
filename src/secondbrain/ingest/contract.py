"""The data contract for a business's real data, version 1.

Every message is an envelope around one record:

    {"schema_version": 1, "event_id": "...", "company": "lumen",
     "brain": "merchandising", "data": {...}}

The record's kind (``sale``, ``inventory``, ...) comes from the MQTT topic or
the HTTP path, not the payload, so a message can't claim one kind and be
routed as another. docs/INTEGRATION.md is the human-readable version of this
file; keep them in step.

Unknown fields are rejected rather than ignored: a typo such as ``qty`` for
``quantity`` should reach the dead-letter table, not silently become a zero.
Version 1 may gain optional fields; anything incompatible becomes version 2
on a new topic prefix (``sb/v2``).
"""

import re
from dataclasses import dataclass
from datetime import date, datetime, time, timezone
from decimal import Decimal
from typing import Annotated, Literal

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, model_validator

SCHEMA_VERSION = 1

Sku = Annotated[str, Field(min_length=1, max_length=100)]
Text = Annotated[str, Field(min_length=1, max_length=200)]
Money = Annotated[Decimal, Field(ge=0, max_digits=12, decimal_places=4)]
Fraction = Annotated[float, Field(ge=0, lt=1)]
SensitivityName = Literal["public", "internal", "restricted"]


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ProductInfo(Strict):
    """Enough to create a product the brain hasn't seen yet."""

    name: Text
    category: Text
    price: Annotated[Money, Field(gt=0)]  # list price
    sensitivity: SensitivityName = "internal"


class Product(ProductInfo):
    sku: Sku


class _ForSku(Strict):
    sku: Sku
    # Optional on every per-SKU record: an unknown SKU is created from it,
    # and without it a record for an unknown SKU is dead-lettered.
    product: ProductInfo | None = None


class Sale(_ForSku):
    """One order line. Returns are lines with ``kind: "return"`` (or a negative quantity)."""

    order_id: Text
    line_id: Text
    quantity: int  # not 0; negative means a return
    unit_price: Money  # list price per unit, before the line's discount
    discount: Fraction = 0.0  # share taken off the line, or give discount_amount
    discount_amount: Money | None = None  # total off the line, in currency
    ts: AwareDatetime
    day: date | None = None  # business day; defaults to ts's date in ts's own offset
    kind: Literal["sale", "return"] = "sale"

    @model_validator(mode="after")
    def _normalize(self):
        if self.quantity == 0:
            raise ValueError("quantity must not be 0")
        if self.quantity < 0:
            self.kind = "return"
            self.quantity = -self.quantity
        if self.discount_amount is not None:
            gross = self.unit_price * self.quantity
            if gross <= 0 or self.discount_amount > gross:
                raise ValueError("discount_amount must be at most quantity * unit_price")
            self.discount = float(self.discount_amount / gross)
        return self

    @property
    def business_day(self) -> date:
        return self.day or self.ts.date()


class DailySales(_ForSku):
    """A day's totals for one SKU. Replaces whatever that day held."""

    day: date
    units: int = Field(ge=0)
    revenue: Money | None = None  # defaults to units * price
    price: Money | None = None  # average selling price after discount; defaults to revenue / units
    discount: Fraction = 0.0
    in_stock: bool | None = None  # false: out of stock for at least part of the day


class InventorySnapshot(_ForSku):
    """Stock position at a moment. Older snapshots than the one applied are ignored."""

    as_of: AwareDatetime | date  # a bare date means the end of that business day
    day: date | None = None
    on_hand: int = Field(ge=0)
    on_order: int = Field(0, ge=0)
    order_eta: date | None = None
    # Required the first time a SKU's stock is reported, kept afterwards.
    supplier: Text | None = None
    lead_time_days: int | None = Field(None, gt=0)
    unit_cost: Money | None = None

    @property
    def at(self) -> datetime:
        return _moment(self.as_of)

    @property
    def business_day(self) -> date:
        return self.day or (self.as_of.date() if isinstance(self.as_of, datetime) else self.as_of)


class Stockout(_ForSku):
    state: Literal["out", "back_in_stock"] = "out"
    at: AwareDatetime
    day: date | None = None

    @property
    def business_day(self) -> date:
        return self.day or self.at.date()


class PurchaseOrder(_ForSku):
    po_id: Text
    quantity: int = Field(gt=0)
    ordered_at: AwareDatetime
    eta: date | None = None
    supplier: Text | None = None
    status: Literal["placed", "cancelled"] = "placed"
    # When this version of the order was made (a change or cancellation); defaults to ordered_at.
    updated_at: AwareDatetime | None = None


class GoodsReceipt(_ForSku):
    po_id: Text | None = None
    quantity: int = Field(gt=0)
    received_at: AwareDatetime
    day: date | None = None

    @property
    def business_day(self) -> date:
        return self.day or self.received_at.date()


class Promotion(Strict):
    promo_id: Text
    name: Text
    sku: Sku | None = None  # None: sitewide
    starts: date
    ends: date
    discount: Fraction
    cancelled: bool = False

    @model_validator(mode="after")
    def _dates(self):
        if self.ends < self.starts:
            raise ValueError("ends is before starts")
        return self


RECORDS: dict[str, type[BaseModel]] = {
    "product": Product,
    "sale": Sale,
    "daily_sales": DailySales,
    "inventory": InventorySnapshot,
    "stockout": Stockout,
    "purchase_order": PurchaseOrder,
    "goods_receipt": GoodsReceipt,
    "promotion": Promotion,
}
KINDS = tuple(RECORDS)

_LABEL = re.compile(r"^[A-Za-z0-9_]+$")


class Envelope(Strict):
    schema_version: Literal[1]
    event_id: Annotated[str, Field(min_length=1, max_length=200)]
    company: Text  # company id ("lumen") or path ("root.lumen")
    brain: Text  # department id ("merchandising"), a deeper path ("sales.emea"), "_" for the company itself, or a full path
    sent_at: AwareDatetime | None = None
    data: dict


@dataclass
class Message:
    kind: str
    envelope: Envelope
    record: BaseModel
    company: str  # root.lumen
    brain: str  # root.lumen.merchandising


def company_path(company: str) -> str:
    """``lumen`` or ``root.lumen`` → ``root.lumen``."""
    path = company if company.startswith("root.") else f"root.{company}"
    labels = path.split(".")
    if len(labels) != 2 or not _LABEL.match(labels[1]):
        raise ValueError(f"company must be a company id like 'lumen', not {company!r}")
    return path


def brain_path(company: str, brain: str) -> str:
    """A brain inside ``company`` (a path): department id, relative path, ``_`` or full path."""
    if brain in ("_", company):
        return company
    path = brain if brain.startswith("root.") else f"{company}.{brain}"
    if not (path + ".").startswith(company + ".") or not all(_LABEL.match(x) for x in path.split(".")):
        raise ValueError(f"brain {brain!r} is not a brain inside {company}")
    return path


def parse(kind: str, raw: object) -> Message:
    """Validate one message; raises ValueError (pydantic's ValidationError is one)."""
    if kind not in RECORDS:
        raise ValueError(f"unknown kind {kind!r}; expected one of {', '.join(KINDS)}")
    if not isinstance(raw, dict):
        raise ValueError("a message must be a JSON object")
    envelope = Envelope.model_validate(raw)
    record = RECORDS[kind].model_validate(envelope.data)
    company = company_path(envelope.company)
    return Message(kind, envelope, record, company, brain_path(company, envelope.brain))


def _moment(value: datetime | date) -> datetime:
    if isinstance(value, datetime):
        return value
    return datetime.combine(value, time(23, 59, 59), tzinfo=timezone.utc)
