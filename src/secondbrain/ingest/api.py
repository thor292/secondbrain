"""HTTP ingestion: JSON messages, CSV backfill, and admin endpoints for ingest keys.

Clients authenticate with their company's ingest key
(``Authorization: Bearer sbk_...``). A key writes only inside its own scope:
a request with any message for another company or brain is refused whole
(403) before anything is written. Messages that are well addressed but
invalid are dead-lettered and reported per message, so one bad line doesn't
block a batch.
"""

from typing import Annotated, Any

import psycopg
from fastapi import APIRouter, BackgroundTasks, Body, Depends, File, Form, Header, HTTPException, Request, UploadFile
from pydantic import BaseModel

from secondbrain.config import Settings
from secondbrain.ingest import backfill, contract, keys, processor, store
from secondbrain.ingest.keys import IngestKey
from secondbrain.ingest.pipeline import IngestSettings, RefreshScheduler, scheduler_for

MAX_BATCH = 1000
MAX_CSV_BYTES = 50 * 1024 * 1024


class NewKey(BaseModel):
    company: str
    name: str
    scope: str | None = None  # a department id or path; default: the whole company


def build_router(settings: Settings, require_admin: Any) -> APIRouter:
    router = APIRouter(tags=["ingest"])
    ingest_settings = IngestSettings()
    state: dict[str, Any] = {"schema": False, "scheduler": None}

    def admin_conn() -> psycopg.Connection:
        conn = psycopg.connect(settings.admin_database_url, autocommit=True)
        if not state["schema"]:
            store.ensure_schema(conn)
            state["schema"] = True
        return conn

    def scheduler(request: Request) -> RefreshScheduler:
        if state["scheduler"] is None:
            brain = request.app.state.brain
            state["scheduler"] = scheduler_for(settings, brain.llm, brain.embedder, ingest_settings).start()
        return state["scheduler"]

    def ingest_key(authorization: Annotated[str | None, Header()] = None) -> IngestKey:
        token = (authorization or "").removeprefix("Bearer ").strip()
        if not token:
            raise HTTPException(401, "Send your ingest key as 'Authorization: Bearer sbk_...'")
        with admin_conn() as conn:
            key = keys.verify(conn, token)
        if key is None:
            raise HTTPException(401, "Invalid or revoked ingest key")
        return key

    Key = Annotated[IngestKey, Depends(ingest_key)]

    def check_scope(key: IngestKey, items: list) -> None:
        """Refuse the whole request if any message is addressed outside the key's scope."""
        for i, item in enumerate(items):
            if not isinstance(item, dict):
                continue
            try:
                company = contract.company_path(str(item.get("company", "")))
                brain = contract.brain_path(company, str(item.get("brain", "")))
            except ValueError:
                continue  # malformed: the processor dead-letters it with the reason
            if company != key.company or not key.allows(brain):
                raise HTTPException(403, f"Message {i} is for {brain}; this key may only write to {key.scope}")

    def run(request: Request, key: IngestKey, kind: str, items: list, source: str) -> dict:
        if kind not in contract.RECORDS:
            raise HTTPException(404, f"Unknown kind {kind!r}; expected one of {', '.join(contract.KINDS)}")
        result = {"processed": 0, "duplicates": 0, "rejected": []}
        with admin_conn() as conn:
            for i, item in enumerate(items):
                outcome = processor.process(conn, kind, item, source)
                if outcome.status == "processed":
                    result["processed"] += 1
                elif outcome.status == "duplicate":
                    result["duplicates"] += 1
                else:
                    result["rejected"].append({"index": i, "event_id": outcome.event_id, "error": outcome.error})
        if result["processed"] and ingest_settings.auto_refresh:
            scheduler(request).mark(key.company)
        return result

    @router.post("/ingest/refresh")
    def refresh(key: Key, request: Request, background: BackgroundTasks):
        """Recalculate forecasts and signals for the key's company now, without waiting."""
        background.add_task(scheduler(request).refresh_now, key.company)
        return {"company": key.company, "refresh": "started"}

    @router.post("/ingest/{kind}")
    def ingest(kind: str, key: Key, request: Request,
               body: Annotated[dict | list[dict], Body(description="one message or a list of messages")]):
        """Receive one message or a batch (a JSON list) of one kind."""
        items = body if isinstance(body, list) else [body]
        if len(items) > MAX_BATCH:
            raise HTTPException(413, f"At most {MAX_BATCH} messages per request; use CSV for backfills")
        check_scope(key, items)
        return run(request, key, kind, items, "http")

    @router.post("/ingest/{kind}/csv")
    async def ingest_csv(kind: str, key: Key, request: Request,
                         file: Annotated[UploadFile, File()], brain: Annotated[str, Form()]):
        """Backfill history from a CSV file whose columns are the record's fields."""
        raw = await file.read(MAX_CSV_BYTES + 1)
        if len(raw) > MAX_CSV_BYTES:
            raise HTTPException(413, "CSV larger than 50 MB; split it")
        try:
            items = backfill.messages(kind, raw.decode("utf-8"), key.company, brain)
        except UnicodeDecodeError:
            raise HTTPException(400, "CSV must be UTF-8")
        check_scope(key, items)
        return run(request, key, kind, items, "csv")

    @router.post("/admin/ingest/keys", dependencies=[Depends(require_admin)])
    def create_key(body: NewKey):
        """Create an ingest key. The key is in this response only; store it safely."""
        with admin_conn() as conn:
            try:
                key_id, key = keys.create(conn, body.company, body.name, body.scope)
            except ValueError as exc:
                raise HTTPException(400, str(exc))
        return {"id": key_id, "key": key}

    @router.get("/admin/ingest/keys", dependencies=[Depends(require_admin)])
    def list_keys(company: str | None = None):
        with admin_conn() as conn:
            return keys.listing(conn, company)

    @router.delete("/admin/ingest/keys/{key_id}", dependencies=[Depends(require_admin)])
    def revoke_key(key_id: str):
        with admin_conn() as conn:
            if not keys.revoke(conn, key_id):
                raise HTTPException(404, "No such active key")
        return {"revoked": key_id}

    @router.get("/admin/ingest/errors", dependencies=[Depends(require_admin)])
    def errors(company: str | None = None, limit: int = 50):
        """The dead-letter queue: recent rejected messages with the reason and the payload."""
        with admin_conn() as conn:
            rows = store.dead_letters(conn, company, min(limit, 500))
        names = ("received_at", "company", "event_id", "kind", "brain", "source", "attempts", "error", "payload")
        return [dict(zip(names, r)) for r in rows]

    return router
