"""HTTP API and chat screen.

    uv run uvicorn secondbrain.api:app --reload

Users are identified by the X-User-Id header. That is a stand-in for SSO:
it must not be exposed beyond a trusted demo network.
"""

import hmac
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Literal

import psycopg
from fastapi import BackgroundTasks, Depends, FastAPI, File, Form, Header, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from psycopg.rows import dict_row
from pydantic import BaseModel

from secondbrain import admin, learning, maintenance, relationships, tree
from secondbrain.documents import UnsupportedDocument, extract_text
from secondbrain.graph import build_full_graph
from secondbrain.config import Settings, get_settings
from secondbrain.memory.short_term import ShortTermMemory
from secondbrain.memory.store import MemoryStore
from secondbrain.models import build_models
from secondbrain.sensitivity import Sensitivity
from secondbrain.relationships import GraphUnavailable, RelationshipGraph
from secondbrain.service import SecondBrain, UnknownUser

STATIC_DIR = Path(__file__).parent / "static"
SensitivityName = Literal["public", "internal", "restricted"]


class RememberRequest(BaseModel):
    content: str
    sensitivity: SensitivityName = "internal"


class AskRequest(BaseModel):
    question: str


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        llm, embedder = build_models(settings)
        store = MemoryStore(settings.app_database_url, settings.embedding_dim)
        short_term = ShortTermMemory(settings.redis_url, settings.short_term_ttl)
        graph = RelationshipGraph(settings.falkordb_url, settings.graph_prefix)
        app.state.brain = SecondBrain(settings, store, short_term, llm, embedder, graph)
        yield
        store.close()

    app = FastAPI(title="SecondBrain", lifespan=lifespan)

    def brain(request: Request) -> SecondBrain:
        return request.app.state.brain

    def require_admin(x_admin_token: Annotated[str | None, Header()] = None) -> None:
        if not settings.admin_token:
            raise HTTPException(404, "Admin API is disabled")
        if not x_admin_token or not hmac.compare_digest(x_admin_token, settings.admin_token):
            raise HTTPException(401, "Invalid admin token")

    UserId = Annotated[str, Header(alias="X-User-Id")]
    Brain = Annotated[SecondBrain, Depends(brain)]

    @app.exception_handler(UnknownUser)
    async def unknown_user(request: Request, exc: UnknownUser):
        return JSONResponse({"detail": f"Unknown user {exc}"}, status_code=401)

    @app.exception_handler(GraphUnavailable)
    async def graph_unavailable(request: Request, exc: GraphUnavailable):
        return JSONResponse({"detail": "The relationship graph (FalkorDB) is unavailable"}, status_code=503)

    # Pages are revalidated on every load, so browsers never show a stale copy after an update.
    NO_CACHE = {"Cache-Control": "no-cache"}

    @app.get("/", include_in_schema=False)
    def chat_screen():
        return FileResponse(STATIC_DIR / "index.html", headers=NO_CACHE)

    @app.get("/graph-view", include_in_schema=False)
    def graph_screen():
        return FileResponse(STATIC_DIR / "graph.html", headers=NO_CACHE)

    @app.get("/graph")
    def graph(user_id: UserId, brain: Brain):
        return brain.graph(user_id)

    @app.get("/health")
    def health(brain: Brain):
        return {"status": "ok", "llm": brain.llm.name, "embedding_model": brain.embedder.name}

    @app.get("/demo/users")
    def demo_users():
        if not settings.demo_mode:
            raise HTTPException(404, "Demo mode is off")
        with psycopg.connect(settings.admin_database_url, row_factory=dict_row) as conn:
            return conn.execute(
                """SELECT u.id, u.name, u.home_brain::text AS home_brain, b.name AS brain_name
                   FROM users u JOIN brains b ON b.path = u.home_brain ORDER BY u.home_brain, u.id"""
            ).fetchall()

    @app.post("/remember")
    def remember(body: RememberRequest, user_id: UserId, brain: Brain):
        return brain.remember(user_id, body.content, Sensitivity.parse(body.sensitivity))

    @app.post("/consolidate")
    def consolidate(user_id: UserId, brain: Brain, background: BackgroundTasks):
        try:
            count = brain.consolidate(user_id)
            if count:
                # New long-term memories may duplicate or update older ones.
                company = tree.company_of(brain.store.get_user(user_id).home_brain)
                background.add_task(maintenance.run, settings, brain.llm, company)
                background.add_task(relationships.run, settings, brain.llm, brain.relationships, company)
            return {"consolidated": count}
        except psycopg.errors.InsufficientPrivilege:
            raise HTTPException(403, "You can't store memories at that sensitivity in your brain")

    @app.post("/conversation/reset")
    def reset_conversation(user_id: UserId, brain: Brain):
        brain.new_conversation(user_id)
        return {"ok": True}

    @app.post("/ask")
    def ask(body: AskRequest, user_id: UserId, brain: Brain):
        result = brain.ask(user_id, body.question)
        return {
            "answer": result.answer,
            "answered_from": result.answered_from,
            "searched_for": result.searched_for,
            "sources": [
                {"brain": h.brain_path, "type": h.memory_type, "content": h.content,
                 "distance": round(h.distance, 3)}
                for h in result.sources
            ],
        }

    @app.get("/experts")
    def experts(topic: str, user_id: UserId, brain: Brain):
        """Who knows about a topic, counting only memories this person may see."""
        return brain.experts(user_id, topic)

    @app.get("/entities/{name}")
    def entity(name: str, user_id: UserId, brain: Brain):
        """What we know about a customer, supplier, product, person or topic."""
        result = brain.entity(user_id, name)
        if result is None:
            # Same answer whether it doesn't exist or this person may not see it.
            raise HTTPException(404, f"Nothing known about {name!r}")
        return result

    @app.get("/admin/graph", dependencies=[Depends(require_admin)])
    def full_graph():
        """The whole SecondBrain: root, every company, every memory."""
        with psycopg.connect(settings.admin_database_url, row_factory=dict_row) as conn:
            return build_full_graph(conn)

    @app.post("/admin/maintain", dependencies=[Depends(require_admin)])
    def maintain(brain: Brain):
        """Merge duplicate memories and retire outdated ones, across all companies."""
        changes = maintenance.run(settings, brain.llm)
        return {"changes": [
            {"relation": c.relation, "old": c.old, "new": c.new, "reason": c.reason} for c in changes
        ]}

    @app.post("/admin/learn", dependencies=[Depends(require_admin)])
    def learn(brain: Brain):
        """Send anonymized patterns from companies that allow sharing up to the root."""
        result = learning.run(settings, brain.llm, brain.embedder)
        return {**result, "stored": [{"pattern": p.text, "topic": p.topic} for p in result["stored"]]}

    @app.post("/admin/companies", dependencies=[Depends(require_admin)])
    def onboard_company(spec: dict):
        with psycopg.connect(settings.admin_database_url) as conn:
            try:
                return {"company": admin.onboard(conn, spec)}
            except (KeyError, ValueError) as exc:
                raise HTTPException(400, f"Invalid company spec: {exc}")

    @app.post("/admin/brains/{brain_path}/documents", dependencies=[Depends(require_admin)])
    async def upload_documents(
        brain_path: str,
        brain: Brain,
        background: BackgroundTasks,
        files: Annotated[list[UploadFile], File()],
        sensitivity: Annotated[SensitivityName, Form()] = "internal",
    ):
        documents = []
        for upload in files:
            try:
                documents.append((upload.filename, extract_text(upload.filename or "", await upload.read())))
            except UnsupportedDocument as exc:
                raise HTTPException(400, str(exc))
        with psycopg.connect(settings.admin_database_url) as conn:
            try:
                count = admin.import_documents(
                    conn, brain.embedder, settings, brain_path, documents,
                    Sensitivity.parse(sensitivity),
                )
            except ValueError as exc:
                raise HTTPException(400, str(exc))
        if count and tree.level(brain_path) >= tree.COMPANY_LEVEL:
            background.add_task(
                relationships.run, settings, brain.llm, brain.relationships, tree.company_of(brain_path)
            )
        return {"brain": brain_path, "chunks": count}

    return app


app = create_app()
