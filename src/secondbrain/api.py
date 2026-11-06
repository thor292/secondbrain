"""HTTP API and chat screen.

    uv run uvicorn secondbrain.api:app --reload

Users are identified by the X-User-Id header. That is a stand-in for SSO:
it must not be exposed beyond a trusted demo network.
"""

import hmac
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Literal

import httpx
import psycopg
from fastapi import BackgroundTasks, Depends, FastAPI, File, Form, Header, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response
from psycopg.rows import dict_row
from pydantic import BaseModel

from secondbrain import admin, demo, evolution, learning, live, maintenance, relationships, signals, tree
from secondbrain.commerce import contract as commerce
from secondbrain.documents import UnsupportedDocument, extract_text
from secondbrain.forecasting.api import build_router as forecast_router
from secondbrain.graph import build_full_graph
from secondbrain.ingest.api import build_router as ingest_router
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
    app.include_router(forecast_router())

    def brain(request: Request) -> SecondBrain:
        return request.app.state.brain

    def require_admin(x_admin_token: Annotated[str | None, Header()] = None) -> None:
        if not settings.admin_token:
            raise HTTPException(404, "Admin API is disabled")
        if not x_admin_token or not hmac.compare_digest(x_admin_token, settings.admin_token):
            raise HTTPException(401, "Invalid admin token")

    app.include_router(ingest_router(settings, require_admin))

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

    @app.get("/static/{name}", include_in_schema=False)
    def static_file(name: str):
        """The shared stylesheet and script (only plain files in the static folder)."""
        path = STATIC_DIR / name
        if "/" in name or name.startswith(".") or path.suffix not in (".css", ".js") or not path.is_file():
            raise HTTPException(404)
        return FileResponse(path, headers=NO_CACHE)

    @app.get("/", include_in_schema=False)
    def chat_screen():
        return FileResponse(STATIC_DIR / "index.html", headers=NO_CACHE)

    @app.get("/explore", include_in_schema=False)
    def explore_screen():
        return FileResponse(STATIC_DIR / "explore.html", headers=NO_CACHE)

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

    class SignalFeedback(BaseModel):
        status: Literal["acted", "dismissed", "open"]
        note: str | None = None

    @app.get("/signals")
    def list_signals(user_id: UserId, brain: Brain, include_closed: bool = False):
        """Decision signals this person may see: only those whose evidence they can all read."""
        brain._user(user_id)
        with brain.store.as_user(user_id) as conn:
            return [{**s, "evidence": signals.evidence_details(conn, s)}
                    for s in signals.visible(conn, include_closed)]

    @app.get("/signals/quality")
    def signal_quality(user_id: UserId, brain: Brain):
        """The brain's measured track record, over the signals this person may see."""
        brain._user(user_id)
        with brain.store.as_user(user_id) as conn:
            return signals.quality(conn)

    @app.post("/signals/{signal_id}/feedback")
    def signal_feedback(signal_id: str, body: SignalFeedback, user_id: UserId, brain: Brain):
        """Mark a signal as acted on or dismissed; the brain uses this to learn what's useful."""
        brain._user(user_id)
        with brain.store.as_user(user_id) as conn:
            if not signals.feedback(conn, signal_id, user_id, body.status, body.note):
                raise HTTPException(404, "No such signal that you can see")
        return {"ok": True}

    @app.get("/live/credentials")
    def live_credentials(user_id: UserId, brain: Brain):
        """This person's own login for live notices: they can only hear brains they may see."""
        brain._user(user_id)
        if not settings.live_host:
            raise HTTPException(404, "Live notifications are switched off")
        return live.credentials(settings, user_id)

    @app.get("/admin/live/monitor", dependencies=[Depends(require_admin)])
    def live_monitor():
        """Read-only login that hears every notice (operations view)."""
        if not settings.live_host:
            raise HTTPException(404, "Live notifications are switched off")
        return live.monitor_credentials(settings)

    @app.post("/admin/live/sync", dependencies=[Depends(require_admin)])
    def live_sync():
        """Regenerate broker logins and ACL from the grants table (after onboarding or a role change)."""
        if not settings.live_host:
            raise HTTPException(404, "Live notifications are switched off")
        return live.sync_broker(settings)

    @app.get("/brain/evolution")
    def brain_evolution(user_id: UserId, brain: Brain):
        """Track record, feedback, outcomes and updated facts this person may see."""
        brain._user(user_id)
        with brain.store.as_user(user_id) as conn:
            return evolution.history(conn)

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

    @app.post("/admin/signals/generate", dependencies=[Depends(require_admin)])
    def generate_signals(brain: Brain):
        """Recalculate every company's decision signals."""
        result = signals.run(settings, brain.llm)
        return {company: len(items) for company, items in result.items()}

    class Simulation(BaseModel):
        company: str = "root.desiliving"
        days: int = 14
        policy: Literal["as_is", "follow", "ignore"] = "as_is"

    @app.post("/admin/simulate", dependencies=[Depends(require_admin)])
    def simulate(body: Simulation, brain: Brain):
        """Demo only: move a company's clock forward, judge outcomes, recalculate."""
        if not 1 <= body.days <= 90:
            raise HTTPException(400, "days must be between 1 and 90")
        return evolution.advance(settings, brain.llm, brain.embedder, body.company, body.days, policy=body.policy)

    @app.post("/admin/demo/reset", dependencies=[Depends(require_admin)])
    def reset_demo():
        """Demo only: put sales, stock, forecasts and signals back to the saved starting point."""
        if not settings.demo_mode:
            raise HTTPException(404, "Not available outside demo mode")
        with psycopg.connect(settings.admin_database_url, row_factory=dict_row) as conn:
            try:
                return demo.restore(conn)
            except demo.NoSnapshot as exc:
                raise HTTPException(409, str(exc))

    class Refresh(BaseModel):
        company: str

    @app.post("/admin/refresh", dependencies=[Depends(require_admin)])
    def refresh(body: Refresh, brain: Brain):
        """After new real sales arrive: grade forecasts, judge outcomes, recalculate (idempotent)."""
        return evolution.refresh(settings, brain.llm, brain.embedder, body.company)

    @app.get("/admin/hq", dependencies=[Depends(require_admin)])
    def hq():
        with psycopg.connect(settings.admin_database_url, row_factory=dict_row) as conn:
            return evolution.hq(conn)

    @app.get("/hq", include_in_schema=False)
    def hq_screen():
        return FileResponse(STATIC_DIR / "hq.html", headers=NO_CACHE)

    @app.get("/commerce", include_in_schema=False)
    def commerce_screen():
        return FileResponse(STATIC_DIR / "commerce.html", headers=NO_CACHE)

    @app.api_route("/commerce/api/{path:path}", methods=["GET", "POST"], include_in_schema=False)
    async def commerce_api(path: str, request: Request):
        """The Live Commerce engine's API, same origin as the pages. Commands need the admin token."""
        if request.method == "POST":
            require_admin(request.headers.get("x-admin-token"))
        url = f"http://127.0.0.1:{commerce.ENGINE_PORT}/{path}"
        try:
            async with httpx.AsyncClient(timeout=5) as client:
                r = await client.request(request.method, url, params=request.query_params, content=await request.body(),
                                         headers={"content-type": request.headers.get("content-type", "application/json")})
        except httpx.HTTPError:
            raise HTTPException(503, "The Live Commerce engine is not running (uv run python -m secondbrain.commerce)")
        return Response(r.content, r.status_code, media_type=r.headers.get("content-type"))

    @app.post("/admin/learn", dependencies=[Depends(require_admin)])
    def learn(brain: Brain):
        """Send anonymized patterns from companies that allow sharing up to the root."""
        result = learning.run(settings, brain.llm, brain.embedder)
        return {**result, "stored": [{"pattern": p.text, "topic": p.topic} for p in result["stored"]]}

    @app.post("/admin/companies", dependencies=[Depends(require_admin)])
    def onboard_company(spec: dict, background: BackgroundTasks):
        with psycopg.connect(settings.admin_database_url) as conn:
            try:
                company = admin.onboard(conn, spec)
            except (KeyError, ValueError) as exc:
                raise HTTPException(400, f"Invalid company spec: {exc}")
        if settings.live_host:  # new people and grants: the broker must know them too
            background.add_task(live.sync_broker, settings)
        return {"company": company}

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
