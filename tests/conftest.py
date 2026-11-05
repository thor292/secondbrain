"""Integration tests run against the docker-compose Postgres and Redis.

Start them first: ``docker compose up -d --wait``.
"""

import os
from pathlib import Path

import psycopg
import pytest
from fastapi.testclient import TestClient

from secondbrain import admin
from secondbrain.api import create_app
from secondbrain.config import Settings
from secondbrain.models import build_models
from secondbrain.schema import create_schema
from secondbrain.sensitivity import Sensitivity

EXAMPLES = Path(__file__).parent / "fixtures"
ADMIN_TOKEN = "test-admin-token"
# Several checkouts can run the suite at once, each on its own database,
# Redis db and graph prefix: SB_TEST_SUFFIX=_ingest SB_TEST_REDIS_DB=14 uv run pytest
SUFFIX = os.environ.get("SB_TEST_SUFFIX", "")
TEST_DB = f"secondbrain_test{SUFFIX}"
ADMIN_SERVER = "postgresql://secondbrain_admin:dev_admin_password@localhost:55432"


def ensure_test_database() -> None:
    with psycopg.connect(f"{ADMIN_SERVER}/secondbrain", autocommit=True) as conn:
        if conn.execute("SELECT 1 FROM pg_database WHERE datname = %s", (TEST_DB,)).fetchone():
            return
        conn.execute(f'CREATE DATABASE "{TEST_DB}"')
        conn.execute(f'GRANT CONNECT ON DATABASE "{TEST_DB}" TO secondbrain_app')
    with psycopg.connect(f"{ADMIN_SERVER}/{TEST_DB}", autocommit=True) as conn:
        conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        conn.execute("CREATE EXTENSION IF NOT EXISTS ltree")
        conn.execute("GRANT USAGE ON SCHEMA public TO secondbrain_app")


def make_test_settings() -> Settings:
    """Settings for this checkout's test database (also used by scripts/evolution_reliability.py)."""
    ensure_test_database()
    return Settings(
        _env_file=None,
        admin_database_url=f"{ADMIN_SERVER}/{TEST_DB}",
        app_database_url=f"postgresql://secondbrain_app:dev_app_password@localhost:55432/{TEST_DB}",
        redis_url=f"redis://localhost:56379/{os.environ.get('SB_TEST_REDIS_DB', '15')}",
        # Test graphs get their own prefix, so they never touch the demo's "sb_" graphs.
        graph_prefix=f"sbtest{SUFFIX.replace('_', '')}",
        model_provider="stub",
        embedding_dim=256,
        admin_token=ADMIN_TOKEN,
        demo_mode=True,
    )


@pytest.fixture(scope="session")
def settings() -> Settings:
    return make_test_settings()


def load_examples(settings: Settings) -> None:
    import json

    _, embedder = build_models(settings)
    with psycopg.connect(settings.admin_database_url) as conn:
        for company in ("acme", "globex"):
            admin.onboard(conn, json.loads((EXAMPLES / company / "company.json").read_text()))
        imports = [
            ("root.acme", "acme/docs/company", Sensitivity.INTERNAL),
            ("root.acme.sales", "acme/docs/sales", Sensitivity.INTERNAL),
            ("root.acme.finance", "acme/docs/finance/internal.md", Sensitivity.INTERNAL),
            ("root.acme.finance", "acme/docs/finance/restricted.md", Sensitivity.RESTRICTED),
            ("root.acme.hr", "acme/docs/hr", Sensitivity.INTERNAL),
            ("root.globex", "globex/docs/company", Sensitivity.INTERNAL),
        ]
        for brain, path, level in imports:
            docs = admin.read_documents(EXAMPLES / path)
            admin.import_documents(conn, embedder, settings, brain, docs, level)
        # Knowledge in our own root brain must never reach a company user.
        admin.import_documents(
            conn, embedder, settings, "root",
            [("root.md", "Root pattern: annual leave policies across clients average 26 days.")],
            Sensitivity.PUBLIC,
        )


@pytest.fixture()
def client(settings: Settings):
    create_schema(settings, reset=True)
    load_examples(settings)
    app = create_app(settings)
    with TestClient(app) as client:
        app.state.brain.short_term._redis.flushdb()
        app.state.brain.relationships.drop_all()
        yield client


def ask(client: TestClient, user: str, question: str) -> dict:
    response = client.post("/ask", json={"question": question}, headers={"X-User-Id": user})
    assert response.status_code == 200, response.text
    return response.json()
