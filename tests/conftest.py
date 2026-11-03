"""Integration tests run against the docker-compose Postgres and Redis.

Start them first: ``docker compose up -d --wait``.
"""

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

EXAMPLES = Path(__file__).parent.parent / "examples"
ADMIN_TOKEN = "test-admin-token"


@pytest.fixture(scope="session")
def settings() -> Settings:
    return Settings(
        _env_file=None,
        admin_database_url="postgresql://secondbrain_admin:dev_admin_password@localhost:55432/secondbrain_test",
        app_database_url="postgresql://secondbrain_app:dev_app_password@localhost:55432/secondbrain_test",
        redis_url="redis://localhost:56379/15",
        model_provider="stub",
        embedding_dim=256,
        admin_token=ADMIN_TOKEN,
        demo_mode=True,
    )


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
        yield client


def ask(client: TestClient, user: str, question: str) -> dict:
    response = client.post("/ask", json={"question": question}, headers={"X-User-Id": user})
    assert response.status_code == 200, response.text
    return response.json()
