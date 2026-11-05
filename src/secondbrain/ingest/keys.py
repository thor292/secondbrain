"""Per-company ingest API keys.

A key belongs to one company and may only write inside its scope: the whole
company, or one department's subtree. The key itself is shown once, when it
is created; only its hash is stored.
"""

import hashlib
import secrets
from dataclasses import dataclass

import psycopg

from secondbrain.ingest.contract import brain_path, company_path

PREFIX = "sbk_"


@dataclass(frozen=True)
class IngestKey:
    id: str
    company: str  # root.lumen
    scope: str  # root.lumen or root.lumen.merchandising

    def allows(self, brain: str) -> bool:
        return brain == self.scope or brain.startswith(self.scope + ".")


def _hash(key: str) -> str:
    return hashlib.sha256(key.encode()).hexdigest()


def create(conn: psycopg.Connection, company: str, name: str, scope: str | None = None) -> tuple[str, str]:
    """Create a key; returns (key id, the key). The key can't be recovered later."""
    company = company_path(company)
    scope = brain_path(company, scope) if scope else company
    if conn.execute("SELECT 1 FROM brains WHERE path = %s::ltree", (scope,)).fetchone() is None:
        raise ValueError(f"No brain {scope!r}; onboard the company first")
    key = PREFIX + secrets.token_urlsafe(32)
    row = conn.execute(
        """INSERT INTO ingest_keys (company, scope, name, prefix, key_hash)
           VALUES (%s::ltree, %s::ltree, %s, %s, %s) RETURNING id::text""",
        (company, scope, name, key[:10], _hash(key)),
    ).fetchone()
    return row[0], key


def verify(conn: psycopg.Connection, key: str) -> IngestKey | None:
    row = conn.execute(
        """SELECT id::text, company::text, scope::text FROM ingest_keys
           WHERE key_hash = %s AND revoked_at IS NULL""",
        (_hash(key),),
    ).fetchone()
    return IngestKey(*row) if row else None


def revoke(conn: psycopg.Connection, key_id: str) -> bool:
    return conn.execute(
        "UPDATE ingest_keys SET revoked_at = now() WHERE id::text = %s AND revoked_at IS NULL", (key_id,)
    ).rowcount == 1


def listing(conn: psycopg.Connection, company: str | None = None) -> list[dict]:
    rows = conn.execute(
        """SELECT id::text, company::text, scope::text, name, prefix, created_at, revoked_at FROM ingest_keys
           WHERE %s::ltree IS NULL OR company = %s::ltree ORDER BY created_at""",
        (company and company_path(company),) * 2,
    ).fetchall()
    return [dict(zip(("id", "company", "scope", "name", "prefix", "created_at", "revoked_at"), r)) for r in rows]
