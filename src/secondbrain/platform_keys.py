"""Read keys for other systems (e.g. Shoppi, Media Planner) to fetch signals and forecasts.

A platform key belongs to one named platform and lists the companies (merchants) it may read.
Each platform gets a service identity (``platform_<name>``) with read grants on exactly those
companies, so every read still goes through row-level security: the key can never see a
merchant it wasn't given.

Clients send ``Authorization: Bearer sbp_...``. The key is shown once, when created; only its
hash is stored. Keys are read-only: they may call GET endpoints and give feedback on a signal,
nothing else (no chat, no writing memories).
"""

import hashlib
import re
import secrets
import time

import psycopg

from secondbrain.sensitivity import Sensitivity

PREFIX = "sbp_"
USER_PREFIX = "platform_"

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS platform_keys (
    id          uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    platform    text NOT NULL,
    user_id     text NOT NULL REFERENCES users(id),
    prefix      text NOT NULL,
    key_hash    text NOT NULL UNIQUE,
    created_at  timestamptz NOT NULL DEFAULT now(),
    revoked_at  timestamptz
);
"""

_CACHE: dict[str, tuple[float, str | None]] = {}
_CACHE_SECONDS = 60


def _hash(key: str) -> str:
    return hashlib.sha256(key.encode()).hexdigest()


def _label(name: str) -> str:
    label = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
    if not label:
        raise ValueError(f"Can't make an id from platform name {name!r}")
    return label


def ensure_schema(conn: psycopg.Connection) -> None:
    conn.execute(SCHEMA_SQL)


def create(conn: psycopg.Connection, platform: str, companies: list[str],
           sensitivity: str = "internal") -> dict:
    """Create a key for a platform that may read these companies. Re-creating for the same
    platform replaces which companies it may read (older keys of that platform keep working)."""
    ensure_schema(conn)
    if not companies:
        raise ValueError("List at least one company the platform may read")
    paths = [c if c.startswith("root.") else f"root.{c}" for c in companies]
    for path in paths:
        row = conn.execute("SELECT kind FROM brains WHERE path = %s::ltree", (path,)).fetchone()
        if row is None or row[0] != "company":
            raise ValueError(f"No company {path!r}; onboard it first")
    level = Sensitivity.parse(sensitivity)
    user_id = USER_PREFIX + _label(platform)
    conn.execute(
        """INSERT INTO users (id, name, home_brain) VALUES (%s, %s, 'root')
           ON CONFLICT (id) DO UPDATE SET name = excluded.name""",
        (user_id, f"{platform} (platform)"),
    )
    conn.execute("DELETE FROM grants WHERE user_id = %s", (user_id,))
    for path in paths:
        conn.execute(
            """INSERT INTO grants (user_id, brain_path, include_children, max_sensitivity)
               VALUES (%s, %s::ltree, true, %s)""",
            (user_id, path, int(level)),
        )
    key = PREFIX + secrets.token_urlsafe(32)
    key_id = conn.execute(
        "INSERT INTO platform_keys (platform, user_id, prefix, key_hash) VALUES (%s, %s, %s, %s) RETURNING id::text",
        (platform, user_id, key[:10], _hash(key)),
    ).fetchone()[0]
    _CACHE.clear()
    return {"id": key_id, "key": key, "platform": platform, "companies": paths, "sensitivity": level.name.lower()}


def verify(conn: psycopg.Connection, key: str) -> str | None:
    """The service identity a key acts as, or None for an unknown or revoked key."""
    now = time.monotonic()
    hit = _CACHE.get(key)
    if hit and now - hit[0] < _CACHE_SECONDS:
        return hit[1]
    ensure_schema(conn)
    row = conn.execute(
        "SELECT user_id FROM platform_keys WHERE key_hash = %s AND revoked_at IS NULL", (_hash(key),)
    ).fetchone()
    user_id = row[0] if row else None
    _CACHE[key] = (now, user_id)
    return user_id


def listing(conn: psycopg.Connection) -> list[dict]:
    ensure_schema(conn)
    rows = conn.execute(
        """SELECT k.id::text, k.platform, k.prefix, k.created_at, k.revoked_at,
                  array(SELECT g.brain_path::text FROM grants g WHERE g.user_id = k.user_id ORDER BY 1) AS companies
           FROM platform_keys k ORDER BY k.created_at"""
    ).fetchall()
    cols = ["id", "platform", "prefix", "created_at", "revoked_at", "companies"]
    return [dict(zip(cols, r)) for r in rows]


def revoke(conn: psycopg.Connection, key_id: str) -> bool:
    ensure_schema(conn)
    done = conn.execute(
        "UPDATE platform_keys SET revoked_at = now() WHERE id::text = %s AND revoked_at IS NULL", (key_id,)
    ).rowcount == 1
    _CACHE.clear()
    return done
