"""Long-term memory in PostgreSQL.

Every read and write runs inside ``as_user``, which tells the database who
is asking. Row-level security then filters rows; this module never checks
permissions itself.
"""

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime

import psycopg
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool

from secondbrain.schema import vector_type


@dataclass
class User:
    id: str
    name: str
    home_brain: str


@dataclass
class Hit:
    id: str
    brain_path: str
    memory_type: str
    content: str
    distance: float
    created_at: datetime
    amended: bool = False  # a newer memory this viewer can see changed one of its facts


def vector_literal(embedding: list[float]) -> str:
    return "[" + ",".join(f"{v:.6f}" for v in embedding) + "]"


class MemoryStore:
    def __init__(self, database_url: str, embedding_dim: int) -> None:
        self._pool = ConnectionPool(database_url, kwargs={"row_factory": dict_row}, open=True)
        self._cast = vector_type(embedding_dim)[0].split("(")[0]

    def close(self) -> None:
        self._pool.close()

    @contextmanager
    def as_user(self, user_id: str) -> Iterator[psycopg.Connection]:
        with self._pool.connection() as conn:
            # Local to this transaction, so it can't leak to the next request
            # that reuses the pooled connection.
            conn.execute("SELECT set_config('app.user_id', %s, true)", (user_id,))
            yield conn

    def get_user(self, user_id: str) -> User | None:
        with self._pool.connection() as conn:
            row = conn.execute(
                "SELECT id, name, home_brain::text FROM users WHERE id = %s", (user_id,)
            ).fetchone()
        return User(**row) if row else None

    def insert(
        self,
        conn: psycopg.Connection,
        *,
        brain_path: str,
        memory_type: str,
        sensitivity: int,
        content: str,
        embedding: list[float],
        embedding_model: str,
        created_by: str,
    ) -> str:
        row = conn.execute(
            f"""
            INSERT INTO memories (brain_path, memory_type, tier, sensitivity, content,
                                  embedding, embedding_model, created_by)
            VALUES (%s::ltree, %s, 'long_term', %s, %s, %s::{self._cast}, %s, %s)
            RETURNING id::text
            """,
            (brain_path, memory_type, sensitivity, content,
             vector_literal(embedding), embedding_model, created_by),
        ).fetchone()
        return row["id"]

    def search(
        self,
        conn: psycopg.Connection,
        embedding: list[float],
        *,
        brain_path: str,
        include_children: bool,
        limit: int,
    ) -> list[Hit]:
        scope = "brain_path <@ %(path)s::ltree" if include_children else "brain_path = %(path)s::ltree"
        rows = conn.execute(
            f"""
            SELECT id::text, brain_path::text, memory_type, content, created_at,
                   embedding <=> %(vec)s::{self._cast} AS distance,
                   (m.amended_by IS NOT NULL
                    AND EXISTS (SELECT 1 FROM memories n WHERE n.id = m.amended_by)) AS amended
            FROM memories m
            WHERE {scope}
              -- A replaced memory is hidden only from people who can see its
              -- replacement (row-level security applies inside the subquery),
              -- so nobody loses a fact because of a note they may not read.
              AND (m.superseded_by IS NULL
                   OR NOT EXISTS (SELECT 1 FROM memories n WHERE n.id = m.superseded_by))
            ORDER BY distance
            LIMIT %(limit)s
            """,
            {"path": brain_path, "vec": vector_literal(embedding), "limit": limit},
        ).fetchall()
        return [Hit(**row) for row in rows]
