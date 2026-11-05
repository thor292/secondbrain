"""Create the database schema.

Run as the admin role: ``uv run python -m secondbrain.schema [--reset]``.

Access control lives here, in the database: the app role can only see a
memory when ``sb_can_see`` says so. Application code never decides who sees
what.
"""

import argparse

import psycopg

from secondbrain.config import Settings, get_settings
from secondbrain.forecasting import store as forecasting_store
from secondbrain import signals as signals_schema
from secondbrain.ingest import store as ingest_store

# pgvector's HNSW index supports up to 2,000 dimensions for `vector` and
# 4,000 for `halfvec` (half precision).
MAX_VECTOR_DIM = 2000
MAX_HALFVEC_DIM = 4000


def vector_type(dim: int) -> tuple[str, str]:
    """Column type and HNSW operator class for an embedding size."""
    if dim <= MAX_VECTOR_DIM:
        return f"vector({dim})", "vector_cosine_ops"
    if dim <= MAX_HALFVEC_DIM:
        return f"halfvec({dim})", "halfvec_cosine_ops"
    raise ValueError(
        f"Embedding dimension {dim} exceeds pgvector's index limit of "
        f"{MAX_HALFVEC_DIM}; use a model or setting with fewer dimensions"
    )


def schema_sql(dim: int) -> str:
    column_type, ops = vector_type(dim)
    return f"""
CREATE TABLE brains (
    path     ltree PRIMARY KEY,
    name     text NOT NULL,
    kind     text NOT NULL CHECK (kind IN ('root', 'company', 'department')),
    -- What a company lets flow up to the root: nothing, or anonymized patterns.
    sharing  text NOT NULL DEFAULT 'none' CHECK (sharing IN ('none', 'patterns'))
);

CREATE TABLE users (
    id          text PRIMARY KEY,
    name        text NOT NULL,
    home_brain  ltree NOT NULL REFERENCES brains(path)
);

-- A grant lets a user see memories in one brain (or its whole subtree) up
-- to a sensitivity level. Access between sibling departments only exists
-- through an explicit grant.
CREATE TABLE grants (
    user_id           text NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    brain_path        ltree NOT NULL REFERENCES brains(path),
    include_children  boolean NOT NULL DEFAULT false,
    max_sensitivity   smallint NOT NULL CHECK (max_sensitivity BETWEEN 0 AND 2),
    PRIMARY KEY (user_id, brain_path)
);

CREATE TABLE memories (
    id               uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    brain_path       ltree NOT NULL REFERENCES brains(path),
    memory_type      text NOT NULL CHECK (memory_type IN ('episodic', 'semantic', 'user')),
    tier             text NOT NULL CHECK (tier IN ('short_term', 'long_term')),
    sensitivity      smallint NOT NULL CHECK (sensitivity BETWEEN 0 AND 2),
    content          text NOT NULL,
    embedding        {column_type} NOT NULL,
    embedding_model  text NOT NULL,
    created_by       text REFERENCES users(id),
    source           text,  -- document a chunk came from; NULL for notes
    -- clock_timestamp, not now(): rows saved in one transaction still get
    -- distinct times, so maintenance can tell which memory is newer.
    created_at       timestamptz NOT NULL DEFAULT clock_timestamp(),
    -- Memory maintenance: a memory replaced by a newer one ("superseded") or
    -- folded into an earlier one ("merged") points at it. Nothing is deleted.
    status           text NOT NULL DEFAULT 'active' CHECK (status IN ('active', 'superseded', 'merged')),
    superseded_by    uuid REFERENCES memories(id),
    status_reason    text,
    -- Still active, but one of its facts was changed by a newer memory.
    amended_by       uuid REFERENCES memories(id),
    maintained_at    timestamptz
);

CREATE INDEX memories_brain_path_idx ON memories USING gist (brain_path);
CREATE INDEX memories_created_at_idx ON memories (created_at);
CREATE INDEX memories_unmaintained_idx ON memories (created_at) WHERE maintained_at IS NULL;
CREATE INDEX memories_embedding_idx ON memories USING hnsw (embedding {ops});

-- Security definer: runs with the owner's rights, so the app role can check
-- grants without being able to read the grants table itself.
CREATE FUNCTION sb_can_see(target ltree, level smallint) RETURNS boolean
LANGUAGE sql STABLE SECURITY DEFINER SET search_path = public AS $$
    SELECT EXISTS (
        SELECT 1 FROM grants g
        WHERE g.user_id = current_setting('app.user_id', true)
          AND (g.brain_path = target
               OR (g.include_children AND target <@ g.brain_path))
          AND level <= g.max_sensitivity
    )
$$;

-- The signed-in user's own grants, without exposing anyone else's.
CREATE FUNCTION sb_my_grants()
RETURNS TABLE (brain_path ltree, include_children boolean, max_sensitivity smallint)
LANGUAGE sql STABLE SECURITY DEFINER SET search_path = public AS $$
    SELECT g.brain_path, g.include_children, g.max_sensitivity FROM grants g
    WHERE g.user_id = current_setting('app.user_id', true)
$$;

ALTER TABLE memories ENABLE ROW LEVEL SECURITY;
ALTER TABLE memories FORCE ROW LEVEL SECURITY;

CREATE POLICY memories_read ON memories FOR SELECT
    USING (sb_can_see(brain_path, sensitivity));
CREATE POLICY memories_write ON memories FOR INSERT
    WITH CHECK (sb_can_see(brain_path, sensitivity)
                AND created_by = current_setting('app.user_id', true));

REVOKE ALL ON grants FROM PUBLIC;
REVOKE ALL ON FUNCTION sb_can_see(ltree, smallint) FROM PUBLIC;
GRANT EXECUTE ON FUNCTION sb_can_see(ltree, smallint) TO secondbrain_app;
REVOKE ALL ON FUNCTION sb_my_grants() FROM PUBLIC;
GRANT EXECUTE ON FUNCTION sb_my_grants() TO secondbrain_app;
GRANT SELECT ON brains, users TO secondbrain_app;
GRANT SELECT, INSERT ON memories TO secondbrain_app;
""" + RELATIONSHIP_SQL


# Entities and relationships extracted from memories (see relationships.py).
# They are the source of truth for the FalkorDB graph, which can always be
# rebuilt from them. Each row is visible only through a memory the reader may
# see, so a name mentioned only in a restricted note stays hidden even here.
RELATIONSHIP_SQL = """
CREATE TABLE entities (
    id       uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    company  ltree NOT NULL REFERENCES brains(path),
    key      text NOT NULL,  -- normalized name, so "Nordlys" and "nordlys" are one entity
    name     text NOT NULL,
    kind     text NOT NULL CHECK (kind IN ('person', 'customer', 'supplier', 'product', 'carrier',
                                           'place', 'organization', 'topic')),
    user_id  text REFERENCES users(id),  -- set when the entity is one of the company's people
    UNIQUE (company, key)
);

CREATE TABLE memory_entities (
    memory_id  uuid NOT NULL REFERENCES memories(id),
    entity_id  uuid NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
    PRIMARY KEY (memory_id, entity_id)
);

-- A relationship is only as visible as the memory that states it.
CREATE TABLE entity_relations (
    memory_id  uuid NOT NULL REFERENCES memories(id),
    source_id  uuid NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
    relation   text NOT NULL,
    target_id  uuid NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
    PRIMARY KEY (memory_id, source_id, relation, target_id)
);

-- Which memories have been through extraction, so each is processed once.
CREATE TABLE entity_extractions (
    memory_id     uuid PRIMARY KEY REFERENCES memories(id),
    model         text NOT NULL,
    extracted_at  timestamptz NOT NULL DEFAULT now()
);

ALTER TABLE memory_entities ENABLE ROW LEVEL SECURITY;
ALTER TABLE memory_entities FORCE ROW LEVEL SECURITY;
ALTER TABLE entity_relations ENABLE ROW LEVEL SECURITY;
ALTER TABLE entity_relations FORCE ROW LEVEL SECURITY;
ALTER TABLE entities ENABLE ROW LEVEL SECURITY;
ALTER TABLE entities FORCE ROW LEVEL SECURITY;

-- The subqueries are themselves filtered by row-level security on memories
-- (and memory_entities), so visibility always comes from sb_can_see.
CREATE POLICY memory_entities_read ON memory_entities FOR SELECT
    USING (EXISTS (SELECT 1 FROM memories m WHERE m.id = memory_id));
CREATE POLICY entity_relations_read ON entity_relations FOR SELECT
    USING (EXISTS (SELECT 1 FROM memories m WHERE m.id = memory_id));
CREATE POLICY entities_read ON entities FOR SELECT
    USING (EXISTS (SELECT 1 FROM memory_entities me WHERE me.entity_id = id));

GRANT SELECT ON entities, memory_entities, entity_relations TO secondbrain_app;
"""


DROP_SQL = """
DROP TABLE IF EXISTS entity_extractions, entity_relations, memory_entities, entities CASCADE;
DROP TABLE IF EXISTS memories, grants, users, brains CASCADE;
DROP FUNCTION IF EXISTS sb_can_see(ltree, smallint);
DROP FUNCTION IF EXISTS sb_my_grants();
"""


def create_schema(settings: Settings | None = None, reset: bool = False) -> None:
    # Imported here: evolution imports modules that import this one.
    from secondbrain import evolution as evolution_schema

    settings = settings or get_settings()
    with psycopg.connect(settings.admin_database_url) as conn:
        if reset:
            conn.execute("DROP SCHEMA IF EXISTS demo_snapshot CASCADE")  # a saved demo state belongs to the old data
            conn.execute(ingest_store.DROP_SQL)
            conn.execute(evolution_schema.DROP_SQL)
            conn.execute(signals_schema.DROP_SQL)
            conn.execute(forecasting_store.DROP_SQL)
            conn.execute(DROP_SQL)
        conn.execute(schema_sql(settings.embedding_dim))
        forecasting_store.create_schema(conn)
        # Signals cite memories and products, so they come last.
        conn.execute(signals_schema.SCHEMA_SQL)
        # Forecast grades cite forecasts and products.
        conn.execute(evolution_schema.SCHEMA_SQL)
        ingest_store.ensure_schema(conn)


def ensure_relationship_schema(settings: Settings | None = None) -> bool:
    """Add the relationship tables to a database created before they existed.

    Purely additive: existing tables and rows are not touched. Returns True
    if the tables were created.
    """
    settings = settings or get_settings()
    with psycopg.connect(settings.admin_database_url) as conn:
        if conn.execute("SELECT to_regclass('entities')").fetchone()[0] is not None:
            return False
        conn.execute(RELATIONSHIP_SQL)
        return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reset", action="store_true", help="drop existing tables first")
    args = parser.parse_args()
    create_schema(reset=args.reset)
    print("Schema created.")


if __name__ == "__main__":
    main()
