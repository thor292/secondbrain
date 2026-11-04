"""Relationship layer: entities extracted from memories, in a FalkorDB graph.

When memories become long-term (consolidation, document import), the model
extracts what they mention (people, customers, suppliers, products, carriers,
places, organizations, topics) and how those relate ("Nordlys supplies the
Halo floor lamp"). Results are stored in PostgreSQL first, the source of
truth, then synced to one FalkorDB graph per company, which can always be
rebuilt from PostgreSQL.

FalkorDB has no row-level security, so:
- the graph holds structure only (ids, names, kinds, edges), never memory text;
- every query runs against the asking person's own company graph and is
  limited to the memory ids they may see, which come from PostgreSQL under
  row-level security (``viewer_scope``);
- an entity, person or relationship is only returned when it is reached
  through one of those memories, and memory text is read from PostgreSQL
  again, as that person.

    uv run python -m secondbrain.relationships            # extract new memories, rebuild graphs
    uv run python -m secondbrain.relationships --company root.lumen
"""

import argparse
import json
import logging
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

import psycopg
import redis
from falkordb import FalkorDB
from psycopg.rows import dict_row

from secondbrain import tree
from secondbrain.config import Settings, get_settings
from secondbrain.models import LLM

log = logging.getLogger(__name__)

KINDS = {"person", "customer", "supplier", "product", "carrier", "place", "organization", "topic"}
MAX_ENTITIES = 12
EXTRACT_WORKERS = 8
SYNC_BATCH = 500

EXTRACT_SYSTEM = f"""You extract the entities and relationships from one memory in a company's knowledge base.

Entities are specific things the memory mentions: people, customers, suppliers, products, carriers
(shipping companies), places (cities, warehouses, regions), other organizations, and topics.
Topics are 1 to 3 short subjects the memory is about, in lowercase, like "free shipping",
"goodwill credits" or "annual leave".

Reply with JSON only:
{{"entities": [{{"name": "...", "kind": "person|customer|supplier|product|carrier|place|organization|topic"}}],
 "relations": [{{"source": "...", "relation": "...", "target": "..."}}]}}

Rules:
- When the memory refers to one of the company's people listed below, even by first name, use their full name.
- A customer who is a person has kind "customer". "person" is for the company's people and other individuals.
- Use a product's most specific name as written, like "Halo floor lamp".
- A relation links two entities from your list, with a short lowercase verb phrase, like "supplies",
  "handled complaint from" or "delayed shipments to".
- Leave out generic words (customer, team, company), dates, amounts, and the company itself.
- At most {MAX_ENTITIES} entities."""

_STOPWORDS = {"the", "a", "an", "of", "and", "or", "for", "to", "in", "on", "about", "who", "knows", "what", "with"}


def normalize(name: str) -> str:
    """Matching key for a name: lowercase, single spaces, no leading article or quotes."""
    key = re.sub(r"\s+", " ", name.strip().strip("\"'“”‘’.,;:").lower())
    return re.sub(r"^the ", "", key)


def _words(text: str) -> list[str]:
    words = re.findall(r"[a-z0-9@#]+", text.lower())
    return [w.rstrip("s") if len(w) > 3 else w for w in words if w not in _STOPWORDS]


def matches(topic: str, key: str) -> bool:
    """Every word of the topic starts a word of the entity's name ("lamps" finds "halo floor lamp")."""
    wanted, have = _words(topic), _words(key)
    return bool(wanted) and all(any(h.startswith(w) for h in have) for w in wanted)


def person_name(display_name: str) -> str:
    """ "Nina Alvarez (Support agent)" -> "Nina Alvarez"."""
    return re.sub(r"\s*\(.*\)\s*$", "", display_name).strip()


@dataclass
class Extraction:
    entities: list[tuple[str, str]] = field(default_factory=list)  # (name, kind)
    relations: list[tuple[str, str, str]] = field(default_factory=list)  # (source, relation, target)


def parse(reply: str) -> Extraction:
    """Read the model's JSON reply, dropping anything malformed instead of failing."""
    match = re.search(r"\{.*\}", reply, re.DOTALL)
    try:
        data = json.loads(match.group(0)) if match else {}
    except json.JSONDecodeError:
        data = {}
    result, keys = Extraction(), set()
    for item in data.get("entities") or []:
        if not isinstance(item, dict) or not isinstance(item.get("name"), str):
            continue
        name, key = item["name"].strip(), normalize(item["name"])
        if not key or len(name) > 80 or key in keys:
            continue
        keys.add(key)
        kind = item.get("kind") if item.get("kind") in KINDS else "topic"
        result.entities.append((name, kind))
        if len(result.entities) == MAX_ENTITIES:
            break
    for item in data.get("relations") or []:
        if not isinstance(item, dict):
            continue
        source, target = normalize(str(item.get("source", ""))), normalize(str(item.get("target", "")))
        relation = re.sub(r"\s+", " ", str(item.get("relation", "")).strip().lower())[:60]
        # Only relations between entities from this same memory, so each one stays tied to it.
        if source in keys and target in keys and source != target and relation:
            result.relations.append((source, relation, target))
    return result


def extract(llm: LLM, content: str, company_name: str, people: list[str]) -> Extraction:
    prompt = f"Company: {company_name}\nPeople: {'; '.join(people)}\n\nMemory:\n{content}"
    result = parse(llm.complete(EXTRACT_SYSTEM, prompt))
    # Models sometimes list the company itself anyway; it would link everything to everything.
    own = normalize(company_name)
    result.entities = [(n, k) for n, k in result.entities if normalize(n) != own]
    result.relations = [r for r in result.relations if own not in (r[0], r[2])]
    return result


def _people(conn: psycopg.Connection, company: str) -> tuple[list[str], dict[str, str]]:
    """The company's people's names, and a key -> user id map (full names, and first names when unique)."""
    rows = conn.execute(
        "SELECT id, name FROM users WHERE home_brain <@ %s::ltree ORDER BY id", (company,)
    ).fetchall()
    names = [(row["id"], person_name(row["name"])) for row in rows]
    by_key = {normalize(name): user_id for user_id, name in names}
    first_names: dict[str, list[str]] = {}
    for user_id, name in names:
        first_names.setdefault(normalize(name.split()[0]), []).append(user_id)
    for key, ids in first_names.items():
        if len(ids) == 1:
            by_key.setdefault(key, ids[0])
    return [name for _, name in names], by_key


def extract_pending(conn: psycopg.Connection, llm: LLM, company: str | None = None) -> dict[str, list[str]]:
    """Extract entities from every memory not processed yet. Runs on the admin connection.

    Returns the processed memory ids per company. The root brain's learned
    patterns are not company knowledge and are skipped.
    """
    conn.row_factory = dict_row
    scope = "AND m.brain_path <@ %(company)s::ltree" if company else ""
    pending = conn.execute(
        f"""SELECT m.id::text, m.content, subpath(m.brain_path, 0, 2)::text AS company, b.name AS company_name
            FROM memories m JOIN brains b ON b.path = subpath(m.brain_path, 0, 2)
            WHERE nlevel(m.brain_path) >= 2 {scope}
              AND NOT EXISTS (SELECT 1 FROM entity_extractions x WHERE x.memory_id = m.id)
            ORDER BY m.created_at, m.id""",
        {"company": company},
    ).fetchall()
    people = {c: _people(conn, c) for c in {row["company"] for row in pending}}

    def run_one(row: dict) -> Extraction | None:
        try:
            return extract(llm, row["content"], row["company_name"], people[row["company"]][0])
        except Exception:  # one failed call shouldn't lose the rest; it is retried next run
            log.exception("Entity extraction failed for memory %s", row["id"])
            return None

    # Model calls are independent, so run them in parallel; save in order.
    with ThreadPoolExecutor(max_workers=EXTRACT_WORKERS) as pool:
        results = list(pool.map(run_one, pending))

    done: dict[str, list[str]] = {}
    for row, extraction in zip(pending, results):
        if extraction is None:
            continue
        _save(conn, row["id"], row["company"], extraction, people[row["company"]][1], llm.name)
        done.setdefault(row["company"], []).append(row["id"])
    conn.commit()
    return done


def _save(
    conn: psycopg.Connection, memory_id: str, company: str, extraction: Extraction,
    people: dict[str, str], model: str,
) -> None:
    ids = {}
    for name, kind in extraction.entities:
        key = normalize(name)
        user_id = people.get(key)
        ids[key] = conn.execute(
            # The first memory to name an entity sets its display name and kind.
            """INSERT INTO entities (company, key, name, kind, user_id)
               VALUES (%s::ltree, %s, %s, %s, %s)
               ON CONFLICT (company, key) DO UPDATE SET user_id = COALESCE(entities.user_id, excluded.user_id)
               RETURNING id::text""",
            (company, key, name, "person" if user_id else kind, user_id),
        ).fetchone()["id"]
        conn.execute(
            "INSERT INTO memory_entities VALUES (%s::uuid, %s::uuid) ON CONFLICT DO NOTHING",
            (memory_id, ids[key]),
        )
    for source, relation, target in extraction.relations:
        conn.execute(
            "INSERT INTO entity_relations VALUES (%s::uuid, %s::uuid, %s, %s::uuid) ON CONFLICT DO NOTHING",
            (memory_id, ids[source], relation, ids[target]),
        )
    conn.execute(
        "INSERT INTO entity_extractions (memory_id, model) VALUES (%s::uuid, %s)", (memory_id, model)
    )


@dataclass(frozen=True)
class Scope:
    """What one person may see: their company, and the memory ids PostgreSQL lets them read."""

    company: str
    visible: tuple[str, ...]


def viewer_scope(conn: psycopg.Connection, home_brain: str) -> Scope:
    """``conn`` must be an ``as_user`` connection: row-level security picks the ids.

    Like search, a replaced memory counts only for people who can't see its
    replacement, so outdated facts don't make someone look like an expert.
    """
    company = tree.company_of(home_brain)
    rows = conn.execute(
        """SELECT m.id::text AS id FROM memories m
           WHERE m.brain_path <@ %s::ltree
             AND (m.superseded_by IS NULL
                  OR NOT EXISTS (SELECT 1 FROM memories n WHERE n.id = m.superseded_by))""",
        (company,),
    ).fetchall()
    return Scope(company, tuple(row["id"] for row in rows))


class GraphUnavailable(Exception):
    pass


class RelationshipGraph:
    """One FalkorDB graph per company: (:Brain), (:Person), (:Memory), (:Entity).

    Edges: PARENT_OF (brain to sub-brain), MEMBER_OF (person to home brain),
    STORED_IN (memory to brain), CREATED (person to memory), MENTIONS (memory
    to entity), SAME_AS (person to the entity naming them) and RELATES_TO
    (entity to entity, with the relation and the id of the memory stating it).
    """

    def __init__(self, url: str, prefix: str) -> None:
        self._db = FalkorDB.from_url(url)
        self.prefix = prefix

    def name(self, company: str) -> str:
        if tree.level(company) != tree.COMPANY_LEVEL:
            raise ValueError(f"{company!r} is not a company brain")
        return f"{self.prefix}_{company.split('.')[1]}"

    def exists(self, company: str) -> bool:
        try:
            return self.name(company) in self._db.list_graphs()
        except redis.exceptions.ConnectionError as exc:
            raise GraphUnavailable(str(exc)) from exc

    def drop_all(self) -> None:
        """Delete every graph with this prefix (tests, or after a schema reset)."""
        for name in self._db.list_graphs():
            if name.startswith(f"{self.prefix}_"):
                self._db.select_graph(name).delete()

    def rebuild(self, conn: psycopg.Connection, company: str) -> None:
        """Recreate a company's graph from PostgreSQL. ``conn`` must be an admin connection."""
        if self.exists(company):
            self._db.select_graph(self.name(company)).delete()
        self.sync(conn, company)

    def sync(self, conn: psycopg.Connection, company: str, memory_ids: list[str] | None = None) -> None:
        """Add or update the company's structure and these memories (all when None)."""
        conn.row_factory = dict_row
        new = not self.exists(company)
        graph = self._db.select_graph(self.name(company))
        if new:
            for label, prop in [("Brain", "path"), ("Person", "id"), ("Memory", "id"), ("Entity", "id")]:
                graph.create_node_range_index(label, prop)

        brains = conn.execute(
            "SELECT path::text, name, kind FROM brains WHERE path <@ %s::ltree", (company,)
        ).fetchall()
        people = conn.execute(
            "SELECT id, name, home_brain::text AS home FROM users WHERE home_brain <@ %s::ltree", (company,)
        ).fetchall()
        only = "" if memory_ids is None else "AND m.id = ANY(%(ids)s::uuid[])"
        params = {"company": company, "ids": memory_ids}
        memories = conn.execute(
            f"""SELECT m.id::text, m.brain_path::text AS brain, m.memory_type AS type, m.created_by
                FROM memories m WHERE m.brain_path <@ %(company)s::ltree {only}""",
            params,
        ).fetchall()
        mentions = conn.execute(
            f"""SELECT me.memory_id::text AS memory, e.id::text AS entity, e.name, e.kind, e.key, e.user_id
                FROM memory_entities me JOIN entities e ON e.id = me.entity_id
                JOIN memories m ON m.id = me.memory_id
                WHERE m.brain_path <@ %(company)s::ltree {only}""",
            params,
        ).fetchall()
        relations = conn.execute(
            f"""SELECT r.memory_id::text AS memory, r.source_id::text AS source, r.relation,
                       r.target_id::text AS target
                FROM entity_relations r JOIN memories m ON m.id = r.memory_id
                WHERE m.brain_path <@ %(company)s::ltree {only}""",
            params,
        ).fetchall()
        entities = list({row["entity"]: row for row in mentions}.values())

        steps = [
            ("UNWIND $rows AS r MERGE (b:Brain {path: r.path}) SET b.name = r.name, b.kind = r.kind", brains),
            ("""UNWIND $rows AS r MATCH (p:Brain {path: r.parent}), (c:Brain {path: r.path})
                MERGE (p)-[:PARENT_OF]->(c)""",
             [{"parent": b["path"].rsplit(".", 1)[0], "path": b["path"]} for b in brains if b["path"] != company]),
            ("""UNWIND $rows AS r MERGE (p:Person {id: r.id}) SET p.name = r.name
                WITH p, r MATCH (b:Brain {path: r.home}) MERGE (p)-[:MEMBER_OF]->(b)""", people),
            ("""UNWIND $rows AS r MERGE (m:Memory {id: r.id}) SET m.type = r.type
                WITH m, r MATCH (b:Brain {path: r.brain}) MERGE (m)-[:STORED_IN]->(b)""", memories),
            ("""UNWIND $rows AS r MATCH (p:Person {id: r.created_by}), (m:Memory {id: r.id})
                MERGE (p)-[:CREATED]->(m)""", [m for m in memories if m["created_by"]]),
            ("""UNWIND $rows AS r MERGE (e:Entity {id: r.entity})
                SET e.name = r.name, e.kind = r.kind, e.key = r.key""", entities),
            ("""UNWIND $rows AS r MATCH (p:Person {id: r.user_id}), (e:Entity {id: r.entity})
                MERGE (p)-[:SAME_AS]->(e)""", [e for e in entities if e["user_id"]]),
            ("""UNWIND $rows AS r MATCH (m:Memory {id: r.memory}), (e:Entity {id: r.entity})
                MERGE (m)-[:MENTIONS]->(e)""", mentions),
            ("""UNWIND $rows AS r MATCH (a:Entity {id: r.source}), (b:Entity {id: r.target})
                MERGE (a)-[:RELATES_TO {relation: r.relation, memory: r.memory}]->(b)""", relations),
        ]
        for cypher, rows in steps:
            for start in range(0, len(rows), SYNC_BATCH):
                graph.query(cypher, {"rows": [dict(r) for r in rows[start:start + SYNC_BATCH]]})

    def _query(self, scope: Scope, cypher: str, **params) -> list[list]:
        """Read-only query in the viewer's company graph, limited to the memories they may see.

        Refusing queries that don't use $visible makes forgetting the
        permission filter a crash instead of a leak.
        """
        if "$visible" not in cypher:
            raise ValueError("Graph queries must be limited to $visible memories")
        try:
            result = self._db.select_graph(self.name(scope.company)).ro_query(
                cypher, {**params, "visible": list(scope.visible)}
            )
        except redis.exceptions.ConnectionError as exc:
            raise GraphUnavailable(str(exc)) from exc
        return result.result_set

    def _visible_entities(self, scope: Scope) -> list[dict]:
        rows = self._query(
            scope,
            """MATCH (m:Memory)-[:MENTIONS]->(e:Entity) WHERE m.id IN $visible
               RETURN e.id, e.name, e.kind, e.key, count(DISTINCT m) ORDER BY e.key""",
        )
        return [{"id": r[0], "name": r[1], "kind": r[2], "key": r[3], "memories": r[4]} for r in rows]

    def _people_of(self, scope: Scope, memory_ids: list[str]) -> dict[str, dict]:
        """People who wrote these memories or are mentioned in them."""
        people: dict[str, dict] = {}
        for role, cypher in [
            ("wrote", """MATCH (p:Person)-[:CREATED]->(m:Memory)
                         WHERE m.id IN $visible AND m.id IN $ids RETURN p.id, p.name, m.id"""),
            ("mentioned_in", """MATCH (p:Person)-[:SAME_AS]->(:Entity)<-[:MENTIONS]-(m:Memory)
                                WHERE m.id IN $visible AND m.id IN $ids RETURN p.id, p.name, m.id"""),
        ]:
            for user_id, name, memory_id in self._query(scope, cypher, ids=memory_ids):
                person = people.setdefault(user_id, {"id": user_id, "name": name, "wrote": set(), "mentioned_in": set()})
                person[role].add(memory_id)
        return people

    def experts(self, scope: Scope, topic: str) -> dict:
        """Who knows about a topic: people and brains linked to visible memories mentioning it."""
        entities = [e for e in self._visible_entities(scope) if matches(topic, e["key"])]
        memory_ids = []
        if entities:
            memory_ids = [r[0] for r in self._query(
                scope,
                """MATCH (m:Memory)-[:MENTIONS]->(e:Entity)
                   WHERE m.id IN $visible AND e.id IN $entities RETURN DISTINCT m.id""",
                entities=[e["id"] for e in entities],
            )]
        people = self._people_of(scope, memory_ids)
        brains: dict[str, dict] = {}
        for path, name, memory_id in self._query(
            scope,
            """MATCH (m:Memory)-[:STORED_IN]->(b:Brain)
               WHERE m.id IN $visible AND m.id IN $ids RETURN b.path, b.name, m.id""",
            ids=memory_ids,
        ):
            brains.setdefault(path, {"path": path, "name": name, "memories": 0})["memories"] += 1
        ranked = sorted(
            ({"id": p["id"], "name": p["name"], "memories": len(p["wrote"] | p["mentioned_in"]),
              "wrote": len(p["wrote"]), "mentioned_in": len(p["mentioned_in"])} for p in people.values()),
            key=lambda p: (-p["memories"], -p["wrote"], p["name"]),
        )
        return {
            "topic": topic,
            "entities": [{"name": e["name"], "kind": e["kind"]} for e in entities],
            "people": ranked,
            "brains": sorted(brains.values(), key=lambda b: (-b["memories"], b["path"])),
            "memory_ids": memory_ids,
        }

    def entity(self, scope: Scope, name: str) -> dict | None:
        """What we know about one entity, or None if the viewer can't see any memory mentioning it.

        "Not found" and "not allowed" look the same, so the answer never
        reveals that a hidden entity exists.
        """
        key = normalize(name)
        candidates = self._visible_entities(scope)
        found = [e for e in candidates if e["key"] == key] or sorted(
            (e for e in candidates if matches(name, e["key"])), key=lambda e: (-e["memories"], len(e["key"]))
        )
        if not found:
            return None
        entity = found[0]
        memory_ids = [r[0] for r in self._query(
            scope,
            "MATCH (m:Memory)-[:MENTIONS]->(e:Entity {id: $id}) WHERE m.id IN $visible RETURN DISTINCT m.id",
            id=entity["id"],
        )]
        relations = [
            {"relation": r[0], "entity": {"name": r[1], "kind": r[2]}, "direction": "out", "memory_id": r[3]}
            for r in self._query(
                scope,
                """MATCH (e:Entity {id: $id})-[r:RELATES_TO]->(o:Entity)
                   WHERE r.memory IN $visible RETURN r.relation, o.name, o.kind, r.memory""",
                id=entity["id"],
            )
        ] + [
            {"relation": r[0], "entity": {"name": r[1], "kind": r[2]}, "direction": "in", "memory_id": r[3]}
            for r in self._query(
                scope,
                """MATCH (o:Entity)-[r:RELATES_TO]->(e:Entity {id: $id})
                   WHERE r.memory IN $visible RETURN r.relation, o.name, o.kind, r.memory""",
                id=entity["id"],
            )
        ]
        related = [
            {"name": r[0], "kind": r[1], "shared_memories": r[2]}
            for r in self._query(
                scope,
                """MATCH (e:Entity {id: $id})<-[:MENTIONS]-(m:Memory)-[:MENTIONS]->(o:Entity)
                   WHERE m.id IN $visible AND o.id <> $id
                   RETURN o.name, o.kind, count(DISTINCT m) AS n ORDER BY n DESC, o.name LIMIT 10""",
                id=entity["id"],
            )
        ]
        people = self._people_of(scope, memory_ids)
        is_person = self._query(
            scope,
            # Only reachable when a visible memory mentions this entity, which `found` already checked.
            """MATCH (p:Person)-[:SAME_AS]->(e:Entity {id: $id})<-[:MENTIONS]-(m:Memory)
               WHERE m.id IN $visible RETURN DISTINCT p.id""",
            id=entity["id"],
        )
        return {
            "name": entity["name"],
            "kind": entity["kind"],
            "person_id": is_person[0][0] if is_person else None,
            "relations": relations,
            "related": related,
            "people": sorted(
                ({"id": p["id"], "name": p["name"], "wrote": len(p["wrote"]),
                  "mentioned_in": len(p["mentioned_in"])} for p in people.values()),
                key=lambda p: (-(p["wrote"] + p["mentioned_in"]), p["name"]),
            ),
            "memory_ids": memory_ids,
        }


def run(settings: Settings, llm: LLM, graph: RelationshipGraph | None, company: str | None = None) -> int:
    """Extract entities from new memories and add them to the graphs (a new graph is built in full).

    PostgreSQL is updated first; if FalkorDB is down, the graph catches up
    with the next rebuild.
    """
    with psycopg.connect(settings.admin_database_url) as conn:
        done = extract_pending(conn, llm, company)
        if graph is not None:
            for name, ids in done.items():
                try:
                    if graph.exists(name):
                        graph.sync(conn, name, ids)
                    else:
                        graph.rebuild(conn, name)
                except (GraphUnavailable, redis.exceptions.ConnectionError):
                    log.warning("FalkorDB unavailable; graph for %s will catch up on the next rebuild", name)
    return sum(len(ids) for ids in done.values())


def main() -> None:
    from secondbrain.models import build_models
    from secondbrain.schema import ensure_relationship_schema

    parser = argparse.ArgumentParser(description="Extract entities from new memories and rebuild company graphs")
    parser.add_argument("--company", help="only this company, e.g. root.lumen")
    args = parser.parse_args()
    settings = get_settings()
    if ensure_relationship_schema(settings):
        print("Added the relationship tables.")
    llm, _ = build_models(settings)
    graph = RelationshipGraph(settings.falkordb_url, settings.graph_prefix)
    with psycopg.connect(settings.admin_database_url, row_factory=dict_row) as conn:
        done = extract_pending(conn, llm, args.company)
        print(f"Extracted entities from {sum(len(ids) for ids in done.values())} new memories.")
        companies = [args.company] if args.company else [
            row["path"] for row in conn.execute("SELECT path::text FROM brains WHERE kind = 'company' ORDER BY path")
        ]
        for company in companies:
            graph.rebuild(conn, company)
            count = conn.execute(
                "SELECT count(*) AS n FROM entities WHERE company = %s::ltree", (company,)
            ).fetchone()["n"]
            print(f"  {company:<16} graph {graph.name(company)}: {count} entities")


if __name__ == "__main__":
    main()
