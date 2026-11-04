"""Graph views of SecondBrain, in the style of a graph database.

Two views:
- ``build_graph``: what one person can see. Built from the same
  permission-checked queries as everything else, limited to their company.
- ``build_full_graph``: the whole tree (root, every company, every memory).
  Admin only, run on the admin connection outside row-level security.
"""

import re

import psycopg

from secondbrain import tree
from secondbrain.memory.store import User
from secondbrain.sensitivity import Sensitivity

MAX_MEMORIES = 300
MAX_MEMORIES_FULL = 2000
# Memories closer than this (cosine distance) get a RELATED_TO edge.
RELATED_MAX_DISTANCE = 0.55
MAX_RELATED_PER_MEMORY = 3


def _title(content: str) -> str:
    first = next((line for line in content.splitlines() if line.strip()), content)
    first = re.sub(r"^#+\s*", "", first).strip()
    return first if len(first) <= 48 else first[:45] + "..."


def _access_edge(user_id: str, row: dict) -> dict:
    level = Sensitivity(row["max_sensitivity"]).name.lower()
    scope = " + sub-brains" if row["include_children"] else ""
    return {"source": f"user:{user_id}", "target": f"brain:{row['brain_path']}",
            "label": "CAN_ACCESS", "props": {"up_to": level + scope}}


def _collect(conn: psycopg.Connection, scope: str, memory_limit: int, viewer: str | None) -> tuple[list, list]:
    """Brains, people and memories under ``scope``, plus their structural edges.

    Which memories come back depends on the connection: the app role only gets
    rows row-level security allows; the admin role gets everything.
    """
    nodes: list[dict] = []
    edges: list[dict] = []

    for row in conn.execute(
        "SELECT path::text, name, kind FROM brains WHERE path <@ %s::ltree ORDER BY path", (scope,)
    ):
        nodes.append({"id": f"brain:{row['path']}", "label": row["name"], "kind": "brain",
                      "props": {"path": row["path"], "kind": row["kind"]}})
        if "." in row["path"] and row["path"] != scope:
            parent = row["path"].rsplit(".", 1)[0]
            edges.append({"source": f"brain:{parent}", "target": f"brain:{row['path']}", "label": "PARENT_OF"})

    for row in conn.execute(
        "SELECT id, name, home_brain::text FROM users WHERE home_brain <@ %s::ltree ORDER BY id", (scope,)
    ):
        nodes.append({"id": f"user:{row['id']}", "label": row["name"], "kind": "user",
                      "props": {"id": row["id"], "home_brain": row["home_brain"]},
                      "me": row["id"] == viewer})
        edges.append({"source": f"user:{row['id']}", "target": f"brain:{row['home_brain']}", "label": "MEMBER_OF"})

    memories = conn.execute(
        """SELECT id::text, brain_path::text, memory_type, sensitivity, content, created_by, created_at
           FROM memories WHERE brain_path <@ %s::ltree ORDER BY created_at DESC LIMIT %s""",
        (scope, memory_limit),
    ).fetchall()
    for row in memories:
        node_id = f"memory:{row['id']}"
        nodes.append({
            # "user" memories (about a person) get their own kind so they aren't confused with people.
            "id": node_id, "label": _title(row["content"]),
            "kind": "user_mem" if row["memory_type"] == "user" else row["memory_type"],
            "props": {"type": row["memory_type"], "tier": "long_term",
                      "sensitivity": Sensitivity(row["sensitivity"]).name.lower(),
                      "brain": row["brain_path"], "created": row["created_at"].isoformat(timespec="seconds"),
                      "content": row["content"]},
        })
        edges.append({"source": node_id, "target": f"brain:{row['brain_path']}", "label": "STORED_IN"})
        if row["created_by"]:
            edges.append({"source": f"user:{row['created_by']}", "target": node_id, "label": "CREATED"})

    edges += _related_edges(conn, [row["id"] for row in memories])
    return nodes, edges


def _related_edges(conn: psycopg.Connection, ids: list[str]) -> list[dict]:
    """Link similar memories, only ever within the same company."""
    if not ids:
        return []
    rows = conn.execute(
        """SELECT a.id::text AS a, b.id::text AS b, a.embedding <=> b.embedding AS distance
           FROM memories a JOIN memories b ON a.id < b.id
           WHERE a.id = ANY(%(ids)s::uuid[]) AND b.id = ANY(%(ids)s::uuid[])
             AND nlevel(a.brain_path) >= 2 AND nlevel(b.brain_path) >= 2
             AND subpath(a.brain_path, 0, 2) = subpath(b.brain_path, 0, 2)
             AND a.embedding <=> b.embedding < %(max)s
           ORDER BY distance""",
        {"ids": ids, "max": RELATED_MAX_DISTANCE},
    ).fetchall()
    edges, per_node = [], {}
    for row in rows:
        if per_node.get(row["a"], 0) >= MAX_RELATED_PER_MEMORY or per_node.get(row["b"], 0) >= MAX_RELATED_PER_MEMORY:
            continue
        per_node[row["a"]] = per_node.get(row["a"], 0) + 1
        per_node[row["b"]] = per_node.get(row["b"], 0) + 1
        edges.append({"source": f"memory:{row['a']}", "target": f"memory:{row['b']}",
                      "label": "RELATED_TO", "props": {"similarity": round(1 - row["distance"], 2)}})
    return edges


def build_graph(conn: psycopg.Connection, user: User, short_term: list[dict]) -> dict:
    """What one person can see in their own company. ``conn`` must be an ``as_user`` connection."""
    company = tree.company_of(user.home_brain)
    nodes, edges = _collect(conn, company, MAX_MEMORIES, viewer=user.id)

    for row in conn.execute("SELECT brain_path::text, include_children, max_sensitivity FROM sb_my_grants()"):
        edges.append(_access_edge(user.id, row))

    for i, item in enumerate(short_term):
        node_id = f"short_term:{i}"
        nodes.append({"id": node_id, "label": _title(item["content"]), "kind": "short_term",
                      "props": {"type": item["memory_type"], "tier": "short_term",
                                "sensitivity": Sensitivity(item["sensitivity"]).name.lower(),
                                "content": item["content"]}})
        edges.append({"source": f"user:{user.id}", "target": node_id, "label": "WORKING_ON"})

    return {"scope": company, "viewer": user.id, "nodes": nodes, "edges": edges}


def build_full_graph(conn: psycopg.Connection) -> dict:
    """The entire SecondBrain tree. ``conn`` must be an admin connection."""
    nodes, edges = _collect(conn, tree.ROOT, MAX_MEMORIES_FULL, viewer=None)
    for row in conn.execute(
        "SELECT user_id, brain_path::text, include_children, max_sensitivity FROM grants ORDER BY user_id"
    ):
        edges.append(_access_edge(row["user_id"], row))
    return {"scope": tree.ROOT, "viewer": None, "nodes": nodes, "edges": edges}
