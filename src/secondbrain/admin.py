"""Onboarding a company and importing its documents.

These run as the admin role, outside row-level security, so they are only
reachable from the CLI or the token-protected admin API.

    uv run python -m secondbrain.admin onboard examples/desiliving/company.json
    uv run python -m secondbrain.admin import root.desiliving.support examples/desiliving/docs/support --sensitivity internal
"""

import argparse
import json
import re
from pathlib import Path

import psycopg

from secondbrain import tree
from secondbrain.config import Settings, get_settings
from secondbrain.documents import SUPPORTED_SUFFIXES, chunk_document, extract_text
from secondbrain.memory.store import vector_literal
from secondbrain.models import Embedder
from secondbrain.schema import vector_type
from secondbrain.sensitivity import Sensitivity

# Roles map to grants. Anything finer-grained (e.g. a Sales lead who may read
# Finance's internal reports) is added as an explicit extra grant.
ROLES = {
    # role: [(brain, include_children, max_sensitivity)]; brain is "home" or "company"
    "member": [("home", False, Sensitivity.INTERNAL), ("company", False, Sensitivity.INTERNAL)],
    "lead": [("home", False, Sensitivity.RESTRICTED), ("company", False, Sensitivity.INTERNAL)],
    "executive": [("company", True, Sensitivity.RESTRICTED)],
}



def _label(name: str) -> str:
    """Turn a display name into an ltree label: lowercase letters, digits, underscores."""
    label = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
    if not label:
        raise ValueError(f"Can't make a brain id from {name!r}")
    return label


def onboard(conn: psycopg.Connection, spec: dict) -> str:
    """Create a company brain, its department brains, users and their grants.

    Re-running with the same spec updates names and adds anything new.
    """
    conn.execute(
        "INSERT INTO brains (path, name, kind) VALUES ('root', 'Root', 'root') ON CONFLICT DO NOTHING"
    )
    company = f"{tree.ROOT}.{spec.get('id') or _label(spec['name'])}"
    sharing = spec.get("sharing", "none")
    if sharing not in {"none", "patterns"}:
        raise ValueError(f"sharing must be 'none' or 'patterns', not {sharing!r}")
    _upsert_brain(conn, company, spec["name"], "company")
    conn.execute("UPDATE brains SET sharing = %s WHERE path = %s::ltree", (sharing, company))
    departments = {}
    for dept in spec.get("departments", []):
        path = f"{company}.{dept.get('id') or _label(dept['name'])}"
        _upsert_brain(conn, path, dept["name"], "department")
        departments[dept.get("id") or _label(dept["name"])] = path

    for user in spec.get("users", []):
        dept_id = user.get("department")
        home = departments[dept_id] if dept_id else company
        role = user.get("role", "member")
        if role not in ROLES:
            raise ValueError(f"Unknown role {role!r} for {user['id']}; use one of {sorted(ROLES)}")
        conn.execute(
            """INSERT INTO users (id, name, home_brain) VALUES (%s, %s, %s::ltree)
               ON CONFLICT (id) DO UPDATE SET name = excluded.name, home_brain = excluded.home_brain""",
            (user["id"], user["name"], home),
        )
        conn.execute("DELETE FROM grants WHERE user_id = %s", (user["id"],))
        grants = [(home if b == "home" else company, c, s) for b, c, s in ROLES[role]]
        for extra in user.get("extra_grants", []):
            grants.append(
                (departments[extra["department"]], False, Sensitivity.parse(extra["max_sensitivity"]))
            )
        for brain, children, level in grants:
            conn.execute(
                """INSERT INTO grants (user_id, brain_path, include_children, max_sensitivity)
                   VALUES (%s, %s::ltree, %s, %s)
                   ON CONFLICT (user_id, brain_path) DO UPDATE
                   SET include_children = grants.include_children OR excluded.include_children,
                       max_sensitivity = GREATEST(grants.max_sensitivity, excluded.max_sensitivity)""",
                (user["id"], brain, children, int(level)),
            )
    return company


def _upsert_brain(conn: psycopg.Connection, path: str, name: str, kind: str) -> None:
    conn.execute(
        """INSERT INTO brains (path, name, kind) VALUES (%s::ltree, %s, %s)
           ON CONFLICT (path) DO UPDATE SET name = excluded.name""",
        (path, name, kind),
    )


def import_documents(
    conn: psycopg.Connection,
    embedder: Embedder,
    settings: Settings,
    brain_path: str,
    documents: list[tuple[str, str]],
    sensitivity: Sensitivity,
) -> int:
    """Store (name, text) documents as long-term semantic memory of one brain."""
    if conn.execute("SELECT 1 FROM brains WHERE path = %s::ltree", (brain_path,)).fetchone() is None:
        raise ValueError(f"No brain {brain_path!r}; onboard the company first")
    cast = vector_type(settings.embedding_dim)[0].split("(")[0]
    pieces = [(name, part) for name, text in documents for part in chunk_document(name, text)]
    if not pieces:
        return 0
    embeddings = embedder.embed([part for _, part in pieces])
    for (name, part), embedding in zip(pieces, embeddings):
        conn.execute(
            f"""INSERT INTO memories (brain_path, memory_type, tier, sensitivity, content,
                                      embedding, embedding_model, source)
                VALUES (%s::ltree, 'semantic', 'long_term', %s, %s, %s::{cast}, %s, %s)""",
            (brain_path, int(sensitivity), f"{part}\n\n(Source: {name})",
             vector_literal(embedding), embedder.name, name),
        )
    return len(pieces)


def read_documents(path: Path) -> list[tuple[str, str]]:
    """(name, text) for every supported document at a path (a file or a folder)."""
    files = [path] if path.is_file() else sorted(p for p in path.rglob("*") if p.is_file())
    return [(f.name, extract_text(f.name, f.read_bytes())) for f in files
            if f.suffix.lower() in SUPPORTED_SUFFIXES and not f.name.startswith(".")]


def main() -> None:
    from secondbrain.models import build_models

    parser = argparse.ArgumentParser(description="SecondBrain admin")
    commands = parser.add_subparsers(dest="command", required=True)
    onboard_cmd = commands.add_parser("onboard", help="create or update a company from a JSON spec")
    onboard_cmd.add_argument("spec", type=Path)
    import_cmd = commands.add_parser("import", help="import .md/.txt/.pdf/.docx documents into a brain")
    import_cmd.add_argument("brain_path")
    import_cmd.add_argument("path", type=Path)
    import_cmd.add_argument("--sensitivity", default="internal", choices=["public", "internal", "restricted"])
    args = parser.parse_args()

    settings = get_settings()
    with psycopg.connect(settings.admin_database_url) as conn:
        if args.command == "onboard":
            company = onboard(conn, json.loads(args.spec.read_text()))
            print(f"Onboarded {company}")
        else:
            llm, embedder = build_models(settings)
            count = import_documents(
                conn, embedder, settings, args.brain_path,
                read_documents(args.path), Sensitivity.parse(args.sensitivity),
            )
            print(f"Imported {count} chunks into {args.brain_path}")
    if args.command == "import" and count and tree.level(args.brain_path) >= tree.COMPANY_LEVEL:
        from secondbrain.relationships import connect_graph, run

        # After the import committed: extraction reads the new chunks on its own connection.
        graph = connect_graph(settings)
        found = run(settings, llm, graph, tree.company_of(args.brain_path))
        print(f"Extracted entities from {found} memories")


if __name__ == "__main__":
    main()
