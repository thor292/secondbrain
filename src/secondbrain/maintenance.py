"""Memory maintenance: merge duplicates and retire outdated facts.

Runs as a system job on the admin connection. Each memory that hasn't been
maintained yet is compared with similar older memories in the same company,
and the model judges each pair:

- "duplicate": the new memory says the same thing, so it is merged into the
  older one;
- "updates": the new memory gives a new value for the same fact (a new price,
  date or rule), so the older one is superseded by it;
- "partial": the older memory holds several facts and one of them changed, so
  it stays active but is marked as partly outdated;
- "unrelated": nothing to do. Events never supersede earlier events.

Nothing is deleted. Search hides a replaced memory only from people who can
see its replacement (see MemoryStore.search).

    uv run python -m secondbrain.maintenance
"""

import json
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import psycopg
from psycopg.rows import dict_row

from secondbrain.config import Settings, get_settings
from secondbrain.models import LLM

# Only reasonably similar pairs are worth asking the model about.
CANDIDATE_MAX_DISTANCE = 0.35
CANDIDATES_PER_MEMORY = 4
JUDGE_WORKERS = 8

JUDGE_SYSTEM = """You maintain a company's memory. Compare an OLD memory with a NEW memory written later.

Reply with JSON only: {"relation": "...", "reason": "..."}, with a one-sentence reason.

relation is one of:
- "duplicate": NEW states the same information as OLD, nothing more.
- "updates": NEW gives a different value for the same specific fact as OLD (same subject, same
  attribute: a new price, date, threshold, rule or status), and nothing else in OLD stays relevant.
- "partial": OLD contains several facts and NEW changes exactly one of them; the others stay true.
- "unrelated": anything else.

Be strict. These are "unrelated", not updates:
- NEW adds details, examples or a new item (OLD is still fully true).
- OLD is a goal, target or plan and NEW is an actual result, or the other way round.
- NEW describes something that happened after OLD (a follow-up event); past events stay true.
- They are about different subjects that merely sound alike."""


@dataclass
class Change:
    relation: str  # "duplicate", "updates" or "partial"
    old_id: str
    new_id: str
    old: str
    new: str
    reason: str


def _judge(llm: LLM, old: str, new: str) -> tuple[str, str]:
    reply = llm.complete(JUDGE_SYSTEM, f"OLD: {old}\n\nNEW: {new}")
    match = re.search(r"\{.*\}", reply, re.DOTALL)
    try:
        data = json.loads(match.group(0)) if match else {}
    except json.JSONDecodeError:
        data = {}
    relation = data.get("relation")
    return (relation if relation in {"duplicate", "updates", "partial"} else "unrelated"), str(data.get("reason", ""))


def maintain(conn: psycopg.Connection, llm: LLM, company: str | None = None) -> list[Change]:
    """Process every memory not yet maintained (optionally within one company)."""
    conn.row_factory = dict_row
    scope = "AND brain_path <@ %(company)s::ltree" if company else ""
    pending = conn.execute(
        f"""SELECT id::text FROM memories
            WHERE maintained_at IS NULL AND status = 'active' {scope}
            ORDER BY created_at, id""",
        {"company": company},
    ).fetchall()
    pairs = []
    for new in pending:
        pairs += [(new["id"], row) for row in conn.execute(
            """SELECT o.id::text AS old_id, o.content AS old, n.content AS new, o.memory_type AS old_type,
                      o.embedding <=> n.embedding AS distance
               FROM memories o, memories n
               WHERE n.id = %(id)s::uuid AND o.id <> n.id AND o.status = 'active'
                 AND o.created_at < n.created_at
                 AND (o.source IS NULL OR n.source IS NULL OR o.source <> n.source)
                 AND nlevel(o.brain_path) >= 2
                 AND subpath(o.brain_path, 0, 2) = subpath(n.brain_path, 0, 2)
                 AND o.embedding <=> n.embedding < %(max)s
               ORDER BY distance LIMIT %(limit)s""",
            {"id": new["id"], "max": CANDIDATE_MAX_DISTANCE, "limit": CANDIDATES_PER_MEMORY},
        )]

    # Model calls are independent, so judge them in parallel; apply in time order.
    with ThreadPoolExecutor(max_workers=JUDGE_WORKERS) as pool:
        verdicts = list(pool.map(lambda pair: _judge(llm, pair[1]["old"], pair[1]["new"]), pairs))

    changes: list[Change] = []
    retired: set[str] = set()  # memories superseded or merged during this run
    for (new_id, old), (relation, reason) in zip(pairs, verdicts):
        # Events are history: a later event never makes an earlier one untrue.
        if old["old_type"] == "episodic" and relation in {"updates", "partial"}:
            relation = "unrelated"
        if relation == "unrelated" or new_id in retired or old["old_id"] in retired:
            continue
        if relation == "partial":
            conn.execute(
                "UPDATE memories SET amended_by = %s::uuid, status_reason = %s WHERE id = %s::uuid",
                (new_id, reason, old["old_id"]),
            )
            changes.append(Change(relation, old["old_id"], new_id, old["old"], old["new"], reason))
            continue
        if relation == "duplicate":
            conn.execute(
                """UPDATE memories SET status = 'merged', superseded_by = %s::uuid, status_reason = %s
                   WHERE id = %s::uuid""",
                (old["old_id"], reason, new_id),
            )
            retired.add(new_id)
        else:
            conn.execute(
                """UPDATE memories SET status = 'superseded', superseded_by = %s::uuid, status_reason = %s
                   WHERE id = %s::uuid""",
                (new_id, reason, old["old_id"]),
            )
            retired.add(old["old_id"])
        changes.append(Change(relation, old["old_id"], new_id, old["old"], old["new"], reason))
    if pending:
        conn.execute(
            "UPDATE memories SET maintained_at = now() WHERE id = ANY(%s::uuid[])",
            ([row["id"] for row in pending],),
        )
    conn.commit()
    return changes


def run(settings: Settings, llm: LLM, company: str | None = None) -> list[Change]:
    with psycopg.connect(settings.admin_database_url) as conn:
        return maintain(conn, llm, company)


def main() -> None:
    from secondbrain.models import build_models

    settings = get_settings()
    llm, _ = build_models(settings)
    changes = run(settings, llm)
    for c in changes:
        verb = "duplicate of" if c.relation == "duplicate" else "replaces"
        print(f"- {c.new[:70]!r}\n    {verb} {c.old[:70]!r}\n    because: {c.reason}")
    print(f"{len(changes)} change(s).")


if __name__ == "__main__":
    main()
