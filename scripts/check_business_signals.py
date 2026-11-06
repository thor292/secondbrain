"""Check the business signals the real model writes, on a database of its own.

Business signals come from the model, so tests with the offline stub can only
check the rules around it. This runs the real model (the LiteLLM settings in
an env file) against secondbrain_test_signals_live, never the demo database,
and reports:

- evidence: every signal cites notes that exist and that the people who see
  it may read; the model's raw replies are compared with what was kept;
- numbers: how many proposals the number check dropped (an unsupported number
  in the title or action) or flagged (in the reason or impact);
- repeats: the same advice keeps its fingerprint across runs, and advice
  someone dismissed isn't raised again in other words;
- little data: one trivial note, and no notes at all.

    uv run python scripts/check_business_signals.py --env ../SecondBrain/.env
"""

import argparse
import json
import re
import sys
from datetime import date
from pathlib import Path

import psycopg
from psycopg.rows import dict_row

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

from secondbrain import admin, signals  # noqa: E402
from secondbrain.config import Settings  # noqa: E402
from secondbrain.memory.store import vector_literal  # noqa: E402
from secondbrain.models import build_models  # noqa: E402
from secondbrain.schema import create_schema, vector_type  # noqa: E402

SERVER = "postgresql://secondbrain_admin:dev_admin_password@localhost:55432"
DB = "secondbrain_test_signals_live"

# Realistic notes in Acme's departments: (brain, author, note).
NOTES = [
    ("root.acme.sales", "asha", "Bluebird Stores wants a bulk order of 400 Cloud throws before December 12, at 15 percent off list"),
    ("root.acme.sales", "asha", "Storm in the Midwest: the carrier delayed ground shipments to Chicago and Detroit by 2 days this week"),
    ("root.acme.sales", "vikram", "Where-is-my-order tickets are up 35 percent since the carrier delays started"),
    ("root.acme.sales", "vikram", "Loyalty early access for the festive sale moves from November 24 to November 23"),
    ("root.acme.sales", "asha", "Received 40 damaged Halo floor lamps from Lampworks today; quarantined and reported"),
    ("root.acme.finance", "meera", "October close finished on the 4th business day as planned"),
    ("root.acme.finance", "meera", "Lampworks will credit us for the 40 damaged Halo lamps on the next invoice"),
    ("root.acme.hr", "sara", "Two support agent roles approved, both to be hired before December"),
]


def ensure_database() -> None:
    with psycopg.connect(f"{SERVER}/secondbrain", autocommit=True) as conn:
        if not conn.execute("SELECT 1 FROM pg_database WHERE datname = %s", (DB,)).fetchone():
            conn.execute(f'CREATE DATABASE "{DB}"')
            conn.execute(f'GRANT CONNECT ON DATABASE "{DB}" TO secondbrain_app')
    with psycopg.connect(f"{SERVER}/{DB}", autocommit=True) as conn:
        conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        conn.execute("CREATE EXTENSION IF NOT EXISTS ltree")
        conn.execute("GRANT USAGE ON SCHEMA public TO secondbrain_app")


def add_notes(conn, embedder, settings, notes) -> None:
    cast = vector_type(settings.embedding_dim)[0].split("(")[0]
    vectors = embedder.embed([n for _, _, n in notes])
    for (brain, author, content), vector in zip(notes, vectors):
        conn.execute(
            f"""INSERT INTO memories (brain_path, memory_type, tier, sensitivity, content, embedding,
                                      embedding_model, created_by)
                VALUES (%s::ltree, 'episodic', 'long_term', 1, %s, %s::{cast}, %s, %s)""",
            (brain, content, vector_literal(vector), embedder.name, author))
    conn.commit()


class Recording:
    """Passes calls to the real model and keeps its raw replies."""

    def __init__(self, llm):
        self.llm, self.name, self.replies, self.prompts = llm, llm.name, [], []

    def complete(self, system, prompt):
        self.prompts.append(prompt)
        reply = self.llm.complete(system, prompt)
        self.replies.append(reply)
        return reply


def raw_items(reply: str) -> list[dict]:
    match = re.search(r"\{.*\}", reply or "", re.DOTALL)
    try:
        return json.loads(match.group(0)).get("signals", []) if match else []
    except json.JSONDecodeError:
        return []


def why_dropped(item: dict, prompt: str) -> str:
    """Re-run the checks business_signals applies, to say which one dropped a proposal."""
    notes = {k: text for k, text in re.findall(r"^(N\d+) \([^)]*\): (.*)$", prompt, re.MULTILINE)}
    lessons = dict(re.findall(r"^(L\d+): (.*)$", prompt, re.MULTILINE))
    cited = [notes[k] for k in item.get("notes") or [] if k in notes]
    if not cited:
        return "cites no note of this business"
    known = signals.evidence_numbers(cited + [lessons[k] for k in item.get("lessons") or [] if k in lessons],
                                     [date.today()])
    bad = signals.unsupported_numbers(f"{item.get('title')} {item.get('action')}", known)
    if bad:
        return f"unsupported number(s) in title or action: {', '.join(bad)}"
    return "same advice as another signal, or dismissed recently"


def visible_to(settings, user: str) -> list[dict]:
    with psycopg.connect(settings.app_database_url) as conn:
        conn.execute("SELECT set_config('app.user_id', %s, false)", (user,))
        return [s for s in signals.visible(conn) if s["kind"] == "business"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--env", required=True, help="env file with the LiteLLM settings (only those are used)")
    parser.add_argument("--runs", type=int, default=3)
    args = parser.parse_args()

    live = Settings(_env_file=args.env)
    settings = live.model_copy(update={"admin_database_url": f"{SERVER}/{DB}",
                                       "app_database_url": f"{SERVER.replace('secondbrain_admin:dev_admin', 'secondbrain_app:dev_app')}/{DB}",
                                       "model_provider": "litellm"})
    llm, embedder = build_models(settings)
    print(f"Model {llm.name}, embeddings {embedder.name} ({settings.embedding_dim} dims), database {DB}")
    ensure_database()
    create_schema(settings, reset=True)
    with psycopg.connect(settings.admin_database_url) as conn:
        for company in ("acme", "globex"):
            admin.onboard(conn, json.loads((ROOT / f"tests/fixtures/{company}/company.json").read_text()))
        add_notes(conn, embedder, settings, NOTES)
        known_ids = {r[0] for r in conn.execute("SELECT id::text FROM memories")}

    recording = Recording(llm)
    runs = []
    for n in range(args.runs):
        kept = signals.run(settings, recording, "root.acme")["root.acme"]
        business = [s for s in kept if s.kind == "business"]
        proposed = raw_items(recording.replies[-1])
        runs.append(business)
        print(f"\nRun {n + 1}: the model proposed {len(proposed)}, {len(business)} kept")
        for item in proposed:
            if not any(s.title in str(item.get("title")) or str(item.get("title")) in s.title for s in business):
                print(f"  dropped: {item.get('title')}  ({why_dropped(item, recording.prompts[-1])})")
        for s in business:
            ids_in_text = re.findall(r"\b[NL]\d+\b", " ".join((s.title, s.action, s.reason, s.impact)))
            assert set(s.evidence["memories"]) <= known_ids and s.evidence["memories"], "cites a note that doesn't exist"
            assert not ids_in_text, f"ids left in text: {ids_in_text}"
            flags = [f for f in s.evidence.get("facts", []) if f.startswith("Check before")]
            print(f"  [{s.urgency}, {s.confidence:.0%}] {s.title}\n      -> {s.action}"
                  f"\n      cites {len(s.evidence['memories'])} note(s), {len(s.evidence['lessons'])} lesson(s)"
                  + (f"; FLAGGED: {flags[0]}" if flags else ""))

    # Same advice, same fingerprint, across runs.
    for n, run in enumerate(runs[1:], 2):
        before = {s.fingerprint for r in runs[:n - 1] for s in r}
        same = sum(s.fingerprint in before for s in run)
        print(f"\nRun {n}: {same} of {len(run)} signals keep the fingerprint of the same advice in an earlier run")

    # Each person sees only advice whose every note they may read.
    for user in ("asha", "vikram", "ravi", "meera", "sara", "lee"):
        seen = visible_to(settings, user)
        print(f"{user} sees {len(seen)} business signal(s)")

    # Dismiss one; the next run must not raise it again, however it is worded.
    if runs[-1]:
        target = runs[-1][0]
        with psycopg.connect(settings.admin_database_url) as conn:
            conn.execute("""UPDATE signals SET status = 'dismissed', feedback_by = 'asha', feedback_note = 'not a priority',
                            feedback_at = now() WHERE fingerprint = %s AND status = 'open'""", (target.fingerprint,))
            conn.commit()
        again = [s for s in signals.run(settings, recording, "root.acme")["root.acme"] if s.kind == "business"]
        proposed = raw_items(recording.replies[-1])
        print(f"\nAfter dismissing '{target.title}': the model proposed {len(proposed)}, {len(again)} kept; "
              f"dismissed advice raised again: {any(s.fingerprint == target.fingerprint for s in again)}")
        for s in again:
            print(f"  {s.title}")

    # Little data: one trivial note; then none at all.
    with psycopg.connect(settings.admin_database_url) as conn:
        add_notes(conn, embedder, settings, [("root.globex.ops", "lee", "Team lunch moved to Friday")])
    tiny = [s for s in signals.run(settings, recording, "root.globex")["root.globex"] if s.kind == "business"]
    print(f"\nOne trivial note (Globex): {len(tiny)} signal(s)" + "".join(f"\n  {s.title} -> {s.action}" for s in tiny))
    with psycopg.connect(settings.admin_database_url, row_factory=dict_row) as conn:
        conn.execute("DELETE FROM memories WHERE brain_path <@ 'root.globex'")
        calls = len(recording.replies)
        empty = signals.business_signals(conn, recording, "root.globex")
        print(f"No notes: {len(empty)} signal(s), model asked: {len(recording.replies) > calls}")


if __name__ == "__main__":
    main()
