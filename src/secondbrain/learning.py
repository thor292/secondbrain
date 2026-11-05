"""Learning across the tree: anonymized patterns flow from companies to the root.

Only companies whose sharing setting is "patterns" take part, and only their
public and internal memories are read; restricted memories never leave the
company. The model turns a company's memories into general business lessons,
then a strict rule-based check rejects any lesson that still contains a
number, a person's name, the company's name, or another identifying term.
What reaches the root records no link back to the company it came from.

Outcomes are learned from too, without the model: how run-out warnings
turned out when people acted quickly, acted late, ignored them, or dismissed
them for a given reason, aggregated over every sharing company and stated in
rough proportions ("about nine times in ten"), never counts, names or products.
These lessons sit in the root brain, where business signals read them.

    uv run python -m secondbrain.learning
"""

import json
import re
from dataclasses import dataclass

import psycopg
from psycopg.rows import dict_row, tuple_row

from secondbrain import tree
from secondbrain.config import Settings, get_settings
from secondbrain.memory.store import vector_literal
from secondbrain.models import LLM, Embedder
from secondbrain.schema import vector_type
from secondbrain.sensitivity import Sensitivity

MAX_MEMORIES = 120
MAX_PATTERNS = 8
# A new pattern this close to an existing root pattern is a repeat, not new knowledge.
REPEAT_MAX_DISTANCE = 0.15
SOURCE = "learned pattern"
OUTCOME_SOURCE = "outcome lesson"
# An outcome lesson needs this many judged warnings behind it: fewer would be
# anecdote, and would come close to describing one company's few signals.
MIN_OUTCOMES = 5
PROMPT_DAYS = 2  # "acted on promptly": within two days of the warning
LONG_LEAD_DAYS = 14
# Generic business acronyms that don't identify anyone.
ACRONYMS_OK = {"VIP", "SLA", "CSAT", "CAC", "ROAS", "MAP", "SKU", "KPI", "SMS", "FAQ", "ETA", "WISMO", "EBITDA", "CEO", "CFO"}

LEARN_SYSTEM = f"""You turn one company's internal knowledge into general, reusable business lessons
for a network of many companies. Write at most {MAX_PATTERNS} lessons that would help OTHER companies.

Rules, all mandatory:
- No names of any kind: no company, person, customer, supplier, partner, product, brand,
  place or tool names. Say "a furniture supplier", "a carrier", "a VIP customer" instead.
- No numbers at all: no prices, percentages, dates, counts or amounts. Use relative words
  such as "a small credit", "a lower threshold", "during peak season".
- Each lesson is one or two sentences and states a practice and why it works.
- Skip anything that only makes sense for this one company.

Reply with JSON only: {{"patterns": [{{"pattern": "...", "topic": "..."}}]}}"""


@dataclass
class Pattern:
    text: str
    topic: str


def _forbidden_terms(conn: psycopg.Connection, company: str, contents: list[str]) -> set[str]:
    """Words that would identify the company: its name, its brains, its people,
    and capitalized names that appear in its memories (products, suppliers, customers)."""
    terms: set[str] = set()
    # Plain tuples: the caller's connection may return rows as dicts, and
    # unpacking a dict gives its column name, not the value.
    with conn.cursor(row_factory=tuple_row) as cur:
        names = [r[0] for r in cur.execute("SELECT name FROM brains WHERE path <@ %s::ltree", (company,))]
        people = [r[0] for r in cur.execute("SELECT name FROM users WHERE home_brain <@ %s::ltree", (company,))]
        # Products and suppliers name the business as surely as its own name
        # does; their capitalized words are the names ("Halo" in "Halo brass
        # floor lamp"), the rest describe ("floor lamp") and stay usable.
        labels = [r[0] for r in cur.execute(
            """SELECT name FROM products WHERE brain_path <@ %s::ltree
               UNION SELECT supplier FROM inventory WHERE brain_path <@ %s::ltree""", (company, company))]
    for label in labels:
        terms |= {w.lower() for w in re.findall(r"\b[A-Z][A-Za-z'&-]+", label) if len(w) > 2}
    for name in names:
        terms |= {w.lower() for w in re.findall(r"[A-Za-z][A-Za-z'&.-]+", name) if len(w) > 2}
    terms |= set(company.split(".")[1:])
    for name in people:
        terms |= {w.lower() for w in re.findall(r"[A-Za-z]+", name.split("(")[0]) if len(w) > 2}
    common = {"the", "and", "for", "our", "we", "from", "with", "this", "that", "every", "each",
              "customer", "customers", "support", "marketing", "finance", "operations", "people",
              "merchandising", "black", "friday", "holiday", "christmas", "home", "team", "board"}
    for text in contents:
        body = [line for line in text.splitlines() if "›" not in line and not line.startswith("(Source:")]
        for sentence in re.split(r"(?<=[.!?])\s+|\n+", " ".join(body)):
            words = re.findall(r"[A-Za-z][A-Za-z'&-]*", sentence)
            # Capitalized words after the first word of a sentence are usually
            # names (people, products, suppliers, places); so are acronyms like UPS.
            for word in words[1:]:
                if (word[0].isupper() and len(word) > 2) and word.upper() not in ACRONYMS_OK:
                    terms.add(word.lower())
        terms |= {w.lower().lstrip("@") for w in re.findall(r"@\w+", text)}
    return terms - common


def is_anonymous(pattern: str, forbidden: set[str]) -> bool:
    if re.search(r"\d", pattern):
        return False
    if re.search(r"@|https?://|\S+@\S+", pattern):
        return False
    words = {w.lower() for w in re.findall(r"[A-Za-z][A-Za-z'-]*", pattern)}
    return not (words & forbidden)


def learn_from_company(conn: psycopg.Connection, llm: LLM, company: str) -> tuple[list[Pattern], int]:
    rows = conn.execute(
        """SELECT content FROM memories
           WHERE brain_path <@ %s::ltree AND status = 'active' AND sensitivity <= %s
           ORDER BY created_at DESC LIMIT %s""",
        (company, int(Sensitivity.INTERNAL), MAX_MEMORIES),
    ).fetchall()
    contents = [row["content"] for row in rows]
    if not contents:
        return [], 0
    reply = llm.complete(LEARN_SYSTEM, "\n\n---\n\n".join(contents))
    match = re.search(r"\{.*\}", reply, re.DOTALL)
    try:
        items = json.loads(match.group(0)).get("patterns", []) if match else []
    except json.JSONDecodeError:
        items = []
    forbidden = _forbidden_terms(conn, company, contents)
    patterns, rejected = [], 0
    for item in items[:MAX_PATTERNS]:
        text = str(item.get("pattern", "")).strip()
        if text and is_anonymous(text, forbidden):
            patterns.append(Pattern(text, str(item.get("topic", "")).strip().lower()))
        elif text:
            rejected += 1
    return patterns, rejected


def _proportion(part: int, whole: int) -> str:
    """A share in words, rounded to tenths: exact counts could identify a company."""
    if part == whole:
        return "every time"
    if part == 0:
        return "never"
    tenths = round(10 * part / whole)
    words = ["almost never", "about one time in ten", "about two times in ten", "about three times in ten",
             "about four times in ten", "about half the time", "about six times in ten", "about seven times in ten",
             "about eight times in ten", "about nine times in ten", "almost every time"]
    return words[tenths]


DISMISSED_BECAUSE = {
    "stock elsewhere": "because stock was said to be available elsewhere",
    "already handled": "as already handled",
    "not a priority": "as not a priority",
    "data is wrong": "because the numbers looked wrong",
    "phasing out": "because the product was being phased out",
}


def outcome_patterns(conn: psycopg.Connection) -> list[tuple[str, str]]:
    """Structured lessons from judged run-out warnings at every sharing company:
    (key, lesson). Each needs MIN_OUTCOMES judged warnings behind it."""
    from secondbrain.signals import dismissal_reason

    rows = conn.execute(
        """SELECT s.status, s.feedback_note, s.feedback_at - s.created_at AS delay,
                  (s.details->>'lead_time_days')::int AS lead, o.result
           FROM signal_outcomes o JOIN signals s ON s.id = o.signal_id
           JOIN brains b ON b.path = s.company
           WHERE b.kind = 'company' AND b.sharing = 'patterns' AND s.kind = 'stockout'"""
    ).fetchall()
    acted = [r for r in rows if r["status"] == "acted"]
    ignored = [r for r in rows if r["status"] in ("dismissed", "expired")]
    prompt = [r for r in acted if r["delay"] is not None and r["delay"].days < PROMPT_DAYS]
    late = [r for r in acted if r not in prompt]
    long_lead = [r for r in acted if (r["lead"] or 0) > LONG_LEAD_DAYS]
    groups = [
        ("acted promptly", prompt, "prevented", "Run-out warnings acted on within two days were prevented {}."),
        ("acted late", late, "prevented", "Run-out warnings acted on more than two days after they were raised "
                                          "were prevented {}."),
        ("ignored", ignored, "happened", "Run-out warnings nobody acted on came true {}."),
        ("long lead time", long_lead, "happened_anyway",
         "When the supplier's lead time is long, stock ordered on a run-out warning still arrived too late {}."),
    ]
    for reason, words in DISMISSED_BECAUSE.items():
        dismissed = [r for r in ignored if r["status"] == "dismissed" and dismissal_reason(r["feedback_note"]) == reason]
        groups.append((f"dismissed: {reason}", dismissed, "happened",
                       f"Run-out warnings dismissed {words} came true {{}}."))
    return [(key, text.format(_proportion(sum(r["result"] == result for r in group), len(group))))
            for key, group, result, text in groups if len(group) >= MIN_OUTCOMES]


def learn_outcomes(conn: psycopg.Connection, embedder: Embedder, settings: Settings) -> list[str]:
    """Store outcome lessons in the root brain; a changed proportion replaces
    the older lesson with the same key. Returns the lessons that are new or changed."""
    forbidden: set[str] = set()
    for row in conn.execute("SELECT path::text AS path FROM brains WHERE kind = 'company' AND sharing = 'patterns'").fetchall():
        forbidden |= _forbidden_terms(conn, row["path"], [])
    cast = vector_type(settings.embedding_dim)[0].split("(")[0]
    changed = []
    for key, text in outcome_patterns(conn):
        if not is_anonymous(text, forbidden):
            continue
        source = f"{OUTCOME_SOURCE}: {key}"
        current = conn.execute(
            "SELECT id, content FROM memories WHERE brain_path = %s::ltree AND source = %s AND status = 'active'",
            (tree.ROOT, source),
        ).fetchone()
        if current and current["content"] == text:
            continue
        [embedding] = embedder.embed([text])
        new_id = conn.execute(
            f"""INSERT INTO memories (brain_path, memory_type, tier, sensitivity, content,
                                      embedding, embedding_model, source, maintained_at)
                VALUES (%s::ltree, 'semantic', 'long_term', %s, %s, %s::{cast}, %s, %s, now()) RETURNING id""",
            (tree.ROOT, int(Sensitivity.INTERNAL), text, vector_literal(embedding), embedder.name, source),
        ).fetchone()["id"]
        if current:
            conn.execute(
                """UPDATE memories SET status = 'superseded', superseded_by = %s,
                          status_reason = 'replaced by a lesson from more outcomes' WHERE id = %s""",
                (new_id, current["id"]),
            )
        changed.append(text)
    conn.commit()
    return changed


def learn(conn: psycopg.Connection, llm: LLM, embedder: Embedder, settings: Settings) -> dict:
    """Learn from every sharing company and store new patterns in the root brain."""
    conn.row_factory = dict_row
    conn.execute("INSERT INTO brains (path, name, kind) VALUES ('root', 'Root', 'root') ON CONFLICT DO NOTHING")
    companies = [row["path"] for row in conn.execute(
        "SELECT path::text FROM brains WHERE kind = 'company' AND sharing = 'patterns' ORDER BY path"
    ).fetchall()]
    cast = vector_type(settings.embedding_dim)[0].split("(")[0]
    stored, repeats, rejected = [], 0, 0
    for company in companies:
        candidates, dropped = learn_from_company(conn, llm, company)
        rejected += dropped
        if not candidates:
            continue
        embeddings = embedder.embed([p.text for p in candidates])
        for pattern, embedding in zip(candidates, embeddings):
            repeat = conn.execute(
                f"""SELECT 1 FROM memories WHERE brain_path = %s::ltree
                    AND embedding <=> %s::{cast} < %s LIMIT 1""",
                (tree.ROOT, vector_literal(embedding), REPEAT_MAX_DISTANCE),
            ).fetchone()
            if repeat:
                repeats += 1
                continue
            # Nothing here points back at the company the pattern came from.
            conn.execute(
                f"""INSERT INTO memories (brain_path, memory_type, tier, sensitivity, content,
                                          embedding, embedding_model, source, maintained_at)
                    VALUES (%s::ltree, 'semantic', 'long_term', %s, %s, %s::{cast}, %s, %s, now())""",
                (tree.ROOT, int(Sensitivity.INTERNAL), pattern.text, vector_literal(embedding),
                 embedder.name, SOURCE),
            )
            stored.append(pattern)
        conn.commit()
    outcome_lessons = learn_outcomes(conn, embedder, settings)
    return {"companies": len(companies), "stored": stored, "repeats": repeats, "rejected": rejected,
            "outcome_lessons": outcome_lessons}


def run(settings: Settings, llm: LLM, embedder: Embedder) -> dict:
    with psycopg.connect(settings.admin_database_url) as conn:
        return learn(conn, llm, embedder, settings)


def main() -> None:
    from secondbrain.models import build_models

    settings = get_settings()
    llm, embedder = build_models(settings)
    result = run(settings, llm, embedder)
    print(f"Learned from {result['companies']} sharing company(ies); "
          f"{len(result['stored'])} new pattern(s), {result['repeats']} repeat(s), "
          f"{result['rejected']} rejected by the anonymity check.")
    for p in result["stored"]:
        print(f"- [{p.topic}] {p.text}")
    for text in result["outcome_lessons"]:
        print(f"- [outcomes] {text}")


if __name__ == "__main__":
    main()
