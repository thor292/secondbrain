"""Learning across the tree: anonymized patterns flow from companies to the root.

Only companies whose sharing setting is "patterns" take part, and only their
public and internal memories are read; restricted memories never leave the
company. The model turns a company's memories into general business lessons,
then a strict rule-based check rejects any lesson that still contains a
number, a person's name, the company's name, or another identifying term.
What reaches the root records no link back to the company it came from.

    uv run python -m secondbrain.learning
"""

import json
import re
from dataclasses import dataclass

import psycopg
from psycopg.rows import dict_row

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
    for (name,) in conn.execute("SELECT name FROM brains WHERE path <@ %s::ltree", (company,)).fetchall():
        terms |= {w.lower() for w in re.findall(r"[A-Za-z][A-Za-z'&.-]+", name) if len(w) > 2}
    terms |= set(company.split(".")[1:])
    for (name,) in conn.execute("SELECT name FROM users WHERE home_brain <@ %s::ltree", (company,)).fetchall():
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
    return {"companies": len(companies), "stored": stored, "repeats": repeats, "rejected": rejected}


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


if __name__ == "__main__":
    main()
