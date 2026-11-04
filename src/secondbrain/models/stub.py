"""Offline stand-ins for tests and development without the company proxy.

They are deterministic, not intelligent: the embedder hashes words, so only
questions sharing words with a memory will match.
"""

import hashlib
import json
import math
import re

STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "of", "for", "to", "in", "on",
    "and", "or", "what", "whats", "how", "many", "much", "our", "we", "do", "does",
    "can", "i", "my", "per", "at", "by", "with", "it", "this", "that",
}
EPISODIC_HINTS = {"meeting", "met", "yesterday", "today", "happened", "called", "visited"}
USER_HINTS = {"prefers", "likes", "role", "manager", "cfo", "ceo", "reports"}


def _stem(word: str) -> str:
    return word.rstrip("s") if len(word) > 3 else word


def _words(text: str) -> list[str]:
    words = re.findall(r"[a-z0-9]+", text.lower())
    return [_stem(w) for w in words if w not in STOPWORDS]


def _relation(prompt: str) -> str:
    """Crude offline judgement: near-identical wording is a duplicate; the same
    words with different numbers is an update."""
    match = re.search(r"OLD: (.*)\n\nNEW: (.*)", prompt, re.DOTALL)
    if not match:
        return "unrelated"
    old, new = (set(_words(part)) for part in match.groups())
    words = lambda ws: {w for w in ws if not w.isdigit()}
    numbers = lambda ws: {w for w in ws if w.isdigit()}
    overlap = len(words(old) & words(new)) / max(len(words(old) | words(new)), 1)
    if overlap > 0.8 and numbers(old) == numbers(new):
        return "duplicate"
    if overlap > 0.5 and numbers(old) != numbers(new):
        return "updates"
    return "unrelated"


def _entities(prompt: str) -> dict:
    """Crude offline extraction: capitalized words and runs of them are entities,
    the company's people (by full or first name) are persons, and the first
    entity is related to each of the others."""
    company = re.search(r"^Company: (.*)$", prompt, re.MULTILINE)
    people = re.search(r"^People: (.*)$", prompt, re.MULTILINE)
    company_words = set(company.group(1).split()) if company else set()
    names = [p for p in people.group(1).split("; ") if p] if people else []
    text = prompt.split("\nMemory:\n", 1)[-1]
    entities: list[dict] = []
    for run in re.findall(r"\b[A-Z][A-Za-z0-9]*(?:[ ]+[A-Z][A-Za-z0-9]*)*", text):
        if run.lower() in STOPWORDS or set(run.split()) <= company_words:
            continue
        person = next((p for p in names if run in (p, p.split()[0])), None)
        name = person or run
        if all(e["name"] != name for e in entities):
            entities.append({"name": name, "kind": "person" if person else "topic"})
    relations = [{"source": entities[0]["name"], "relation": "mentioned with", "target": e["name"]}
                 for e in entities[1:]]
    return {"entities": entities, "relations": relations}


class StubEmbedder:
    name = "stub-hash"

    def __init__(self, dim: int) -> None:
        self.dim = dim

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._embed_one(text) for text in texts]

    def _embed_one(self, text: str) -> list[float]:
        vector = [0.0] * self.dim
        for word in _words(text):
            digest = hashlib.sha256(word.encode()).digest()
            vector[int.from_bytes(digest[:4], "big") % self.dim] += 1.0
        norm = math.sqrt(sum(v * v for v in vector)) or 1.0
        return [v / norm for v in vector]


class StubLLM:
    name = "stub-rules"

    def complete(self, system: str, prompt: str) -> str:
        if "extract the entities" in system.lower():
            return json.dumps(_entities(prompt))
        if "classify" in system.lower():
            words = set(_words(prompt))
            if words & {_stem(w) for w in USER_HINTS}:
                memory_type = "user"
            elif words & {_stem(w) for w in EPISODIC_HINTS}:
                memory_type = "episodic"
            else:
                memory_type = "semantic"
            return json.dumps({"memory_type": memory_type})
        if "decision signals" in system.lower():
            # Offline: one signal citing the first note and lesson, one citing nothing (must be dropped).
            cites = [k for k in ("N1",) if "\nN1 " in prompt]
            lessons = [k for k in ("L1",) if "\nL1:" in prompt]
            return json.dumps({"signals": [
                {"title": "Follow up on the latest note", "action": "Review it this week", "reason": "stub",
                 "impact": "stub", "urgency": "medium", "confidence": 0.5, "notes": cites, "lessons": lessons},
                {"title": "Unsupported idea", "action": "Do something", "reason": "no evidence", "impact": "?",
                 "urgency": "high", "confidence": 0.9, "notes": [], "lessons": []},
            ]})
        if "reusable business lessons" in system.lower():
            # Offline: one clean lesson, plus two the anonymity check must reject.
            # The company's name is the first word of its documents' titles.
            titles = re.findall(r"^(\w+)[^\n]*›", prompt, re.MULTILINE)
            name = titles[0] if titles else "Acme"
            return json.dumps({"patterns": [
                {"pattern": "Escalating larger customer credits to a lead keeps goodwill spending consistent.", "topic": "support"},
                {"pattern": "Free shipping above 60 dollars lifts conversion.", "topic": "pricing"},
                {"pattern": f"Teams at {name} prefer short briefs.", "topic": "process"},
            ]})
        if "maintain a company's memory" in system.lower():
            return json.dumps({"relation": _relation(prompt), "reason": "stub judgement"})
        if "rewrite" in system.lower():
            # No real understanding offline: search with the follow-up as written.
            match = re.search(r"^Follow-up question: (.+)$", prompt, re.MULTILINE)
            return match.group(1) if match else prompt
        # Answering: echo the most relevant memory, which is listed first.
        match = re.search(
            r"^\[1\] (?:\(\d{4}-\d{2}-\d{2}[^)]*\) )?(.*?)\n*(?=^\[2\] |^Recent context:|^Conversation so far:|^Question:)",
            prompt, re.MULTILINE | re.DOTALL
        )
        return match.group(1).strip() if match else "NOT_FOUND"
