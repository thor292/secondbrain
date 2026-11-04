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
        if "classify" in system.lower():
            words = set(_words(prompt))
            if words & {_stem(w) for w in USER_HINTS}:
                memory_type = "user"
            elif words & {_stem(w) for w in EPISODIC_HINTS}:
                memory_type = "episodic"
            else:
                memory_type = "semantic"
            return json.dumps({"memory_type": memory_type})
        # Answering: echo the most relevant memory, which is listed first.
        match = re.search(
            r"^\[1\] (.*?)\n*(?=^\[2\] |^Recent context:|^Question:)", prompt, re.MULTILINE | re.DOTALL
        )
        return match.group(1).strip() if match else "NOT_FOUND"
