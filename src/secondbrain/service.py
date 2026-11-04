"""The brain's operations: remember, consolidate, ask."""

import json
import re
from dataclasses import dataclass, field

from secondbrain import tree
from secondbrain.graph import build_graph
from secondbrain.config import Settings
from secondbrain.memory.short_term import ShortTermMemory
from secondbrain.memory.store import Hit, MemoryStore, User
from secondbrain.models import LLM, Embedder
from secondbrain.sensitivity import Sensitivity

MEMORY_TYPES = {"episodic", "semantic", "user"}

CLASSIFY_SYSTEM = (
    "Classify the memory into exactly one type: "
    '"episodic" (an event that happened), "semantic" (a fact or piece of knowledge), '
    'or "user" (information about a person). '
    'Reply with JSON only, like {"memory_type": "semantic"}.'
)

NOT_FOUND = "NOT_FOUND"
NO_ANSWER = "I don't have that information."

REWRITE_SYSTEM = (
    "Rewrite the follow-up question as a standalone question, using the conversation "
    "to fill in what it refers to. Reply with the rewritten question only."
)

ANSWER_SYSTEM = (
    "You are a company's Second Brain. Answer the question using only the numbered "
    "memories and recent context provided. Cite memory numbers like [1]. If they don't "
    f"contain the answer, reply with exactly {NOT_FOUND} and nothing else."
)


class UnknownUser(Exception):
    pass


@dataclass
class Answer:
    answer: str
    answered_from: str  # "department", "company", or "none"
    sources: list[Hit] = field(default_factory=list)
    searched_for: str = ""  # the question as searched, after resolving follow-ups


class SecondBrain:
    def __init__(
        self, settings: Settings, store: MemoryStore, short_term: ShortTermMemory,
        llm: LLM, embedder: Embedder,
    ) -> None:
        self.settings = settings
        self.store = store
        self.short_term = short_term
        self.llm = llm
        self.embedder = embedder

    def _user(self, user_id: str) -> User:
        user = self.store.get_user(user_id)
        if user is None:
            raise UnknownUser(user_id)
        return user

    def classify(self, content: str) -> str:
        reply = self.llm.complete(CLASSIFY_SYSTEM, content)
        # Models sometimes wrap JSON in prose or a code fence; take the object itself.
        match = re.search(r"\{.*\}", reply, re.DOTALL)
        try:
            memory_type = json.loads(match.group(0)).get("memory_type") if match else None
        except (json.JSONDecodeError, AttributeError):
            memory_type = None
        return memory_type if memory_type in MEMORY_TYPES else "semantic"

    def remember(self, user_id: str, content: str, sensitivity: Sensitivity) -> dict:
        """Capture new input into the user's short-term memory."""
        self._user(user_id)
        item = {
            "content": content,
            "memory_type": self.classify(content),
            "sensitivity": int(sensitivity),
        }
        self.short_term.push(user_id, item)
        return item

    def consolidate(self, user_id: str) -> int:
        """Move the user's short-term memory into their home brain's long-term memory.

        This is the simplest V4 rule (keep everything); importance scoring,
        merging and decay are later variants.
        """
        user = self._user(user_id)
        items = self.short_term.all(user_id)
        if not items:
            return 0
        embeddings = self.embedder.embed([item["content"] for item in items])
        with self.store.as_user(user_id) as conn:
            for item, embedding in zip(items, embeddings):
                self.store.insert(
                    conn,
                    brain_path=user.home_brain,
                    memory_type=item["memory_type"],
                    sensitivity=item["sensitivity"],
                    content=item["content"],
                    embedding=embedding,
                    embedding_model=self.embedder.name,
                    created_by=user_id,
                )
        # Only after the insert committed, so a failure loses nothing.
        self.short_term.drop_oldest(user_id, len(items))
        return len(items)

    def _standalone(self, question: str, history: list[dict]) -> str:
        """Turn a follow-up ("and for express?") into a question that can be searched on its own."""
        if not history:
            return question
        lines = ["Conversation:"]
        for turn in history[-3:]:
            lines += [f"User: {turn['question']}", f"Assistant: {turn['answer'][:400]}"]
        lines += ["", f"Follow-up question: {question}"]
        rewritten = self.llm.complete(REWRITE_SYSTEM, "\n".join(lines)).strip()
        return rewritten or question

    def _answer(
        self, question: str, hits: list[Hit], recent: list[dict], history: list[dict] | None = None,
    ) -> str | None:
        """The model's answer from these memories, or None if they don't contain it."""
        if not hits and not recent:
            return None
        lines = ["Memories:"]
        # Dates let the model prefer the newer of two memories that disagree.
        lines += [
            f"[{i}] ({hit.created_at:%Y-%m-%d}{', partly outdated by a newer memory' if hit.amended else ''}) {hit.content}"
            for i, hit in enumerate(hits, start=1)
        ]
        if recent:
            lines += ["", "Recent context:"] + [f"- {item['content']}" for item in recent]
        if history:
            lines += ["", "Conversation so far:"]
            for turn in history[-3:]:
                lines += [f"User: {turn['question']}", f"Assistant: {turn['answer'][:400]}"]
        lines += ["", f"Question: {question}"]
        reply = self.llm.complete(ANSWER_SYSTEM, "\n".join(lines)).strip()
        return None if reply.startswith(NOT_FOUND) else reply

    def ask(self, user_id: str, question: str) -> Answer:
        """Answer from the user's own brain, else from their company brain. Never the root.

        Falling back is decided by whether the model finds the answer in the
        memories, not by a distance threshold alone: loosely related memories
        are often close in embedding space without answering the question.
        """
        user = self._user(user_id)
        company = tree.company_of(user.home_brain)
        history = self.short_term.history(user_id)
        standalone = self._standalone(question, history)
        [embedding] = self.embedder.embed([standalone])
        recent = self.short_term.recent(user_id)
        if user.home_brain == company:
            # People placed at the company level (e.g. the CEO) search the whole
            # company; row-level security still limits it to what they may see.
            levels = [(company, True, "company")]
        else:
            levels = [(user.home_brain, False, "department"), (company, True, "company")]

        for path, include_children, answered_from in levels:
            # Search inside a short transaction; the model call happens outside it.
            with self.store.as_user(user_id) as conn:
                hits = self.store.search(
                    conn, embedding, brain_path=path, include_children=include_children,
                    limit=self.settings.search_limit,
                )
            hits = [h for h in hits if h.distance <= self.settings.fallback_max_distance]
            answer = self._answer(standalone, hits, recent, history)
            if answer is not None:
                self.short_term.add_turn(user_id, question, answer)
                return Answer(answer, answered_from, hits, standalone)
        self.short_term.add_turn(user_id, question, NO_ANSWER)
        return Answer(NO_ANSWER, "none", [], standalone)

    def new_conversation(self, user_id: str) -> None:
        self._user(user_id)
        self.short_term.clear_history(user_id)

    def graph(self, user_id: str) -> dict:
        """Everything this person can see in their company, as nodes and edges."""
        user = self._user(user_id)
        with self.store.as_user(user_id) as conn:
            return build_graph(conn, user, self.short_term.all(user_id))
