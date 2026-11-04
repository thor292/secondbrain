"""Add realistic notes from Acme people through the normal remember -> consolidate flow.

Gives the demo and graph view episodic and person memories, not only imported
documents. Run after scripts/load_examples.sh.
"""

from secondbrain.config import get_settings
from secondbrain.memory.short_term import ShortTermMemory
from secondbrain.memory.store import MemoryStore
from secondbrain.models import build_models
from secondbrain.sensitivity import Sensitivity
from secondbrain.service import SecondBrain

NOTES = {
    "asha": [
        ("Met the Zenith Foods buyer yesterday, they want a 12 percent discount on the annual contract", "internal"),
        ("Zenith Foods buyer prefers calls over email and decides by end of month", "internal"),
        ("Called Bluebird Stores today, they are interested in the Q4 bundle", "internal"),
    ],
    "vikram": [
        ("Approved the 12 percent discount for Zenith Foods because the contract is above 500k", "internal"),
        ("Asha is the account owner for Zenith Foods and Bluebird Stores", "internal"),
        ("Q4 pipeline review meeting happened today, 1.4 million dollars committed so far", "restricted"),
    ],
    "meera": [
        ("Month-end close for October happened on the 3rd working day as planned", "internal"),
        ("Ravi prefers the expense summary as a one page PDF every Friday", "internal"),
    ],
    "ravi": [
        ("Board meeting yesterday asked for a plan to recover the profit margin to 21 percent", "restricted"),
    ],
    "sara": [
        ("Interviewed two candidates for the sales role today, one moves to the final round", "internal"),
        ("Vikram is the hiring manager for the new sales role", "internal"),
    ],
}


def main() -> None:
    settings = get_settings()
    llm, embedder = build_models(settings)
    store = MemoryStore(settings.app_database_url, settings.embedding_dim)
    brain = SecondBrain(settings, store, ShortTermMemory(settings.redis_url, settings.short_term_ttl), llm, embedder)
    try:
        for user, notes in NOTES.items():
            for content, level in notes:
                brain.remember(user, content, Sensitivity.parse(level))
            print(f"{user}: saved {brain.consolidate(user)} notes")
    finally:
        store.close()


if __name__ == "__main__":
    main()
