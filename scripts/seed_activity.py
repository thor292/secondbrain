"""Add realistic notes from Lumen Home's people through the normal remember -> consolidate flow.

Gives the demo episodic and person memories, not only imported documents.
A few notes deliberately update facts from the documents (free shipping
threshold, early access date) or repeat earlier notes, so memory maintenance
has something to resolve. Run after scripts/load_examples.py.
"""

from secondbrain import learning, maintenance
from secondbrain.config import get_settings
from secondbrain.memory.short_term import ShortTermMemory
from secondbrain.memory.store import MemoryStore
from secondbrain.models import build_models
from secondbrain.sensitivity import Sensitivity
from secondbrain.service import SecondBrain

NOTES = {
    "daniel": [
        ("Storm in the Midwest: UPS delayed ground shipments to Chicago and Detroit by 2 days this week", "internal"),
        ("From November 1 the free shipping threshold drops from 75 dollars to 60 dollars for the holiday season", "internal"),
        ("Reno warehouse adds a second shift from November 15 to December 20 to handle holiday volume", "internal"),
    ],
    "ivy": [
        ("Received 40 damaged Halo floor lamps from Nordlys today, quarantined in aisle Q and reported to Merchandising", "internal"),
        ("Daniel prefers incident updates in the #ops-alerts Slack channel, not email", "internal"),
    ],
    "nina": [
        ("Customer Priya Shah reported a second damaged Aria lamp today; sent a replacement and a 25 dollar goodwill credit", "internal"),
        ("Priya Shah is a VIP customer and prefers order updates by SMS", "internal"),
        ("Priya Shah wants updates by text message rather than email", "internal"),
    ],
    "arjun": [
        ("Weekly support review today: where-is-my-order tickets up 35 percent after the UPS delays in the Midwest", "internal"),
        ("Approved a 120 dollar credit for a customer whose Aria sofa delivery was missed twice by Hearth Freight", "internal"),
    ],
    "leo": [
        ("Creator partnership with @homebyjules confirmed for Black Friday: 3 Instagram posts and 1 reel", "internal"),
        ("Loyalty early access for Black Friday moves from November 24 to November 23 to get ahead of competitors", "internal"),
    ],
    "sofia": [
        ("Leo prefers campaign briefs as one page Notion documents with the offer in the first line", "internal"),
        ("Gift guide email draft for December 7 is ready for review, featuring lighting under 100 dollars", "internal"),
    ],
    "omar": [
        ("Casa Forma confirmed the Aria linen sofa is back in stock on December 2; lead time shortened to 6 weeks", "internal"),
        ("Nordlys will credit us for the 40 damaged Halo lamps on the next invoice", "internal"),
    ],
    "tom": [
        ("October month-end close finished on the 4th business day as planned", "internal"),
    ],
    "rachel": [
        ("Board meeting today: keep Black Friday discount depth at 20 percent to protect the 45 percent gross margin floor", "restricted"),
    ],
    "hana": [
        ("Two open roles approved: Support agent (2 hires before December) and Performance marketer", "internal"),
        ("Arjun is the hiring manager for the two Support agent roles", "internal"),
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
    changes = maintenance.run(settings, llm)
    for c in changes:
        verb = {"duplicate": "duplicates", "updates": "replaces", "partial": "partly updates"}[c.relation]
        print(f"maintenance: {c.new[:60]!r} {verb} {c.old.splitlines()[0][:50]!r}")
    result = learning.run(settings, llm, embedder)
    print(f"learning: {len(result['stored'])} pattern(s) sent to the root, "
          f"{result['rejected']} rejected by the anonymity check")
    for p in result["stored"]:
        print(f"  - {p.text}")


if __name__ == "__main__":
    main()
