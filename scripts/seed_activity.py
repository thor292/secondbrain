"""Add realistic notes from Desi Living's people through the normal remember -> consolidate flow.

Gives the demo episodic and person memories, not only imported documents.
A few notes deliberately update facts from the documents (free shipping
threshold, Ghar Club early access date, sofa restock) or repeat earlier notes
(a VIP customer's contact preference, said twice in different words), so
memory maintenance has something to resolve. Run after scripts/load_examples.py.
"""

from secondbrain import learning, maintenance
from secondbrain.config import get_settings
from secondbrain.memory.short_term import ShortTermMemory
from secondbrain.memory.store import MemoryStore
from secondbrain.models import build_models
from secondbrain.sensitivity import Sensitivity
from secondbrain.service import SecondBrain

NOTES = {
    "vikram": [
        ("Festival rush: Delhivery delayed parcels to Delhi NCR and Kolkata pincodes by 2 days this week; Blue Dart express is on time", "internal"),
        ("From Dhanteras (6 November) to 15 November the free shipping threshold drops from ₹999 to ₹699 for Diwali and the start of the wedding season", "internal"),
        ("Hoskote warehouse keeps the second shift and Sunday working until 15 November to clear Diwali and wedding season volume", "internal"),
    ],
    "priya": [
        ("Received 36 damaged Moradabad brass floor lamps from Noor Metalcraft today with dented bases, quarantined in bay Q at Hoskote and reported to Kabir in Merchandising", "internal"),
        ("Vikram prefers incident updates in the #ops-alerts Slack channel, not email", "internal"),
    ],
    "neha": [
        ("Customer Ananya Reddy from Bengaluru reported a second damaged brass floor lamp today; sent a replacement and a ₹500 goodwill credit", "internal"),
        ("Ananya Reddy is a VIP customer and prefers order updates on WhatsApp", "internal"),
        ("Ananya Reddy wants updates as WhatsApp messages rather than phone calls or email", "internal"),
    ],
    "arjun": [
        ("Weekly support review today: where-is-my-order tickets up 40 percent after the Delhivery festival delays in Delhi NCR", "internal"),
        ("Approved a ₹2,500 credit for a customer in Pune whose sheesham sofa delivery slot was missed twice by BharatFleet", "internal"),
    ],
    "rohan": [
        ("Creator partnership with @ghar.by.tara confirmed for Diwali: 3 Instagram posts and 1 reel featuring the brass diya & sandalwood candle set", "internal"),
        ("Second Diwali creator, @chai.and.decor, posts a Hindi reel on the kantha throw and brass floor lamp on Dhanteras", "internal"),
        ("Ghar Club early access to the Wedding Edit moves from 14 November to 12 November to get ahead of competing marketplace sales", "internal"),
    ],
    "sana": [
        ("Rohan prefers campaign briefs as one page Notion documents with the offer in the first line", "internal"),
        ("Wedding gifting email draft for 16 November is ready for review, featuring brass and textiles under ₹5,000", "internal"),
    ],
    "kabir": [
        ("Saharanpur Woodcraft confirmed the sheesham sofa is back in stock on 30 November; lead time shortened to 5 weeks", "internal"),
        ("Noor Metalcraft will issue a credit note for the 36 damaged brass floor lamps on the next invoice and will add corner protectors to the cartons", "internal"),
    ],
    "tanvi": [
        ("October month-end close finished on the 5th business day as planned, despite the festive volume", "internal"),
    ],
    "anand": [
        ("Board meeting today: keep the Diwali discount depth at 20 percent to protect the 45 percent gross margin", "restricted"),
    ],
    "farah": [
        ("Two open roles approved: Support agent speaking Hindi and Kannada (2 hires before the wedding season) and Performance marketer", "internal"),
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
