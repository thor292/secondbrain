import psycopg

from secondbrain import learning
from secondbrain.learning import is_anonymous
from tests.conftest import ADMIN_TOKEN, ask


def test_anonymity_check():
    forbidden = {"acme", "asha", "zenith"}
    assert is_anonymous("Escalating large credits to a lead keeps spending consistent.", forbidden)
    assert not is_anonymous("Free shipping above 60 dollars lifts conversion.", forbidden)
    assert not is_anonymous("Zenith buyers respond to early offers.", forbidden)
    assert not is_anonymous("Ask asha@acme.com for details.", forbidden)


def test_learning_sends_only_anonymous_patterns_from_sharing_companies(client, settings):
    seen_prompts = []
    brain = client.app.state.brain
    original = brain.llm.complete

    def spy(system, prompt):
        if "reusable business lessons" in system:
            seen_prompts.append(prompt)
        return original(system, prompt)

    brain.llm.complete = spy
    response = client.post("/admin/learn", headers={"X-Admin-Token": ADMIN_TOKEN}).json()

    # Only Acme shares patterns; Globex (sharing "none") is never read.
    assert response["companies"] == 1 and len(seen_prompts) == 1
    assert "Globex" not in seen_prompts[0]
    # Restricted memories never leave the company.
    assert "18 percent" not in seen_prompts[0]
    # The stub returns one clean lesson and two identifying ones.
    assert [p["pattern"] for p in response["stored"]] == [
        "Escalating larger customer credits to a lead keeps goodwill spending consistent."
    ]
    assert response["rejected"] == 2

    with psycopg.connect(settings.admin_database_url) as conn:
        rows = conn.execute("SELECT content, source, created_by FROM memories WHERE brain_path = 'root'").fetchall()
    learned = [r for r in rows if r[1] == learning.SOURCE]
    assert len(learned) == 1 and learned[0][2] is None  # no link back to a company or person

    # Running again doesn't store the same lesson twice.
    again = client.post("/admin/learn", headers={"X-Admin-Token": ADMIN_TOKEN}).json()
    assert again["stored"] == [] and again["repeats"] == 1

    # Company users still never get answers from the root.
    assert "Escalating" not in ask(client, "asha", "How should larger customer credits be handled?")["answer"]


def test_learning_needs_admin_token(client):
    assert client.post("/admin/learn").status_code == 401
