import psycopg
import pytest

from tests.conftest import ADMIN_TOKEN, ask


def test_department_answers_its_own_question(client):
    result = ask(client, "asha", "What is the Q4 sales target?")
    assert result["answered_from"] == "department"
    assert "2 million" in result["answer"]


def test_falls_back_to_company_brain(client):
    result = ask(client, "asha", "How many days of annual leave do employees get?")
    assert result["answered_from"] == "company"
    assert "24 days" in result["answer"]
    assert all(s["brain"].startswith("root.acme") for s in result["sources"])


def test_never_reaches_other_companies_or_root(client):
    result = ask(client, "asha", "How many days of annual leave do employees get?")
    text = result["answer"] + " ".join(s["content"] for s in result["sources"])
    assert "Globex" not in text
    assert "Root pattern" not in text
    globex = ask(client, "lee", "How many days of annual leave do employees get?")
    assert "Globex employees get 30 days" in globex["answer"]
    assert "Acme" not in globex["answer"]


def test_sibling_department_is_hidden_without_a_grant(client):
    result = ask(client, "asha", "When is the month-end close done?")
    assert all(not s["brain"].endswith("finance") for s in result["sources"])


def test_explicit_grant_opens_sibling_internal_but_not_restricted(client):
    close = ask(client, "vikram", "When is the month-end close done?")
    assert any(s["brain"] == "root.acme.finance" for s in close["sources"])
    margin = ask(client, "vikram", "What was the Q3 profit margin?")
    assert "18 percent" not in margin["answer"]


def test_restricted_memory_only_for_permitted_people(client):
    assert "18 percent" not in ask(client, "meera", "What was the Q3 profit margin?")["answer"]
    assert "18 percent" in ask(client, "ravi", "What was the Q3 profit margin?")["answer"]


def test_database_enforces_permissions_directly(client, settings):
    with psycopg.connect(settings.app_database_url) as conn:
        assert conn.execute("SELECT count(*) FROM memories").fetchone()[0] == 0
        conn.execute("SELECT set_config('app.user_id', 'asha', true)")
        paths = {r[0] for r in conn.execute("SELECT DISTINCT brain_path::text FROM memories")}
        assert paths == {"root.acme", "root.acme.sales"}
        with pytest.raises(psycopg.errors.InsufficientPrivilege):
            conn.execute("SELECT * FROM grants")


def test_cannot_write_into_another_department(client, settings):
    with psycopg.connect(settings.app_database_url) as conn:
        conn.execute("SELECT set_config('app.user_id', 'asha', true)")
        with pytest.raises(psycopg.errors.InsufficientPrivilege):
            conn.execute(
                "INSERT INTO memories (brain_path, memory_type, tier, sensitivity, content,"
                " embedding, embedding_model, created_by) VALUES ('root.acme.finance', 'semantic',"
                " 'long_term', 1, 'x', array_fill(0.1, ARRAY[256])::vector, 'stub', 'asha')"
            )


def test_remember_then_consolidate(client):
    headers = {"X-User-Id": "asha"}
    item = client.post("/remember", json={"content": "Met the vendor yesterday, price agreed at 40 dollars"},
                       headers=headers).json()
    assert item["memory_type"] == "episodic"
    assert client.post("/consolidate", headers=headers).json() == {"consolidated": 1}
    assert client.post("/consolidate", headers=headers).json() == {"consolidated": 0}
    result = ask(client, "asha", "What price was agreed with the vendor?")
    assert result["answered_from"] == "department"
    assert "40 dollars" in result["answer"]


def test_unknown_user_is_rejected(client):
    response = client.post("/ask", json={"question": "hi"}, headers={"X-User-Id": "nobody"})
    assert response.status_code == 401


def test_admin_onboards_and_uploads_documents(client):
    headers = {"X-Admin-Token": ADMIN_TOKEN}
    spec = {"id": "initech", "name": "Initech", "departments": [{"id": "it", "name": "IT"}],
            "users": [{"id": "peter", "name": "Peter", "department": "it"}]}
    assert client.post("/admin/companies", json=spec).status_code == 401
    assert client.post("/admin/companies", json=spec, headers=headers).json() == {"company": "root.initech"}
    upload = client.post(
        "/admin/brains/root.initech.it/documents", headers=headers,
        files=[("files", ("it.md", b"The wifi password rotates every Monday.", "text/markdown"))],
        data={"sensitivity": "internal"},
    )
    assert upload.json() == {"brain": "root.initech.it", "chunks": 1}
    assert "Monday" in ask(client, "peter", "When does the wifi password rotate?")["answer"]


def test_chat_screen_and_demo_users(client):
    assert "SecondBrain" in client.get("/").text
    users = {u["id"] for u in client.get("/demo/users").json()}
    assert {"asha", "ravi", "lee"} <= users


def test_conversation_history_is_kept_and_can_be_reset(client):
    headers = {"X-User-Id": "asha"}
    ask(client, "asha", "What is the Q4 sales target?")
    ask(client, "asha", "And who approves bigger discounts?")
    brain = client.app.state.brain
    history = brain.short_term.history("asha")
    assert [t["question"] for t in history] == ["What is the Q4 sales target?", "And who approves bigger discounts?"]
    assert "2 million" in history[0]["answer"]
    assert brain.short_term.history("vikram") == []  # each person has their own conversation
    assert client.post("/conversation/reset", headers=headers).json() == {"ok": True}
    assert brain.short_term.history("asha") == []


def test_maintenance_retires_outdated_facts_and_merges_duplicates(client, settings):
    from secondbrain import maintenance

    def note(user, text):
        client.post("/remember", json={"content": text}, headers={"X-User-Id": user})
        client.post("/consolidate", headers={"X-User-Id": user})

    note("asha", "The demo booth budget for the trade fair is 4000 dollars")
    note("asha", "The demo booth budget for the trade fair is 6500 dollars")
    note("asha", "Bluebird buyer prefers phone calls in the morning")
    note("asha", "Bluebird buyer prefers phone calls in the morning")
    brain = client.app.state.brain
    maintenance.run(settings, brain.llm)  # also runs in the background after /consolidate

    import psycopg
    with psycopg.connect(settings.admin_database_url) as conn:
        rows = dict(conn.execute(
            "SELECT content, status FROM memories WHERE created_by = 'asha' ORDER BY created_at"
        ).fetchall())
        statuses = conn.execute(
            "SELECT status FROM memories WHERE content = 'Bluebird buyer prefers phone calls in the morning' ORDER BY created_at"
        ).fetchall()
    assert rows["The demo booth budget for the trade fair is 4000 dollars"] == "superseded"
    assert rows["The demo booth budget for the trade fair is 6500 dollars"] == "active"
    assert [s for (s,) in statuses] == ["active", "merged"]

    answer = ask(client, "asha", "What is the demo booth budget for the trade fair?")
    assert "6500" in answer["answer"]
    assert all("4000" not in s["content"] for s in answer["sources"])


def test_replaced_fact_stays_visible_to_people_who_cannot_see_the_update(client, settings):
    """A company fact updated in a department note must not vanish for other departments."""
    from secondbrain import admin, maintenance
    from secondbrain.sensitivity import Sensitivity
    import psycopg

    brain = client.app.state.brain
    with psycopg.connect(settings.admin_database_url) as conn:
        admin.import_documents(conn, brain.embedder, settings, "root.acme",
                               [("parking.md", "Office parking costs 12 dollars per day")], Sensitivity.INTERNAL)
    client.post("/remember", json={"content": "Office parking costs 15 dollars per day"}, headers={"X-User-Id": "meera"})
    client.post("/consolidate", headers={"X-User-Id": "meera"})  # stored in the Finance brain
    maintenance.run(settings, brain.llm)

    assert "15 dollars" in ask(client, "meera", "How much does office parking cost per day?")["answer"]
    assert "12 dollars" in ask(client, "asha", "How much does office parking cost per day?")["answer"]
