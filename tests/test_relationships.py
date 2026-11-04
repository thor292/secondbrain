"""Entities, the FalkorDB relationship graph, and permission-safe graph queries.

The stub model treats capitalized words as entities, so in the fixtures
"December", "Q3" and "Board" appear only in Acme's restricted finance memory,
and "Friday" only in Finance's internal notes.
"""

import psycopg
import pytest

from secondbrain import relationships
from secondbrain.relationships import Scope


def remember(client, user, content, sensitivity="internal"):
    headers = {"X-User-Id": user}
    assert client.post("/remember", json={"content": content, "sensitivity": sensitivity}, headers=headers).status_code == 200
    assert client.post("/consolidate", headers=headers).status_code == 200


def entity(client, user, name):
    return client.get(f"/entities/{name}", headers={"X-User-Id": user})


def experts(client, user, topic):
    response = client.get("/experts", params={"topic": topic}, headers={"X-User-Id": user})
    assert response.status_code == 200, response.text
    return response.json()


@pytest.fixture()
def graph_client(client, settings):
    """The fixture companies with entities extracted and graphs built, plus a few notes."""
    brain = client.app.state.brain
    relationships.run(settings, brain.llm, brain.relationships)
    # Consolidating runs extraction and the incremental graph update in the background.
    remember(client, "asha", "Bluebird wants a bigger discount before the holidays")
    remember(client, "ravi", "Sara is leaving the company in March", sensitivity="restricted")
    return client


def test_extraction_is_stored_in_postgres(graph_client, settings):
    with psycopg.connect(settings.admin_database_url) as conn:
        pending = conn.execute(
            """SELECT count(*) FROM memories m WHERE nlevel(brain_path) >= 2
               AND NOT EXISTS (SELECT 1 FROM entity_extractions x WHERE x.memory_id = m.id)"""
        ).fetchone()[0]
        assert pending == 0
        rows = dict(conn.execute(
            "SELECT key, company::text FROM entities WHERE key IN ('december', 'bluebird', 'annual')"
        ).fetchall())
        assert rows["december"] == "root.acme" and rows["bluebird"] == "root.acme"
        # Same name in two companies: two separate entities.
        assert conn.execute("SELECT count(*) FROM entities WHERE key = 'annual'").fetchone()[0] == 2
        sara = conn.execute("SELECT kind, user_id FROM entities WHERE key = 'sara'").fetchone()
        assert sara == ("person", "sara")
        # The root brain's own knowledge is not company knowledge and isn't extracted.
        assert conn.execute(
            "SELECT count(*) FROM entity_extractions x JOIN memories m ON m.id = x.memory_id WHERE m.brain_path = 'root'"
        ).fetchone()[0] == 0


def test_entity_only_in_restricted_memory_is_hidden(graph_client):
    for user in ("asha", "meera", "vikram"):
        assert entity(graph_client, user, "December").status_code == 404
    result = entity(graph_client, "ravi", "December").json()
    assert result["kind"] == "topic"
    assert any("18 percent" in m["content"] for m in result["memories"])


def test_hidden_and_unknown_entities_look_the_same(graph_client):
    hidden, unknown = entity(graph_client, "asha", "Q3"), entity(graph_client, "asha", "Zebra")
    assert hidden.status_code == unknown.status_code == 404
    assert hidden.json()["detail"].replace("Q3", "X") == unknown.json()["detail"].replace("Zebra", "X")


def test_relations_only_from_visible_memories(graph_client):
    # "Finance" is in Finance's internal notes (Vikram has a grant) and in its restricted results.
    vikram = entity(graph_client, "vikram", "Finance").json()
    seen = {r["entity"]["name"] for r in vikram["relations"]} | {r["name"] for r in vikram["related"]}
    assert "Friday" in seen
    assert not seen & {"Q3", "Board", "December"}
    assert all(m["brain"] == "root.acme.finance" and "18 percent" not in m["content"] for m in vikram["memories"])

    ravi = entity(graph_client, "ravi", "Finance").json()
    assert {"Q3", "Board", "December"} <= {r["entity"]["name"] for r in ravi["relations"]}


def test_experts_follow_permissions(graph_client):
    hidden = experts(graph_client, "asha", "december")
    assert hidden["entities"] == [] and hidden["people"] == [] and hidden["memory_count"] == 0

    ravi = experts(graph_client, "ravi", "december")
    assert ravi["entities"] == [{"name": "December", "kind": "topic"}]
    assert [b["path"] for b in ravi["brains"]] == ["root.acme.finance"]


def test_experts_finds_the_people_who_wrote_about_a_topic(graph_client):
    result = experts(graph_client, "vikram", "bluebird")
    assert [(p["id"], p["wrote"]) for p in result["people"]] == [("asha", 1)]
    assert result["brains"] == [{"path": "root.acme.sales", "name": "Sales", "memories": 1}]
    assert "Bluebird" in result["memories"][0]["content"]
    # Finance can't read Sales, so for Meera nobody knows about Bluebird.
    assert experts(graph_client, "meera", "bluebird")["people"] == []


def test_people_from_restricted_memories_do_not_leak(graph_client):
    # Ravi's restricted note mentions Sara; only people who can read it learn anything from it.
    assert experts(graph_client, "asha", "march")["people"] == []
    assert entity(graph_client, "asha", "Sara").status_code == 404
    assert entity(graph_client, "sara", "March").status_code == 404
    ravi = entity(graph_client, "ravi", "Sara").json()
    assert ravi["person_id"] == "sara"
    assert {p["id"] for p in ravi["people"]} == {"ravi", "sara"}
    assert {p["id"] for p in experts(graph_client, "ravi", "march")["people"]} == {"ravi", "sara"}


def test_no_queries_across_companies(graph_client):
    assert entity(graph_client, "lee", "December").status_code == 404
    assert entity(graph_client, "lee", "Bluebird").status_code == 404
    leave = experts(graph_client, "lee", "annual")
    assert leave["brains"] == [{"path": "root.globex", "name": "Globex Corp", "memories": 1}]
    assert all("Acme" not in m["content"] for m in leave["memories"])
    # Each company has its own graph, and Acme's holds no Globex memories.
    graph = graph_client.app.state.brain.relationships
    assert graph.name("root.acme") != graph.name("root.globex")
    acme = graph._db.select_graph(graph.name("root.acme"))
    brains = {r[0] for r in acme.ro_query("MATCH (b:Brain) RETURN b.path").result_set}
    assert brains == {"root.acme", "root.acme.sales", "root.acme.finance", "root.acme.hr"}


def test_graph_holds_no_memory_text(graph_client):
    graph = graph_client.app.state.brain.relationships
    acme = graph._db.select_graph(graph.name("root.acme"))
    keys = {k for [props] in acme.ro_query("MATCH (m:Memory) RETURN properties(m)").result_set for k in props}
    assert keys == {"id", "type"}


def test_database_filters_entities_too(graph_client, settings):
    with psycopg.connect(settings.app_database_url) as conn:
        for table in ("entities", "memory_entities", "entity_relations"):
            assert conn.execute(f"SELECT count(*) FROM {table}").fetchone()[0] == 0
        conn.execute("SELECT set_config('app.user_id', 'asha', true)")
        keys = {r[0] for r in conn.execute("SELECT key FROM entities")}
        assert "bluebird" in keys and not keys & {"december", "q3", "sara", "march"}
        assert all(not k.startswith("globex") for k in keys)
        conn.execute("SELECT set_config('app.user_id', 'ravi', true)")
        assert {"december", "sara"} <= {r[0] for r in conn.execute("SELECT key FROM entities")}


def test_queries_must_be_limited_to_visible_memories(graph_client):
    graph = graph_client.app.state.brain.relationships
    with pytest.raises(ValueError):
        graph._query(Scope("root.acme", ()), "MATCH (e:Entity) RETURN e.name")


def test_missing_graph_is_rebuilt_with_the_same_answers(graph_client):
    before = experts(graph_client, "vikram", "bluebird"), entity(graph_client, "ravi", "Finance").json()
    graph_client.app.state.brain.relationships.drop_all()
    after = experts(graph_client, "vikram", "bluebird"), entity(graph_client, "ravi", "Finance").json()
    normalize = lambda e: {**e, "relations": sorted(e["relations"], key=str), "related": sorted(e["related"], key=str)}
    assert before[0] == after[0]
    assert normalize(before[1]) == normalize(after[1])


def test_unknown_user_is_rejected(graph_client):
    assert graph_client.get("/experts", params={"topic": "x"}, headers={"X-User-Id": "nobody"}).status_code == 401


@pytest.mark.parametrize("topic,key,expected", [
    ("lamps", "halo floor lamp", True),
    ("shipping", "free shipping threshold", True),
    ("halo lamp", "halo floor lamp", True),
    ("sofa", "halo floor lamp", False),
    ("the", "the thing", False),
])
def test_topic_matching(topic, key, expected):
    assert relationships.matches(topic, key) is expected


def test_parse_drops_malformed_output():
    result = relationships.parse(
        'Sure! {"entities": [{"name": "Nordlys", "kind": "supplier"}, {"name": "nordlys"}, {"name": "Halo floor lamp", '
        '"kind": "gadget"}, "junk"], "relations": [{"source": "Nordlys", "relation": "Supplies", "target": "Halo floor lamp"}, '
        '{"source": "Nordlys", "relation": "knows", "target": "Somebody else"}]}'
    )
    assert result.entities == [("Nordlys", "supplier"), ("Halo floor lamp", "topic")]
    assert result.relations == [("nordlys", "supplies", "halo floor lamp")]
