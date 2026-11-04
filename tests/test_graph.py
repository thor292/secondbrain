from tests.conftest import ADMIN_TOKEN


def graph(client, user):
    response = client.get("/graph", headers={"X-User-Id": user})
    assert response.status_code == 200, response.text
    return response.json()


def node_ids(g, prefix):
    return {n["id"] for n in g["nodes"] if n["id"].startswith(prefix)}


def test_graph_shows_only_own_company(client):
    g = graph(client, "asha")
    assert g["scope"] == "root.acme"
    brains = node_ids(g, "brain:")
    assert brains == {"brain:root.acme", "brain:root.acme.sales", "brain:root.acme.finance", "brain:root.acme.hr"}
    assert all("globex" not in n["id"] and n["id"] != "brain:root" for n in g["nodes"])
    assert "user:lee" not in node_ids(g, "user:")


def test_graph_memories_follow_permissions(client):
    asha = graph(client, "asha")
    asha_brains = {n["props"]["brain"] for n in asha["nodes"] if n["id"].startswith("memory:")}
    assert asha_brains == {"root.acme", "root.acme.sales"}

    ravi = graph(client, "ravi")
    restricted = [n for n in ravi["nodes"] if n.get("props", {}).get("sensitivity") == "restricted"]
    assert restricted and all(n["props"]["brain"] == "root.acme.finance" for n in restricted)
    assert not any(n.get("props", {}).get("sensitivity") == "restricted" for n in asha["nodes"])


def test_graph_shows_only_viewers_own_grants(client):
    g = graph(client, "vikram")
    access = [e for e in g["edges"] if e["label"] == "CAN_ACCESS"]
    assert access and all(e["source"] == "user:vikram" for e in access)
    assert {e["target"] for e in access} == {"brain:root.acme", "brain:root.acme.sales", "brain:root.acme.finance"}


def test_graph_edges_point_at_existing_nodes(client):
    headers = {"X-User-Id": "asha"}
    client.post("/remember", json={"content": "Met the vendor yesterday"}, headers=headers)
    client.post("/consolidate", headers=headers)
    client.post("/remember", json={"content": "Preparing the Q4 deck"}, headers=headers)
    g = graph(client, "asha")
    ids = {n["id"] for n in g["nodes"]}
    assert all(e["source"] in ids and e["target"] in ids for e in g["edges"])
    labels = {e["label"] for e in g["edges"]}
    assert {"PARENT_OF", "MEMBER_OF", "CAN_ACCESS", "STORED_IN", "CREATED", "WORKING_ON"} <= labels


def test_graph_screen_is_served(client):
    assert "SecondBrain Graph" in client.get("/graph-view").text


def test_full_graph_needs_admin_token(client):
    assert client.get("/admin/graph").status_code == 401
    assert client.get("/admin/graph", headers={"X-Admin-Token": "wrong"}).status_code == 401


def test_full_graph_shows_whole_tree(client):
    g = client.get("/admin/graph", headers={"X-Admin-Token": ADMIN_TOKEN}).json()
    brains = node_ids(g, "brain:")
    assert {"brain:root", "brain:root.acme", "brain:root.acme.finance", "brain:root.globex"} <= brains
    assert {"user:asha", "user:ravi", "user:lee"} <= node_ids(g, "user:")
    sensitivities = {n["props"].get("sensitivity") for n in g["nodes"] if n["id"].startswith("memory:")}
    assert "restricted" in sensitivities
    assert {"source": "brain:root", "target": "brain:root.acme", "label": "PARENT_OF"} in g["edges"]
    ids = {n["id"] for n in g["nodes"]}
    assert all(e["source"] in ids and e["target"] in ids for e in g["edges"])
    brain_of = {n["id"]: n["props"]["brain"] for n in g["nodes"] if n["id"].startswith("memory:")}
    for e in g["edges"]:
        if e["label"] == "RELATED_TO":
            a, b = brain_of[e["source"]], brain_of[e["target"]]
            assert a.split(".")[:2] == b.split(".")[:2], "related memories must stay within one company"


def test_explore_screen_is_served(client):
    assert "SecondBrain Explore" in client.get("/explore").text
