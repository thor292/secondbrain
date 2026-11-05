"""Live notifications (from the SecondBrain MQTT kit). Unit tests run anywhere; the broker
test needs the live broker up:

    SB_LIVE_TEST_BROKER=1 uv run pytest tests/test_live.py
(it regenerates docker/live-broker/passwd + acl with test users and reloads the broker;
run `uv run python -m secondbrain.live sync` afterwards to restore the demo's logins)
"""
import base64
import hashlib
import json
import os
import threading
import time

import pytest

from secondbrain import live
from secondbrain.live.sync import down_topic, sync_password, up_topic

S = live.MqttSettings(secret="test-secret", publisher_password="pub-pass", port=51884)


def test_topics_mirror_the_tree():
    assert live.memory_topic("root.acme.sales", 1) == "sb/root/acme/sales/memories/internal"
    assert live.memory_topic("root.acme", 1, "signals") == "sb/root/acme/signals/internal"
    assert live.inbox_topic("riya") == "sb/users/riya/inbox"


def test_grant_allows_its_level_and_below_only():
    t = live.topics_for_grants([{"brain_path": "root.acme.sales", "include_children": False, "max_sensitivity": 1}])
    assert t == ["sb/root/acme/sales/memories/internal", "sb/root/acme/sales/memories/public",
                 "sb/root/acme/sales/signals/internal", "sb/root/acme/sales/signals/public"]


def test_include_children_adds_wildcards_down_to_max_depth():
    t = live.topics_for_grants([{"brain_path": "root.acme", "include_children": True, "max_sensitivity": 0}])
    assert t == ["sb/root/acme/+/memories/public", "sb/root/acme/+/signals/public",
                 "sb/root/acme/memories/public", "sb/root/acme/signals/public"]


def test_passwords_are_stable_and_distinct():
    assert live.user_password(S, "riya") == live.user_password(S, "riya")
    assert live.user_password(S, "riya") != live.user_password(S, "arjun")
    assert live.monitor_password(S) != sync_password(S, "globex")
    other = live.MqttSettings(secret="another-secret", publisher_password="x")
    assert live.user_password(S, "riya") != live.user_password(other, "riya")


def test_mosquitto_hash_format_verifies():
    _, kind, iters, salt, digest = live.mosquitto_hash("pw").split("$")
    assert kind == "7" and iters == "101"
    assert base64.b64decode(digest) == hashlib.pbkdf2_hmac("sha512", b"pw", base64.b64decode(salt), 101)


def test_payload_never_contains_content():
    p = live.public_payload({"event": "created", "memory_id": "m1", "brain_path": "root.acme.sales",
                             "sensitivity": 2, "user_id": "riya", "time": "t", "content": "SECRET",
                             "detail": {"source": "chat", "text": "SECRET"}})
    assert "SECRET" not in json.dumps(p) and p["sensitivity"] == "restricted"
    s = live.public_payload({"event": "opened", "signal_id": "s1", "brain_path": "root.acme", "sensitivity": 1,
                             "time": "t", "title": "SECRET", "detail": {"urgency": "high"}})
    assert s["signal_id"] == "s1" and "SECRET" not in json.dumps(s) and s["detail"] == {"urgency": "high"}


def test_events_published_only_after_commit():
    bus, got = live.EventBus(), []
    bus.listeners.append(got.extend)
    with pytest.raises(RuntimeError):
        with bus.collect():
            bus.emit({"event": "created"})
            raise RuntimeError("transaction rolled back")
    assert got == []
    with bus.collect():
        bus.emit({"event": "created"})
    assert got == [{"event": "created"}]


def test_switched_off_without_a_host(settings):
    live.publish(settings, [{"event": "created"}])  # live_host is empty in tests: nothing happens


def test_broker_files(tmp_path):
    grants = [{"user_id": "riya", "brain_path": "root.acme.sales", "include_children": False, "max_sensitivity": 1}]
    info = live.write_broker_files(S, ["riya", "nogrant"], grants, tmp_path, self_hosted=["globex"])
    acl = (tmp_path / "acl").read_text()
    users = [line.split(":")[0] for line in (tmp_path / "passwd").read_text().splitlines()]
    assert users == ["sb_publisher", "nogrant", "riya", "sb_monitor", "sync_globex"]
    assert "topic read sb/root/acme/sales/signals/internal" in acl and "restricted" not in acl
    assert f"user sync_globex\ntopic write {up_topic('globex')}\ntopic read {down_topic('globex')}" in acl
    assert info["users"] == 2


def test_credentials_follow_the_grants_table(settings, client):
    s = settings.model_copy(update={"live_host": "localhost", "live_secret": "x", "live_publisher_password": "y"})
    c = live.credentials(s, "asha")
    assert c["username"] == "asha" and c["password"] == live.user_password(live.MqttSettings.from_app(s), "asha")
    assert "sb/root/acme/sales/signals/internal" in c["topics"] and "sb/users/asha/inbox" in c["topics"]
    assert not any("restricted" in t or "finance" in t for t in c["topics"])
    assert client.get("/live/credentials", headers={"X-User-Id": "asha"}).status_code == 404  # switched off


def test_saving_memories_announces_them_after_commit(client, monkeypatch):
    sent = []
    monkeypatch.setattr(live, "publish", lambda settings, events: sent.extend(events))
    client.post("/remember", json={"content": "Big order from Initech next week"}, headers={"X-User-Id": "asha"})
    assert client.post("/consolidate", headers={"X-User-Id": "asha"}).json() == {"consolidated": 1}
    created, inbox = sent
    assert created["event"] == "created" and created["brain_path"] == "root.acme.sales" and created["memory_id"]
    assert inbox == {"event": "inbox", "user_id": "asha", "outcome": "saved", "count": 1, "time": inbox["time"]}


@pytest.mark.skipif(not os.environ.get("SB_LIVE_TEST_BROKER"), reason="set SB_LIVE_TEST_BROKER=1 with the live broker up")
def test_broker_delivers_only_what_grants_allow():
    import paho.mqtt.client as mqtt
    from secondbrain.config import get_settings

    s = live.MqttSettings.from_app(get_settings())
    grants = [
        {"user_id": "t_sales", "brain_path": "root.acme.sales", "include_children": False, "max_sensitivity": 1},
        {"user_id": "t_ceo", "brain_path": "root.acme", "include_children": True, "max_sensitivity": 2},
    ]
    live.write_broker_files(s, ["t_sales", "t_ceo"], grants, live.BROKER_DIR, self_hosted=["globex"])
    assert live.reload_broker(live.REPO, live.SERVICE)
    time.sleep(1)

    received: dict[str, list[str]] = {"t_sales": [], "t_ceo": [], "sb_monitor": [], "sync_globex": []}
    clients = []

    def subscriber(user, password):
        ready = threading.Event()
        c = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
        c.username_pw_set(user, password)
        c.on_connect = lambda cl, *_: cl.subscribe("sb/#", qos=1) and None
        c.on_subscribe = lambda *a: ready.set()
        c.on_message = lambda cl, u, m: received[user].append(m.topic)
        c.connect(s.host, s.port)
        c.loop_start()
        assert ready.wait(5), f"{user} could not subscribe"
        clients.append(c)
        return c

    subscriber("t_sales", live.user_password(s, "t_sales"))
    subscriber("t_ceo", live.user_password(s, "t_ceo"))
    subscriber("sb_monitor", live.monitor_password(s))
    sync = subscriber("sync_globex", sync_password(s, "globex"))

    pub = live.MqttPublisher(s)
    time.sleep(0.5)
    now = "2026-01-01T00:00:00Z"
    pub([{"event": "created", "memory_id": "a", "brain_path": "root.acme.sales", "sensitivity": 1, "user_id": "x", "time": now},
         {"event": "created", "memory_id": "b", "brain_path": "root.acme.finance", "sensitivity": 2, "user_id": "x", "time": now},
         {"event": "opened", "signal_id": "c", "brain_path": "root.acme", "sensitivity": 1, "time": now},
         {"event": "inbox", "user_id": "t_sales", "outcome": "created"}])
    clients[0].publish("sb/root/acme/finance/memories/restricted", "{}", qos=1)  # t_sales cannot write
    sync.publish(up_topic("globex"), json.dumps({"type": "pattern"}), qos=1)      # allowed
    sync.publish(up_topic("initech"), json.dumps({"type": "pattern"}), qos=1)     # not its topic
    time.sleep(2)
    pub.close()
    for c in clients:
        c.loop_stop(); c.disconnect()

    assert received["t_sales"] == ["sb/root/acme/sales/memories/internal", "sb/users/t_sales/inbox"]
    assert sorted(received["t_ceo"]) == ["sb/root/acme/finance/memories/restricted",
                                         "sb/root/acme/sales/memories/internal", "sb/root/acme/signals/internal"]
    assert sorted(received["sb_monitor"]) == sorted([
        "sb/root/acme/sales/memories/internal", "sb/root/acme/finance/memories/restricted",
        "sb/root/acme/signals/internal", "sb/users/t_sales/inbox", "sb/sync/globex/up"])
    assert received["sync_globex"] == []
