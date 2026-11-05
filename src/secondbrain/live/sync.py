"""OPTIONAL: self-hosted brains sync with the root over MQTT.

Only needed if some companies run their whole brain on their own servers.

    sb/sync/<company slug>/up     company -> root: anonymized, privacy-checked patterns only
    sb/sync/<company slug>/down   root -> company: acknowledgements, skills, best practices

The broker ACL gives each company's sync account write access to its own `up` topic and read
access to its own `down` topic, nothing else - so the topic a message arrives on proves which
company sent it. Persistent sessions (clean_session=False, QoS 1) plus broker persistence mean a
brain that is offline gets its messages when it reconnects.

This file is transport only. What to do with a message (store a pattern, apply a skill) is your
brain's learning logic: pass it in as callbacks.

    # on the company's servers
    agent = CompanySyncAgent(settings, "globex", on_message=handle_ack_or_push).start()
    agent.send_pattern("p-17", "Teams that review contracts yearly negotiate lower prices.")

    # at the root
    root = RootSyncAgent(settings, is_registered=lambda slug: slug in SELF_HOSTED,
                         on_pattern=store_pattern_return_result).start()
    root.push("globex", {"type": "push", "root_id": "...", "kind": "skill", "text": "..."})
"""
import hashlib
import hmac
import json
import logging
import threading
import time
from typing import Callable

from .settings import MqttSettings

log = logging.getLogger(__name__)


def up_topic(slug: str) -> str:
    return f"sb/sync/{slug}/up"


def down_topic(slug: str) -> str:
    return f"sb/sync/{slug}/down"


def sync_user(slug: str) -> str:
    return f"sync_{slug}"


def sync_password(settings: MqttSettings, slug: str) -> str:
    return hmac.new(settings.secret.encode(), f"sync:{slug}".encode(), hashlib.sha256).hexdigest()[:32]


class _Agent:
    def __init__(self, settings: MqttSettings, client_id: str, username: str, password: str, subscribe: str):
        import paho.mqtt.client as mqtt

        self.settings = settings
        self.connected = threading.Event()
        self.client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id=client_id, clean_session=False)
        self.client.username_pw_set(username, password)
        self._subscribe = subscribe
        self.client.on_connect = self._on_connect
        self.client.on_message = self._on_message_safe
        self.client.on_disconnect = lambda *a: self.connected.clear()
        self._pending: list = []

    def _on_connect(self, client, userdata, flags, rc, props=None):
        if rc.is_failure:
            log.error("sync agent could not connect: %s", rc)
            return
        client.subscribe(self._subscribe, qos=1)
        self.connected.set()

    def _on_message_safe(self, client, userdata, msg):
        try:
            self.handle(msg.topic, json.loads(msg.payload))
        except Exception:
            log.exception("sync message failed: %s", msg.topic)

    def handle(self, topic: str, payload: dict) -> None:
        raise NotImplementedError

    def start(self, timeout: float = 5) -> "_Agent":
        self.client.connect(self.settings.host, self.settings.port)
        self.client.loop_start()
        if not self.connected.wait(timeout):
            raise ConnectionError("sync agent could not connect to the MQTT broker")
        return self

    def stop(self, timeout: float = 3.0) -> None:
        deadline = time.monotonic() + timeout  # let queued messages go out first
        while any(not m.is_published() for m in self._pending) and time.monotonic() < deadline:
            time.sleep(0.05)
        self.client.loop_stop()
        self.client.disconnect()

    def publish(self, topic: str, payload: dict) -> None:
        # Never block here: this is also called from the network thread (message callbacks),
        # which is the thread that has to process the broker's acknowledgement.
        self._pending = [m for m in self._pending if not m.is_published()]
        self._pending.append(self.client.publish(topic, json.dumps(payload, default=str), qos=1))


class CompanySyncAgent(_Agent):
    """Runs next to a self-hosted company's brain. Logs in as sync_<slug>."""

    def __init__(self, settings: MqttSettings, slug: str, on_message: Callable[[dict], None]):
        self.slug, self.on_message = slug, on_message
        super().__init__(settings, f"sb-sync-{slug}", sync_user(slug), sync_password(settings, slug), down_topic(slug))

    def send_pattern(self, promotion_id: str, content: str) -> None:
        """Send one ANONYMIZED pattern up. Run your privacy checks before calling this.
        Safe to resend: the root should ignore duplicates by promotion_id."""
        self.publish(up_topic(self.slug), {"type": "pattern", "promotion_id": promotion_id, "content": content})

    def handle(self, topic: str, payload: dict) -> None:
        self.on_message(payload)  # {"type": "ack", ...} or {"type": "push", ...}


class RootSyncAgent(_Agent):
    """Runs at the root. Logs in as the publisher account (it may read sb/sync/+/up)."""

    def __init__(self, settings: MqttSettings, is_registered: Callable[[str], bool],
                 on_pattern: Callable[[str, dict], dict]):
        from .realtime import PUBLISHER

        self.is_registered, self.on_pattern = is_registered, on_pattern
        super().__init__(settings, "sb-root-sync", PUBLISHER, settings.publisher_password, "sb/sync/+/up")

    def handle(self, topic: str, payload: dict) -> None:
        slug = topic.split("/")[2]
        if not self.is_registered(slug):
            log.warning("ignoring sync message from unregistered company %s", slug)
            return
        if payload.get("type") != "pattern":
            return
        result = self.on_pattern(slug, payload) or {}  # check again, merge, return e.g. {"status": "merged"}
        self.publish(down_topic(slug), {"type": "ack", "promotion_id": payload["promotion_id"], **result})

    def push(self, slug: str, message: dict) -> None:
        """Send a skill / best practice down to one self-hosted company."""
        self.publish(down_topic(slug), message)
