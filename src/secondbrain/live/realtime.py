"""Real-time notifications over MQTT.

Topics mirror the brain tree, with sensitivity as the last level, so the broker itself can stop
a restricted event from reaching someone who may only read internal memories:

    sb/root/acme/sales/memories/internal     a memory changed in Acme Sales (internal)
    sb/root/acme/finance/memories/restricted
    sb/root/acme/signals/internal            Acme's decision signals changed
    sb/users/<user id>/inbox                 private notices (e.g. "your note was promoted")

Messages carry ids and metadata only, never memory content. A client that wants the content
calls your API, where your database permission check decides again.

The broker's users and ACL are generated from YOUR grants (`write_broker_files`), so your
database stays the only place permissions are decided. MQTT has no replay; it is for
notifications, not for the durable pipeline.
"""
import base64
import hashlib
import hmac
import json
import logging
import os
import subprocess
import time
from pathlib import Path
from typing import Iterable

from .settings import SENSITIVITY_NAMES, MqttSettings

log = logging.getLogger(__name__)
PUBLISHER = "sb_publisher"  # backend service account: may publish and read everything under sb/
MONITOR = "sb_monitor"      # operations / demo: may read everything, may publish nothing


# ------------------------------------------------------------------ topics

# What can change in a brain. Each kind is one topic level, so grants cover all of them.
KINDS = ("memories", "signals")


def memory_topic(brain_path: str, sensitivity: int, kind: str = "memories") -> str:
    """root.acme.sales + 1 -> sb/root/acme/sales/memories/internal"""
    return f"sb/{brain_path.replace('.', '/')}/{kind}/{SENSITIVITY_NAMES[sensitivity]}"


def event_kind(event: dict) -> str:
    return "signals" if "signal_id" in event else "memories"


def inbox_topic(user_id: str) -> str:
    return f"sb/users/{user_id}/inbox"


def topics_for_grants(grants: Iterable[dict], max_depth: int = 3) -> list[str]:
    """Every memory topic a set of grants allows reading (with + wildcards for subtrees).

    A grant is a dict with: brain_path ("root.acme.sales"), max_sensitivity (int) and
    include_children (bool: also every brain below brain_path).
    """
    topics = []
    for g in grants:
        base = g["brain_path"].replace(".", "/")
        depth = len(g["brain_path"].split("."))
        for level in range(g["max_sensitivity"] + 1):
            name = SENSITIVITY_NAMES[level]
            for kind in KINDS:
                topics.append(f"sb/{base}/{kind}/{name}")
                if g.get("include_children"):
                    for extra in range(1, max_depth - depth + 1):
                        topics.append(f"sb/{base}/{'/'.join(['+'] * extra)}/{kind}/{name}")
    return sorted(set(topics))


# ------------------------------------------------------------------ logins

def user_password(settings: MqttSettings, user_id: str) -> str:
    """Per-user broker password derived from the server secret (stand-in for SSO-issued tokens).
    The server can recompute it at any time, so it never has to be stored."""
    return hmac.new(settings.secret.encode(), user_id.encode(), hashlib.sha256).hexdigest()[:32]


def monitor_password(settings: MqttSettings) -> str:
    return hmac.new(settings.secret.encode(), b"monitor", hashlib.sha256).hexdigest()[:32]


def mosquitto_hash(password: str, iterations: int = 101) -> str:
    """Mosquitto 2.x password-file format: $7$ = PBKDF2-SHA512 with a random salt."""
    salt = os.urandom(12)
    digest = hashlib.pbkdf2_hmac("sha512", password.encode(), salt, iterations)
    return f"$7${iterations}${base64.b64encode(salt).decode()}${base64.b64encode(digest).decode()}"


# ------------------------------------------------------------------ broker files

def write_broker_files(settings: MqttSettings, users: Iterable[str], grants: Iterable[dict],
                       out_dir: Path, self_hosted: Iterable[str] = ()) -> dict:
    """Generate Mosquitto's `passwd` and `acl` files.

    users        every user id in your brain
    grants       dicts with user_id, brain_path, include_children, max_sensitivity
    self_hosted  company slugs that run their own brain and sync with the root (optional)

    Re-run whenever users or grants change, then `reload_broker()`.
    """
    from .sync import down_topic, sync_password, sync_user, up_topic

    users = sorted(set(users))
    by_user: dict[str, list[dict]] = {u: [] for u in users}
    for g in grants:
        by_user.setdefault(g["user_id"], []).append(g)

    passwd = [f"{PUBLISHER}:{mosquitto_hash(settings.publisher_password)}"]
    acl = ["# Generated from the grants table by secondbrain.live.write_broker_files. Do not edit.",
           f"user {PUBLISHER}", "topic readwrite sb/#", ""]
    for user_id in sorted(by_user):
        passwd.append(f"{user_id}:{mosquitto_hash(user_password(settings, user_id))}")
        acl.append(f"user {user_id}")
        acl += [f"topic read {t}" for t in topics_for_grants(by_user[user_id], settings.max_depth)]
        acl += [f"topic read {inbox_topic(user_id)}", ""]

    passwd.append(f"{MONITOR}:{mosquitto_hash(monitor_password(settings))}")
    acl += [f"user {MONITOR}", "topic read sb/#", ""]

    self_hosted = sorted(set(self_hosted))
    for slug in self_hosted:  # each self-hosted brain may only talk on its own sync topics
        passwd.append(f"{sync_user(slug)}:{mosquitto_hash(sync_password(settings, slug))}")
        acl += [f"user {sync_user(slug)}", f"topic write {up_topic(slug)}", f"topic read {down_topic(slug)}", ""]

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, lines in (("passwd", passwd), ("acl", acl)):
        path = out_dir / name
        path.write_text("\n".join(lines) + "\n")
        path.chmod(0o644)  # the broker runs as a different user inside its container
    return {"users": len(by_user), "self_hosted": self_hosted, "acl_lines": len(acl), "dir": str(out_dir)}


def reload_broker(compose_dir: Path | str = ".", service: str = "mqtt") -> bool:
    """Ask a docker-compose broker to re-read passwd/acl (SIGHUP). No restart, no dropped clients."""
    try:
        subprocess.run(["docker", "compose", "kill", "-s", "SIGHUP", service], cwd=compose_dir,
                       check=True, capture_output=True, timeout=30)
        return True
    except (OSError, subprocess.SubprocessError) as exc:
        log.warning("could not reload the MQTT broker: %s", exc)
        return False


# ------------------------------------------------------------------ publishing

def public_payload(event: dict) -> dict:
    """What subscribers receive: ids and metadata, never content."""
    key = "signal_id" if event_kind(event) == "signals" else "memory_id"
    return {
        "event": event["event"],
        key: event[key],
        "brain": event["brain_path"],
        "sensitivity": SENSITIVITY_NAMES[event["sensitivity"]],
        "by_user": event.get("user_id"),
        "time": event["time"],
        "detail": {k: v for k, v in (event.get("detail") or {}).items()
                   if k in ("by", "source", "short_term_id", "urgency", "signal_kind")},
    }


class MqttPublisher:
    """Event listener: publishes committed change events. Reconnects on its own.

    Events it understands:
      {"event": "created"|"updated"|..., "memory_id", "brain_path", "sensitivity", "user_id", "time"}
      {"event": "opened", "signal_id", "brain_path" (the company), "sensitivity", "time"}
      {"event": "inbox", "user_id", ...anything for that user...}
    """

    def __init__(self, settings: MqttSettings):
        import paho.mqtt.client as mqtt

        self.client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id=f"sb-live-{os.getpid()}-{os.urandom(3).hex()}")
        self.client.username_pw_set(PUBLISHER, settings.publisher_password)
        self.client.connect_async(settings.host, settings.port)
        self.client.loop_start()
        self._pending: list = []

    def __call__(self, events: list[dict]) -> None:
        for e in events:
            if e["event"] == "inbox":
                topic, payload = inbox_topic(e["user_id"]), e  # private: only that user can read it
            else:
                topic = memory_topic(e["brain_path"], e["sensitivity"], event_kind(e))
                payload = public_payload(e)
            self._pending.append(self.client.publish(topic, json.dumps(payload, default=str), qos=1))
        self._pending = [m for m in self._pending if not m.is_published()]

    def close(self, timeout: float = 3.0) -> None:
        """Give queued messages a moment to go out (matters for short-lived scripts)."""
        deadline = time.monotonic() + timeout
        while self._pending and time.monotonic() < deadline:
            self._pending = [m for m in self._pending if not m.is_published()]
            time.sleep(0.05)
        self.client.loop_stop()
        self.client.disconnect()


def listen(settings: MqttSettings, username: str = MONITOR, password: str | None = None,
           topic: str = "sb/#") -> None:
    """Print every message this login may read (default: the read-only monitor). Ctrl-C to stop."""
    import paho.mqtt.client as mqtt

    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
    client.username_pw_set(username, password if password is not None else monitor_password(settings))
    client.on_connect = lambda c, *_: c.subscribe(topic, qos=1)
    client.on_message = lambda c, u, msg: print(msg.topic, msg.payload.decode(), flush=True)
    client.connect(settings.host, settings.port)
    client.loop_forever()
