"""Live notifications: tell people the moment something they may see changes.

Built on the SecondBrain MQTT kit (realtime.py, outbox.py, sync.py). The data-ingest
broker brings business events IN; this one pushes permission-checked notices OUT to
people's screens. The broker's logins and ACL are generated from the grants table, so
the broker itself stops a notice from reaching someone who may not see that brain.

Messages carry ids only. The page then reloads from the API, which checks again.
Everything here is a no-op while SB_LIVE_HOST is empty, and publishing never raises.
"""
import logging
from datetime import datetime, timezone
from pathlib import Path

import psycopg
from psycopg.rows import dict_row

from secondbrain.config import Settings

from .outbox import EventBus
from .realtime import (
    KINDS, MONITOR, PUBLISHER, MqttPublisher, inbox_topic, listen, memory_topic, monitor_password,
    mosquitto_hash, public_payload, reload_broker, topics_for_grants, user_password, write_broker_files,
)
from .settings import SENSITIVITY_NAMES, MqttSettings

log = logging.getLogger(__name__)
REPO = Path(__file__).resolve().parents[3]
BROKER_DIR = REPO / "docker" / "live-broker"
SERVICE = "live-broker"

_publishers: dict[MqttSettings, MqttPublisher] = {}


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def publish(settings: Settings, events: list[dict]) -> None:
    """Send committed change events. Call only after the transaction committed."""
    if not events or not settings.live_host:
        return
    try:
        s = MqttSettings.from_app(settings)
        if s not in _publishers:
            _publishers[s] = MqttPublisher(s)
        _publishers[s](events)
    except Exception:  # a notification must never undo or fail a committed write
        log.exception("live notification failed")


def _grants(settings: Settings) -> tuple[list[str], list[dict]]:
    with psycopg.connect(settings.admin_database_url, row_factory=dict_row) as conn:
        users = [r["id"] for r in conn.execute("SELECT id FROM users ORDER BY id").fetchall()]
        grants = conn.execute(
            """SELECT user_id, brain_path::text AS brain_path, include_children, max_sensitivity
               FROM grants ORDER BY user_id, brain_path""").fetchall()
    return users, grants


def sync_broker(settings: Settings, out_dir: Path = BROKER_DIR, reload: bool = True) -> dict:
    """Regenerate the broker's logins and ACL from the grants table, then reload it.
    Run after onboarding, a role change, or someone leaving."""
    users, grants = _grants(settings)
    info = write_broker_files(MqttSettings.from_app(settings), users, grants, out_dir)
    info["reloaded"] = reload_broker(REPO, SERVICE) if reload else False
    return info


def credentials(settings: Settings, user_id: str) -> dict:
    """What a signed-in person's browser needs to listen: their own login and topics."""
    s = MqttSettings.from_app(settings)
    with psycopg.connect(settings.admin_database_url, row_factory=dict_row) as conn:
        grants = conn.execute(
            """SELECT brain_path::text AS brain_path, include_children, max_sensitivity
               FROM grants WHERE user_id = %s""", (user_id,)).fetchall()
    return {"url": s.ws_url, "username": user_id, "password": user_password(s, user_id),
            "topics": topics_for_grants(grants, s.max_depth) + [inbox_topic(user_id)]}


def monitor_credentials(settings: Settings) -> dict:
    """The read-only operations login (sees every notice, can publish nothing)."""
    s = MqttSettings.from_app(settings)
    return {"url": s.ws_url, "username": MONITOR, "password": monitor_password(s), "topics": ["sb/#"]}


__all__ = [
    "BROKER_DIR", "KINDS", "MONITOR", "PUBLISHER", "SENSITIVITY_NAMES", "EventBus", "MqttPublisher", "MqttSettings",
    "credentials", "inbox_topic", "listen", "memory_topic", "monitor_credentials", "monitor_password",
    "mosquitto_hash", "now", "public_payload", "publish", "reload_broker", "sync_broker", "topics_for_grants",
    "user_password", "write_broker_files",
]
