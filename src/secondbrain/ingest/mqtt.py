"""MQTT bridge: businesses publish events, SecondBrain stores them.

    uv run python -m secondbrain.ingest.mqtt

Topics are ``sb/v1/{company}/{brain}/{kind}``, e.g.
``sb/v1/desiliving/merchandising/sale``; ``{brain}`` is a department id, a deeper
path with dots (``sales.emea``) or ``_`` for the company brain itself. A
payload is one message or a JSON list of messages, and must name the same
company and brain as its topic. Broker ACLs should let each client publish
only under its own ``sb/v1/{company}/#``; this bridge checks again anyway.

Delivery is QoS 1 with manual acknowledgement: a message is acked only after
its database transaction committed (or it was dead-lettered). If the
database is unreachable, nothing is acked and the bridge reconnects after a
pause, and the broker redelivers from the persistent session. Event ids make
redelivery harmless.

Several bridges can share the load with a shared subscription
(``SB_MQTT_SHARE_GROUP=secondbrain``): the broker gives each message to one
of them.

Rejected messages go to the dead-letter table and, unless turned off, are
reported on ``sb/v1/{company}/_errors`` so the publisher sees them.

Settings (environment or .env), production uses MQTTS on 8883:

    SB_MQTT_HOST, SB_MQTT_PORT=8883, SB_MQTT_TLS=true, SB_MQTT_CA_FILE,
    SB_MQTT_CERT_FILE, SB_MQTT_KEY_FILE (client certificate, optional),
    SB_MQTT_USERNAME, SB_MQTT_PASSWORD, SB_MQTT_CLIENT_ID, SB_MQTT_SHARE_GROUP

Local development against docker compose's broker (plain, no TLS):

    SB_MQTT_HOST=localhost SB_MQTT_PORT=51883 SB_MQTT_TLS=false uv run python -m secondbrain.ingest.mqtt
"""

import json
import logging
import ssl
import time
from collections.abc import Callable
from dataclasses import dataclass

import paho.mqtt.client as mqtt
import psycopg
from paho.mqtt.enums import CallbackAPIVersion
from paho.mqtt.packettypes import PacketTypes
from paho.mqtt.properties import Properties
from pydantic_settings import BaseSettings, SettingsConfigDict

from secondbrain.config import Settings, get_settings
from secondbrain.ingest import contract, processor, store

log = logging.getLogger(__name__)


class MqttSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="SB_MQTT_", env_file=".env", extra="ignore")

    host: str = "localhost"
    port: int = 8883
    tls: bool = True
    ca_file: str = ""  # empty: the system's trusted CAs
    cert_file: str = ""
    key_file: str = ""
    username: str = ""
    password: str = ""
    client_id: str = "secondbrain-ingest"
    share_group: str = ""  # e.g. "secondbrain" to subscribe as $share/secondbrain/...
    topic_prefix: str = "sb/v1"
    publish_errors: bool = True
    keepalive: int = 60
    session_expiry: int = 24 * 3600  # how long the broker keeps unacked messages while we're away


@dataclass
class Topic:
    company: str  # root.desiliving
    brain: str  # root.desiliving.merchandising
    kind: str


def parse_topic(topic: str, prefix: str = "sb/v1") -> Topic:
    """``sb/v1/desiliving/merchandising/sale`` → Topic; raises ValueError."""
    rest = topic.removeprefix(prefix + "/") if topic.startswith(prefix + "/") else None
    parts = rest.split("/") if rest else []
    if len(parts) != 3:
        raise ValueError(f"topic {topic!r} is not {prefix}/{{company}}/{{brain}}/{{kind}}")
    company_id, brain, kind = parts
    if kind not in contract.RECORDS:
        raise ValueError(f"unknown kind {kind!r} in topic; expected one of {', '.join(contract.KINDS)}")
    company = contract.company_path(company_id)
    return Topic(company, contract.brain_path(company, brain), kind)


class Bridge:
    """Turns MQTT messages into processed events. Has no MQTT code, so it's testable directly."""

    def __init__(self, settings: Settings, prefix: str = "sb/v1",
                 on_processed: Callable[[str], None] | None = None,
                 publish: Callable[[str, bytes], None] | None = None):
        self.settings = settings
        self.prefix = prefix
        self.on_processed = on_processed
        self.publish = publish
        self._conn: psycopg.Connection | None = None

    def _db(self) -> psycopg.Connection:
        if self._conn is None or self._conn.closed:
            self._conn = psycopg.connect(self.settings.admin_database_url, autocommit=True)
            store.ensure_schema(self._conn)
        return self._conn

    def handle(self, topic: str, payload: bytes) -> bool:
        """Process one MQTT message. True: done (ack it). False: database trouble, redeliver later."""
        try:
            conn = self._db()
            try:
                where = parse_topic(topic, self.prefix)
                body = json.loads(payload)
            except (ValueError, UnicodeDecodeError) as exc:
                raw = {"topic": topic, "raw": payload[:10000].decode("utf-8", "replace")}
                outcome = processor.dead_letter(conn, _kind_of(topic), raw, "mqtt", f"unreadable message: {exc}",
                                                company=_company_segment(topic, self.prefix) or None)
                self._report(topic, [outcome])
                return True
            items = body if isinstance(body, list) else [body]
            outcomes = [processor.process(conn, where.kind, item, "mqtt", where.company, where.brain)
                        for item in items]
        except psycopg.OperationalError:
            log.exception("Database unavailable; message on %s will be redelivered", topic)
            if self._conn is not None:
                self._conn.close()
            return False
        if self.on_processed and any(o.status == "processed" for o in outcomes):
            self.on_processed(where.company)
        self._report(topic, outcomes)
        return True

    def _report(self, topic: str, outcomes: list[processor.Outcome]) -> None:
        if not self.publish:
            return
        for o in outcomes:
            if o.status != "rejected":
                continue
            # To the company whose topic it came on: that's the publisher who needs to know.
            company = _company_segment(topic, self.prefix) or (o.company or "").removeprefix("root.")
            if not company:
                continue
            body = {"topic": topic, "event_id": o.event_id, "error": o.error}
            self.publish(f"{self.prefix}/{company}/_errors", json.dumps(body).encode())


def _kind_of(topic: str) -> str:
    return topic.rsplit("/", 1)[-1][:50]


def _company_segment(topic: str, prefix: str) -> str:
    parts = topic.removeprefix(prefix + "/").split("/")
    return parts[0] if topic.startswith(prefix + "/") and parts[0] else ""


def subscription(m: MqttSettings) -> str:
    topic = f"{m.topic_prefix}/+/+/+"
    return f"$share/{m.share_group}/{topic}" if m.share_group else topic


def build_client(m: MqttSettings, bridge: Bridge) -> mqtt.Client:
    client = mqtt.Client(CallbackAPIVersion.VERSION2, client_id=m.client_id,
                         protocol=mqtt.MQTTv5, manual_ack=True)
    if m.username:
        client.username_pw_set(m.username, m.password or None)
    if m.tls:
        client.tls_set(ca_certs=m.ca_file or None, certfile=m.cert_file or None, keyfile=m.key_file or None,
                       cert_reqs=ssl.CERT_REQUIRED, tls_version=ssl.PROTOCOL_TLS_CLIENT)
    client.reconnect_delay_set(min_delay=1, max_delay=60)
    bridge.publish = (lambda t, p: client.publish(t, p, qos=1)) if m.publish_errors else None

    def on_connect(client, userdata, flags, reason_code, properties):
        if reason_code.is_failure:
            log.error("MQTT connection refused: %s", reason_code)
            return
        log.info("Connected to %s:%s, subscribing to %s", m.host, m.port, subscription(m))
        client.subscribe(subscription(m), qos=1)

    def on_message(client, userdata, msg):
        if bridge.handle(msg.topic, msg.payload):
            client.ack(msg.mid, msg.qos)
        else:
            # Unacked messages stay in the broker's session; reconnecting gets them again.
            client.disconnect()

    client.on_connect = on_connect
    client.on_message = on_message
    return client


def run(settings: Settings, m: MqttSettings, on_processed: Callable[[str], None] | None = None) -> None:
    bridge = Bridge(settings, m.topic_prefix, on_processed)
    client = build_client(m, bridge)
    connect = Properties(PacketTypes.CONNECT)
    connect.SessionExpiryInterval = m.session_expiry
    # A persistent session (clean_start False) keeps unacked QoS 1 messages for us across reconnects.
    client.connect(m.host, m.port, m.keepalive, clean_start=False, properties=connect)
    backoff = 1
    while True:
        client.loop_forever(retry_first_connection=True)
        # loop_forever only returns after disconnect(): the database was down.
        log.warning("Reconnecting in %s s", backoff)
        time.sleep(backoff)
        backoff = min(backoff * 2, 60)
        try:
            client.reconnect()
            backoff = 1
        except OSError:
            log.exception("Reconnect failed")


def main() -> None:
    from secondbrain.ingest.pipeline import IngestSettings, scheduler_for
    from secondbrain.models import build_models

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    settings = get_settings()
    ingest = IngestSettings()
    on_processed = None
    if ingest.auto_refresh:
        llm, embedder = build_models(settings)
        on_processed = scheduler_for(settings, llm, embedder, ingest).start().mark
    run(settings, MqttSettings(), on_processed)


if __name__ == "__main__":
    main()
