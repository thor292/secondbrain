"""Settings for live notifications, taken from the app's own settings (SB_LIVE_* variables)."""
from dataclasses import dataclass

from secondbrain.config import Settings
from secondbrain.sensitivity import Sensitivity

# Sensitivity levels, lowest first. The name is the last level of every topic.
SENSITIVITY_NAMES = {int(s): s.name.lower() for s in Sensitivity}


@dataclass(frozen=True)
class MqttSettings:
    host: str = "localhost"          # empty = live notifications switched off
    port: int = 1883
    ws_url: str = "ws://localhost:9001"
    secret: str = ""                 # per-user passwords are derived from this; keep it private
    publisher_password: str = ""     # the backend's own broker account (sb_publisher)
    max_depth: int = 3               # root . company . department

    @classmethod
    def from_app(cls, settings: Settings) -> "MqttSettings":
        s = cls(host=settings.live_host, port=settings.live_port, ws_url=settings.live_ws_url,
                secret=settings.live_secret, publisher_password=settings.live_publisher_password)
        if s.host and (not s.secret or not s.publisher_password):
            raise ValueError("Set SB_LIVE_SECRET and SB_LIVE_PUBLISHER_PASSWORD (see .env.example)")
        return s
