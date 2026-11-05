"""uv run python -m secondbrain.live sync      regenerate broker logins + ACL from the grants table
   uv run python -m secondbrain.live listen    print every notice (read-only monitor login)"""
import sys

from secondbrain.config import get_settings

from . import MqttSettings, listen, sync_broker


def main() -> None:
    settings = get_settings()
    command = sys.argv[1] if len(sys.argv) > 1 else ""
    if command == "sync":
        print(sync_broker(settings, reload="--no-reload" not in sys.argv))
    elif command == "listen":
        listen(MqttSettings.from_app(settings))
    else:
        sys.exit(__doc__)


if __name__ == "__main__":
    main()
