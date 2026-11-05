# Live notifications

People see new decision signals and new knowledge the moment they happen, without
refreshing. Built from the SecondBrain MQTT kit (`src/secondbrain/live/`).

Two brokers, two jobs:

| Broker | Port (dev) | Direction | Who connects |
|---|---|---|---|
| Ingest (`mosquitto`) | 51883 | business systems → brain | POS, ERP, warehouse (see INTEGRATION.md) |
| Live (`live-broker`) | 51884, ws 59001 | brain → people | the backend publishes; browsers listen |

## How permissions are enforced

- The broker's logins and ACL are **generated from the `grants` table**
  (`python -m secondbrain.live sync`), never written by hand. Onboarding a company resyncs automatically.
- Topics mirror the brain tree, with sensitivity last:
  `sb/root/lumen/signals/internal`, `sb/root/lumen/support/memories/internal`, `sb/users/<id>/inbox`.
  The broker never delivers a topic outside someone's grants.
- Messages carry **ids only, never content**. The page reloads from the API, which checks permissions again.
- Notices are sent **after the database commit**, so a rolled-back write is never announced.
- Each person's broker password is HMAC-SHA256(`SB_LIVE_SECRET`, user id); the broker stores PBKDF2-SHA512 hashes.

## What is announced

| Event | Topic | Shown as |
|---|---|---|
| A new decision signal | `sb/<company>/signals/internal` | "🚨 3 new decision signals" (Explore, Chat, HQ) |
| New long-term knowledge | `sb/<brain>/memories/<sensitivity>` | "🧠 New knowledge just arrived" |
| Your notes were saved | `sb/users/<id>/inbox` | "✅ Saved to long-term memory" |

## Run it

```bash
uv run python -m secondbrain.live sync --no-reload   # first time: write passwd + acl
docker compose up -d live-broker
uv run python -m secondbrain.live listen             # watch every notice (read-only monitor)
SB_LIVE_TEST_BROKER=1 uv run pytest tests/test_live.py   # permission test against the real broker
```

## Self-hosted companies (ready, not yet wired)

`live/sync.py` lets a company that runs its brain on its own servers send anonymized
patterns up to the root and receive lessons down, over `sb/sync/<slug>/up|down`, each
company locked to its own topics. Wire it to `learning.py` when the first self-hosted client signs.

## Before production

TLS (8883 and `wss://`), real secrets, short-lived logins from SSO via a Mosquitto auth plugin,
client certificates for sync accounts, and `passwd`/`acl` owned by the mosquitto user (mode 0700).
