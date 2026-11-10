#!/usr/bin/env bash
# Demo 2 (Live Commerce on TimescaleDB), next to demo 1 on the same laptop.
#
#   docker compose -f docker-compose.timescale.yml -p secondbrain-ts up -d --wait   # once
#   scripts/demo2.sh            # app on :8775, engine on :8776, TimescaleDB writer + watcher
#
# Its own database (TimescaleDB on 55433, set in this checkout's .env), its own MQTT topics
# (commerce2/v1 in, sb/commerce2 out) and ports, so demo 1 (:8765) keeps running untouched.
set -euo pipefail
cd "$(dirname "$0")/.."
export SB_COMMERCE_ENGINE_PORT=8776 SB_COMMERCE_PREFIX=commerce2/v1 SB_COMMERCE_LIVE_PREFIX=sb/commerce2
uv run uvicorn secondbrain.api:app --port 8775 &
app=$!
trap 'kill $app 2>/dev/null' EXIT
uv run python -m secondbrain.commerce --tsdb --speed 1h "$@"
