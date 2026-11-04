#!/usr/bin/env bash
# Load the example companies (Lumen Home, Brightpath Logistics) and realistic
# activity notes from Lumen's people, then extract entities into the FalkorDB
# graphs. Run after `secondbrain.schema --reset`.
set -euo pipefail
cd "$(dirname "$0")/.."
uv run python scripts/load_examples.py
uv run python scripts/seed_activity.py
uv run python -m secondbrain.relationships
