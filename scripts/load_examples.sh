#!/usr/bin/env bash
# Load the example companies (Lumen Home, Brightpath Logistics) and realistic
# activity notes from Lumen's people. Run after `secondbrain.schema --reset`.
set -euo pipefail
cd "$(dirname "$0")/.."
uv run python scripts/load_examples.py
uv run python scripts/seed_activity.py
