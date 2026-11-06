#!/usr/bin/env bash
# Load the example companies (Desi Living, BharatFleet Logistics) and realistic
# activity notes from Desi Living's people, extract entities into the FalkorDB
# graphs, generate sample sales history, forecasts and stock, and calculate
# decision signals, and save the demo starting point (Reset demo in HQ). Run after
# `secondbrain.schema --reset`.
set -euo pipefail
cd "$(dirname "$0")/.."
uv run python scripts/load_examples.py
uv run python scripts/seed_activity.py
uv run python -m secondbrain.relationships
uv run python scripts/generate_sales.py
uv run python -m secondbrain.forecasting --under root.desiliving --horizon 120
uv run python scripts/generate_inventory.py
uv run python -m secondbrain.signals
uv run python -m secondbrain.demo save
