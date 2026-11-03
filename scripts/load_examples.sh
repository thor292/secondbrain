#!/usr/bin/env bash
# Load the example companies. To demo with a real company, write its own
# company.json and docs folder in the same shape and run the same commands.
set -euo pipefail
cd "$(dirname "$0")/.."
run() { uv run python -m secondbrain.admin "$@"; }

run onboard examples/acme/company.json
run import root.acme examples/acme/docs/company --sensitivity internal
run import root.acme.sales examples/acme/docs/sales --sensitivity internal
run import root.acme.finance examples/acme/docs/finance/internal.md --sensitivity internal
run import root.acme.finance examples/acme/docs/finance/restricted.md --sensitivity restricted
run import root.acme.hr examples/acme/docs/hr --sensitivity internal

run onboard examples/globex/company.json
run import root.globex examples/globex/docs/company --sensitivity internal
