# SecondBrain

A tree of connected Second Brains (root → companies → departments), each with
five memory types. See [ARCHITECTURE.md](ARCHITECTURE.md) for the design.

## What works now

- **Tree of brains**: company and department brains, stored as ltree paths
  (`root.acme.sales`).
- **Permissions enforced by PostgreSQL**: row-level security decides what each
  person sees. Departments can't read each other without an explicit grant,
  and restricted memories need a high enough grant.
- **Fallback to the company brain**: questions search the user's department
  first, then the company, never other companies or the root.
- **Short-term → long-term memory**: notes go into Redis with an expiry, and
  consolidation moves them into PostgreSQL.
- **Company onboarding and document import**: from a JSON spec and `.md`/`.txt`
  files, through the CLI or the admin API.
- **Chat screen** at `http://localhost:8765` for live demos.
- **Model layer**: an offline `stub` provider for development and tests, and a
  `litellm` provider for the company proxy.

## Quick start

Requires Docker and [uv](https://docs.astral.sh/uv/).

```bash
docker compose up -d --wait
uv sync
cp .env.example .env               # set SB_DEMO_MODE=true and an SB_ADMIN_TOKEN
uv run python -m secondbrain.schema --reset
./scripts/load_examples.sh         # Acme Retail and Globex sample companies
uv run uvicorn secondbrain.api:app --port 8765
```

Open `http://localhost:8765`, pick a user and ask, for example:

| User | Question | Expected |
|------|----------|----------|
| Asha (Sales) | What is the Q4 sales target? | From her department |
| Asha (Sales) | How many days of annual leave do we get? | Falls back to the company: 24 days (not Globex's 30) |
| Asha (Sales) | What was the Q3 profit margin? | Not found: restricted Finance data |
| Ravi (CFO) | What was the Q3 profit margin? | 18 percent |

## Demoing with a real company

1. Write `company.json` in the shape of `examples/acme/company.json`
   (departments, users, roles `member` / `lead` / `executive`, extra grants).
2. Put the company's documents in folders per brain, like `examples/acme/docs`.
3. Load them:

   ```bash
   uv run python -m secondbrain.admin onboard path/to/company.json
   uv run python -m secondbrain.admin import root.<company> path/to/docs/company --sensitivity internal
   uv run python -m secondbrain.admin import root.<company>.<dept> path/to/docs/<dept> --sensitivity internal
   ```

   Or use the admin API (`X-Admin-Token` header): `POST /admin/companies` and
   `POST /admin/brains/{brain_path}/documents`.

## Connecting the company models

In `.env`:

```
SB_MODEL_PROVIDER=litellm
SB_LITELLM_BASE_URL=<LiteLLM proxy URL>
SB_LITELLM_API_KEY=<key>
SB_LLM_MODEL=<chat model name in LiteLLM>
SB_EMBEDDING_MODEL=<embedding model name>
SB_EMBEDDING_DIM=<its dimensions>
```

Changing the embedding model or its dimensions requires recreating the schema
and re-importing documents (`schema --reset`, then import again), because
stored vectors must all come from the same model.

The stub provider only matches questions that share words with a memory and
answers by quoting the best match. Real answers need the LiteLLM provider.

## Tests

```bash
docker compose up -d --wait
uv run pytest
```

Tests use a separate `secondbrain_test` database and never touch demo data.

## Not built yet

- Real sign-in (SSO). Users are identified by an `X-User-Id` header, so only
  run the demo on a trusted network.
- Root brain learning, anonymization, FalkorDB relationships, MQTT, smarter
  consolidation (variants V1–V6 in ARCHITECTURE.md).
