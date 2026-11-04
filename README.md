# SecondBrain

A tree of connected Second Brains (root → companies → departments), each with
five memory types. See [ARCHITECTURE.md](ARCHITECTURE.md) for the design.

## What works now

- **Tree of brains**: company and department brains, stored as ltree paths
  (`root.lumen.support`).
- **Permissions enforced by PostgreSQL**: row-level security decides what each
  person sees. Departments can't read each other without an explicit grant,
  and restricted memories need a high enough grant.
- **Fallback to the company brain**: questions search the user's department
  first, then the company, never other companies or the root.
- **Conversation memory**: follow-up questions ("and can I carry any over?")
  are rewritten into full questions using the conversation so far.
- **Short-term → long-term memory**: notes go into Redis with an expiry, and
  consolidation moves them into PostgreSQL.
- **Memory maintenance**: new memories are compared with similar older ones.
  Duplicates are merged, outdated facts are replaced, and sections where one
  fact changed are marked "partly outdated". Nothing is deleted, and a replaced
  fact is only hidden from people who can see its replacement.
- **Root learning**: companies whose `sharing` is `patterns` send anonymized,
  general lessons to the root. Restricted memories are never used, and any
  lesson with a number, a person's name or a company name is rejected.
- **Documents**: Markdown, plain text, PDF and Word (.docx), split along their
  own sections ("Employee Handbook › Expenses"), with tables kept.
- **Company onboarding** from a JSON spec, through the CLI or the admin API.
- **Chat screen** at `http://localhost:8765` for live demos.
- **Graph view** at `http://localhost:8765/graph-view`, in the style of a graph
  database browser:
  - **Whole SecondBrain (admin)**: the root with its learned patterns, every
    company, department, person and memory. Needs the admin token.
  - **As a person**: only what that person is allowed to see.
  - Replaced and duplicate memories appear faded, linked to what replaced them.
- **Relationships ("who knows what")**: when memories become long-term
  (documents imported, notes saved), the model extracts the people, customers,
  suppliers, products, carriers, places and topics they mention, and how those
  relate ("Casa Forma supplies the Aria linen sofa"). The results are stored in
  PostgreSQL and synced to one **FalkorDB** graph per company.
  - `GET /experts?topic=Black Friday`: who wrote or is mentioned in memories
    about a topic, and which brains hold them.
  - `GET /entities/Hearth Freight`: what we know about a customer, supplier,
    product, person or topic: relationships, related entities, people, memories.
  - Both only count memories the asking person may read. FalkorDB has no
    permissions of its own, so every graph query is limited to memory ids that
    PostgreSQL returns for that person, and the graph holds no memory text.
- **Model layer**: an offline `stub` provider for tests, and a `litellm`
  provider for the company proxy.

## Quick start

Requires Docker, [uv](https://docs.astral.sh/uv/), and the VPN for the company
models.

```bash
docker compose up -d --wait
uv sync
cp .env.example .env               # fill in the LiteLLM settings, SB_DEMO_MODE=true, an SB_ADMIN_TOKEN
uv run python -m secondbrain.schema --reset
./scripts/load_examples.sh         # companies, documents, notes, maintenance, root learning, entities
uv run uvicorn secondbrain.api:app --port 8765
```

## Example companies

- **Lumen Home** (`examples/lumen`): an online home decor and furniture brand.
  6 departments, 11 people, 18 documents (handbook and campaign plan in Word,
  returns policy, fulfillment SOP and Q3 results in PDF, the rest Markdown),
  some restricted, plus notes from its people that update older facts.
  Shares patterns with the root.
- **Brightpath Logistics** (`examples/brightpath`): a small logistics company
  that does not share anything, used to show isolation.

The PDF and Word files are generated from `examples/lumen/source/` with
`uv run python scripts/build_example_docs.py`.

### Demo questions

| Person | Ask | Shows |
|--------|-----|-------|
| Nina (Support) | Can I give a customer a 50 dollar credit without asking anyone? | Department policy: no, agents can give up to $25 |
| Nina (Support) | How many vacation days do I get? → *and can I carry any over?* | Company fallback, then a follow-up understood from the conversation |
| Nina (Support) | What's the free shipping threshold right now? | $60: an Operations note replaced the $75 policy (Nina may read Operations) |
| Sofia (Marketing) | What's the free shipping threshold right now? | $75: she can't see the Operations note, so she keeps the official policy |
| Leo (Marketing) | When does loyalty early access for Black Friday start? | November 23, the newer date wins over the plan's November 24 |
| Tom (Finance) | What was gross margin in Q3? | Not found: restricted |
| Rachel (CFO) | What was gross margin in Q3? | 46 percent, from the restricted PDF |
| Sofia (Marketing) | What is the hotel cap for a manager? | $190, from a table in the Word handbook |
| Karim (Brightpath) | How many days of annual leave do we get? | 20 days, never Lumen's 25 |

## Demoing with a real company

1. Write `company.json` in the shape of `examples/lumen/company.json`
   (departments, users, roles `member` / `lead` / `executive`, extra grants,
   `sharing`: `none` or `patterns`).
2. Put its documents in `docs/<brain>/<sensitivity>/`, where `<brain>` is
   `company` or a department id and `<sensitivity>` is `public`, `internal` or
   `restricted`, like `examples/lumen/docs`.
3. Put the folder under `examples/` and run `uv run python scripts/load_examples.py`,
   or load it piece by piece:

   ```bash
   uv run python -m secondbrain.admin onboard path/to/company.json
   uv run python -m secondbrain.admin import root.<company>.<dept> path/to/docs --sensitivity internal
   ```

   Or use the admin API (`X-Admin-Token` header): `POST /admin/companies` and
   `POST /admin/brains/{brain_path}/documents`.

Before loading real client documents, make sure the client agrees to them
being processed by the models behind the LiteLLM proxy.

## Maintenance and learning

Both run automatically at the end of `load_examples.sh`; maintenance also runs
in the background after "Save to long-term" in the chat.

```bash
uv run python -m secondbrain.maintenance    # merge duplicates, retire outdated facts
uv run python -m secondbrain.learning       # send anonymized patterns to the root
```

Admin API: `POST /admin/maintain` and `POST /admin/learn`.

## Relationships (FalkorDB)

FalkorDB runs in `docker compose` on port 56380. Entity extraction runs by
itself after "Save to long-term" in the chat and after a document upload or
`admin import`. To extract anything not processed yet and rebuild the graphs:

```bash
uv run python -m secondbrain.relationships                      # every company
uv run python -m secondbrain.relationships --company root.lumen # one company
```

On a database created before this feature, the same command first adds the
new tables (`entities`, `memory_entities`, `entity_relations`,
`entity_extractions`). Existing tables and rows are not changed. With the
LiteLLM models it takes about 30 seconds for the Lumen demo data.

PostgreSQL is the source of truth and the graphs are rebuilt from it, so
FalkorDB keeps no volume: a company graph that is missing (for example after
a restart) is rebuilt automatically on the first query.

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

`SB_LITELLM_BASE_URL` includes the `/v1` suffix if the proxy uses one (for
example `http://10.21.64.70:4000/v1`). The company proxy is only reachable on
the company network or VPN.

The demo uses `claude-sonnet` for answers and `gemini-embedding-001` at 1536
dimensions (it supports 768, 1536 or 3072; we request the size explicitly).

Changing the embedding model or its dimensions requires recreating the schema
and re-importing documents (`schema --reset`, then import again), because
stored vectors must all come from the same model.

The stub provider (used by the tests) only matches questions that share words
with a memory and answers by quoting the best match. Real answers need the
LiteLLM provider.

## Tests

```bash
docker compose up -d --wait
uv run pytest
```

Tests use a separate `secondbrain_test` database and their own small companies
(`tests/fixtures`), and never touch demo data. Their FalkorDB graphs are named
`sbtest_*`, the demo's `sb_*`.

## Not built yet

- Real sign-in (SSO). Users are identified by an `X-User-Id` header, so only
  run the demo on a trusted network. Needs the company's identity provider.
- Connectors (Google Drive, Slack, email). Need app credentials from IT.
- Pushing root patterns down to companies (variant V1), MQTT, self-hosting
  (see ARCHITECTURE.md).
- Using the relationship graph when answering questions, and showing entities
  in the graph view (see ARCHITECTURE.md, phase 3).
