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

- **Sales forecasting** (first version): daily units per product, eight weeks
  ahead, with 80% and 95% ranges. Four model variants (last week, last year
  scaled by growth, exponential smoothing, LightGBM with promotions and
  holidays) are backtested on each merchant's history and the best one per
  product is kept, with its reason. Sales are permission-checked like memories.
  See [docs/FORECASTING.md](docs/FORECASTING.md).
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

## Explore page

`http://localhost:8765/explore`, with "Looking as" choosing whose view it is:

- **Decision signals**: what the brain recommends doing next, each with the
  reason, the evidence, the impact, a confidence and a deadline. Stock signals
  are calculated from the forecast, stock on hand and on order, and supplier
  lead times (run-outs, backorders, excess stock). Business signals come from
  the model reading the company's recent notes and the lessons learned at the
  root; every one must cite its evidence. People mark signals as acted on or
  dismissed, and those aren't raised again for two weeks. A signal is only
  visible to someone who can see all of its evidence.

- **Sales forecast**: what is forecast (units sold per day), the stock outlook
  and the decision it leads to, how the forecast is made in four steps, what
  drives it (recent level, growth, weekday rhythm, last year's same weeks,
  promotions, Black Friday, measured from the product's own sales), the chart
  with 80% and 95% ranges, and every model's backtest scores. Any horizon from
  7 to 120 days.
- **Who knows what**: a search drawn as a graph in the style of FalkorDB's own
  browser (labelled nodes, typed relationship arrows, properties on click,
  double-click to expand), built only from what the selected person may see.

## The evolving brain and SecondBrain HQ

- **Outcomes**: every run-out warning is judged once its date comes: prevented
  (acted on, didn't run out), happened (ignored, ran out), happened anyway, or
  false alarm. Each outcome is saved as a memory in the product's brain, so the
  brain remembers it. Signals nobody answered before their deadline expire.
- **Learning**: a run-out warning's confidence is the forecast's chance it
  comes true if ignored, blended with how earlier warnings for that product
  (and, less, other products) turned out; each signal says why in its evidence
  ("Why 84% confidence: ..."). A product that ran out although someone acted
  is raised more urgently; one dismissed repeatedly for the same reason ("we
  have a second warehouse") gets lower priority and is raised only close to
  its deadline, unless those warnings came true anyway. Business signals see
  feedback with reasons, keep the same fingerprint when the model rewords
  advice (so dismissed advice stays dismissed), must cite a note of the
  business, and are dropped if their title or action uses a number that isn't
  in the cited evidence (flagged if only the reason or impact does).
  `GET /signals/quality` gives the measured track record for what the viewer
  may see, shown in one line on the Decision signals tab. Root learning adds
  outcome lessons from sharing companies, in rough proportions only ("Run-out
  warnings nobody acted on came true about nine times in ten").
- **Measuring signals**: `uv run python scripts/eval_signals.py` runs 18
  simulated futures (seeds, lead times, demand shocks) under three policies
  (ignore, act when due, a naive reorder-point rule) on its own databases and
  reports precision/recall, lead-time and quantity adequacy, calibration and
  money. `scripts/check_business_signals.py --env <.env>` checks the real
  model's business signals on its own database.
- **Brain evolution** tab on Explore: sales saved, sales lost when ignored,
  how often warnings came true, and a timeline of outcomes, feedback and facts
  the brain updated.
- **SecondBrain HQ** at `http://localhost:8765/hq` (admin token): the root
  brain's view of every connected business — signals, money at stake, outcomes,
  track record, lessons at the root. Aggregates only, never a business's notes.
- **Moving time forward (demo only)**: HQ's "Simulate 14/28 days" or
  `uv run python -m secondbrain.evolution --days 28` draws sales from the
  forecast with randomness, runs stock down, delivers orders placed by acting
  on signals after the lead time, judges outcomes, and recalculates forecasts
  and signals. With real businesses the same judging runs on real sales.
  Reset to Nov 4 with `schema --reset` and `load_examples.sh`.

## Decision signals

Recalculated by `load_examples.sh`, or on demand:

```bash
uv run python -m secondbrain.signals
```

Admin API: `POST /admin/signals/generate`. Stock data is sample data from
`scripts/generate_lumen_inventory.py`, set against the forecasts.

## Relationships (FalkorDB)

FalkorDB runs in `docker compose` on port 56380. Its own **FalkorDB Browser**
is on `http://localhost:53000` (log in with host `localhost`, port `6379`,
then pick the graph `sb_lumen`). It is an admin tool: it shows a company's
whole graph without per-person permissions, so never show it to a client with
another company's data loaded. Entity extraction runs by
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

## Forecasting

Lumen's sales history is synthetic: two years of daily sales for eight
products, with seasons, Black Friday, promotions and a stockout. Both steps
also run at the end of `load_examples.sh`.

```bash
uv run python scripts/generate_lumen_sales.py         # load history and promotions into root.lumen.merchandising
uv run python -m secondbrain.forecasting              # backtest, choose a model per product, store forecasts
uv run python scripts/generate_lumen_sales.py --csv data/lumen_sales.csv --no-db   # CSV only
```

The forecasting job prints each model's backtest error (WAPE) per product and
is meant to run nightly. Then, as someone allowed to see Merchandising's data:

```bash
curl -H 'X-User-Id: omar' 'localhost:8765/forecast?product=halo-lamp&horizon=28'
curl -H 'X-User-Id: omar' localhost:8765/forecast/products
```

The answer has the forecast with ranges, the chosen model and why, every
model's backtest scores, recent sales, and a nowcast adjustment when sales
since the last run differ from the forecast. Nina (Support) gets a 404: she
can't see Merchandising.

`sales_daily` becomes a TimescaleDB hypertable when the extension is there.
To try it next to the running stack (port 55433):
`docker compose -f docker-compose.timescale.yml -p secondbrain-ts up -d --wait`.

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
