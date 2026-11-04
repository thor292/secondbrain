# SecondBrain — Architecture

> Status: **Draft, under discussion.** Nothing here is built yet.
> Last updated: 2026-11-04
>
> Principle: the demo is the first slice of the real product. Every demo
> choice must carry over to production; anything demo-only is marked.

## 1. Vision

SecondBrain is a memory system we provide to businesses as a product.
Every customer gets its own Second Brain, its departments can get their own,
and all of them are connected in a **tree** whose root is the Second Brain
our organization maintains.

```
                    ROOT  (our organization)
                   /        |         \
           Company A    Company B    Company C
           /      \          |
      Sales    Finance      HR
        |
     (teams / individuals — possible deeper levels)
```

## 2. Decided

These are fixed across every version we build.

### 2.1 Tree structure
- **Root**: our organization's Second Brain, the largest one.
- **Company brains**: one per customer, connected to the root.
- **Department brains**: optional, connected to their company brain.
- The tree can grow deeper (teams, individuals) using the same model.

### 2.2 Five memory types in every brain

| Memory      | Holds |
|-------------|-------|
| Short-term  | Current working context: recent conversations, active tasks |
| Episodic    | Events: what happened, when, who was involved |
| Semantic    | Facts and knowledge ("refund policy is 30 days") |
| User        | Memory about people: preferences, roles, history |
| Long-term   | Consolidated, durable knowledge |

Proposed refinement (to validate): treat **short-term / long-term as tiers**
(how long a memory lasts) and **episodic / semantic / user as types** (what
a memory is about), so every memory has one of each.

| Type     | Short-term (working)            | Long-term (consolidated) |
|----------|---------------------------------|--------------------------|
| Episodic | "Meeting with vendor today"     | "Q3 vendor dispute and how it was resolved" |
| Semantic | "Someone said price is $40"     | "Standard price is $40 (verified)" |
| User     | "Ravi seems to prefer email"    | "Ravi: CFO, prefers email, approves budgets" |

### 2.3 Data flows up
- Data, or at least what is learned from it, eventually flows upward:
  Department → Company → Root.
- What reaches the root depends on each company's restrictions:
  - **No restrictions**: data may flow up.
  - **Restricted**: only skills and patterns go up, never raw data.
- Suggested default: patterns only, with raw data sharing as an opt-in.

### 2.4 Queries stop at the company
- A department brain that can't answer falls back to its **company brain**.
- It **never** falls back to the root. Company-specific questions are
  answered only from that company's knowledge.
  (Like querying your company's Google workspace, not all of Google.)

### 2.5 Access between siblings goes through the parent, with permission
- Sibling brains (e.g. Sales and Finance) never access each other directly.
- A request goes through the parent brain, which checks whether **that
  specific person** is allowed.
- Every memory carries:
  - an **owner** (which brain it belongs to)
  - a **sensitivity label** (e.g. public / internal / restricted / finance-only)
  - **per-user permissions**

### 2.6 Interface: API and chat
- **API first**: the core is an API.
- **Chat assistant**: our own client, built on that API.
- **Connectors**: pull in existing company knowledge (Google Drive,
  databases, Slack, email) so a new brain doesn't start empty.

## 3. Technology stack

### 3.1 Storage and messaging

| Layer | Technology | Role |
|-------|------------|------|
| Core memory + permissions | **PostgreSQL + pgvector** | Source of truth for all memory types, search by meaning |
| Short-term memory | **Redis** | Fast session memory with automatic expiry |
| Relationships | **FalkorDB** | Graph of people, teams, topics (user memory, "who knows what") |
| Raw files | **S3 / MinIO** (self-hosted) | Documents pulled in by connectors |
| Durable internal pipeline | **Redis Streams** | Consolidation and promotion up the tree (replayable) |
| Real-time messaging + edge sync | **MQTT** | Live updates, syncing self-hosted brains |

Introduce gradually: PostgreSQL + Redis first; FalkorDB when user
relationships are built; MQTT when self-hosted brains and live updates are
built.

**PostgreSQL features we rely on**
- **pgvector**: vector type + HNSW index for search by meaning.
- **ltree**: tree path per memory (`root.companyA.finance`), so
  "fall back to my company, never the root" is a path filter.
- **Row-level security**: the database enforces who sees each memory.
- **Time indexes**: episodic event history.

**FalkorDB notes**
- One graph per company → isolation between companies.
- No per-record permissions → our code must check permissions on every
  graph query.
- Runs in memory → watch RAM cost at scale.
- PostgreSQL stays the source of truth; the graph can be rebuilt from it.
- **Open item:** license is believed to be SSPL. Legal must confirm it's
  fine for a commercial service before we commit.

**FalkorDB as built (phase 3)**
- Extracted entities and relationships are stored in PostgreSQL first
  (`entities`, `memory_entities`, `entity_relations`, `entity_extractions`),
  then synced to the graph `sb_<company>`. Tests use `sbtest_<company>`.
- Graph: `(:Brain)`, `(:Person)`, `(:Memory)`, `(:Entity)` with `PARENT_OF`,
  `MEMBER_OF`, `STORED_IN`, `CREATED`, `MENTIONS`, `SAME_AS` (a person and the
  entity naming them) and `RELATES_TO` (carries the id of the memory stating it).
- **Permission rule:** every graph query takes the viewer's *scope*: their
  company (picks the graph) and the memory ids PostgreSQL returns for them
  under row-level security. Queries must filter on those ids (code refuses a
  query that doesn't), so an entity, person or relationship is only reached
  through a memory the viewer may read. Memory text is never stored in the
  graph; it is read from PostgreSQL again, as the viewer.
- The new PostgreSQL tables have row-level security too: a row is visible only
  through a memory the reader may see. Two independent layers.
- Hidden and unknown entities give the same "not found", so an answer never
  reveals that something hidden exists.
- No volume: a missing graph is rebuilt from PostgreSQL on first use.
- Rejected: copying grants and sensitivity into the graph and filtering
  there. It would duplicate the permission logic outside the database that
  enforces it, and the two could drift.
- Scale note: the visible id list is sent with each query. Fine for
  thousands of memories per person; at much larger sizes, filter by brain and
  sensitivity in the graph *in addition* and keep the id check as the final
  gate, or cache scopes per person.

**MQTT notes**
- Topics mirror the tree: `root/companyA/finance/memory/new`,
  `root/companyA/#`.
- Broker topic permissions block companies from each other's topics.
- No replayable history → not used for the durable pipeline (Redis
  Streams does that).

### 3.2 Where embeddings live

Embeddings are stored **in PostgreSQL via pgvector, in the same row as the
memory**. No separate vector database.

```sql
CREATE TABLE memories (
  id               UUID PRIMARY KEY,
  brain_path       LTREE,          -- root.companyA.finance
  memory_type      TEXT,           -- episodic / semantic / user
  tier             TEXT,           -- short_term / long_term
  sensitivity      TEXT,           -- public / internal / finance_only
  content          TEXT,
  embedding        VECTOR(n),      -- n = embedding model's dimensions
  embedding_model  TEXT,           -- which model produced the vector
  created_at       TIMESTAMPTZ
);

CREATE INDEX ON memories USING hnsw (embedding vector_cosine_ops);
```

Why this holds up in production:
1. **Security**: row-level security protects the embedding along with the
   memory. No second system to keep permissions in sync with.
2. **Consistency**: memory and embedding are saved in one transaction.
3. **One query**: tree filter + permissions + similarity in one step.
4. **Self-hosting**: one database to install instead of two.

Scale path: pgvector handles millions of vectors per company. If one very
large customer reaches tens or hundreds of millions, move *that customer's*
vectors to a dedicated vector DB (e.g. Qdrant) behind the same storage
layer.

Constraints:
- pgvector's HNSW index supports up to ~2,000 dimensions (~4,000 as
  half precision). The embedding model must fit.
- Changing the embedding model means re-embedding stored memories;
  `embedding_model` makes that migration traceable.

### 3.3 Backend and identity (proposed)
- **Python + FastAPI**: most AI tooling is in Python; fits API-first.
- **Identity**: SSO with companies' existing systems (Google, Microsoft,
  Okta). Keycloak is an open-source option that also works self-hosted.

## 4. The AI layer

The databases store and fetch; the AI does the thinking. It has four jobs.

| Job | What it does |
|-----|--------------|
| **Organize** (on input) | Understand, extract, classify into memory type and sensitivity, embed |
| **Maintain** (background) | Consolidate, score importance, resolve conflicts, merge duplicates, build the graph |
| **Learn** (up the tree) | Pick department knowledge worth promoting; **anonymize** before anything reaches the root; turn patterns into skills |
| **Answer** (on query) | Understand the question, rerank results, write the answer with sources |

Query flow:
```
Question
  → [AI]  understand
  → [DB]  permission check            (never AI)
  → [DB]  search department → company (pgvector + FalkorDB)
  → [AI]  rerank
  → [AI]  write answer with source
```

**Rule: AI never makes permission decisions.** Access control is enforced
by the database (row-level security). AI can be tricked by prompt
injection; the database can't.

Anonymization is the most privacy-critical AI step: it needs the strongest
available model plus a rule-based check for names and numbers as a backup.

Model size per job (production guidance):

| Job | Model size |
|-----|------------|
| Embedding | Embedding model |
| Classification, extraction | Small / medium (runs on every input) |
| Consolidation, conflicts | Medium |
| Answers, anonymization | Largest available |

All AI calls go through **one model interface**, so providers are swapped
by configuration.

### 4.1 Models

**Demo** (decided by our senior; demo only):
- LLM: the company's **LiteLLM** proxy.
- Embeddings: the company's **in-house embedding model**.
- Still needed: LiteLLM endpoint and key, which models it exposes, the
  embedding model's name and dimensions.

**Production candidates** (variant V6):
- LLM: **Qwen3.5** family (Apache 2.0, many sizes) as default; gpt-oss
  (Apache 2.0) and GLM-5.2 (MIT) as alternatives. Avoid custom-license
  models (e.g. Llama) for commercial use.
- Embeddings: **Qwen3-Embedding-8B** at 1024 dimensions (Apache 2.0,
  multilingual, smaller sizes available, fits pgvector's index).
- Optional: Claude for companies that allow cloud AI.
- Open-source models cost GPU servers instead of license fees. Development
  runs small models locally (Ollama); production serves with vLLM.

## 5. Open questions → variants

Not decided yet. These will be **pluggable settings** so we can build
different versions, compare them, and keep the best.

| # | Open question | Variants to try |
|---|---------------|-----------------|
| V1 | Does the root push skills down? | Off / skills only / skills + best practices |
| V2 | Who approves promotion up the tree? | Automatic / admin approval / confidence threshold |
| V3 | Hosting | Our cloud / their servers / mix |
| V4 | How memory moves between types | Different rules for consolidation and decay (see 5.1) |
| V5 | Isolation between companies | Shared tables + row-level security / schema per company / database per company |
| V6 | Model provider | LiteLLM (demo) / self-hosted open-source / Claude / mix |

### 5.1 Starting point for V4: memory pipeline (one candidate)

1. **Capture**: new input lands in short-term memory with an expiry time.
2. **Extract**: at session end, events → episodic, facts → semantic,
   people → user memory.
3. **Consolidate**: a periodic job promotes important, repeated or
   confirmed memories to long-term; merges duplicates; resolves conflicts
   (newer fact replaces older).
4. **Decay**: unused, low-importance memories fade out.
5. **Promote up the tree**: department long-term memory becomes a
   candidate for the company brain (approval per V2); the company sends
   anonymized patterns to the root (per 2.3).

### 5.2 Comparing versions

To keep the best version, "best" is defined before building:
- A fixed test set: companies, departments, memories and questions.
- Scores: answer accuracy, **permission leaks (must be zero)**, speed,
  cost.
- Every variant runs against the same test set.

## 6. Plan

| Phase | What we build | Ends with |
|-------|---------------|-----------|
| **0. Finish design** | AI models, language, identity, how versions are scored | This document complete |
| **1. One brain** | Five memory types, memory pipeline, API, PostgreSQL + Redis | A single brain that remembers and answers |
| **2. The tree** | Company + departments, fallback, access through parent, per-person permissions | A company with two departments; zero leaks |
| **3. Relationships** | User memory graph in FalkorDB | "Who knows what" queries (**first version built**, see 6.1) |
| **4. Compare versions** | Alternatives for V2 and V4, scored | First variants chosen with data |
| **5. The root** | Learning flows up, patterns from restricted companies, V1 | Learning across the tree |
| **6. Edge and scale** | MQTT, self-hosted brains (V3), isolation options (V5) | Deployable on a company's own servers |
| **7. Product** | Connectors, chat assistant | Usable by a real customer |

### 6.1 Phase 3 status

Built: entity and relationship extraction on consolidation and document
import (LiteLLM; a deterministic stub for tests), storage in PostgreSQL,
per-company FalkorDB graphs with incremental sync and full rebuild
(`python -m secondbrain.relationships`), and two APIs: `GET /experts?topic=`
and `GET /entities/{name}`. Tests prove no entity, person or relationship
leaks across permissions or companies, and that the leak tests fail if the
permission filter is removed.

On the Lumen demo data, extraction takes about 30 seconds and finds about 200
entities (suppliers, carriers, products, customers, places, topics).

Not done yet:
- **Entity resolution**: "Halo floor lamp" and "Halo brass floor lamp", or
  "Nordlys" and "Nordlys lighting", are separate entities. Merge by aliases or
  an AI pass, measured on the demo data.
- **Answering with the graph**: feeding related entities into `ask` was not
  added, because it hasn't been measured. Plan: a fixed question set, compare
  answers with and without graph context, keep it only if it is better.
- **Graph view**: entities are not shown in `graph-view` yet; the current
  fixed nested layout should stay readable, so this needs its own design.
- Memories retired by maintenance stay in the graph; queries already skip
  them for people who can see the replacement, like search does.

## 7. Open items

- LiteLLM endpoint, key and available models (demo).
- Company embedding model name and dimensions (demo).
- FalkorDB license check with legal.
- Confirm backend language (Python + FastAPI proposed).
- Confirm identity approach (SSO / Keycloak proposed).
- Define the test set and scoring for comparing versions.
- Agentic commerce: the open questions in 8.8.

## 8. Use case: Agentic commerce (proposed)

> Status: **Concept under discussion.** Nothing here is built yet.

SecondBrain connected to merchants on an agentic commerce platform, with
two capabilities: **forecasting** (time series predictions) and **real-time
offers at checkout**, each decision carrying a reason.

### 8.1 Mapping onto SecondBrain

| SecondBrain concept | In agentic commerce |
|---|---|
| Company brain | A merchant |
| Department brains | The merchant's stores, channels or regions |
| Episodic memory | Events: product views, add-to-cart, checkout started, abandoned, purchased |
| Semantic memory | Catalog, prices, margins, discount rules |
| User memory | Customer profiles: history, preferences, price sensitivity |
| Root brain | Anonymized patterns across merchants (e.g. "for carts under $50, free shipping converts better than 10% off") |

Cross-merchant patterns at the root give a new merchant useful behaviour on
day one (cold start), without any merchant's raw data being shared.

### 8.2 Forecasting

Examples: sales per product next week, demand spikes, stock running out,
traffic by hour.

| Piece | Choice | Why |
|---|---|---|
| Storage | **TimescaleDB** | PostgreSQL extension: same database, permissions and merchant isolation; hypertables, continuous aggregates (per minute/hour/day), compression |
| Baseline models | Statistical (ETS, ARIMA via `statsforecast`) | Fast, reliable, explainable |
| Stronger models | LightGBM with features (promotions, holidays, price changes) | Usually most accurate in retail |
| Cold start | Time series foundation models (Chronos, TimesFM) | Forecast with little history |

All three run as variants and are **backtested** on past data; the best one is
kept per merchant.

### 8.3 Real-time offer at checkout

The decision must take **under ~100–200 ms**. An LLM call takes 1–3 s, so the
live decision is made by a fast decision engine; the LLM works around it
(explanations, insights, memory), never on the checkout's critical path.

```
Customer at checkout
  → live signals (ms): time on checkout, hesitation, shipping cost reveal,
                       cart value, customer history, merchant margins
  → decision engine (< 50 ms):
       1. abandonment risk
       2. best offer: none / free shipping / 5% / 10% / ...
       3. merchant guardrails: max discount, margin floor, budget  (hard rules, never AI)
  → offer + reason codes
       ├─ shown to the customer
       └─ decision log (inputs, model version, offer, reasons) → outcome later
```

**Reasoning on every decision, right or wrong:**
- **Reason codes** come from the decision model: which signals drove it and how
  much. Fast and exact.
- **A readable explanation** is built from the codes, by template instantly,
  or richer by the LLM afterwards for the merchant's dashboard.
- **Every decision is logged with its reason and later its outcome**,
  including wrong ones, which are what the system learns from.

**Learning while running:** a **contextual bandit** tries offers, observes
outcomes and shifts toward what works per customer and merchant type, while
still exploring. It updates on every outcome, with no retraining job. Later,
**uplift models** estimate whether a discount *changed* the outcome, so
discounts don't go to people who would have bought anyway.

### 8.4 "Evolving every second": what is realistic

| What | Update frequency |
|---|---|
| Live signals (behaviour, cart, stock) | Every second or faster |
| Decision engine (bandit) | On every outcome, seconds to minutes after checkout |
| Memory (events, customer profiles) | Continuously |
| Forecasts | Retrained hourly or daily, with real-time adjustments in between |
| The LLM itself | Doesn't learn by the second; the memory and data it reads do |

The brain's knowledge and decisions update continuously; its models retrain
on a schedule.

### 8.5 Stack additions

| Need | Choice | Notes |
|---|---|---|
| Time series | TimescaleDB | Inside our PostgreSQL |
| Event stream | Redpanda or Kafka | Redis Streams is fine for a pilot; merchant event volume needs a real event platform |
| Live features | Redis (already in stack) | Customer and cart signals in milliseconds |
| Decision engine | Python service: bandit + guardrails | Vowpal Wabbit or own implementation |
| Forecasting | `statsforecast`, LightGBM, Chronos / TimesFM | Built and compared as variants |
| Very large analytics (later) | ClickHouse | Only if a merchant outgrows TimescaleDB |

### 8.6 Risks to design for from day one

1. **Gaming**: if abandoning a cart reliably earns a discount, customers learn
   it. Guardrails need per-customer limits and some randomness.
2. **Legal and fairness**: personalized pricing is regulated in some regions
   (e.g. the EU). Decisions must never use protected attributes (gender, age,
   ethnicity, ...). The decision log makes every decision auditable.
3. **Merchant control**: merchants set the guardrails; the AI only chooses
   inside them.
4. **Proving value**: always keep a control group with no offer, to measure
   real uplift.

### 8.7 Suggested order

1. **Forecasting first**: needs only historical data and shows value fast.
2. **Offline simulation** of the discount engine on historical checkouts.
3. **Live offers**, once real-time event integration is in place.

### 8.8 Open questions

- Is the platform our own, or do we connect to merchants on Shopify,
  WooCommerce, etc.?
- What data do merchants provide: order history only, or live browsing and
  checkout events? (Live offers need live events.)
- Who shows the offer: our checkout, or the merchant's checkout calling our API?
- Which regions? This sets the legal constraints on personalized discounts.
- What is the latency budget at checkout?
