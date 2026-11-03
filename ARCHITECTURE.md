# SecondBrain — Architecture

> Status: **Draft, under discussion.** Nothing here is built yet.
> Last updated: 2026-11-03
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
| **3. Relationships** | User memory graph in FalkorDB | "Who knows what" queries |
| **4. Compare versions** | Alternatives for V2 and V4, scored | First variants chosen with data |
| **5. The root** | Learning flows up, patterns from restricted companies, V1 | Learning across the tree |
| **6. Edge and scale** | MQTT, self-hosted brains (V3), isolation options (V5) | Deployable on a company's own servers |
| **7. Product** | Connectors, chat assistant | Usable by a real customer |

## 7. Open items

- LiteLLM endpoint, key and available models (demo).
- Company embedding model name and dimensions (demo).
- FalkorDB license check with legal.
- Confirm backend language (Python + FastAPI proposed).
- Confirm identity approach (SSO / Keycloak proposed).
- Define the test set and scoring for comparing versions.
