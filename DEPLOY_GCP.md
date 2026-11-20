# Deploying SecondBrain (v2, `shoppi` branch) to GCP

What runs where:

| Piece | GCP service | Notes |
|---|---|---|
| API + web pages | **Cloud Run** service `secondbrain-api` | This repo's `Dockerfile`. Reaches everything else through the VPC. |
| Database | **Cloud SQL for PostgreSQL** (15+) | Extensions `vector` and `ltree`. TimescaleDB is optional and not available on Cloud SQL; the core works without it. |
| Short-term chat memory | **Memorystore for Redis** | Small basic tier is enough. |
| Relationship graph | FalkorDB on a small **Compute Engine VM** (Docker) | Optional at first: only the "who knows what" graph needs it. |
| AI models | Company **LiteLLM** server, over the VPC | Same as today: chat, signals and embeddings. |
| Nightly jobs | **Cloud Run jobs** + **Cloud Scheduler** | Same image, different command. |
| Secrets | **Secret Manager** | Database passwords, LiteLLM key, admin token. |

Not deployed: the Live Commerce simulator, the MQTT brokers and the TimescaleDB demo page. In production, Shoppi sends real data to `/ingest`.

Replace `PROJECT`, `REGION`, `VPC`, `SUBNET`, `DB_PRIVATE_IP` below with your values.

---

## 0. Minimum to start

The image has **no** built-in addresses. With nothing set it stops at once with
`SecondBrain can't start: SB_ADMIN_DATABASE_URL is not set`. The only required settings are:

- `SB_ADMIN_DATABASE_URL` (secret): `postgresql://secondbrain_admin:<pw>@<cloud sql private ip>:5432/secondbrain`
- the LiteLLM settings (`SB_MODEL_PROVIDER`, `SB_LITELLM_BASE_URL`, `SB_LLM_MODEL`, `SB_EMBEDDING_MODEL`, `SB_EMBEDDING_DIM`, and the `SB_LITELLM_API_KEY` secret)

and the schema job (step 5) must have run once. If the database can't be reached, the log says
`Can't connect to PostgreSQL at <host>: ...` (then check the VPC egress and the Cloud SQL private IP).

## 1. What the GCP team must set up (or confirm)

1. **Cloud SQL Postgres** with a **private IP** in `VPC`, database `secondbrain`.
2. **One database login**, `secondbrain_admin` (create it in the console). Everything uses it: the API, the schema job and the nightly jobs.

   Plus one **role without a login**, `secondbrain_app`. The API switches to it on every connection, so row-level security (each company sees only its own data) still applies even though the login owns the tables. **The name must be exactly `secondbrain_app`**: the schema grants permissions to it.

   Connected as `secondbrain_admin`:
   ```sql
   CREATE EXTENSION IF NOT EXISTS vector;
   CREATE EXTENSION IF NOT EXISTS ltree;
   CREATE ROLE secondbrain_app NOLOGIN;
   GRANT secondbrain_app TO secondbrain_admin;   -- lets the login switch to it
   GRANT USAGE ON SCHEMA public TO secondbrain_app;
   ```
   The API refuses to start if it can't switch to `secondbrain_app`, so a missing `GRANT` shows up straight away.

3. **Cloud Run → VPC**: Direct VPC egress (or a Serverless VPC Access connector) on `VPC`/`SUBNET`, with routes and firewall open to:
   - Cloud SQL private IP, port 5432
   - Memorystore Redis, port 6379
   - the company LiteLLM server (address from the local `.env`), port 4000
   - FalkorDB VM, port 6379 (if used)
4. **Artifact Registry** Docker repository, e.g. `REGION-docker.pkg.dev/PROJECT/secondbrain`.
5. A **service account** for Cloud Run, e.g. `secondbrain-run@PROJECT.iam.gserviceaccount.com`, with `roles/secretmanager.secretAccessor` and `roles/cloudsql.client`.

## 2. Build and push the image

From this folder (Cloud Run needs `linux/amd64`):

```bash
gcloud auth configure-docker REGION-docker.pkg.dev
docker build --platform linux/amd64 -t REGION-docker.pkg.dev/PROJECT/secondbrain/api:shoppi .
docker push REGION-docker.pkg.dev/PROJECT/secondbrain/api:shoppi
```

Or let Cloud Build do it (no local Docker needed):
```bash
gcloud builds submit --tag REGION-docker.pkg.dev/PROJECT/secondbrain/api:shoppi .
```

## 3. Secrets

```bash
printf '%s' 'postgresql://secondbrain_admin:<pw>@DB_PRIVATE_IP:5432/secondbrain' | gcloud secrets create sb-admin-db-url --data-file=-
printf '%s' '<litellm key>'   | gcloud secrets create sb-litellm-key  --data-file=-
printf '%s' "$(openssl rand -hex 24)" | gcloud secrets create sb-admin-token --data-file=-
```

## 4. Settings (plain environment variables)

| Variable | Value |
|---|---|
| `SB_MODEL_PROVIDER` | `litellm` |
| `SB_LITELLM_BASE_URL` | same as the local `.env` (the company LiteLLM server) |
| `SB_LLM_MODEL` | same as the local `.env` |
| `SB_EMBEDDING_MODEL` | same as the local `.env` |
| `SB_EMBEDDING_DIM` | same as the local `.env` (must match the model) |
| `SB_REDIS_URL` | `redis://<memorystore ip>:6379/0`. Optional: left out, recent chat is kept inside each instance (fine with one instance). |
| `SB_FALKORDB_URL` | `redis://<falkordb vm ip>:6379`. Optional: left out, only the "who knows what" graph is off (503). |
| `SB_GRAPH_PREFIX` | `sb` |
| `SB_DEMO_MODE` | `false` (already the image default) |
| `SB_APP_DATABASE_URL` | leave it out (one login). Set it only if you later want a separate login for the API. |

## 5. Create the schema (once, and after each release)

```bash
gcloud run jobs create sb-schema \
  --image REGION-docker.pkg.dev/PROJECT/secondbrain/api:shoppi --region REGION \
  --service-account secondbrain-run@PROJECT.iam.gserviceaccount.com \
  --network VPC --subnet SUBNET --vpc-egress private-ranges-only \
  --set-secrets SB_ADMIN_DATABASE_URL=sb-admin-db-url:latest \
  --command python --args=-m,secondbrain.schema
gcloud run jobs execute sb-schema --region REGION --wait
```

## 5b. Load the Shoppi merchants (once, on an empty database)

Creates the schema if needed, then the 4 Shoppi merchants (companies, owner logins, 400 days
of sales, stock, catalog documents), forecasts and signals. Safe to run again.

```bash
gcloud run jobs create sb-seed-shoppi \
  --image REGION-docker.pkg.dev/PROJECT/secondbrain/api:shoppi --region REGION \
  --network VPC --subnet SUBNET --vpc-egress private-ranges-only \
  --set-cloudsql-instances <instance connection name> \
  --set-secrets SB_ADMIN_DATABASE_URL=sb-admin-db-url:latest,SB_LITELLM_API_KEY=sb-litellm-key:latest \
  --set-env-vars SB_MODEL_PROVIDER=litellm,SB_LITELLM_BASE_URL=<as in .env>,SB_LLM_MODEL=<model>,SB_EMBEDDING_MODEL=<model>,SB_EMBEDDING_DIM=<dim> \
  --task-timeout 30m --memory 2Gi \
  --command python --args=-m,secondbrain.seed_shoppi
gcloud run jobs execute sb-seed-shoppi --region REGION --wait
```
Without LiteLLM reachable, add `--no-ai` to the args (skips documents and signals; run again later without it).

## 6. Deploy the API

```bash
gcloud run deploy secondbrain-api \
  --image REGION-docker.pkg.dev/PROJECT/secondbrain/api:shoppi --region REGION \
  --service-account secondbrain-run@PROJECT.iam.gserviceaccount.com \
  --network VPC --subnet SUBNET --vpc-egress private-ranges-only \
  --ingress internal-and-cloud-load-balancing --no-allow-unauthenticated \
  --cpu 2 --memory 2Gi --min-instances 1 --max-instances 5 \
  --set-env-vars SB_MODEL_PROVIDER=litellm,SB_LITELLM_BASE_URL=<as in .env>,SB_LLM_MODEL=<model>,SB_EMBEDDING_MODEL=<model>,SB_EMBEDDING_DIM=<dim>,SB_REDIS_URL=redis://<ip>:6379/0 \
  --set-secrets SB_ADMIN_DATABASE_URL=sb-admin-db-url:latest,SB_LITELLM_API_KEY=sb-litellm-key:latest,SB_ADMIN_TOKEN=sb-admin-token:latest
```

`--ingress internal-and-cloud-load-balancing` keeps it off the public internet: Shoppi and Media Planner reach it inside the VPC. Check:
```bash
curl -H "Authorization: Bearer $(gcloud auth print-identity-token)" https://<service-url>/health
```

## 7. Nightly jobs

Same image as jobs, with `SB_ADMIN_DATABASE_URL`, the LiteLLM settings and the same `--network`/`--subnet` flags as step 5:

| Job | Command | Schedule (Cloud Scheduler, IST) |
|---|---|---|
| `sb-forecast` | `python -m secondbrain.forecasting --under root` | 01:00 daily |
| `sb-signals` | `python -m secondbrain.signals` | 02:00 daily |
| `sb-relationships` | `python -m secondbrain.relationships` | 03:00 daily (only with FalkorDB) |

## 8. Onboard merchants and connect Shoppi

1. Create each merchant as a company (admin API):
   ```bash
   curl -X POST https://<service-url>/admin/companies -H "X-Admin-Token: <token>" -H 'Content-Type: application/json' \
     -d '{"id":"haldirams","name":"Haldiram'\''s","sharing":"patterns","departments":[{"id":"merchandising","name":"Merchandising"}]}'
   ```
2. Issue Shoppi an ingest key per merchant: `POST /admin/ingest/keys` (see `docs/INTEGRATION.md`).
3. Shoppi sends history once with `POST /ingest/{kind}/csv`, then live events with `POST /ingest/{kind}` (`product`, `sale`, `daily_sales`, `inventory`, `purchase_order`, `promotion`). Field definitions: `docs/INTEGRATION.md`.

## Not ready yet (before Shoppi / Media Planner *read* from it)

- `/signals` and `/forecast` identify the caller with an `X-User-Id` header and no password. That's fine inside a demo, **not** for other systems. A read API with per-merchant read keys is the next piece to build.
- The Live Commerce, HQ and TimescaleDB pages show browser-generated demo data when the simulator isn't running. Ignore them in production until they read real data.
