# SecondBrain on GCP: what to do, in order

Who: **Srinivas** runs every step. After steps 3, 4 and 5, send the result to Santh, who checks it.
Where: Cloud Shell, project `di-ai-lab`, service `secondbrain`, region `us-east4`.
Already done: Cloud SQL role setup (`secondbrain_app`), first deploy.

---

## Step 1. Get the latest code

```bash
cd ~/secondbrain && git pull origin shoppi && git log --oneline -1
```
✅ Done when the last line starts with `99cc8a7` (or newer).

## Step 2. Build and push the image

```bash
docker build -t gcr.io/di-ai-lab/second-brain:shoppi . && docker push gcr.io/di-ai-lab/second-brain:shoppi
```
✅ Done when the push ends with a `digest: sha256:...` line.

## Step 3. Deploy the new image

Cloud Run → `secondbrain` → **Edit & deploy new revision**:

| Setting | Value |
|---|---|
| Container image | `gcr.io/di-ai-lab/second-brain:shoppi` (pick the newest digest) |
| Variables | keep the existing ones, and make sure these exist: `SB_ADMIN_DATABASE_URL` (starts with `postgresql://`), `SB_LITELLM_API_KEY`, `SB_LITELLM_BASE_URL`, `SB_MODEL_PROVIDER=litellm`, `SB_LLM_MODEL=claude-sonnet-4-5`, `SB_EMBEDDING_MODEL=gemini-embedding-001`, `SB_EMBEDDING_DIM=1536`, `SB_DEMO_MODE=true`, `SB_ADMIN_TOKEN=<a long random value>` |
| Cloud SQL connections | `di-lab-dev-pg-sql-01` |
| Networking | the same VPC as now (needed for LiteLLM) |

Click **Deploy**.
✅ Done when `https://secondbrain-610993990979.us-east4.run.app/health` shows `"status":"ok"`.

## Step 4. Load the Shoppi data (once)

Cloud Run → **Jobs** → **Create job**:

| Setting | Value |
|---|---|
| Name | `sb-seed-shoppi` |
| Container image | the same image as step 3 |
| Container command | `python` |
| Container arguments | `-m` and `secondbrain.seed_shoppi` (two separate arguments) |
| Variables | the same as the service in step 3 |
| Cloud SQL connections | `di-lab-dev-pg-sql-01` |
| Networking | the same VPC as the service |
| Task timeout | 30 minutes; memory 2 GiB |

Click **Create**, then **Execute**. It takes a few minutes.
✅ Done when the job's log ends with:
```
root.bata: 8 open signals
root.bigbasket: 20 open signals
root.fabindia: 42 open signals
root.haldirams: 18 open signals
Done.
```
(Counts can differ slightly with the real model.) Then open the site: **Chat** lists the 4 owners, **Explore** shows signals.
❌ If it fails: send the last 20 log lines to Santh.

## Step 5. Create the keys for Shoppi and Media Planner

Use the `SB_ADMIN_TOKEN` value from step 3:

```bash
curl -X POST https://secondbrain-610993990979.us-east4.run.app/admin/platform-keys -H "X-Admin-Token: <admin token>" -H 'Content-Type: application/json' -d '{"platform":"Shoppi","companies":["haldirams","bigbasket","fabindia","bata"]}'
```
```bash
curl -X POST https://secondbrain-610993990979.us-east4.run.app/admin/platform-keys -H "X-Admin-Token: <admin token>" -H 'Content-Type: application/json' -d '{"platform":"Media Planner","companies":["haldirams","bigbasket","fabindia","bata"]}'
```
Each answer contains `"key": "sbp_..."`. It is shown **only once**: save it in Secret Manager.

Check one key:
```bash
curl https://secondbrain-610993990979.us-east4.run.app/signals -H "Authorization: Bearer sbp_..."
```
✅ Done when it returns a list of signals.

## Step 6. Hand over to the Shoppi and Media Planner teams

Give each team its own key and this:

| Call (header `Authorization: Bearer sbp_...`) | Returns |
|---|---|
| `GET /signals` | open decision signals for its merchants |
| `GET /forecast/products` | its merchants' products |
| `GET /forecast?product=<SKU>&horizon=28` | the daily forecast for one product |
| `POST /signals/<id>/feedback` with `{"status":"acted"}` | tells SecondBrain a signal was acted on |

Base URL: the service URL (or its internal address, after step 7).

## Step 7. Close it to the internet

Cloud Run → `secondbrain` → **Networking** → Ingress: **Internal**. Save.
After this, only systems inside the VPC (Shoppi, Media Planner) can reach it; open the web pages over the VPN.
✅ Done when the URL no longer opens from a normal browser off-VPN, and Shoppi's call from step 5 still works from inside the VPC.
