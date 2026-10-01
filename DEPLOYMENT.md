# Deployment

Production setup, running on free tiers:

| Component | Host | Notes |
|---|---|---|
| React UI (`rag-ui/`) | **Vercel** | Static Vite build, `rag-ui/vercel.json` |
| FastAPI backend | **Render** (free, no card) *or* **Google Cloud Run** (needs billing) | see "Two backend options" below |
| PostgreSQL | **Supabase** | Tables are created automatically on first use |
| Vector store | **Qdrant Cloud** | Free 1 GB cluster |
| Redis (optional) | Upstash | Only needed for multi-worker memory |

Ollama isn't needed on these hosts. Answer generation and scanned-PDF/image
OCR both use Groq first (`GROQ_API_KEY`, or the per-task `GROQ_*_API_KEY`
overrides) and only fall back to Ollama when no key is set.

### Two backend options

| | **3a. Render free** (lightweight) | **3b. Google Cloud Run** (full) |
|---|---|---|
| Card / billing account | Not needed | Needed |
| Image | `Dockerfile.render` + `requirements-render.txt` | `Dockerfile` + `requirements.txt` |
| BGE-M3 embeddings | Hugging Face Inference API (`EMBEDDING_BACKEND=hf_api`) | Loaded in-process (BGE-M3 baked into the image) |
| Reranker | Jina API (`JINA_API_KEY`) | Jina API, or the local cross-encoder |
| Memory | ~260 MB (fits 512 MB) | 4 GiB |
| Idle behaviour | Sleeps after 15 min; ~50 s to wake | Scales to zero; ~30–60 s to start |

Both use the same code: `file_preparation/embedding/backend.py` picks
`embedder.py` (local) or `remote_embedder.py` (API) from `EMBEDDING_BACKEND`.
Dense vectors come from the same BGE-M3 model either way, so moving between
the two later doesn't require re-indexing documents.

---

## 1. PostgreSQL — Supabase

1. Create a project at <https://supabase.com/dashboard> and set a database password.
2. **Connect → Session pooler** (IPv4-compatible) gives you:
   - `PG_HOST` = `aws-0-<region>.pooler.supabase.com` (copy it exactly)
   - `PG_PORT` = `5432`
   - `PG_DB` = `postgres`
   - `PG_USER` = `postgres.<project-ref>`
   - `PG_PASSWORD` = the database password

## 2. Vector store — Qdrant Cloud

1. Create a free cluster at <https://cloud.qdrant.io>.
2. `QDRANT_URL` = the cluster endpoint **with `:6333`**, e.g.
   `https://xxxx.eu-central-1-0.aws.cloud.qdrant.io:6333`.
3. Create an API key → `QDRANT_API_KEY`.
4. To start from an empty store (every user then uploads their own documents),
   with `QDRANT_URL` / `QDRANT_API_KEY` in a local `.env`:
   ```bash
   python scripts/reset_qdrant.py            # dry run: lists collections and counts
   python scripts/reset_qdrant.py --confirm  # deletes them
   ```
   Collections are recreated automatically on the next upload. Users only see
   their own uploads plus documents an admin uploaded to the shared knowledge base.

## 3a. Backend — Render (free, no card)

1. **Hugging Face token** (embeddings): <https://huggingface.co/settings/tokens>
   → *Create new token* → *Fine-grained* → tick **"Make calls to Inference
   Providers"** → copy it (`HF_TOKEN`). The free monthly allowance covers light
   demo use; if it runs out, uploads and questions fail with an embedding error
   until the next month.
2. **Jina API key** (reranker): <https://jina.ai/> → *API* → copy the free key
   (`JINA_API_KEY`). Without it the backend skips reranking; it would not have
   the memory to load the local reranker.
3. **Render**: sign up at <https://render.com> with GitHub → **New → Blueprint**
   → pick this repo (and the branch to deploy). Render reads `render.yaml` and
   asks for each secret: `HF_TOKEN`, `JINA_API_KEY`, `GROQ_API_KEY`, the Supabase
   `PG_HOST` / `PG_USER` / `PG_PASSWORD`, `QDRANT_URL` / `QDRANT_API_KEY`, and
   `FRONTEND_URL` (put your Vercel URL here once step 4 is done).
   `JWT_SECRET` is generated automatically.
4. Groq retires models over time. `…/health` lists any configured model your key
   can't use under `groq.unavailable_models`, together with `available_models`;
   set `GROQ_MEMORY_MODEL` (query rewriting / memory) and `JUDGE_MODEL` to one of
   the available ids in the Render environment.
5. The first build takes ~5–10 minutes. The service URL looks like
   `https://support-api-xxxx.onrender.com`; check `…/health`. Ollama showing
   "down" there is expected. Every push to the deployed branch redeploys.

Free-plan limits: 512 MB RAM and a small CPU share. Very large uploads (e.g.
PDFs with hundreds of pages) may be slow or run out of memory. Uploaded files
live on temporary disk; documents and chats persist in Qdrant and Postgres.

## 3b. Backend — Google Cloud Run (needs billing)

### One-time Google Cloud setup

1. Go to <https://console.cloud.google.com>, create a project (e.g. `support-ai`),
   and note its **Project ID**. Cloud Run needs a billing account linked to the
   project even when usage stays within the free tier. Consider adding a
   budget alert under **Billing → Budgets & alerts**, e.g. at $5.
2. Enable the APIs for the project (you can do this in **Cloud Shell**, the `>_` icon at the top right):
   ```bash
   gcloud services enable run.googleapis.com cloudbuild.googleapis.com artifactregistry.googleapis.com
   ```
3. Create a deploy service account and a key for GitHub Actions:
   ```bash
   PROJECT_ID=$(gcloud config get-value project)
   gcloud iam service-accounts create github-deployer --display-name "GitHub deployer"
   SA=github-deployer@$PROJECT_ID.iam.gserviceaccount.com
   for role in run.admin cloudbuild.builds.editor artifactregistry.admin \
               iam.serviceAccountUser storage.admin serviceusage.serviceUsageConsumer; do
     gcloud projects add-iam-policy-binding $PROJECT_ID --member "serviceAccount:$SA" --role "roles/$role" --condition=None -q >/dev/null
   done
   # Cloud Build's default runtime account also needs to push images and deploy
   PROJECT_NUMBER=$(gcloud projects describe $PROJECT_ID --format 'value(projectNumber)')
   for role in run.admin iam.serviceAccountUser artifactregistry.writer storage.admin logging.logWriter; do
     gcloud projects add-iam-policy-binding $PROJECT_ID --member "serviceAccount:$PROJECT_NUMBER-compute@developer.gserviceaccount.com" --role "roles/$role" --condition=None -q >/dev/null
   done
   gcloud iam service-accounts keys create key.json --iam-account $SA
   cat key.json   # copy the whole JSON, then: rm key.json
   ```

### GitHub settings

Repo → **Settings → Secrets and variables → Actions**:

- **Secret** `GCP_SA_KEY` = the full contents of `key.json`
- **Variable** `GCP_PROJECT_ID` = your project ID
- Optional **variables**: `GCP_REGION` (default `europe-west1`, near Supabase
  Frankfurt) and `CLOUD_RUN_SERVICE` (default `support-api`)

Each push to `main` (or **Actions → Deploy backend to Cloud Run → Run workflow**)
builds the Docker image with Cloud Build and deploys it. The first build takes
about 15–25 minutes, because it installs PyTorch and bakes in the ~2.3 GB
BGE-M3 model. The job log ends with the service URL, e.g.
`https://support-api-xxxxx-ew.a.run.app`.

### Environment variables

After the first deploy, go to **Cloud Run → support-api → Edit & deploy new
revision → Variables & secrets** and add everything you use from
`.env.example`. At minimum:

`GROQ_API_KEY`, `JWT_SECRET`, `PG_HOST`, `PG_PORT`, `PG_DB`, `PG_USER`,
`PG_PASSWORD`, `QDRANT_URL`, `QDRANT_API_KEY`, `FRONTEND_URL` (your Vercel URL),
`CORS_ORIGIN_REGEX` = `https://.*\.vercel\.app`, and optionally `JINA_API_KEY`
(uses hosted reranking instead of loading the local model, which saves RAM).

Later deploys from GitHub keep these variables. Check `https://<service-url>/health`
afterwards; Ollama reporting "down" there is expected.

### Service settings and cost

The workflow deploys with 2 vCPU / 4 GiB, `min-instances 0` (it scales to zero
when idle, so the first request after a pause waits ~30–60 s while it starts),
`max-instances 1`, and CPU kept on after responses (`--no-cpu-throttling`) so
background document indexing finishes. Light demo use stays within Cloud Run's
monthly free allowance. Heavy use is billed per second while the instance is
running; see <https://cloud.google.com/run/pricing>. The image (~4 GB) in
Artifact Registry costs a few cents a month beyond the free 0.5 GB.

Uploaded files and extracted images live on the container's temporary disk and
are lost when it scales down. Durable data lives in Postgres and Qdrant.

## 4. Frontend — Vercel

1. <https://vercel.com/new> → import this GitHub repo.
2. **Root Directory**: `rag-ui` (Vite is auto-detected).
3. Environment variable `VITE_API_URL` = your backend URL (Render or Cloud Run, no trailing slash).
4. Deploy, then set the resulting URL as `FRONTEND_URL` on the backend service.

## 5. First admin user

With the backend's env vars in a local `.env`:

```bash
python scripts/create_admin.py
```

## 6. Google OAuth (optional)

Add `https://<service-url>/auth/google/callback` as an authorized redirect URI in
Google Cloud Console, then set `GOOGLE_CLIENT_ID`, `GOOGLE_CLIENT_SECRET` and
`GOOGLE_REDIRECT_URI` on the backend service.

---

## Running the backend image anywhere else

```bash
docker build -t support-api .
docker run -p 8080:8080 --env-file .env support-api
```

Any container host with about 4 GB of RAM works. Set `PORT` if the platform expects
a different port. For small hosts, use the lightweight image instead:

```bash
docker build -f Dockerfile.render -t support-api-lite .
docker run -p 10000:10000 --env-file .env support-api-lite   # needs HF_TOKEN, JINA_API_KEY
```
