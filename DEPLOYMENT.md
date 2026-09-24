# Deployment

Production setup, running on free tiers:

| Component | Host | Notes |
|---|---|---|
| React UI (`rag-ui/`) | **Vercel** | Static Vite build, `rag-ui/vercel.json` |
| FastAPI backend | **Google Cloud Run** | `Dockerfile`, deployed by `.github/workflows/deploy-cloud-run.yml` |
| PostgreSQL | **Supabase** | Tables are created automatically on first use |
| Vector store | **Qdrant Cloud** | Free 1 GB cluster |
| Redis (optional) | Upstash | Only needed for multi-worker memory |

Ollama isn't needed on these hosts. Answer generation and scanned-PDF/image
OCR both use Groq first (`GROQ_API_KEY`, or the per-task `GROQ_*_API_KEY`
overrides) and only fall back to Ollama when no key is set.

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

## 3. Backend — Google Cloud Run

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
3. Environment variable `VITE_API_URL` = your Cloud Run URL (no trailing slash).
4. Deploy, then set the resulting URL as `FRONTEND_URL` on the Cloud Run service.

## 5. First admin user

With the backend's env vars in a local `.env`:

```bash
python scripts/create_admin.py
```

## 6. Google OAuth (optional)

Add `https://<service-url>/auth/google/callback` as an authorized redirect URI in
Google Cloud Console, then set `GOOGLE_CLIENT_ID`, `GOOGLE_CLIENT_SECRET` and
`GOOGLE_REDIRECT_URI` on the Cloud Run service.

---

## Running the backend image anywhere else

```bash
docker build -t support-api .
docker run -p 8080:8080 --env-file .env support-api
```

Any container host with about 4 GB of RAM works. Set `PORT` if the platform expects
a different port.
