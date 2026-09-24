# Deployment

Free-tier production setup:

| Component | Host | Notes |
|---|---|---|
| React UI (`rag-ui/`) | **Vercel** | Static Vite build, `rag-ui/vercel.json` |
| FastAPI backend | **Hugging Face Docker Space** | `Dockerfile`, free CPU tier (16 GB RAM, 2 vCPU) |
| PostgreSQL | **Supabase** | Tables are created automatically on first use |
| Vector store | **Qdrant Cloud** | Free 1 GB cluster |
| Redis (optional) | Upstash | Only needed for multi-worker memory |

Ollama isn't needed on these hosts. Answer generation and scanned-PDF/image
OCR both use Groq first (`GROQ_API_KEY`, or the per-task `GROQ_*_API_KEY`
overrides) and only fall back to Ollama when no key is set.

---

## 1. PostgreSQL — Supabase

1. Create a project at <https://supabase.com/dashboard>.
2. **Connect → Session pooler** gives you the values below (IPv4-compatible,
   which Hugging Face needs):
   - `PG_HOST` = `aws-0-<region>.pooler.supabase.com`
   - `PG_PORT` = `5432`
   - `PG_DB` = `postgres`
   - `PG_USER` = `postgres.<project-ref>`
   - `PG_PASSWORD` = the database password you set

## 2. Vector store — Qdrant Cloud

1. Create a free cluster at <https://cloud.qdrant.io>.
2. Copy the cluster URL (`https://xxxx.<region>.cloud.qdrant.io:6333`) → `QDRANT_URL`
   and create an API key → `QDRANT_API_KEY`.

## 3. Backend — Hugging Face Space

1. Create a Space at <https://huggingface.co/new-space>: SDK **Docker**,
   template **Blank**, hardware **CPU basic (free)**. For example `MA29/customer-support-api`.
2. In the Space's **Settings → Variables and secrets**, add as **secrets**
   everything from `.env.example` you use; at minimum:
   `GROQ_API_KEY`, `JWT_SECRET`, `PG_*`, `QDRANT_URL`, `QDRANT_API_KEY`,
   `FRONTEND_URL` (your Vercel URL, set after step 4), and optionally
   `JINA_API_KEY` (uses hosted reranking instead of downloading the local
   model, which saves RAM and cold-start time).
3. Automatic deploys from GitHub:
   - Create a **write** token at <https://huggingface.co/settings/tokens>.
   - GitHub repo → **Settings → Secrets and variables → Actions**:
     - Secret `HF_TOKEN` = that token
     - Variable `HF_SPACE` = `MA29/customer-support-api`
   - Each push to `main` (or **Actions → Deploy backend to Hugging Face Space → Run workflow**)
     pushes the backend to the Space, and the Space then rebuilds the Docker image.
4. The API is served at `https://<user>-<space-name>.hf.space`
   (e.g. `https://ma29-customer-support-api.hf.space`). Check
   `…/health` once the build finishes. The first request that embeds
   text downloads BGE-M3 (~2 GB), so it will be slow.

Free Spaces sleep after 48 h without traffic and their disk is ephemeral:
uploaded files, extracted images and the model cache are lost on restart.
Durable data lives in Postgres and Qdrant.

## 4. Frontend — Vercel

1. <https://vercel.com/new> → import this GitHub repo.
2. **Root Directory**: `rag-ui` (Vite is auto-detected).
3. Environment variable `VITE_API_URL` = your Space URL (no trailing slash).
4. Deploy, then put the resulting URL (e.g. `https://customer-support-ai-assistant.vercel.app`)
   into the Space secret `FRONTEND_URL`. To allow preview deployments too, set
   `CORS_ORIGIN_REGEX` = `https://.*\.vercel\.app`.

## 5. First admin user

With the backend's env vars loaded locally (or from the Space's terminal):

```bash
python scripts/create_admin.py
```

## 6. Google OAuth (optional)

Add `https://<space-url>/auth/google/callback` as an authorized redirect URI in
Google Cloud Console, then set `GOOGLE_CLIENT_ID`, `GOOGLE_CLIENT_SECRET` and
`GOOGLE_REDIRECT_URI` on the Space.

---

## Running the backend image anywhere else

```bash
docker build -t support-api .
docker run -p 7860:7860 --env-file .env support-api
```

Any container host with about 4 GB+ of RAM works (Render, Railway, Fly.io, a VPS).
Set `PORT` if the platform expects a different port.
