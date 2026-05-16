# Containerization Design

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Containerize the full app (FastAPI backend, Next.js frontend, MLflow) so it runs on any machine with a single `docker compose up --build`, with no local Python or Node installation required.

**Architecture:** Three services in docker-compose — `mlflow` (existing, unchanged), `api` (FastAPI + LangGraph agents, built from the existing `Dockerfile`), `frontend` (Next.js, built from a new `Dockerfile.frontend`). The browser talks to both `api` (port 8000) and `frontend` (port 3000) via localhost. Services talk to each other via Docker's internal network.

**Tech stack:** Docker multi-stage builds, UV (Python), Node 20 Alpine (Next.js standalone output), docker compose watch for optional live sync.

---

## Services

| Service | Build | Exposed port | Internal name |
|---|---|---|---|
| `mlflow` | `ghcr.io/mlflow/mlflow:v2.18.0` (unchanged) | `5000` | `mlflow` |
| `api` | `Dockerfile` (repurposed from Streamlit) | `8000` | `api` |
| `frontend` | `Dockerfile.frontend` (new) | `3000` | `frontend` |

**Startup order:** `api` declares `depends_on: mlflow`. `frontend` has no dependency — the browser connects to `api` directly, not the frontend container.

**Volumes:**
- `./data:/app/data` mounted into `api` — agents read/write datasets and trained models here
- `mlflow_data` named volume — MLflow SQLite + artifacts persist across restarts

---

## Dockerfile (FastAPI backend)

The existing multi-stage UV build is kept as-is. Only the final two lines change:

```dockerfile
EXPOSE 8000
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

The non-root user, slim runtime image, and UV layer caching are unchanged.

---

## Dockerfile.frontend (Next.js — new)

Three-stage build:

1. **deps** (`node:20-alpine`) — install `node_modules` from `package-lock.json` only (cached unless lock changes)
2. **builder** (`node:20-alpine`) — copy source, set `NEXT_PUBLIC_API_URL=http://localhost:8000` as build arg, run `next build`
3. **runtime** (`node:20-alpine`) — copy only `.next/standalone` + `.next/static` + `public/`; run `node server.js`

The standalone output bundles everything needed to run Next.js with no `node_modules` at runtime, producing a significantly smaller final image.

**Requires:** `output: 'standalone'` added to `frontend/next.config.ts`.

**Why `http://localhost:8000` for `NEXT_PUBLIC_API_URL`:** this variable is baked into the static bundle at build time and used by the browser (which is outside Docker). The browser reaches the API at `localhost:8000`, not via Docker's internal network.

---

## docker-compose.yml

The old `app` (Streamlit) service is removed and replaced:

```yaml
api:
  build:
    context: .
    dockerfile: Dockerfile
  ports: ["8000:8000"]
  depends_on: [mlflow]
  env_file: .env
  environment:
    MLFLOW_TRACKING_URI: http://mlflow:5000
  volumes:
    - ./data:/app/data

frontend:
  build:
    context: ./frontend
    dockerfile: Dockerfile.frontend
  ports: ["3000:3000"]
  environment:
    NEXT_PUBLIC_API_URL: http://localhost:8000
```

The `mlflow` service and `mlops-net` network are unchanged.

---

## .dockerignore files

**`.dockerignore`** (repo root — covers `Dockerfile`/`api` build):
- Excludes: `.venv/`, `__pycache__/`, `*.pyc`, `data/`, `mlruns/`, `frontend/node_modules/`, `.git/`, `*.db`

**`frontend/.dockerignore`** (covers `Dockerfile.frontend` build):
- Excludes: `node_modules/`, `.next/`, `.git/`

---

## Environment & Setup

**`.env.example`** (new — committed to repo):

```bash
# ============================================================
# LLM — OpenAI API
# Get your key at: https://platform.openai.com/api-keys
# ============================================================
OPENAI_API_KEY=your_openai_api_key_here

# Per-agent model overrides (adjust to your OpenAI plan)
OPENAI_MODEL_SUPERVISOR=gpt-4.1-mini
OPENAI_MODEL_DATA_VALIDATOR=gpt-4.1-mini
OPENAI_MODEL_PLANNER=gpt-4.1-mini
OPENAI_MODEL_EVALUATOR=gpt-4.1-mini
OPENAI_MODEL_DEPLOYER=gpt-4.1-mini

# ============================================================
# FREE ALTERNATIVE: GitHub Models
# See README.md → "Using GitHub Models" for migration guide
# ============================================================
```

**`README.md`** gets a new "Getting Started" section:

1. Prerequisites: Docker Desktop only (no Python/Node needed)
2. Clone repo
3. Copy `.env.example` → `.env`, fill in `OPENAI_API_KEY`
4. `docker compose up --build`
5. Open `http://localhost:3000`
6. To update after code changes: `docker compose up --build`

Plus a **"Using GitHub Models (free alternative)"** subsection:
1. Create a GitHub Personal Access Token (free, github.com/settings/tokens)
2. In `.env`: set `OPENAI_API_KEY=<github_pat>`, add `OPENAI_BASE_URL=https://models.inference.ai.azure.com`
3. Change model names to `openai/gpt-4.1-mini` format
4. No code changes needed — the OpenAI SDK reads `OPENAI_BASE_URL` automatically

---

## Files Changed

| File | Action |
|---|---|
| `Dockerfile` | Change last 2 lines: `EXPOSE 8000` + uvicorn CMD |
| `Dockerfile.frontend` | Create — 3-stage Node 20 Alpine build |
| `docker-compose.yml` | Remove Streamlit `app`; add `api` + `frontend` services |
| `frontend/next.config.ts` | Add `output: 'standalone'` |
| `.dockerignore` | Create |
| `frontend/.dockerignore` | Create |
| `.env.example` | Create |
| `README.md` | Add "Getting Started" + GitHub Models guide sections |
