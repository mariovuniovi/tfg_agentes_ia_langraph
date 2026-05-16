# Containerization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the Streamlit Docker setup with a three-service compose (MLflow + FastAPI + Next.js) so the full app runs on any machine with `docker compose up --build`.

**Architecture:** `Dockerfile` is repurposed for FastAPI; a new `Dockerfile.frontend` builds Next.js in standalone mode. `docker-compose.yml` removes the old Streamlit service and adds `api` and `frontend`. Two `.dockerignore` files keep build contexts lean. `.env.example` and a new README section document setup for reproducers.

**Tech Stack:** Docker multi-stage builds, UV (Python 3.12), Node 20 Alpine, Next.js standalone output, docker compose.

---

## Files

| File | Action |
|---|---|
| `Dockerfile` | Modify — change EXPOSE + CMD (lines 37-39) |
| `Dockerfile.frontend` | Create — 3-stage Node 20 Alpine build |
| `docker-compose.yml` | Modify — remove `app` service, add `api` + `frontend` |
| `frontend/next.config.ts` | Modify — add `output: 'standalone'` |
| `.dockerignore` | Create |
| `frontend/.dockerignore` | Create |
| `.env.example` | Create |
| `README.md` | Modify — replace Quick start, add GitHub Models guide |

---

### Task 1: Repurpose Dockerfile for FastAPI

**Files:**
- Modify: `Dockerfile` (lines 37-39 only)

- [ ] **Step 1: Replace the EXPOSE and CMD lines**

Open `Dockerfile`. The current lines 37-39 are:
```dockerfile
USER app
EXPOSE 8501
CMD ["streamlit", "run", "dashboard/app.py", \
     "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true"]
```

Replace with:
```dockerfile
USER app
EXPOSE 8000
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

Everything above line 37 is unchanged.

- [ ] **Step 2: Verify the full Dockerfile looks correct**

The complete file should be:
```dockerfile
# =============================================================================
# Stage 1: Build — install deps with UV
# =============================================================================
FROM python:3.12-slim-bookworm AS build

COPY --from=ghcr.io/astral-sh/uv:0.10 /uv /uvx /bin/

WORKDIR /app

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never

# Layer 1: Install dependencies only (cached unless lock changes)
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --locked --no-install-project --no-dev

# Layer 2: Copy source and install the project itself
COPY . /app
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev --no-editable

# =============================================================================
# Stage 2: Runtime — minimal image without UV or build tools
# =============================================================================
FROM python:3.12-slim-bookworm AS runtime

RUN groupadd -g 1001 app && useradd -u 1001 -g app -m app
WORKDIR /app

COPY --from=build --chown=app:app /app /app
ENV PATH="/app/.venv/bin:$PATH"

USER app
EXPOSE 8000
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

- [ ] **Step 3: Commit**

```bash
git add Dockerfile
git commit -m "chore: repurpose Dockerfile for FastAPI backend"
```

---

### Task 2: Create .dockerignore files

**Files:**
- Create: `.dockerignore`
- Create: `frontend/.dockerignore`

These must be created before the Dockerfiles are tested, or Docker copies gigabytes of unwanted files into the build context.

- [ ] **Step 1: Create `.dockerignore` at the repo root**

```
# Python
.venv/
__pycache__/
*.pyc
*.pyo
.pytest_cache/
.mypy_cache/
.ruff_cache/

# Data & ML artifacts (mounted as volume at runtime)
data/
mlruns/
mlflow.db
evidently_workspace/
storage/
*.db

# Frontend (has its own build)
frontend/node_modules/
frontend/.next/

# Git & docs
.git/
docs/
*.md

# Secrets
.env
```

- [ ] **Step 2: Create `frontend/.dockerignore`**

```
node_modules/
.next/
.git/
```

- [ ] **Step 3: Commit**

```bash
git add .dockerignore frontend/.dockerignore
git commit -m "chore: add dockerignore files for api and frontend builds"
```

---

### Task 3: Enable Next.js standalone output

**Files:**
- Modify: `frontend/next.config.ts`

The Next.js standalone build mode bundles everything needed to run the server into `.next/standalone` — no `node_modules` required at runtime, which shrinks the final image significantly.

- [ ] **Step 1: Add `output: 'standalone'` to next.config.ts**

Replace the entire file with:
```typescript
import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  output: "standalone",
};

export default nextConfig;
```

- [ ] **Step 2: Verify the local build still works**

```bash
cd frontend && npm run build
```

Expected: build completes, `.next/standalone/` directory is created.

- [ ] **Step 3: Commit**

```bash
git add frontend/next.config.ts
git commit -m "chore: enable Next.js standalone output for Docker"
```

---

### Task 4: Create Dockerfile.frontend

**Files:**
- Create: `Dockerfile.frontend`

Three-stage build: install deps → build → minimal runtime. The build arg `NEXT_PUBLIC_API_URL` is baked into the static bundle at compile time (Next.js `NEXT_PUBLIC_*` vars are inlined at build, not read at runtime).

- [ ] **Step 1: Create `Dockerfile.frontend`**

```dockerfile
# =============================================================================
# Stage 1: Install dependencies
# =============================================================================
FROM node:20-alpine AS deps

WORKDIR /app
COPY package.json package-lock.json ./
RUN npm ci

# =============================================================================
# Stage 2: Build
# =============================================================================
FROM node:20-alpine AS builder

WORKDIR /app
COPY --from=deps /app/node_modules ./node_modules
COPY . .

# Baked into the static bundle — must be the URL the browser uses
ARG NEXT_PUBLIC_API_URL=http://localhost:8000
ENV NEXT_PUBLIC_API_URL=$NEXT_PUBLIC_API_URL

RUN npm run build

# =============================================================================
# Stage 3: Runtime — standalone output only, no node_modules
# =============================================================================
FROM node:20-alpine AS runtime

WORKDIR /app

RUN addgroup -g 1001 app && adduser -u 1001 -G app -s /bin/sh -D app

COPY --from=builder --chown=app:app /app/.next/standalone ./
COPY --from=builder --chown=app:app /app/.next/static ./.next/static
COPY --from=builder --chown=app:app /app/public ./public

USER app
EXPOSE 3000
ENV PORT=3000
ENV HOSTNAME=0.0.0.0

CMD ["node", "server.js"]
```

- [ ] **Step 2: Commit**

```bash
git add Dockerfile.frontend
git commit -m "chore: add Next.js multi-stage Dockerfile"
```

---

### Task 5: Update docker-compose.yml

**Files:**
- Modify: `docker-compose.yml`

Remove the old Streamlit `app` service. Add `api` (FastAPI) and `frontend` (Next.js). The `mlflow` service, named volume, and network are unchanged.

- [ ] **Step 1: Replace docker-compose.yml with the new version**

```yaml
services:
  # ===========================================================================
  # MLflow Tracking Server (SQLite backend)
  # Access at: http://localhost:5000
  # ===========================================================================
  mlflow:
    image: ghcr.io/mlflow/mlflow:v2.18.0
    container_name: mlops-mlflow
    restart: unless-stopped
    command: >
      mlflow server
        --host 0.0.0.0
        --port 5000
        --backend-store-uri sqlite:///mlflow/mlflow.db
        --artifacts-destination /mlflow/artifacts
        --serve-artifacts
    volumes:
      - mlflow_data:/mlflow
    ports:
      - "5000:5000"
    networks:
      - mlops-net

  # ===========================================================================
  # FastAPI Backend + LangGraph Agents
  # Access at: http://localhost:8000
  # ===========================================================================
  api:
    build:
      context: .
      dockerfile: Dockerfile
    container_name: mlops-api
    restart: unless-stopped
    depends_on:
      - mlflow
    env_file:
      - .env
    environment:
      MLFLOW_TRACKING_URI: http://mlflow:5000
      PYTHONUNBUFFERED: "1"
    ports:
      - "8000:8000"
    volumes:
      - ./data:/app/data
    networks:
      - mlops-net

  # ===========================================================================
  # Next.js Frontend
  # Access at: http://localhost:3000
  # ===========================================================================
  frontend:
    build:
      context: ./frontend
      dockerfile: Dockerfile.frontend
      args:
        NEXT_PUBLIC_API_URL: http://localhost:8000
    container_name: mlops-frontend
    restart: unless-stopped
    ports:
      - "3000:3000"
    networks:
      - mlops-net

volumes:
  mlflow_data:

networks:
  mlops-net:
    driver: bridge
```

- [ ] **Step 2: Commit**

```bash
git add docker-compose.yml
git commit -m "chore: replace Streamlit service with FastAPI + Next.js in docker-compose"
```

---

### Task 6: Create .env.example

**Files:**
- Create: `.env.example`

This file is committed to the repo. It documents every required variable so someone cloning the repo knows exactly what to fill in.

- [ ] **Step 1: Create `.env.example`**

```bash
# ============================================================
# LLM — OpenAI API (paid)
# Get your key at: https://platform.openai.com/api-keys
# ============================================================
OPENAI_API_KEY=your_openai_api_key_here

# Per-agent model overrides — adjust to your OpenAI plan
OPENAI_MODEL_SUPERVISOR=gpt-4.1-mini
OPENAI_MODEL_DATA_VALIDATOR=gpt-4.1-mini
OPENAI_MODEL_PLANNER=gpt-4.1-mini
OPENAI_MODEL_EVALUATOR=gpt-4.1-mini
OPENAI_MODEL_DEPLOYER=gpt-4.1-mini

# ============================================================
# FREE ALTERNATIVE: GitHub Models
# See README.md → "Using GitHub Models (free)" for full guide
# Requires changing OPENAI_API_KEY + adding OPENAI_BASE_URL
# ============================================================

# ============================================================
# MLflow (do not change when running via docker compose)
# ============================================================
MLFLOW_TRACKING_URI=http://localhost:5000

# ============================================================
# Optional: LangSmith tracing
# ============================================================
LANGCHAIN_TRACING_V2=false
# LANGCHAIN_API_KEY=your_langsmith_key_here
# LANGCHAIN_PROJECT=mlops-multi-agent
```

- [ ] **Step 2: Verify .env is in .gitignore**

```bash
grep "^\.env$" .gitignore
```

Expected: `.env` appears. If not, add it:
```bash
echo ".env" >> .gitignore
git add .gitignore
```

- [ ] **Step 3: Commit**

```bash
git add .env.example
git commit -m "chore: add .env.example with OpenAI defaults and GitHub Models note"
```

---

### Task 7: Update README.md

**Files:**
- Modify: `README.md`

Replace the existing "Quick start" section with a Docker-first "Getting Started" section. Keep all other sections (Status, Tech stack, Tests, Project structure, etc.) unchanged.

- [ ] **Step 1: Replace the "Quick start" section**

Find this block in `README.md`:
```markdown
## Quick start
...
## Seeding the experience pool
```

Replace everything from `## Quick start` up to (but not including) `## Seeding the experience pool` with:

```markdown
## Getting started

**Prerequisites:** [Docker Desktop](https://www.docker.com/products/docker-desktop/) — no Python, Node, or any other tool needed.

```bash
git clone https://github.com/mariovuniovi/tfg_agentes_ia_langraph.git
cd tfg_agentes_ia_langraph
cp .env.example .env          # then edit .env and fill in OPENAI_API_KEY
docker compose up --build
```

Open **http://localhost:3000** — the pipeline UI is ready.

| Service | URL |
|---|---|
| Next.js UI | http://localhost:3000 |
| FastAPI backend | http://localhost:8000 |
| MLflow | http://localhost:5000 |

**After making code changes:**
```bash
docker compose up --build     # rebuilds only what changed, takes ~1-2 min
```

---

### Using GitHub Models (free alternative)

> **Note on model quality:** This project was developed and tested against OpenAI's API. GitHub Models offers compatible model names but is a different hosted service with a **150 requests/day rate limit** on the free tier. Agent reliability — especially the data validator and planner, which require precise instruction-following — may be lower than with OpenAI's API. Use GitHub Models to explore the system; use OpenAI's API for consistent results.

1. Create a free [GitHub Personal Access Token](https://github.com/settings/tokens) (no special scopes needed)
2. In your `.env`, make these changes:
   ```bash
   OPENAI_API_KEY=your_github_personal_access_token
   OPENAI_BASE_URL=https://models.inference.ai.azure.com
   OPENAI_MODEL_SUPERVISOR=openai/gpt-4.1-mini
   OPENAI_MODEL_DATA_VALIDATOR=openai/gpt-4.1-mini
   OPENAI_MODEL_PLANNER=openai/gpt-4.1-mini
   OPENAI_MODEL_EVALUATOR=openai/gpt-4.1-mini
   OPENAI_MODEL_DEPLOYER=openai/gpt-4.1-mini
   ```
3. Run `docker compose up --build` — no code changes needed. The OpenAI SDK reads `OPENAI_BASE_URL` automatically.

---

### Local development (without Docker)

```bash
uv sync                                    # install all Python dependencies
cp .env.example .env                       # fill in OPENAI_API_KEY
```

Two terminals:
```bash
# Terminal 1 — FastAPI backend (port 8000)
uv run uvicorn api.main:app --reload --port 8000

# Terminal 2 — Next.js frontend (port 3000)
cd frontend && npm install && npm run dev
```

```

- [ ] **Step 2: Update the Tech stack section**

Find the line:
```markdown
- **GitHub Models** (gpt-4.1-mini) — free LLM tier
```

Replace with:
```markdown
- **OpenAI API** (gpt-4.1-mini) — LLM provider (GitHub Models free tier also supported, see Getting started)
```

- [ ] **Step 3: Commit**

```bash
git add README.md
git commit -m "docs: add Docker getting-started guide and GitHub Models alternative"
```

---

### Task 8: Smoke test — verify full stack starts

No automated test exists for Docker compose. Verify manually.

- [ ] **Step 1: Build and start all services**

```bash
docker compose up --build
```

Expected output (order may vary):
```
mlops-mlflow  | [INFO] Uvicorn running on http://0.0.0.0:5000
mlops-api     | INFO:     Uvicorn running on http://0.0.0.0:8000
mlops-frontend | ready - started server on 0.0.0.0:3000
```

All three lines must appear before continuing.

- [ ] **Step 2: Verify each service responds**

```bash
curl http://localhost:5000/health          # MLflow — expect {"status":"OK"}
curl http://localhost:8000/docs            # FastAPI — expect HTML (Swagger UI)
curl http://localhost:3000                 # Next.js — expect HTML
```

- [ ] **Step 3: Verify the UI loads and can reach the API**

Open `http://localhost:3000` in a browser. The pipeline page should load without console errors about failed API calls to `localhost:8000`.

- [ ] **Step 4: Stop and clean up**

```bash
docker compose down
```

- [ ] **Step 5: Final commit + push**

```bash
git push origin claude/develop
```
