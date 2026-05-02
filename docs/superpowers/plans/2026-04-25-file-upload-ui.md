# File Upload UI Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace text-input-based file selection on the Pipeline and Monitoring pages with a button-driven file picker (solid navy button + file rows), and add sonner toast notifications for success/error feedback.

**Architecture:** New `POST /uploads` FastAPI endpoint stores uploaded files to `data/uploads/` and returns server paths; `TriggerPanel` calls it before starting a run; `AdHocForm` keeps sending files directly to the existing drift endpoint. Both components share the same visual upload pattern. Toast notifications are wired app-wide via sonner's `<Toaster />` in the root layout.

**Tech Stack:** FastAPI (Python), Next.js 16 App Router, React 19, sonner (toast), Tailwind CSS 4, Vitest, pytest-asyncio + httpx

**Working directory:** All paths below are relative to `.worktrees/nextjs-frontend/` unless prefixed with `frontend/`.

---

## File Map

| File | Action | Responsibility |
|---|---|---|
| `api/routers/uploads.py` | Create | `POST /uploads` — store files, return paths |
| `api/tests/test_uploads.py` | Create | Unit tests for uploads endpoint |
| `api/main.py` | Modify | Register uploads router |
| `frontend/package.json` | Modify | Add `sonner` dependency |
| `frontend/app/layout.tsx` | Modify | Add `<Toaster />` to app root |
| `frontend/lib/api.ts` | Modify | Add `uploadFiles()` function |
| `frontend/components/pipeline/TriggerPanel.tsx` | Modify | Replace text input with file upload UI |
| `frontend/components/monitoring/AdHocForm.tsx` | Modify | Replace dashed drop zone with button + file rows |

---

## Task 1: `POST /uploads` backend endpoint

**Files:**
- Create: `api/routers/uploads.py`
- Create: `api/tests/test_uploads.py`
- Modify: `api/main.py`

- [ ] **Step 1: Write the failing tests**

Create `api/tests/test_uploads.py`:

```python
"""Tests for POST /uploads endpoint."""
import io
import pytest
from httpx import AsyncClient, ASGITransport
from api.main import app


@pytest.fixture()
async def client():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        yield c


@pytest.mark.asyncio
async def test_upload_csv_returns_paths(client, tmp_path, monkeypatch):
    monkeypatch.setattr("api.routers.uploads.UPLOAD_DIR", str(tmp_path))
    csv_bytes = b"col1,col2\n1,2\n3,4\n"
    resp = await client.post(
        "/uploads",
        files=[
            ("files", ("train.csv", io.BytesIO(csv_bytes), "text/csv")),
            ("files", ("test.csv",  io.BytesIO(csv_bytes), "text/csv")),
        ],
    )
    assert resp.status_code == 200
    data = resp.json()
    assert "paths" in data
    assert len(data["paths"]) == 2
    assert all(p.endswith(".csv") for p in data["paths"])


@pytest.mark.asyncio
async def test_upload_no_files_returns_400(client):
    resp = await client.post("/uploads", files=[])
    assert resp.status_code == 400


@pytest.mark.asyncio
async def test_upload_non_csv_returns_422(client, tmp_path, monkeypatch):
    monkeypatch.setattr("api.routers.uploads.UPLOAD_DIR", str(tmp_path))
    resp = await client.post(
        "/uploads",
        files=[("files", ("data.txt", io.BytesIO(b"hello"), "text/plain"))],
    )
    assert resp.status_code == 422
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
cd .worktrees/nextjs-frontend
uv run pytest api/tests/test_uploads.py -v
```

Expected: `ModuleNotFoundError` or `404` — `uploads` router doesn't exist yet.

- [ ] **Step 3: Create the uploads router**

Create `api/routers/uploads.py`:

```python
"""Uploads router: receive CSV files, store them, return server paths."""
import os
import uuid

from fastapi import APIRouter, HTTPException, UploadFile

UPLOAD_DIR = "data/uploads"

router = APIRouter()


@router.post("/uploads")
async def upload_files(files: list[UploadFile]) -> dict:
    if not files:
        raise HTTPException(status_code=400, detail="No files provided")

    for f in files:
        filename = f.filename or ""
        if not filename.lower().endswith(".csv"):
            raise HTTPException(
                status_code=422,
                detail=f"Only CSV files are accepted, got: {filename}",
            )

    os.makedirs(UPLOAD_DIR, exist_ok=True)

    saved_paths: list[str] = []
    for f in files:
        prefix = uuid.uuid4().hex[:8]
        dest = os.path.join(UPLOAD_DIR, f"{prefix}_{f.filename}")
        content = await f.read()
        with open(dest, "wb") as fh:
            fh.write(content)
        saved_paths.append(dest)

    return {"paths": saved_paths}
```

- [ ] **Step 4: Register the router in `api/main.py`**

Current `api/main.py` line 5:
```python
from api.routers import runs, experiments, monitoring
```
Change to:
```python
from api.routers import runs, experiments, monitoring, uploads
```

Current line 19 (after `app.include_router(monitoring.router)`):
```python
app.include_router(monitoring.router)
```
Change to:
```python
app.include_router(monitoring.router)
app.include_router(uploads.router)
```

- [ ] **Step 5: Run tests to confirm they pass**

```bash
uv run pytest api/tests/test_uploads.py -v
```

Expected: all 3 tests `PASSED`.

- [ ] **Step 6: Commit**

```bash
git add api/routers/uploads.py api/tests/test_uploads.py api/main.py
git commit -m "feat: add POST /uploads endpoint for CSV file storage"
```

---

## Task 2: Install sonner and add Toaster to app root

**Files:**
- Modify: `frontend/package.json`
- Modify: `frontend/app/layout.tsx`

- [ ] **Step 1: Install sonner**

```bash
cd .worktrees/nextjs-frontend/frontend
npm install sonner
```

Expected: `sonner` appears in `package.json` dependencies.

- [ ] **Step 2: Add `<Toaster />` to the root layout**

Current `frontend/app/layout.tsx`:
```tsx
import type { Metadata } from 'next'
import './globals.css'
import { TopNav } from '@/components/TopNav'
import { Providers } from '@/components/Providers'

export const metadata: Metadata = { title: 'MLOps Dashboard' }

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <head>
        <link rel="preconnect" href="https://fonts.googleapis.com" />
        <link rel="preconnect" href="https://fonts.gstatic.com" crossOrigin="anonymous" />
        <link
          href="https://fonts.googleapis.com/css2?family=Fira+Code:wght@400;500&family=Fira+Sans:wght@400;500;600&display=swap"
          rel="stylesheet"
        />
      </head>
      <body className="min-h-screen bg-slate-50 font-sans text-slate-900">
        <Providers>
          <TopNav />
          <main className="p-6">{children}</main>
        </Providers>
      </body>
    </html>
  )
}
```

Replace with:
```tsx
import type { Metadata } from 'next'
import './globals.css'
import { TopNav } from '@/components/TopNav'
import { Providers } from '@/components/Providers'
import { Toaster } from 'sonner'

export const metadata: Metadata = { title: 'MLOps Dashboard' }

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <head>
        <link rel="preconnect" href="https://fonts.googleapis.com" />
        <link rel="preconnect" href="https://fonts.gstatic.com" crossOrigin="anonymous" />
        <link
          href="https://fonts.googleapis.com/css2?family=Fira+Code:wght@400;500&family=Fira+Sans:wght@400;500;600&display=swap"
          rel="stylesheet"
        />
      </head>
      <body className="min-h-screen bg-slate-50 font-sans text-slate-900">
        <Providers>
          <TopNav />
          <main className="p-6">{children}</main>
        </Providers>
        <Toaster position="bottom-right" richColors />
      </body>
    </html>
  )
}
```

- [ ] **Step 3: Verify the dev server starts without errors**

```bash
cd .worktrees/nextjs-frontend/frontend
npm run dev
```

Expected: server starts on port 3000, no TypeScript or import errors in terminal.

Kill the server (Ctrl+C) once confirmed.

- [ ] **Step 4: Commit**

```bash
git add frontend/package.json frontend/package-lock.json frontend/app/layout.tsx
git commit -m "feat: add sonner toast library and Toaster to root layout"
```

---

## Task 3: Add `uploadFiles` to `frontend/lib/api.ts`

**Files:**
- Modify: `frontend/lib/api.ts`

- [ ] **Step 1: Add the function**

Open `frontend/lib/api.ts`. After the closing brace of `runAdHocDrift` (line 54), add:

```typescript
export async function uploadFiles(files: File[]): Promise<{ paths: string[] }> {
  const form = new FormData()
  files.forEach(f => form.append('files', f))
  return json(await fetch(`${BASE}/uploads`, { method: 'POST', body: form }))
}
```

The end of the file should now look like:

```typescript
export async function runAdHocDrift(
  files: File[],
  referenceIndex: number,
  currentIndex: number,
): Promise<DriftReport> {
  const form = new FormData()
  files.forEach(f => form.append('files', f))
  form.append('reference_index', String(referenceIndex))
  form.append('current_index', String(currentIndex))
  return json(await fetch(`${BASE}/monitoring/drift`, { method: 'POST', body: form }))
}

export async function uploadFiles(files: File[]): Promise<{ paths: string[] }> {
  const form = new FormData()
  files.forEach(f => form.append('files', f))
  return json(await fetch(`${BASE}/uploads`, { method: 'POST', body: form }))
}

export async function fetchHealth(): Promise<HealthResponse> {
  return json(await fetch(`${BASE}/health`))
}
```

- [ ] **Step 2: Type-check**

```bash
cd .worktrees/nextjs-frontend/frontend
npx tsc --noEmit
```

Expected: no errors.

- [ ] **Step 3: Commit**

```bash
git add frontend/lib/api.ts
git commit -m "feat: add uploadFiles API function"
```

---

## Task 4: Rework `TriggerPanel` with file upload UI

**Files:**
- Modify: `frontend/components/pipeline/TriggerPanel.tsx`

- [ ] **Step 1: Replace the component**

Full replacement for `frontend/components/pipeline/TriggerPanel.tsx`:

```tsx
'use client'
import { useRef, useState } from 'react'
import { toast } from 'sonner'
import { uploadFiles, startRun } from '@/lib/api'
import { useRunStore } from '@/stores/run-store'
import { RunStatusBadge } from './RunStatusBadge'

function formatBytes(bytes: number): string {
  return bytes < 1024 ? `${bytes} B` : `${(bytes / 1024).toFixed(0)} KB`
}

export function TriggerPanel({ onRunStarted }: { onRunStarted: (id: string) => void }) {
  const [files, setFiles] = useState<File[]>([])
  const [loading, setLoading] = useState(false)
  const inputRef = useRef<HTMLInputElement>(null)
  const status = useRunStore((s) => s.status)

  function removeFile(index: number) {
    setFiles((prev) => prev.filter((_, i) => i !== index))
  }

  async function handleRun() {
    if (!files.length) return
    setLoading(true)
    try {
      const { paths } = await uploadFiles(files)
      const { run_id } = await startRun(paths)
      useRunStore.getState().setRunId(run_id)
      onRunStarted(run_id)
      toast.success('Pipeline started', {
        description: `${files.length} file${files.length > 1 ? 's' : ''} uploaded · ${run_id}`,
      })
    } catch (err) {
      toast.error('Upload failed', {
        description: err instanceof Error ? err.message : 'Check file format and try again',
      })
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="rounded-lg border border-slate-200 bg-white p-4">
      <h2 className="mb-3 text-sm font-semibold text-navy-900">Start Pipeline Run</h2>

      <button
        type="button"
        onClick={() => inputRef.current?.click()}
        className="mb-3 flex w-full items-center justify-center gap-2 rounded bg-navy px-4 py-2 text-sm font-semibold text-white hover:opacity-90"
      >
        ↑ Upload CSV files
      </button>
      <input
        ref={inputRef}
        type="file"
        multiple
        accept=".csv"
        className="hidden"
        onChange={(e) => setFiles(Array.from(e.target.files ?? []))}
      />

      {files.length > 0 && (
        <ul className="mb-3 space-y-1.5">
          {files.map((f, i) => (
            <li
              key={i}
              className="flex items-center gap-2 rounded bg-slate-100 px-3 py-2 text-sm text-slate-700"
            >
              <span className="truncate flex-1">{f.name}</span>
              <span className="shrink-0 text-xs text-slate-400">{formatBytes(f.size)}</span>
              <button
                type="button"
                onClick={() => removeFile(i)}
                className="shrink-0 text-slate-300 hover:text-red-500"
                aria-label={`Remove ${f.name}`}
              >
                ×
              </button>
            </li>
          ))}
        </ul>
      )}

      <div className="flex items-center gap-3">
        <button
          onClick={handleRun}
          disabled={loading || !files.length || status === 'running'}
          className="rounded bg-navy px-4 py-2 text-sm font-medium text-white hover:opacity-90 disabled:opacity-50"
        >
          {loading ? 'Starting...' : '▶ Run Pipeline'}
        </button>
        <RunStatusBadge />
      </div>
    </div>
  )
}
```

- [ ] **Step 2: Type-check**

```bash
cd .worktrees/nextjs-frontend/frontend
npx tsc --noEmit
```

Expected: no errors.

- [ ] **Step 3: Smoke-test in browser**

Start both servers:
```bash
# Terminal 1
cd .worktrees/nextjs-frontend
uv run uvicorn api.main:app --reload --port 8000

# Terminal 2
cd .worktrees/nextjs-frontend/frontend
npm run dev
```

Navigate to `http://localhost:3000/pipeline`. Verify:
- "Upload CSV files" navy button is visible (no more text input)
- Clicking the button opens the OS file picker
- Selecting `data/samples/iris_measurements.csv` shows a file row with name + size + × button
- Clicking × removes the row
- "Run Pipeline" button is disabled when no files are selected

- [ ] **Step 4: Commit**

```bash
git add frontend/components/pipeline/TriggerPanel.tsx
git commit -m "feat: replace text input with file upload UI in TriggerPanel"
```

---

## Task 5: Rework `AdHocForm` with file upload UI

**Files:**
- Modify: `frontend/components/monitoring/AdHocForm.tsx`

- [ ] **Step 1: Replace the component**

Full replacement for `frontend/components/monitoring/AdHocForm.tsx`:

```tsx
'use client'
import { useRef, useState } from 'react'
import { useMutation } from '@tanstack/react-query'
import { toast } from 'sonner'
import { runAdHocDrift } from '@/lib/api'
import type { DriftReport } from '@/types/api'
import { DriftTable } from './DriftTable'

function formatBytes(bytes: number): string {
  return bytes < 1024 ? `${bytes} B` : `${(bytes / 1024).toFixed(0)} KB`
}

export function AdHocForm() {
  const [files, setFiles] = useState<File[]>([])
  const [refIdx, setRefIdx] = useState(0)
  const [curIdx, setCurIdx] = useState(1)
  const [result, setResult] = useState<DriftReport | null>(null)
  const inputRef = useRef<HTMLInputElement>(null)

  function removeFile(index: number) {
    setFiles((prev) => {
      const next = prev.filter((_, i) => i !== index)
      if (refIdx >= next.length) setRefIdx(0)
      if (curIdx >= next.length) setCurIdx(Math.min(1, next.length - 1))
      return next
    })
  }

  const mutation = useMutation({
    mutationFn: () => runAdHocDrift(files, refIdx, curIdx),
    onSuccess: (data) => {
      setResult(data)
      toast.success('Analysis complete', {
        description: data.dataset_drift ? 'Drift detected in dataset' : 'No drift detected',
      })
    },
    onError: (err) => {
      toast.error('Analysis failed', {
        description: err instanceof Error ? err.message : 'Check that both files are valid CSVs with matching columns',
      })
    },
  })

  return (
    <div className="space-y-4">
      <button
        type="button"
        onClick={() => inputRef.current?.click()}
        className="flex w-full items-center justify-center gap-2 rounded bg-navy px-4 py-2 text-sm font-semibold text-white hover:opacity-90"
      >
        ↑ Upload CSV files
      </button>
      <input
        ref={inputRef}
        type="file"
        multiple
        accept=".csv"
        className="hidden"
        onChange={(e) => setFiles(Array.from(e.target.files ?? []))}
      />

      {files.length > 0 && (
        <ul className="space-y-1.5">
          {files.map((f, i) => (
            <li
              key={i}
              className="flex items-center gap-2 rounded bg-slate-100 px-3 py-2 text-sm text-slate-700"
            >
              <span className="truncate flex-1">{f.name}</span>
              <span className="shrink-0 text-xs text-slate-400">{formatBytes(f.size)}</span>
              <button
                type="button"
                onClick={() => removeFile(i)}
                className="shrink-0 text-slate-300 hover:text-red-500"
                aria-label={`Remove ${f.name}`}
              >
                ×
              </button>
            </li>
          ))}
        </ul>
      )}

      {files.length >= 2 && (
        <div className="flex flex-wrap items-center gap-4">
          <label className="flex items-center gap-2 text-sm text-slate-600">
            Reference:
            <select
              value={refIdx}
              onChange={(e) => setRefIdx(Number(e.target.value))}
              className="rounded border border-slate-300 px-2 py-1 text-sm"
            >
              {files.map((f, i) => <option key={i} value={i}>{f.name}</option>)}
            </select>
          </label>
          <label className="flex items-center gap-2 text-sm text-slate-600">
            Current:
            <select
              value={curIdx}
              onChange={(e) => setCurIdx(Number(e.target.value))}
              className="rounded border border-slate-300 px-2 py-1 text-sm"
            >
              {files.map((f, i) => <option key={i} value={i}>{f.name}</option>)}
            </select>
          </label>
          <button
            onClick={() => mutation.mutate()}
            disabled={mutation.isPending}
            className="rounded bg-amber-600 px-4 py-2 text-sm font-medium text-white hover:bg-amber-700 disabled:opacity-50"
          >
            {mutation.isPending ? 'Running...' : 'Run Drift'}
          </button>
        </div>
      )}

      {result && (
        <div className="space-y-3 rounded-lg border border-slate-200 bg-white p-4">
          <div className="flex items-center gap-4">
            <span className={`rounded-full px-3 py-1 text-sm font-medium ${
              result.dataset_drift ? 'bg-red-100 text-red-700' : 'bg-green-100 text-green-700'
            }`}>
              {result.dataset_drift ? 'Drift detected' : 'No drift'}
            </span>
            <span className="text-xl font-semibold">{(result.drift_share * 100).toFixed(1)}%</span>
            <span className="text-sm text-slate-400">columns with drift</span>
          </div>
          <DriftTable columns={result.columns} />
        </div>
      )}
    </div>
  )
}
```

- [ ] **Step 2: Type-check**

```bash
cd .worktrees/nextjs-frontend/frontend
npx tsc --noEmit
```

Expected: no errors.

- [ ] **Step 3: Smoke-test in browser**

With both servers running, navigate to `http://localhost:3000/monitoring`. Verify:
- Navy "Upload CSV files" button replaces the old dashed box
- Selecting two CSV files shows two file rows
- Reference/Current dropdowns appear when 2+ files are selected
- "Run Drift" button triggers the analysis
- On success, a toast appears bottom-right and the results table renders below
- On error (mismatched columns), an error toast appears instead of the old inline red box

- [ ] **Step 4: Commit**

```bash
git add frontend/components/monitoring/AdHocForm.tsx
git commit -m "feat: replace dashed drop zone with file upload UI in AdHocForm"
```

---

## Task 6: Run full test suite and verify

- [ ] **Step 1: Run backend tests**

```bash
cd .worktrees/nextjs-frontend
uv run pytest api/tests/ -v -m "not integration"
```

Expected: all tests pass including the 3 new upload tests.

- [ ] **Step 2: Run frontend type-check**

```bash
cd .worktrees/nextjs-frontend/frontend
npx tsc --noEmit
```

Expected: no errors.

- [ ] **Step 3: End-to-end smoke test**

With both servers running:
1. Go to `/pipeline` — confirm "Run Pipeline" is disabled with no files; upload one or more CSVs, click "Run Pipeline", watch for success toast with run ID
2. Go to `/monitoring` — upload two CSVs, run drift, watch for "Analysis complete" toast

- [ ] **Step 4: Final commit (if any stray changes)**

```bash
git status
# commit anything uncommitted
```
