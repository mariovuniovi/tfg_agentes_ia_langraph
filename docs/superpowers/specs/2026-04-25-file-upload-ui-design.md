# File Upload UI — Design Spec

## Overview

Replace the current file-input patterns on the Pipeline and Monitoring pages with a consistent upload button + file rows UI, plus a toast notification system.

## Scope

Two frontend components + one new backend endpoint:

- `frontend/components/pipeline/TriggerPanel.tsx` — replace text input with file upload
- `frontend/components/monitoring/AdHocForm.tsx` — replace dashed drop zone with button + rows
- `api/routers/uploads.py` (new) — `POST /uploads` endpoint for the Pipeline page
- Toast infrastructure — installed library wired into the app shell

---

## Upload UI Pattern (both pages)

**Style:** Option B — solid navy button + stacked file rows.

### Button
```
[ ↑  Upload CSV files ]   ← full-width, navy (#1e3a5f), white text
```
Clicking the button opens the native OS file picker (`input[type=file]`, hidden, triggered by ref). Accepts `.csv` only. Multiple files allowed.

### File rows (shown after selection)
Each selected file renders as a row:
```
📄  filename.csv          12 KB   ×
```
- Background: `#f1f5f9`, border-radius 6px
- File size shown in KB
- `×` button removes that file from the selection
- Rows appear immediately after file picker closes (no upload yet)

---

## Pipeline Page Flow

1. User clicks **Upload CSV files** → file picker opens → selects one or more CSVs
2. Selected files appear as rows (client-side only at this point)
3. User clicks **▶ Run Pipeline**
4. Frontend calls `POST /uploads` with the files as multipart form data
5. Backend stores the files in a temporary directory, returns their server-side paths
6. Frontend calls `POST /runs` with those paths (existing `startRun` API)
7. On success → success toast appears bottom-right
8. On failure (upload or run start) → error toast appears

---

## Monitoring Page Flow

No backend change. The existing `POST /monitoring/drift` already accepts files as multipart form.

1. User clicks **Upload CSV files** → selects CSVs → rows appear
2. User picks Reference and Current from the dropdowns (populated from selected files)
3. User clicks **⚡ Run Drift** → files + indexes sent to existing endpoint
4. On success → drift results render below (existing behaviour) + success toast
5. On failure → error toast (replaces current inline error box)

---

## Backend — `POST /uploads`

**New file:** `api/routers/uploads.py`  
**Route:** `POST /uploads`  
**Auth:** none (consistent with the rest of the API)

**Request:** `multipart/form-data` with one or more `files` fields.

**Response:**
```json
{ "paths": ["data/uploads/abc123_iris_measurements.csv", "data/uploads/abc123_iris_labels.csv"] }
```

**Storage:** files saved to `data/uploads/` with a UUID prefix to avoid collisions. The directory is created on first use. No cleanup logic needed for now.

**Errors:** 400 if no files provided; 422 if a file is not a CSV (check content-type or extension).

Register the router in `api/main.py` under the existing router setup.

---

## Toast System

Install **`sonner`** — the standard toast library for Next.js/React projects. Lightweight, zero-config, supports auto-dismiss.

**Setup:**
- Add `<Toaster />` to `frontend/app/layout.tsx` (renders once at app root)
- Import `toast` from `sonner` wherever needed

**Toast messages:**

| Trigger | Type | Title | Subtitle |
|---|---|---|---|
| Pipeline started | success | "Pipeline started" | "2 files uploaded · {run_id}" |
| Upload failed | error | "Upload failed" | "Check file format and try again" |
| Drift run started | success | "Analysis running" | "{n} files ready" |
| Drift failed | error | "Analysis failed" | "Check that both files are valid CSVs" |

**Behaviour:** auto-dismiss after 3 seconds. Appears bottom-right. User can click `×` to dismiss early.

---

## Files Changed

| File | Change |
|---|---|
| `frontend/components/pipeline/TriggerPanel.tsx` | Replace text input + error paragraph with file upload UI |
| `frontend/components/monitoring/AdHocForm.tsx` | Replace dashed drop zone with button + file rows |
| `frontend/lib/api.ts` | Add `uploadFiles(files: File[])` function |
| `frontend/app/layout.tsx` | Add `<Toaster />` from sonner |
| `api/routers/uploads.py` | New file — `POST /uploads` endpoint |
| `api/main.py` | Register uploads router |
| `frontend/package.json` | Add `sonner` dependency |

---

## Out of Scope

- File upload progress bar
- Drag-and-drop on the Pipeline page
- Persistent file storage / file management UI
- Upload size limits
