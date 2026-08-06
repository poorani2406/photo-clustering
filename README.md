<!-- Badges: fill in once the repo is public. See "GitHub Best Practices" for how to generate these. -->
<!-- ![License](https://img.shields.io/github/license/<you>/<repo>) ![Python](https://img.shields.io/badge/python-3.10%2B-blue) -->

# Contact — Face Detection & Grouping

Point this at a Google Drive folder. It downloads every photo, finds every
face, and groups matching faces into "people" so you can see who's in the
folder and every photo they appear in.

> **Stage 1 scope.** Deliberately *not* included yet: login/accounts,
> billing, editable galleries, ZIP downloads, multi-user support. This
> version does one thing — detect faces and group them.

<!-- Screenshot: hero shot of the "people" grid after a folder finishes processing.
     Place it here, e.g. ![People grid](docs/screenshots/people-grid.png) -->

## Features

- **One-click ingestion** — paste a Google Drive folder ID, the app downloads every image in it.
- **Automatic face detection & embedding** — every face in every photo becomes a 512-d embedding via InsightFace (ArcFace).
- **Unsupervised person grouping** — DBSCAN clusters embeddings that belong to the same person, with no need to know the number of people in advance.
- **Custom iterative reassignment pass** — a second pass re-evaluates faces DBSCAN left as outliers (or in too-small clusters) against every established cluster's centroid *and* individual face similarities, catching matches DBSCAN's density model misses on its own. See [`docs/CLUSTERING.md`](docs/CLUSTERING.md) for the full walkthrough.
- **Singleton people preserved** — someone who appears in only one photo still gets their own "person" card instead of being dropped as noise.
- **Local-only storage** — SQLite + a local folder of images. No external database, no cloud service beyond Google Drive itself.

## Architecture

```
Drive folder
   -> list every image file
      -> download each one to disk
         -> detect faces + generate a 512-d embedding per face
            -> DBSCAN clusters embeddings that are close together
               -> iterative reassignment pass rescues remaining outliers
                  -> each final cluster = one person, shown with every photo they're in
```

Everything is stored locally in a SQLite file (`data/app.db`) and downloaded
photos live in `data/images/`.

## Tech Stack

| Layer | Choice |
|---|---|
| Backend | FastAPI + Uvicorn |
| Face detection & embeddings | InsightFace (ArcFace), CPU via ONNX Runtime by default |
| Clustering | scikit-learn (DBSCAN) + custom iterative reassignment |
| Storage | SQLite (metadata) + local filesystem (images) |
| External API | Google Drive API (OAuth 2.0, read-only) |
| Frontend | Vanilla HTML/CSS/JS, no build step |

## Installation

```bash
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

**Heads up on `insightface`**: no compiler needed, it's a plain pip install
on Windows/macOS/Linux (it runs on `onnxruntime`, CPU by default). On first
run it downloads its model pack (~350MB) to `~/.insightface` and caches it
there, so the very first job will pause briefly on "Processing..." while
that download happens.

## Configuration

### 1. Get Google Drive API access

This app skips building its own login system, but Drive itself still requires
OAuth — Google won't hand over file contents to an anonymous script.

1. Go to the [Google Cloud Console](https://console.cloud.google.com/), create
   a project (or use an existing one).
2. Enable the **Google Drive API** (APIs & Services → Library → search "Google
   Drive API" → Enable).
3. Go to APIs & Services → Credentials → **Create Credentials → OAuth client
   ID**.
   - Application type: **Desktop app**.
   - If prompted, configure the OAuth consent screen first — choose
     "External", fill in an app name, and add your own Google account as a
     test user. You don't need to submit it for verification for personal use.
4. Download the resulting JSON and save it as `credentials.json` in the
   project root (same folder as `run.py`).

The first time you process a folder, a browser tab will open asking you to
log in and approve read-only Drive access. After that, a `token.json` is
cached so you won't be asked again.

> `credentials.json` and `token.json` both hold access to your Google
> account — never commit them. Both are already in `.gitignore`.

### 2. Share the folder

Make sure the Google account you authenticate with has at least **viewer**
access to the photographer's folder (either it's in their own Drive, or the
folder's been shared with that account).

### 3. Environment variables

Copy `.env.example` to `.env` and adjust if needed:

```bash
cp .env.example .env
```

- `CLUSTER_EPS` (default `0.9`) — how close two faces' embeddings must be to
  count as the same person. Lower = stricter (more, smaller people-clusters;
  risk of splitting one real person into two). Higher = looser (risk of
  merging two different people together).
- `CLUSTER_MIN_SAMPLES` (default `2`) — minimum photos before a cluster is
  established the normal way. People who appear in only one photo still show
  up (as a "person" of their own), they just aren't used to help define the
  cluster boundary.
- `CLUSTER_MERGE_THRESHOLD`, `CLUSTER_FACE_THRESHOLD`, `CLUSTER_MIN_MERGE_SIZE`
  — control the iterative reassignment pass described in
  [`docs/CLUSTERING.md`](docs/CLUSTERING.md). Confirm the defaults against
  `app/config.py`.

If the app is merging two people who look somewhat alike, lower `CLUSTER_EPS`
and reprocess. If it's splitting the same person into multiple groups, raise
it slightly.

## Usage

```bash
python run.py
```

Open **http://127.0.0.1:8000** in your browser.

<!-- Screenshot: the folder-ID input screen.
     ![Folder input](docs/screenshots/folder-input.png) -->

Paste the folder ID from its Drive URL — the part after `/folders/`:

```
https://drive.google.com/drive/folders/1a2B3cD4EfGhIjKlmNoPQrstuVWxyz
                                        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ this
```

Click **Process folder** and watch the status bar. Depending on folder size
this can take a while — face detection runs on CPU by default (roughly
0.5–2 seconds per photo on a normal laptop).

<!-- Screenshot: processing status bar mid-run.
     ![Processing status](docs/screenshots/processing-status.png) -->

When it finishes, you'll see a grid of every distinct person found. Click one
to see every photo they appear in.

<!-- Screenshot: a person's detail view with their matched photos.
     ![Person detail](docs/screenshots/person-detail.png) -->

## Folder Structure

```
app/
  main.py                FastAPI routes (process folder, list people, serve photos/thumbnails)
  pipeline.py             Orchestrates download -> detect -> cluster for one job
  drive_service.py        All Google Drive API calls (auth, list, download)
  face_engine/            Face detection + embeddings, behind a FaceEngine interface
    base.py                 Abstract FaceEngine interface
    insightface_engine.py   InsightFace implementation (current default)
  clustering.py           DBSCAN + iterative reassignment grouping of embeddings into people
  db.py                    SQLite persistence
  config.py                Paths, env vars, clustering thresholds
static/                   Single-page frontend (vanilla HTML/CSS/JS, no build step)
docs/                      Algorithm write-ups, screenshots
data/                      Created at runtime: downloaded images + app.db (gitignored)
```

## Known Limits of This Stage (by design)

- One folder processed at a time, one job at a time.
- No re-auth / multi-account handling — one Google account, one login.
- Reprocessing the same folder re-downloads and re-detects from scratch (no
  incremental sync yet).
- Face detection runs on CPU by default (`ctx_id=-1`). If you have a GPU
  and an `onnxruntime-gpu` install, pass `ctx_id=0` when constructing
  `InsightFaceEngine` in `app/face_engine/__init__.py` for faster processing.

## Future Improvements

- [ ] Incremental sync (only process new/changed files on reprocess)
- [ ] Background job queue so multiple folders can process concurrently
- [ ] Swap `print()` debug output for structured logging with a debug flag
- [ ] Basic auth / multi-user support ahead of any real deployment
- [ ] Automated tests around the clustering merge logic
- [ ] Optional GPU path documented end-to-end

## License

[MIT](LICENSE) — see the License section of the accompanying repo review for reasoning.
