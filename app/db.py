import sqlite3
import numpy as np
import secrets
from contextlib import contextmanager

from app.config import DB_PATH

SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    public_job_token TEXT UNIQUE NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending',
    total_files INTEGER DEFAULT 0,
    processed_files INTEGER DEFAULT 0,
    duplicate_files_skipped INTEGER DEFAULT 0,
    oauth_token TEXT,
    message TEXT DEFAULT '',
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS job_sources (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id INTEGER NOT NULL,
    raw_link TEXT NOT NULL,
    folder_id TEXT NOT NULL,
    resourcekey TEXT,
    status TEXT DEFAULT 'pending',
    message TEXT DEFAULT '',
    FOREIGN KEY(job_id) REFERENCES jobs(id)
);

CREATE TABLE IF NOT EXISTS photos (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id INTEGER NOT NULL,
    source_id INTEGER DEFAULT 0,
    drive_file_id TEXT NOT NULL,
    filename TEXT NOT NULL,
    mime_type TEXT,
    size_bytes INTEGER,
    content_hash TEXT,
    face_count INTEGER DEFAULT 0,
    storage_path TEXT,
    UNIQUE(job_id, drive_file_id),
    FOREIGN KEY(job_id) REFERENCES jobs(id)
);

CREATE INDEX IF NOT EXISTS idx_photos_job_content_hash ON photos(job_id, content_hash);

CREATE TABLE IF NOT EXISTS faces (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    photo_id INTEGER NOT NULL,
    top INTEGER, "right" INTEGER, bottom INTEGER, left INTEGER,
    embedding BLOB NOT NULL,
    person_id INTEGER,
    FOREIGN KEY(photo_id) REFERENCES photos(id)
);

CREATE TABLE IF NOT EXISTS people (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id INTEGER NOT NULL,
    label TEXT NOT NULL,
    representative_face_id INTEGER,
    FOREIGN KEY(job_id) REFERENCES jobs(id)
);

CREATE TABLE IF NOT EXISTS oauth_pending_states (
    state TEXT PRIMARY KEY,
    public_job_token TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
"""


# In-memory thread-safe cache for robust job retrieval across concurrent workers/requests
_JOB_CACHE: dict[str, dict] = {}
_ID_TO_TOKEN_CACHE: dict[int, str] = {}


@contextmanager
def get_conn():
    conn = sqlite3.connect(str(DB_PATH), timeout=10.0, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA busy_timeout=10000;")
    try:
        yield conn
        conn.commit()
    except Exception:
        try:
            conn.rollback()
        except Exception:
            pass
        raise
    finally:
        conn.close()


def init_db():
    print(f"[PROCESS STARTUP] Initializing SQLite database at {DB_PATH}...", flush=True)
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with get_conn() as conn:
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        conn.execute("PRAGMA busy_timeout=30000;")
        conn.executescript(SCHEMA)
        
        # Ensure optional/extended columns exist in existing deployments
        try:
            cursor = conn.execute("PRAGMA table_info(jobs)")
            job_cols = [row["name"] for row in cursor.fetchall()]
            if "public_job_token" not in job_cols:
                conn.execute("ALTER TABLE jobs ADD COLUMN public_job_token TEXT UNIQUE;")
            if "duplicate_files_skipped" not in job_cols:
                conn.execute("ALTER TABLE jobs ADD COLUMN duplicate_files_skipped INTEGER DEFAULT 0;")
        except Exception as e:
            print(f"[DB INIT WARNING] jobs table check: {e}", flush=True)

        try:
            cursor = conn.execute("PRAGMA table_info(photos)")
            photo_cols = [row["name"] for row in cursor.fetchall()]
            if "content_hash" not in photo_cols:
                conn.execute("ALTER TABLE photos ADD COLUMN content_hash TEXT;")
            if "storage_path" not in photo_cols:
                conn.execute("ALTER TABLE photos ADD COLUMN storage_path TEXT;")
        except Exception as e:
            print(f"[DB INIT WARNING] photos table check: {e}", flush=True)
            
    print("[PROCESS STARTUP] SQLite database initialized and ready.", flush=True)




# ---------- jobs ----------

def create_job() -> tuple[int, str]:
    token = secrets.token_urlsafe(32)
    with get_conn() as conn:
        cur = conn.execute(
            "INSERT INTO jobs (public_job_token, status, message) VALUES (?, 'pending', 'Initializing...')", (token,)
        )
        job_id = cur.lastrowid

    job_dict = {
        "id": job_id,
        "public_job_token": token,
        "status": "pending",
        "total_files": 0,
        "processed_files": 0,
        "duplicate_files_skipped": 0,
        "message": "Initializing...",
        "created_at": None,
    }
    _JOB_CACHE[token] = job_dict
    _ID_TO_TOKEN_CACHE[job_id] = token
    print(f"[JOB CREATE] job_id={job_id} public_token={token}")
    return job_id, token


def get_job_by_token(public_job_token: str) -> dict | None:
    if not public_job_token:
        print("[JOB GET] job_id=None -> 404 Not Found", flush=True)
        return None

    clean_token = str(public_job_token).strip()

    # 1. Fast in-memory cache
    if clean_token in _JOB_CACHE:
        cached = _JOB_CACHE[clean_token]
        return dict(cached)

    # 2. Database read
    try:
        with get_conn() as conn:
            row = conn.execute(
                "SELECT * FROM jobs WHERE public_job_token = ?", (clean_token,)
            ).fetchone()
            if row:
                res = dict(row)
                _JOB_CACHE[clean_token] = res
                _ID_TO_TOKEN_CACHE[res["id"]] = clean_token
                return res

            # Fallback: lookup by integer ID if passed
            if clean_token.isdigit():
                row = conn.execute("SELECT * FROM jobs WHERE id = ?", (int(clean_token),)).fetchone()
                if row:
                    res = dict(row)
                    return res
    except Exception as e:
        print(f"[ERROR] [get_job_by_token db read error]: {e}", flush=True)

    print(f"[JOB GET] job_id={clean_token} -> 404 Not Found", flush=True)
    return None


def update_job(job_id: int, **fields):
    if not fields:
        return
    token = _ID_TO_TOKEN_CACHE.get(job_id)
    # Update in-memory cache IMMEDIATELY
    if token and token in _JOB_CACHE:
        _JOB_CACHE[token].update(fields)
    elif str(job_id) in _JOB_CACHE:
        _JOB_CACHE[str(job_id)].update(fields)

    cols = ", ".join(f"{k} = ?" for k in fields)
    try:
        with get_conn() as conn:
            conn.execute(f"UPDATE jobs SET {cols} WHERE id = ?", (*fields.values(), job_id))
            if token:
                conn.execute(f"UPDATE jobs SET {cols} WHERE public_job_token = ?", (*fields.values(), token))
    except Exception as e:
        print(f"[ERROR] [update_job db error for job {job_id}]: {e}", flush=True)

    status_val = fields.get("status", "")
    print(f"[JOB UPDATE] job_id={job_id} status={status_val}", flush=True)





def get_job(job_id: int):
    with get_conn() as conn:
        row = conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
        return dict(row) if row else None


# ---------- job_sources ----------

def create_job_source(job_id: int, raw_link: str) -> int:
    with get_conn() as conn:
        cur = conn.execute(
            "INSERT INTO job_sources (job_id, raw_link, folder_id, status) VALUES (?, ?, '', 'pending')",
            (job_id, raw_link)
        )
        return cur.lastrowid


def update_job_source(source_id: int, **fields):
    if not fields:
        return
    cols = ", ".join(f"{k} = ?" for k in fields)
    with get_conn() as conn:
        conn.execute(f"UPDATE job_sources SET {cols} WHERE id = ?", (*fields.values(), source_id))


def get_sources_for_job(job_id: int) -> list[dict]:
    with get_conn() as conn:
        rows = conn.execute("SELECT * FROM job_sources WHERE job_id = ?", (job_id,)).fetchall()
        return [dict(r) for r in rows]


# ---------- photos ----------

def insert_photo(job_id: int, source_id: int = 0, drive_file_id: str = "", filename: str = "", mime_type: str = None, size_bytes: int = None, content_hash: str = None, storage_path: str = None) -> int:
    file_identifier = drive_file_id or filename
    with get_conn() as conn:
        cur = conn.execute(
            "INSERT OR IGNORE INTO photos (job_id, source_id, drive_file_id, filename, mime_type, size_bytes, content_hash, storage_path) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (job_id, source_id, file_identifier, filename, mime_type, size_bytes, content_hash, storage_path),
        )
        if cur.lastrowid:
            return cur.lastrowid
        row = conn.execute(
            "SELECT id FROM photos WHERE job_id = ? AND drive_file_id = ?", (job_id, file_identifier)
        ).fetchone()
        return row["id"]



def set_photo_face_count(photo_id: int, count: int):
    with get_conn() as conn:
        conn.execute("UPDATE photos SET face_count = ? WHERE id = ?", (count, photo_id))


def get_photo(photo_id: int):
    with get_conn() as conn:
        row = conn.execute("SELECT * FROM photos WHERE id = ?", (photo_id,)).fetchone()
        return dict(row) if row else None


def get_photos_by_ids(photo_ids: list[int]) -> list[dict]:
    if not photo_ids:
        return []
    placeholders = ", ".join("?" for _ in photo_ids)
    with get_conn() as conn:
        rows = conn.execute(
            f"SELECT * FROM photos WHERE id IN ({placeholders})",
            photo_ids
        ).fetchall()
        return [dict(r) for r in rows]


# ---------- faces ----------

def insert_face(photo_id: int, top: int, right: int, bottom: int, left: int, embedding: np.ndarray) -> int:
    with get_conn() as conn:
        cur = conn.execute(
            "INSERT INTO faces (photo_id, top, \"right\", bottom, left, embedding) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (photo_id, top, right, bottom, left, embedding.astype(np.float64).tobytes()),
        )
        return cur.lastrowid


def get_faces_for_job(job_id: int):
    with get_conn() as conn:
        rows = conn.execute(
            """
            SELECT faces.*, photos.filename FROM faces
            JOIN photos ON photos.id = faces.photo_id
            WHERE photos.job_id = ?
            """,
            (job_id,),
        ).fetchall()
        return [dict(r) for r in rows]


def set_face_person(face_id: int, person_id: int):
    with get_conn() as conn:
        conn.execute("UPDATE faces SET person_id = ? WHERE id = ?", (person_id, face_id))


def get_face(face_id: int):
    with get_conn() as conn:
        row = conn.execute("SELECT * FROM faces WHERE id = ?", (face_id,)).fetchone()
        return dict(row) if row else None


# ---------- people ----------

def clear_people_for_job(job_id: int):
    with get_conn() as conn:
        conn.execute("DELETE FROM people WHERE job_id = ?", (job_id,))


def create_person(job_id: int, label: str, representative_face_id: int) -> int:
    with get_conn() as conn:
        cur = conn.execute(
            "INSERT INTO people (job_id, label, representative_face_id) VALUES (?, ?, ?)",
            (job_id, label, representative_face_id),
        )
        return cur.lastrowid


def list_people(job_id: int):
    with get_conn() as conn:
        rows = conn.execute(
            """
            SELECT people.id, people.label, people.representative_face_id,
                   COUNT(DISTINCT faces.photo_id) AS photo_count
            FROM people
            LEFT JOIN faces ON faces.person_id = people.id
            WHERE people.job_id = ?
            GROUP BY people.id
            ORDER BY photo_count DESC
            """,
            (job_id,),
        ).fetchall()
        return [dict(r) for r in rows]


def rename_person(person_id: int, label: str):
    with get_conn() as conn:
        conn.execute("UPDATE people SET label = ? WHERE id = ?", (label, person_id))


def get_photos_for_person(person_id: int):
    with get_conn() as conn:
        rows = conn.execute(
            """
            SELECT DISTINCT photos.*
            FROM faces
            JOIN photos ON photos.id = faces.photo_id
            WHERE faces.person_id = ?
            """,
            (person_id,),
        ).fetchall()
        return [dict(r) for r in rows]


# ---------- OAuth helpers ----------

def save_oauth_state(state: str, public_job_token: str):
    with get_conn() as conn:
        conn.execute(
            "INSERT INTO oauth_pending_states (state, public_job_token) VALUES (?, ?)",
            (state, public_job_token),
        )


def pop_oauth_state(state: str) -> str:
    """Retrieves the associated public_job_token for a given state, then deletes it."""
    with get_conn() as conn:
        row = conn.execute(
            "SELECT public_job_token FROM oauth_pending_states WHERE state = ?",
            (state,),
        ).fetchone()
        if row:
            token = row["public_job_token"]
            conn.execute("DELETE FROM oauth_pending_states WHERE state = ?", (state,))
            return token
        return None


def save_oauth_token(job_id: int, token_json: str):
    with get_conn() as conn:
        conn.execute("UPDATE jobs SET oauth_token = ? WHERE id = ?", (token_json, job_id))


def get_oauth_token(job_id: int) -> str:
    with get_conn() as conn:
        row = conn.execute("SELECT oauth_token FROM jobs WHERE id = ?", (job_id,)).fetchone()
        return row["oauth_token"] if row else None
