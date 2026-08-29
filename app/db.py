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
    source_id INTEGER NOT NULL,
    drive_file_id TEXT NOT NULL,
    filename TEXT NOT NULL,
    mime_type TEXT,
    size_bytes INTEGER,
    content_hash TEXT,
    face_count INTEGER DEFAULT 0,
    UNIQUE(job_id, drive_file_id),
    FOREIGN KEY(job_id) REFERENCES jobs(id),
    FOREIGN KEY(source_id) REFERENCES job_sources(id)
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


@contextmanager
def get_conn():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def migrate_database(conn):
    print("[MIGRATION] Starting database migration...")
    
    # 1. Turn off foreign keys temporarily so we can restructure
    conn.execute("PRAGMA foreign_keys = OFF;")
    
    # 2. Rename old tables
    conn.execute("ALTER TABLE jobs RENAME TO jobs_old;")
    conn.execute("ALTER TABLE photos RENAME TO photos_old;")
    
    # 3. Create new tables
    conn.executescript(SCHEMA)
    
    # 4. Migrate jobs and create job_sources
    cursor = conn.execute("SELECT * FROM jobs_old")
    old_jobs = cursor.fetchall()
    
    for job in old_jobs:
        job_id = job["id"]
        folder_id = job["folder_id"]
        status = job["status"]
        total_files = job["total_files"]
        processed_files = job["processed_files"]
        message = job["message"]
        created_at = job["created_at"]
        
        # Generate token
        token = secrets.token_urlsafe(32)
        
        # Insert into new jobs table
        conn.execute(
            """
            INSERT INTO jobs (id, public_job_token, status, total_files, processed_files, duplicate_files_skipped, message, created_at)
            VALUES (?, ?, ?, ?, ?, 0, ?, ?)
            """,
            (job_id, token, status, total_files, processed_files, message, created_at)
        )
        
        # Create a job source row
        source_status = "completed" if status == "done" else ("failed" if status == "error" else "pending")
        conn.execute(
            """
            INSERT INTO job_sources (job_id, raw_link, folder_id, resourcekey, status, message)
            VALUES (?, ?, ?, NULL, ?, ?)
            """,
            (job_id, folder_id, folder_id, source_status, message)
        )
        
    # 5. Migrate photos
    conn.execute(
        """
        INSERT INTO photos (id, job_id, source_id, drive_file_id, filename, mime_type, size_bytes, face_count)
        SELECT p.id, p.job_id, s.id, p.drive_file_id, p.filename, NULL, NULL, p.face_count
        FROM photos_old p
        LEFT JOIN job_sources s ON s.job_id = p.job_id;
        """
    )
    
    # 6. Drop old tables
    conn.execute("DROP TABLE jobs_old;")
    conn.execute("DROP TABLE photos_old;")
    
    # 7. Turn on foreign keys
    conn.execute("PRAGMA foreign_keys = ON;")
    print("[MIGRATION] Database migration completed successfully.")


def init_db():
    with get_conn() as conn:
        # Check if we need to migrate from old schema
        cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='jobs'")
        has_jobs = cursor.fetchone()
        
        needs_migration = False
        if has_jobs:
            cursor = conn.execute("PRAGMA table_info(jobs)")
            columns = [row["name"] for row in cursor.fetchall()]
            if "public_job_token" not in columns:
                needs_migration = True
        
        if needs_migration:
            migrate_database(conn)
            
        # Check if jobs table is missing oauth_token
        cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='jobs'")
        has_jobs = cursor.fetchone()
        if has_jobs:
            cursor = conn.execute("PRAGMA table_info(jobs)")
            columns = [row["name"] for row in cursor.fetchall()]
            if "oauth_token" not in columns:
                print("[MIGRATION] jobs table is missing oauth_token. Adding column...")
                conn.execute("ALTER TABLE jobs ADD COLUMN oauth_token TEXT;")
                print("[MIGRATION] oauth_token column added successfully.")
            
        # Also check if photos table is missing content_hash
        cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='photos'")
        has_photos = cursor.fetchone()
        if has_photos:
            cursor = conn.execute("PRAGMA table_info(photos)")
            columns = [row["name"] for row in cursor.fetchall()]
            if "content_hash" not in columns:
                print("[MIGRATION] photos table is missing content_hash. Migrating photos table...")
                conn.execute("PRAGMA foreign_keys = OFF;")
                conn.execute("ALTER TABLE photos RENAME TO photos_old;")
                conn.executescript(SCHEMA)
                conn.execute(
                    """
                    INSERT INTO photos (id, job_id, source_id, drive_file_id, filename, mime_type, size_bytes, face_count, content_hash)
                    SELECT id, job_id, source_id, drive_file_id, filename, mime_type, size_bytes, face_count, NULL
                    FROM photos_old;
                    """
                )
                conn.execute("DROP TABLE photos_old;")
                conn.execute("PRAGMA foreign_keys = ON;")
                print("[MIGRATION] photos table migration completed successfully.")
                
        # Ensure all tables (including any new ones like oauth_pending_states) are created
        conn.executescript(SCHEMA)


# ---------- jobs ----------

def create_job() -> tuple[int, str]:
    token = secrets.token_urlsafe(32)
    with get_conn() as conn:
        cur = conn.execute(
            "INSERT INTO jobs (public_job_token, status) VALUES (?, 'pending')", (token,)
        )
        return cur.lastrowid, token


def get_job_by_token(public_job_token: str) -> dict | None:
    with get_conn() as conn:
        row = conn.execute(
            "SELECT * FROM jobs WHERE public_job_token = ?", (public_job_token,)
        ).fetchone()
        return dict(row) if row else None


def update_job(job_id: int, **fields):
    if not fields:
        return
    cols = ", ".join(f"{k} = ?" for k in fields)
    with get_conn() as conn:
        conn.execute(f"UPDATE jobs SET {cols} WHERE id = ?", (*fields.values(), job_id))


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

def insert_photo(job_id: int, source_id: int, drive_file_id: str, filename: str, mime_type: str = None, size_bytes: int = None, content_hash: str = None) -> int:
    with get_conn() as conn:
        cur = conn.execute(
            "INSERT OR IGNORE INTO photos (job_id, source_id, drive_file_id, filename, mime_type, size_bytes, content_hash) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (job_id, source_id, drive_file_id, filename, mime_type, size_bytes, content_hash),
        )
        if cur.lastrowid:
            return cur.lastrowid
        row = conn.execute(
            "SELECT id FROM photos WHERE job_id = ? AND drive_file_id = ?", (job_id, drive_file_id)
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
