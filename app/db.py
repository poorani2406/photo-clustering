import sqlite3
import numpy as np
from contextlib import contextmanager

from app.config import DB_PATH

SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    folder_id TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending',
    total_files INTEGER DEFAULT 0,
    processed_files INTEGER DEFAULT 0,
    message TEXT DEFAULT '',
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS photos (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id INTEGER NOT NULL,
    drive_file_id TEXT NOT NULL,
    filename TEXT NOT NULL,
    local_path TEXT NOT NULL,
    face_count INTEGER DEFAULT 0,
    UNIQUE(job_id, drive_file_id)
);

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
    representative_face_id INTEGER
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


def init_db():
    with get_conn() as conn:
        # Check if the photos table has the old constraint and needs migration
        cursor = conn.execute("SELECT sql FROM sqlite_schema WHERE name = 'photos'")
        row = cursor.fetchone()
        if row and "drive_file_id TEXT UNIQUE" in row[0]:
            conn.execute("PRAGMA foreign_keys = OFF;")
            conn.execute("ALTER TABLE photos RENAME TO photos_old;")
            conn.execute("""
            CREATE TABLE photos (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                job_id INTEGER NOT NULL,
                drive_file_id TEXT NOT NULL,
                filename TEXT NOT NULL,
                local_path TEXT NOT NULL,
                face_count INTEGER DEFAULT 0,
                UNIQUE(job_id, drive_file_id)
            );
            """)
            conn.execute("""
            INSERT INTO photos (id, job_id, drive_file_id, filename, local_path, face_count)
            SELECT id, job_id, drive_file_id, filename, local_path, face_count FROM photos_old;
            """)
            conn.execute("DROP TABLE photos_old;")
            conn.execute("PRAGMA foreign_keys = ON;")
        
        conn.executescript(SCHEMA)


# ---------- jobs ----------

def create_job(folder_id: str) -> int:
    with get_conn() as conn:
        cur = conn.execute(
            "INSERT INTO jobs (folder_id, status) VALUES (?, 'pending')", (folder_id,)
        )
        return cur.lastrowid


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


# ---------- photos ----------

def insert_photo(job_id: int, drive_file_id: str, filename: str, local_path: str) -> int:
    with get_conn() as conn:
        cur = conn.execute(
            "INSERT OR IGNORE INTO photos (job_id, drive_file_id, filename, local_path) "
            "VALUES (?, ?, ?, ?)",
            (job_id, drive_file_id, filename, local_path),
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


# ---------- faces ----------

def insert_face(photo_id: int, top: int, right: int, bottom: int, left: int, embedding: np.ndarray) -> int:
    with get_conn() as conn:
        cur = conn.execute(
            "INSERT INTO faces (photo_id, top, \"right\", bottom, left, embedding) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            # Stored as float64 regardless of the engine's native dtype (InsightFace
            # produces float32) so clustering.py's np.frombuffer(..., dtype=np.float64)
            # read-back stays correct no matter which FaceEngine wrote the row.
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
            SELECT DISTINCT photos.id, photos.filename, photos.local_path
            FROM faces
            JOIN photos ON photos.id = faces.photo_id
            WHERE faces.person_id = ?
            """,
            (person_id,),
        ).fetchall()
        return [dict(r) for r in rows]
