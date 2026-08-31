import os
# Configure CPU thread limits before importing heavy math/C libraries
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import io
import secrets
import json
import threading
import datetime
import zipfile
import shutil
import traceback
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, Request, Query, UploadFile, File, BackgroundTasks
from fastapi.responses import StreamingResponse, RedirectResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from PIL import Image
from stream_zip import stream_zip, ZIP_32

from app import db
from app.config import DATA_DIR, MAX_FILE_SIZE_BYTES, MAX_FILES_PER_JOB, IMAGE_MIME_TYPES
from app.pipeline import run_job, run_direct_upload_job


app = FastAPI(title="Photo Clustering")


# Initialize database on application startup
db.init_db()

# Pre-initialize Face Engine synchronously on application boot
from app.face_engine import get_face_engine
print("[PROCESS STARTUP] Initializing and warming up InsightFace CPU engine on boot...")
try:
    get_face_engine()
    print("[PROCESS STARTUP] InsightFace CPU engine ready.")
except Exception as e:
    print(f"[PROCESS STARTUP WARNING] Lazy loading will be used: {e}")

print("[PROCESS STARTUP] Photo Clustering FastAPI application loaded and ready.")



@app.get("/health")
@app.get("/api/health")
def health_check():
    return JSONResponse({"status": "ok", "app": "Photo Clustering"})


@app.get("/api/debug/info")
def debug_info():
    with db.get_conn() as conn:
        jobs = [dict(r) for r in conn.execute("SELECT * FROM jobs ORDER BY id DESC LIMIT 10").fetchall()]
    return {
        "env": os.environ.get("ENV", "production"),
        "has_google_api_key": bool(os.environ.get("GOOGLE_DRIVE_API_KEY")),
        "model_name": os.environ.get("INSIGHTFACE_MODEL", "buffalo_s"),
        "models_bundled": (BASE_DIR / "models" / "buffalo_s" / "det_500m.onnx").exists(),
        "recent_jobs": jobs
    }






class ProcessRequest(BaseModel):
    folder_links: list[str]


class StartProcessRequest(BaseModel):
    public_job_token: str


class RenameRequest(BaseModel):
    label: str


class PhotosDownloadRequest(BaseModel):
    photo_ids: list[int]
    public_job_token: str


class PersonDownloadRequest(BaseModel):
    public_job_token: str


def _fetch_image_bytes(photo: dict) -> bytes:
    """Retrieves original image bytes dynamically from local job storage or Google Drive via API key."""
    # 1. Check if stored on disk via storage_path
    storage_path = photo.get("storage_path")
    if storage_path and os.path.exists(storage_path):
        with open(storage_path, "rb") as f:
            return f.read()

    # 2. Check in DATA_DIR / images / {job_id} /
    job_dir = DATA_DIR / "images" / str(photo["job_id"])
    candidate_id = job_dir / f"{photo['drive_file_id']}_{photo['filename']}"
    if candidate_id.exists():
        with open(candidate_id, "rb") as f:
            return f.read()

    candidate_direct_id = job_dir / photo["drive_file_id"]
    if candidate_direct_id.exists():
        with open(candidate_direct_id, "rb") as f:
            return f.read()

    candidate_name = job_dir / photo["filename"]
    if candidate_name.exists():
        with open(candidate_name, "rb") as f:
            return f.read()

    # 3. Fallback: Fetch directly from Google Drive using server API key
    source_id = photo.get("source_id")
    resourcekey = None
    if source_id and source_id > 0:
        with db.get_conn() as conn:
            row = conn.execute("SELECT resourcekey FROM job_sources WHERE id = ?", (source_id,)).fetchone()
            resourcekey = row["resourcekey"] if row else None
            
    try:
        from app.drive_service import fetch_file_bytes, get_drive_service
        service = get_drive_service()
        file_bytes = fetch_file_bytes(service, photo["drive_file_id"], resourcekey)
        return file_bytes
    except Exception as e:
        print(f"[ERROR] Failed to fetch image bytes for photo ID {photo['id']}: {e}")

    raise HTTPException(404, f"Photo image file for ID {photo['id']} not found.")


def _stream_zip(photos: list[dict]):
    """Generates ZIP archive bytes on the fly, downloading one photo at a time."""
    def file_entries():
        seen_filenames = {}
        for p in photos:
            filename = p["filename"]
            # Avoid duplicate filenames in the zip
            base, ext = os.path.splitext(filename)
            count = seen_filenames.get(filename, 0)
            if count > 0:
                filename = f"{base} ({count}){ext}"
                seen_filenames[p["filename"]] += 1
            else:
                seen_filenames[filename] = 1
                
            def chunks():
                try:
                    file_bytes = _fetch_image_bytes(p)
                    yield file_bytes
                except Exception as e:
                    print(f"[ERROR] Failed to fetch bytes for photo_id={p['id']} during ZIP stream: {e}")
                    
            yield (filename, datetime.datetime.now(), 0o600, ZIP_32, chunks())
            
    return stream_zip(file_entries())


from concurrent.futures import ThreadPoolExecutor

WORKER_POOL = ThreadPoolExecutor(max_workers=2, thread_name_prefix="JobWorker")


def run_job_safe(job_id: int, public_token: str, links: list[str]):
    print(f"[WORKER START] Spawning worker task for job_id={job_id} token={public_token}", flush=True)
    try:
        run_job(job_id, links, public_token)
        print(f"[WORKER DONE] Completed worker task for job_id={job_id}", flush=True)
    except Exception as e:
        print(f"[FATAL WORKER ERROR job_id={job_id}]: {e}", flush=True)
        traceback.print_exc()
        try:
            db.update_job(job_id, status="error", message=f"Processing failed: {e}")
            db.update_job_by_token(public_token, status="error", message=f"Processing failed: {e}")
        except Exception:
            pass


# ---------- Route Handlers ----------

@app.post("/api/process")
def process_folder(req: ProcessRequest):
    """
    Initiates asynchronous face clustering for one or more publicly shared Google Drive folders.
    Requires no OAuth or user login.
    """
    # Validate folder_links
    if req.folder_links is None:
        raise HTTPException(400, "folder_links is required")
    if len(req.folder_links) == 0:
        raise HTTPException(400, "At least one non-empty folder link must be provided")
        
    links = [lnk.strip() for lnk in req.folder_links]
    
    # Reject blank links
    for l in req.folder_links:
        if not l.strip():
            raise HTTPException(400, "Blank links are not allowed")
            
    # Reject duplicate links
    if len(links) != len(set(links)):
        raise HTTPException(400, "Duplicate folder links are not allowed")
        
    # Create job (obtaining internal ID and public token)
    job_id, public_token = db.create_job()
    
    # Create job sources
    for link in links:
        db.create_job_source(job_id, link)
        
    db.update_job_by_token(public_token, status="connecting", message="Connecting to Google Drive...")
    WORKER_POOL.submit(run_job_safe, job_id, public_token, links)
    
    return {"job_id": public_token}













@app.get("/api/jobs/{public_job_token}")
def get_job(public_job_token: str):
    job = db.get_job_by_token(public_job_token)
    if not job:
        raise HTTPException(
            status_code=404,
            detail="Job not found. The job ID may be invalid or expired."
        )
        
    job_dict = dict(job)
    # Hide internal integer id from the browser
    job_dict["id"] = public_job_token
    try:
        with db.get_conn() as conn:
            row = conn.execute(
                "SELECT COUNT(faces.id) AS cnt FROM faces JOIN photos ON faces.photo_id = photos.id WHERE photos.job_id = ?",
                (job["id"],)
            ).fetchone()
            job_dict["faces_count"] = row["cnt"] if row else 0
    except Exception:
        job_dict["faces_count"] = 0
    return job_dict





@app.get("/api/jobs/{public_job_token}/people")
def get_people(public_job_token: str):
    job = db.get_job_by_token(public_job_token)
    if not job:
        raise HTTPException(404, "Job not found")
    return db.list_people(job["id"])


@app.get("/api/people/{person_id}/photos")
def get_person_photos(person_id: int, token: str):
    # Verify ownership
    job = db.get_job_by_token(token)
    if not job:
        raise HTTPException(404, "Job not found")
        
    with db.get_conn() as conn:
        row = conn.execute("SELECT job_id FROM people WHERE id = ?", (person_id,)).fetchone()
        if not row or row["job_id"] != job["id"]:
            raise HTTPException(404, "Person not found")
            
    photos = db.get_photos_for_person(person_id)
    return [
        {
            "id": p["id"],
            "filename": p["filename"],
            "url": f"/api/photos/{p['id']}/image?token={token}"
        }
        for p in photos
    ]


@app.patch("/api/people/{person_id}")
def rename_person(person_id: int, req: RenameRequest, token: str):
    # Verify ownership
    job = db.get_job_by_token(token)
    if not job:
        raise HTTPException(404, "Job not found")
        
    with db.get_conn() as conn:
        row = conn.execute("SELECT job_id FROM people WHERE id = ?", (person_id,)).fetchone()
        if not row or row["job_id"] != job["id"]:
            raise HTTPException(404, "Person not found")
            
    db.rename_person(person_id, req.label.strip() or "Unnamed")
    return {"ok": True}


@app.get("/api/photos/{photo_id}/image")
def get_photo_image(photo_id: int, token: str):
    # Verify ownership
    job = db.get_job_by_token(token)
    if not job:
        raise HTTPException(404, "Job not found")
        
    photo = db.get_photo(photo_id)
    if not photo or photo["job_id"] != job["id"]:
        raise HTTPException(404, "Photo not found")
        
    try:
        file_bytes = _fetch_image_bytes(photo)
    except Exception as e:
        raise HTTPException(500, f"Failed to fetch photo from Google Drive: {e}")
        
    return StreamingResponse(io.BytesIO(file_bytes), media_type="image/jpeg")


@app.get("/api/faces/{face_id}/thumbnail")
def get_face_thumbnail(face_id: int, token: str):
    # Verify ownership
    job = db.get_job_by_token(token)
    if not job:
        raise HTTPException(404, "Job not found")
        
    face = db.get_face(face_id)
    if not face:
        raise HTTPException(404, "Face not found")
        
    photo = db.get_photo(face["photo_id"])
    if not photo or photo["job_id"] != job["id"]:
        raise HTTPException(404, "Photo not found")
        
    try:
        file_bytes = _fetch_image_bytes(photo)
    except Exception as e:
        raise HTTPException(500, f"Failed to fetch photo from Google Drive: {e}")
        
    # Crop the face thumbnail in-memory
    image = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    top, right, bottom, left = face["top"], face["right"], face["bottom"], face["left"]
    
    pad_y = int((bottom - top) * 0.35)
    pad_x = int((right - left) * 0.35)
    box = (
        max(0, left - pad_x),
        max(0, top - pad_y),
        min(image.width, right + pad_x),
        min(image.height, bottom + pad_y),
    )
    cropped = image.crop(box)
    cropped.thumbnail((300, 300))
    
    buffer = io.BytesIO()
    cropped.save(buffer, format="JPEG", quality=85)
    buffer.seek(0)
    return StreamingResponse(buffer, media_type="image/jpeg")


@app.get("/api/photos/{photo_id}/download")
def download_photo(photo_id: int, token: str):
    # Verify ownership
    job = db.get_job_by_token(token)
    if not job:
        raise HTTPException(404, "Job not found")
        
    photo = db.get_photo(photo_id)
    if not photo or photo["job_id"] != job["id"]:
        raise HTTPException(404, "Photo not found")
        
    try:
        file_bytes = _fetch_image_bytes(photo)
    except Exception as e:
        raise HTTPException(500, f"Failed to fetch photo from Google Drive: {e}")
        
    return StreamingResponse(
        io.BytesIO(file_bytes),
        media_type="application/octet-stream",
        headers={"Content-Disposition": f"attachment; filename={photo['filename']}"}
    )


@app.post("/api/photos/download")
def download_photos_zip(req: PhotosDownloadRequest):
    # Verify ownership
    job = db.get_job_by_token(req.public_job_token)
    if not job:
        raise HTTPException(404, "Job not found")
        
    photos = db.get_photos_by_ids(req.photo_ids)
    if len(photos) != len(req.photo_ids):
        raise HTTPException(404, "Some photos not found")
        
    for p in photos:
        if p["job_id"] != job["id"]:
            raise HTTPException(404, "Unauthorized photo access")
            
    # Stream the ZIP of original photos
    zip_stream = _stream_zip(photos)
    return StreamingResponse(
        zip_stream,
        media_type="application/zip",
        headers={"Content-Disposition": "attachment; filename=photos.zip"}
    )


@app.post("/api/people/{person_id}/download")
def download_person_zip(person_id: int, req: PersonDownloadRequest):
    # Verify ownership
    job = db.get_job_by_token(req.public_job_token)
    if not job:
        raise HTTPException(404, "Job not found")
        
    with db.get_conn() as conn:
        row = conn.execute("SELECT job_id FROM people WHERE id = ?", (person_id,)).fetchone()
        if not row or row["job_id"] != job["id"]:
            raise HTTPException(404, "Person not found")
            
    photos = db.get_photos_for_person(person_id)
    zip_stream = _stream_zip(photos)
    return StreamingResponse(
        zip_stream,
        media_type="application/zip",
        headers={"Content-Disposition": f"attachment; filename=person_{person_id}.zip"}
    )

@app.get("/privacy", response_class=HTMLResponse)
def privacy_policy():
    return """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Privacy Policy - Photo Clustering</title>
    <style>
        :root {
            --bg: #0b0c10;
            --panel: #1f2833;
            --ink: #f5f5f7;
            --ink-dim: #c5c6c7;
            --accent: #66fcf1;
            --line: rgba(255, 255, 255, 0.1);
        }
        body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            background-color: var(--bg);
            color: var(--ink);
            max-width: 860px;
            margin: 0 auto;
            padding: 40px 24px;
            line-height: 1.7;
        }
        h1 {
            color: var(--accent);
            font-size: 2.2rem;
            margin-bottom: 8px;
        }
        h2 {
            color: var(--ink);
            font-size: 1.3rem;
            margin-top: 32px;
            margin-bottom: 12px;
            border-bottom: 1px solid var(--line);
            padding-bottom: 6px;
        }
        p, li {
            color: var(--ink-dim);
            margin-bottom: 14px;
            font-size: 1rem;
        }
        ul {
            padding-left: 24px;
            margin-bottom: 16px;
        }
        li {
            margin-bottom: 8px;
        }
        a {
            color: var(--accent);
            text-decoration: none;
        }
        a:hover {
            text-decoration: underline;
        }
        .meta-date {
            color: var(--ink-dim);
            font-size: 0.9rem;
            margin-bottom: 24px;
        }
        .nav-link {
            display: inline-block;
            margin-bottom: 24px;
            font-weight: 500;
        }
    </style>
</head>
<body>
    <a href="/" class="nav-link">&larr; Back to Photo Clustering</a>
    <h1>Privacy Policy</h1>
    <div class="meta-date"><strong>Last updated:</strong> August 30, 2026</div>

    <p>
        <strong>Photo Clustering</strong> (<a href="https://photo-clustering.onrender.com">https://photo-clustering.onrender.com</a>) is a privacy-conscious web application designed to help users organize and cluster photos from Google Drive based on detected faces.
    </p>

    <h2>1. Information We Access</h2>
    <p>
        Photo Clustering accesses <strong>only publicly shared Google Drive folders</strong> ("Anyone with the link &rarr; Viewer") that you submit. We do <strong>not</strong> require you to sign in with a Google account, and we never request or store personal Google OAuth tokens or login credentials.
    </p>

    <h2>2. How We Process and Use Your Photos</h2>
    <p>
        Photos in submitted public folders are processed strictly to detect faces, compute mathematical facial embeddings, and group photos containing the same person:
    </p>
    <ul>
        <li><strong>Session Isolation:</strong> Each processing session is assigned an opaque, cryptographically random job token that prevents other users from accessing or viewing your photos or clustering results.</li>
        <li><strong>Transient Processing:</strong> Facial detection and embeddings are computed during your session. Photos are cached in isolated transient storage solely to generate thumbnails and allow you to download organized groups.</li>
        <li><strong>No Model Training:</strong> Your photos and facial embeddings are never used to train generalized artificial intelligence or machine learning models.</li>
    </ul>

    <h2>3. Data Sharing and Third Parties</h2>
    <p>
        We do not sell, rent, monetize, or disclose your photos, metadata, or clustering results to any third party, advertiser, or data broker.
    </p>

    <h2>4. Contact Us</h2>
    <p>
        If you have any questions or privacy concerns regarding Photo Clustering, please contact:
    </p>
    <p>
        <strong>Email:</strong> <a href="mailto:poorani24official@gmail.com">poorani24official@gmail.com</a>
    </p>
</body>
</html>"""


@app.get("/terms", response_class=HTMLResponse)
def terms_of_service():
    return """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Terms of Service - Photo Clustering</title>
    <style>
        :root {
            --bg: #0b0c10;
            --panel: #1f2833;
            --ink: #f5f5f7;
            --ink-dim: #c5c6c7;
            --accent: #66fcf1;
            --line: rgba(255, 255, 255, 0.1);
        }
        body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            background-color: var(--bg);
            color: var(--ink);
            max-width: 860px;
            margin: 0 auto;
            padding: 40px 24px;
            line-height: 1.7;
        }
        h1 {
            color: var(--accent);
            font-size: 2.2rem;
            margin-bottom: 8px;
        }
        h2 {
            color: var(--ink);
            font-size: 1.3rem;
            margin-top: 32px;
            margin-bottom: 12px;
            border-bottom: 1px solid var(--line);
            padding-bottom: 6px;
        }
        p, li {
            color: var(--ink-dim);
            margin-bottom: 14px;
            font-size: 1rem;
        }
        ul {
            padding-left: 24px;
            margin-bottom: 16px;
        }
        li {
            margin-bottom: 8px;
        }
        a {
            color: var(--accent);
            text-decoration: none;
        }
        a:hover {
            text-decoration: underline;
        }
        .meta-date {
            color: var(--ink-dim);
            font-size: 0.9rem;
            margin-bottom: 24px;
        }
        .nav-link {
            display: inline-block;
            margin-bottom: 24px;
            font-weight: 500;
        }
    </style>
</head>
<body>
    <a href="/" class="nav-link">&larr; Back to Photo Clustering</a>
    <h1>Terms of Service</h1>
    <div class="meta-date"><strong>Last updated:</strong> August 30, 2026</div>

    <h2>1. Acceptance of Terms</h2>
    <p>
        By accessing or using <strong>Photo Clustering</strong> (<a href="https://photo-clustering.onrender.com">https://photo-clustering.onrender.com</a>), you agree to these Terms of Service. If you do not agree to these terms, please do not use the service.
    </p>

    <h2>2. Description of Service</h2>
    <p>
        Photo Clustering provides automated tools to analyze images in publicly accessible Google Drive folders, detect faces, cluster photos by individual, and enable previewing and downloading organized photo groups.
    </p>

    <h2>3. User Responsibilities</h2>
    <p>
        You are solely responsible for ensuring that any Google Drive folder you submit is appropriately configured with "Anyone with the link &rarr; Viewer" permissions and that you possess all necessary rights to process the photos.
    </p>

    <h2>4. Acceptable Use</h2>
    <p>
        You agree not to misuse the service, upload malicious software or unlawful materials, or attempt unauthorized access to other sessions or backend systems.
    </p>

    <h2>5. Disclaimer of Warranties & Limitation of Liability</h2>
    <p>
        Photo Clustering is provided on an "AS IS" and "AS AVAILABLE" basis without warranties of any kind. We do not guarantee uninterrupted availability or flawless facial recognition accuracy.
    </p>

    <h2>6. Contact Information</h2>
    <p>
        For inquiries regarding these Terms of Service, please contact:
    </p>
    <p>
        <strong>Email:</strong> <a href="mailto:poorani24official@gmail.com">poorani24official@gmail.com</a>
    </p>
</body>
</html>"""




class NoCacheStaticFiles(StaticFiles):
    """Custom StaticFiles handler ensuring browsers always receive fresh JS and CSS."""
    async def get_response(self, path: str, scope):
        response = await super().get_response(path, scope)
        response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
        return response


# Serve frontend static assets with no-cache headers
app.mount("/", NoCacheStaticFiles(directory="static", html=True), name="static")

