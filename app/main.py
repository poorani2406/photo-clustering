import io
import os
import secrets
import json
import threading
import datetime
from typing import Optional

from fastapi import FastAPI, HTTPException, Request, Query
from fastapi.responses import StreamingResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from PIL import Image
from stream_zip import stream_zip, ZIP_32

from app import db
from app.pipeline import run_job


app = FastAPI(title="Face Clustering - Phase 5")

# Initialize/migrate database on application startup
db.init_db()


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


# ---------- Helpers ----------

def _fetch_image_bytes(photo: dict) -> bytes:
    """Retrieves original image bytes dynamically from Google Drive in memory."""
    source_id = photo["source_id"]
    
    # Retrieve the source's resourcekey from the database
    with db.get_conn() as conn:
        row = conn.execute("SELECT resourcekey FROM job_sources WHERE id = ?", (source_id,)).fetchone()
        resourcekey = row["resourcekey"] if row else None
        
    from app.drive_service import fetch_file_bytes, get_drive_service
    token_json = db.get_oauth_token(photo["job_id"])
    service = get_drive_service(token_json)
    return fetch_file_bytes(service, photo["drive_file_id"], resourcekey)


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


# ---------- Route Handlers ----------

@app.post("/api/process")
def process_folder(req: ProcessRequest):
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
        
    # For local test suites backward compatibility, copy local token.json if exists
    from app.config import GOOGLE_TOKEN_FILE
    if os.path.exists(GOOGLE_TOKEN_FILE):
        try:
            with open(GOOGLE_TOKEN_FILE, "r") as tf:
                db.save_oauth_token(job_id, tf.read())
        except Exception as e:
            print(f"[WARNING] Failed to seed job with local token file: {e}")
            
    # Start the job sequentially in a background thread
    thread = threading.Thread(target=run_job, args=(job_id, links), daemon=True)
    thread.start()
    
    return {"job_id": public_token}


@app.post("/api/oauth/initiate")
def initiate_oauth(req: ProcessRequest, request: Request):
    # Validate folder_links
    if req.folder_links is None:
        raise HTTPException(400, "folder_links is required")
    if len(req.folder_links) == 0:
        raise HTTPException(400, "At least one non-empty folder link must be provided")
        
    links = [lnk.strip() for lnk in req.folder_links]
    for l in links:
        if not l:
            raise HTTPException(400, "Blank links are not allowed")
            
    if len(links) != len(set(links)):
        raise HTTPException(400, "Duplicate folder links are not allowed")
        
    # Create job
    job_id, public_token = db.create_job()
    for link in links:
        db.create_job_source(job_id, link)
        
    from app.config import GOOGLE_CREDENTIALS_FILE, GOOGLE_SCOPES
    if not os.path.exists(GOOGLE_CREDENTIALS_FILE):
        raise HTTPException(500, f"Google client credentials file '{GOOGLE_CREDENTIALS_FILE}' is missing on the server.")
        
    redirect_uri = f"{request.url.scheme}://{request.url.netloc}/api/oauth/callback"
    if "onrender.com" in request.url.netloc:
        redirect_uri = f"https://{request.url.netloc}/api/oauth/callback"
        
    from google_auth_oauthlib.flow import Flow
    flow = Flow.from_client_secrets_file(
        GOOGLE_CREDENTIALS_FILE,
        scopes=GOOGLE_SCOPES,
        redirect_uri=redirect_uri
    )
    
    state = secrets.token_urlsafe(16)
    db.save_oauth_state(state, public_token)
    
    authorization_url, _ = flow.authorization_url(
        access_type='offline',
        include_granted_scopes='true',
        state=state,
        prompt='consent'
    )
    
    return {"auth_url": authorization_url, "public_job_token": public_token}


@app.get("/api/oauth/callback")
def oauth_callback(request: Request,code: str = Query(...),state: str = Query(...),):    
    public_token = db.pop_oauth_state(state)
    if not public_token:
        raise HTTPException(400, "Invalid or expired state token.")
        
    job = db.get_job_by_token(public_token)
    if not job:
        raise HTTPException(404, "Job not found.")
        
    from app.config import GOOGLE_CREDENTIALS_FILE, GOOGLE_SCOPES
    redirect_uri = f"{request.url.scheme}://{request.url.netloc}/api/oauth/callback"
    if "onrender.com" in request.url.netloc:
        redirect_uri = f"https://{request.url.netloc}/api/oauth/callback"
        
    from google_auth_oauthlib.flow import Flow
    flow = Flow.from_client_secrets_file(
        GOOGLE_CREDENTIALS_FILE,
        scopes=GOOGLE_SCOPES,
        redirect_uri=redirect_uri
    )
    
    try:
        flow.fetch_token(code=code)
        credentials = flow.credentials
        db.save_oauth_token(job["id"], credentials.to_json())
    except Exception as e:
        raise HTTPException(500, f"Token exchange failed: {e}")
        
    return RedirectResponse(f"/?job={public_token}&auth=success")


@app.post("/api/process/start")
def start_processing_job(req: StartProcessRequest):
    job = db.get_job_by_token(req.public_job_token)
    if not job:
        raise HTTPException(404, "Job not found")
        
    token_json = db.get_oauth_token(job["id"])
    if not token_json:
        raise HTTPException(400, "Job is not authorized. Please connect your Google account.")
        
    sources = db.get_sources_for_job(job["id"])
    links = [s["raw_link"] for s in sources]
    
    if job["status"] not in ("connecting", "listing", "downloading", "detecting", "clustering"):
        thread = threading.Thread(target=run_job, args=(job["id"], links), daemon=True)
        thread.start()
        
    return {"status": "started"}


@app.get("/api/jobs/{public_job_token}")
def get_job(public_job_token: str):
    job = db.get_job_by_token(public_job_token)
    if not job:
        raise HTTPException(404, "Job not found")
        
    job_dict = dict(job)
    # Hide internal integer id from the browser
    job_dict["id"] = public_job_token
    
    # Calculate faces count for this job in real-time
    with db.get_conn() as conn:
        row = conn.execute(
            "SELECT COUNT(faces.id) AS c FROM faces JOIN photos ON photos.id = faces.photo_id WHERE photos.job_id = ?",
            (job["id"],)
        ).fetchone()
        faces_count = row["c"] if row else 0
    job_dict["faces_count"] = faces_count
    
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


# Serve frontend static assets last
app.mount("/", StaticFiles(directory="static", html=True), name="static")
