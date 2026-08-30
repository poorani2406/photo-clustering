import io
import os
import secrets
import json
import threading
import datetime
from typing import Optional

from fastapi import FastAPI, HTTPException, Request, Query
from fastapi.responses import StreamingResponse, RedirectResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from PIL import Image
from stream_zip import stream_zip, ZIP_32

from app import db
from app.pipeline import run_job


app = FastAPI(title="Photo Clustering")

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

def _get_redirect_uri(request: Request) -> str:
    """Builds the canonical OAuth redirect URI for this server."""
    redirect_override = os.getenv("OAUTH_REDIRECT_URI")
    if redirect_override and redirect_override.strip():
        return redirect_override.strip()

    # Detect protocol from proxy headers (Render, Cloudflare, Nginx, etc.)
    proto = request.headers.get("x-forwarded-proto", request.url.scheme)
    host = request.headers.get("x-forwarded-host", request.url.netloc)
    if "onrender.com" in host or proto == "https":
        proto = "https"

    return f"{proto}://{host}/api/oauth/callback"


def _create_oauth_flow(request: Request):
    """Creates a Google OAuth Flow using either client JSON env var or credentials file."""
    from app.config import get_google_client_config, GOOGLE_SCOPES
    from google_auth_oauthlib.flow import Flow

    client_config = get_google_client_config()
    if not client_config:
        raise HTTPException(
            500,
            "Google OAuth credentials are not configured on the server. "
            "Please set GOOGLE_CLIENT_SECRET_JSON environment variable or provide credentials.json file."
        )

    redirect_uri = _get_redirect_uri(request)

    if isinstance(client_config, dict):
        return Flow.from_client_config(
            client_config,
            scopes=GOOGLE_SCOPES,
            redirect_uri=redirect_uri
        )
    else:
        return Flow.from_client_secrets_file(
            client_config,
            scopes=GOOGLE_SCOPES,
            redirect_uri=redirect_uri
        )


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
        
    flow = _create_oauth_flow(request)
    
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
def oauth_callback(request: Request, code: str = Query(...), state: str = Query(...)):    
    public_token = db.pop_oauth_state(state)
    if not public_token:
        raise HTTPException(400, "Invalid or expired state token.")
        
    job = db.get_job_by_token(public_token)
    if not job:
        raise HTTPException(404, "Job not found.")
        
    flow = _create_oauth_flow(request)
    
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
        .highlight-box {
            background: var(--panel);
            border: 1px solid var(--line);
            border-left: 4px solid var(--accent);
            padding: 16px 20px;
            border-radius: 8px;
            margin: 20px 0;
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
        <strong>Photo Clustering</strong> (<a href="https://photo-clustering.onrender.com">https://photo-clustering.onrender.com</a>) is a web application designed to help users organize and cluster photos from Google Drive based on detected faces.
    </p>

    <h2>1. Information We Access via Google OAuth</h2>
    <p>
        When you choose to connect your Google account, Photo Clustering requests the following read-only OAuth permission:
    </p>
    <ul>
        <li><code>https://www.googleapis.com/auth/drive.readonly</code>: Used strictly to read image files and folder metadata (such as file names, IDs, and MIME types) from the Google Drive folders you provide.</li>
    </ul>

    <h2>2. How We Use and Process Your Data</h2>
    <p>
        Our application uses Google Drive data solely to perform facial detection, feature extraction, and clustering for your requested session:
    </p>
    <ul>
        <li><strong>In-Memory Processing:</strong> Photos are downloaded into transient memory buffers, analyzed for face embeddings, and clustered. Original photos are not permanently stored on our application servers or database disks.</li>
        <li><strong>Transient Job Tokens:</strong> A session-isolated job token allows you to review clustered face groups and stream original photos or ZIP archives directly during your session.</li>
    </ul>

    <div class="highlight-box">
        <h3 style="color: var(--accent); margin-bottom: 8px;">Google API Services User Data Policy Compliance</h3>
        <p style="margin-bottom: 0;">
            Photo Clustering's use and transfer of information received from Google APIs to any other app will adhere to the <a href="https://developers.google.com/terms/api-services-user-data-policy" target="_blank" rel="noopener">Google API Services User Data Policy</a>, including the Limited Use requirements.
        </p>
    </div>

    <h2>3. Data Sharing, Selling, and Disclosure</h2>
    <p>
        We value your privacy:
    </p>
    <ul>
        <li>We <strong>do not sell</strong> user data or photos to third parties.</li>
        <li>We <strong>do not share</strong> your photos or personal information with advertisers or data brokers.</li>
        <li>We <strong>do not use</strong> your photos or facial embeddings to train generalized artificial intelligence models.</li>
    </ul>

    <h2>4. Data Retention and Security</h2>
    <p>
        We employ reasonable security safeguards to protect your session. Authorization tokens are stored in isolated per-job database records and are used exclusively to fulfill your clustering requests. No photos are retained permanently on disk.
    </p>

    <h2>5. User Control and Data Deletion</h2>
    <p>
        You maintain complete control over your Google account permissions at all times:
    </p>
    <ul>
        <li>You can revoke Photo Clustering's access to your Google Drive at any time via your <a href="https://myaccount.google.com/permissions" target="_blank" rel="noopener">Google Account Security Permissions</a> page.</li>
        <li>You can request deletion of any session data or records associated with your use of the service by contacting us at the email below.</li>
    </ul>

    <h2>6. Contact Us</h2>
    <p>
        If you have questions about this Privacy Policy, your data, or our compliance with Google API policies, please contact:
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
        By accessing or using <strong>Photo Clustering</strong> (<a href="https://photo-clustering.onrender.com">https://photo-clustering.onrender.com</a>), you agree to be bound by these Terms of Service. If you do not agree to these terms, please do not use the service.
    </p>

    <h2>2. Description of Service</h2>
    <p>
        Photo Clustering provides tools to analyze images stored in your Google Drive, detect faces, cluster them by individual, and enable previewing and downloading organized photo groups.
    </p>

    <h2>3. User Responsibilities & Google Drive Content</h2>
    <p>
        You are solely responsible for any Google Drive folders and image files you submit for processing. You represent and warrant that you have all necessary rights, licenses, and permissions to access and process the images you provide.
    </p>

    <h2>4. Acceptable Use</h2>
    <p>
        You agree not to misuse the service, attempt unauthorized access to our systems, or use the service for any unlawful or harmful purposes.
    </p>

    <h2>5. Disclaimer of Warranties & Limitation of Liability</h2>
    <p>
        Photo Clustering is provided on an "AS IS" and "AS AVAILABLE" basis without warranties of any kind, whether express or implied. We do not guarantee that the service will be uninterrupted, error-free, or entirely accurate in facial recognition results.
    </p>

    <h2>6. Changes to Terms</h2>
    <p>
        We reserve the right to update these terms at any time. Changes will be posted on this page with an updated revision date.
    </p>

    <h2>7. Contact Information</h2>
    <p>
        For inquiries regarding these Terms of Service, please contact:
    </p>
    <p>
        <strong>Email:</strong> <a href="mailto:poorani24official@gmail.com">poorani24official@gmail.com</a>
    </p>
</body>
</html>"""




# Serve frontend static assets last
app.mount("/", StaticFiles(directory="static", html=True), name="static")
