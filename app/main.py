import io
import threading

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from PIL import Image

from app import db
from app.config import IMAGES_DIR
from app.pipeline import run_job

app = FastAPI(title="Face Clustering - Stage 1")

db.init_db()


class ProcessRequest(BaseModel):
    folder_id: str


class RenameRequest(BaseModel):
    label: str


@app.post("/api/process")
def process_folder(req: ProcessRequest):
    folder_id = req.folder_id.strip()
    if not folder_id:
        raise HTTPException(400, "folder_id is required")

    job_id = db.create_job(folder_id)
    thread = threading.Thread(target=run_job, args=(job_id, folder_id), daemon=True)
    thread.start()
    return {"job_id": job_id}


@app.get("/api/jobs/{job_id}")
def get_job(job_id: int):
    job = db.get_job(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    return job


@app.get("/api/jobs/{job_id}/people")
def get_people(job_id: int):
    job = db.get_job(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    return db.list_people(job_id)


@app.get("/api/people/{person_id}/photos")
def get_person_photos(person_id: int):
    photos = db.get_photos_for_person(person_id)
    return [{"id": p["id"], "filename": p["filename"], "url": f"/api/photos/{p['id']}/image"} for p in photos]


@app.patch("/api/people/{person_id}")
def rename_person(person_id: int, req: RenameRequest):
    db.rename_person(person_id, req.label.strip() or "Unnamed")
    return {"ok": True}


@app.get("/api/photos/{photo_id}/image")
def get_photo_image(photo_id: int):
    photo = db.get_photo(photo_id)
    if not photo:
        raise HTTPException(404, "Photo not found")
    return _serve_image_file(photo["local_path"])


@app.get("/api/faces/{face_id}/thumbnail")
def get_face_thumbnail(face_id: int):
    """Crops just the face out of its source photo, so the people grid
    can show a tight headshot instead of the whole picture."""
    face = db.get_face(face_id)
    if not face:
        raise HTTPException(404, "Face not found")
    photo = db.get_photo(face["photo_id"])
    if not photo:
        raise HTTPException(404, "Source photo not found")

    image = Image.open(photo["local_path"]).convert("RGB")

    top, right, bottom, left = face["top"], face["right"], face["bottom"], face["left"]
    # Pad the crop a bit so it doesn't feel like a mugshot.
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


def _serve_image_file(path: str):
    with open(path, "rb") as f:
        data = f.read()
    return StreamingResponse(io.BytesIO(data), media_type="image/jpeg")


# Serve the frontend last, so /api/* routes above take priority.
app.mount("/", StaticFiles(directory="static", html=True), name="static")
