"""
Orchestrates the full Stage 1 flow for one job:

  list files in the Drive folder
    -> download each image
      -> detect faces + generate embeddings
        -> cluster every embedding in the job
          -> save "people" and which photos belong to each of them

Runs on a background thread so the HTTP request that kicks it off can
return immediately, and the frontend polls /api/jobs/{id} for progress.
"""
import traceback
from pathlib import Path

from app import db
from app.config import IMAGES_DIR
from app.drive_service import get_drive_service, list_images_in_folder, download_file
from app.face_engine import get_face_engine
from app.clustering import cluster_faces


def run_job(job_id: int, folder_id: str):
    face_engine = get_face_engine()
    try:
        db.update_job(job_id, status="connecting", message="Connecting to Google Drive...")
        service = get_drive_service()

        db.update_job(job_id, status="listing", message="Listing images in folder...")
        files = list_images_in_folder(service, folder_id)

        if not files:
            db.update_job(
                job_id,
                status="error",
                message="No images found in that folder. Check the folder ID and sharing access.",
            )
            return

        db.update_job(job_id, total_files=len(files), status="downloading", processed_files=0)

        job_dir: Path = IMAGES_DIR / str(job_id)
        job_dir.mkdir(parents=True, exist_ok=True)

        for i, f in enumerate(files, start=1):
            dest = job_dir / f"{f['id']}_{f['name']}"
            if not dest.exists():
                download_file(service, f["id"], dest)

            photo_id = db.insert_photo(job_id, f["id"], f["name"], str(dest))

            db.update_job(
                job_id,
                status="detecting",
                processed_files=i,
                message=f"Processing {f['name']} ({i}/{len(files)})",
            )

            faces = face_engine.detect_faces(str(dest))
            for face in faces:
                db.insert_face(
                    photo_id,
                    face["top"],
                    face["right"],
                    face["bottom"],
                    face["left"],
                    face["embedding"],
                )
            db.set_photo_face_count(photo_id, len(faces))

        db.update_job(job_id, status="clustering", message="Grouping faces into people...")

        print(f"[DEBUG] [run_job {job_id}] Starting clustering stage.")
        face_rows = db.get_faces_for_job(job_id)
        print(f"[DEBUG] [run_job {job_id}] db.get_faces_for_job returned {len(face_rows)} face rows.")

        assignments = cluster_faces(face_rows)  # face_id -> cluster_index
        print(f"[DEBUG] [run_job {job_id}] cluster_faces returned {len(assignments)} assignments.")

        db.clear_people_for_job(job_id)
        print(f"[DEBUG] [run_job {job_id}] Cleared existing people for job.")

        cluster_to_person = {}
        for face in face_rows:
            cluster_idx = assignments[face["id"]]
            print(f"[DEBUG] [run_job {job_id}] Mapping face_id={face['id']} to cluster_idx={cluster_idx}")
            if cluster_idx not in cluster_to_person:
                person_id = db.create_person(job_id, f"Person {cluster_idx + 1}", face["id"])
                cluster_to_person[cluster_idx] = person_id
                print(f"[DEBUG] [run_job {job_id}] Created new person_id={person_id} for cluster_idx={cluster_idx}")
            db.set_face_person(face["id"], cluster_to_person[cluster_idx])

        people_count = len(cluster_to_person)
        print(f"[DEBUG] [run_job {job_id}] Clustering completed. Found {people_count} distinct people.")
        db.update_job(
            job_id,
            status="done",
            message=f"Done. Found {people_count} distinct people across {len(files)} photos.",
        )

    except Exception as e:
        db.update_job(job_id, status="error", message=f"{type(e).__name__}: {e}")
        traceback.print_exc()
