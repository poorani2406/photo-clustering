"""
Orchestrates the full Stage 1 flow for one job:

  list files in the Drive folders
    -> merge and deduplicate files across sources
    -> fetch each image in memory
    -> detect faces + generate embeddings
    -> cluster every embedding in the job
    -> save "people" and which photos belong to each of them

Runs on a background thread so the HTTP request that kicks it off can
return immediately, and the frontend polls /api/jobs/{id} for progress.
"""
import traceback
import cv2
import numpy as np

from app import db
from app.config import MAX_FILE_SIZE_BYTES, MAX_FILES_PER_JOB
from app.drive_service import (
    get_drive_service,
    list_images_in_folder,
    parse_drive_link,
    fetch_file_bytes,
    DriveApiError,
)
from app.face_engine import get_face_engine
from app.clustering import cluster_faces


def _process_source(job_id: int, source_id: int, raw_link: str) -> list[dict]:
    """Parses, connects, and lists files for a single source.
    Handles individual source failure gracefully without failing the entire job."""
    try:
        # Parse Drive Link
        link_info = parse_drive_link(raw_link)
        folder_id = link_info.folder_id
        resourcekey = link_info.resourcekey

        # Update source state
        db.update_job_source(
            source_id,
            folder_id=folder_id,
            resourcekey=resourcekey,
            status="listing",
            message="Listing files in Drive folder..."
        )

        token_json = db.get_oauth_token(job_id)
        service = get_drive_service(token_json)
        files = list_images_in_folder(service, folder_id, resourcekey)

        # Update source to completed
        db.update_job_source(
            source_id,
            status="completed",
            message=f"Found {len(files)} files."
        )

        # Attach source metadata to each file entry
        for f in files:
            f["source_id"] = source_id
            f["resourcekey"] = resourcekey

        return files

    except Exception as e:
        err_msg = f"{type(e).__name__}: {e}"
        print(f"[ERROR] [_process_source job_id={job_id} source_id={source_id}] {err_msg}")
        db.update_job_source(
            source_id,
            status="failed",
            message=err_msg
        )
        return []


def _merge_and_dedupe(all_source_files: list[dict]) -> tuple[list[dict], int]:
    """Merges all source files, deduplicating them using drive_file_id.
    Returns a tuple of (deduplicated_files, duplicate_count)."""
    seen_ids = set()
    deduped = []
    skipped = 0
    for f in all_source_files:
        file_id = f["id"]
        if file_id in seen_ids:
            skipped += 1
        else:
            seen_ids.add(file_id)
            deduped.append(f)
    return deduped, skipped


def _process_one_file_sequential(service, face_engine, job_id: int, file_entry: dict, seen_hashes: set) -> bool:
    """Processes a single Google Drive file sequentially in memory.
    Returns True if skipped as a content duplicate, False otherwise."""
    import hashlib
    file_id = file_entry["id"]
    filename = file_entry["name"]
    source_id = file_entry["source_id"]
    resourcekey = file_entry.get("resourcekey")
    mime_type = file_entry.get("mimeType")

    # 1. Fetch file bytes
    file_bytes = fetch_file_bytes(service, file_id, resourcekey)

    # 2. Check MAX_FILE_SIZE_BYTES
    size_bytes = len(file_bytes)
    if size_bytes > MAX_FILE_SIZE_BYTES:
        print(f"[WARNING] Skipping {filename} ({file_id}) because size {size_bytes} exceeds limit of {MAX_FILE_SIZE_BYTES} bytes.")
        return False

    # 3. Calculate content hash to check Level 2 duplicates
    content_hash = hashlib.sha256(file_bytes).hexdigest()
    if content_hash in seen_hashes:
        print(f"[DEBUG] Skipping content duplicate for job {job_id}: {filename} (hash={content_hash})")
        return True
    seen_hashes.add(content_hash)

    # 4. Decode image from bytes in memory
    img = cv2.imdecode(np.frombuffer(file_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        print(f"[WARNING] Skipping {filename} ({file_id}) because image decoding failed.")
        return False

    # 5. Pass image bytes to face_engine.detect_faces()
    faces = face_engine.detect_faces(file_bytes)

    # 6. Persist photo using new Phase 2 schema with content_hash
    photo_id = db.insert_photo(
        job_id=job_id,
        source_id=source_id,
        drive_file_id=file_id,
        filename=filename,
        mime_type=mime_type,
        size_bytes=size_bytes,
        content_hash=content_hash
    )

    # 7. Persist faces
    for face in faces:
        db.insert_face(
            photo_id,
            face["top"],
            face["right"],
            face["bottom"],
            face["left"],
            face["embedding"]
        )
    db.set_photo_face_count(photo_id, len(faces))
    return False


def run_job(job_id: int, folder_links: list[str]):
    face_engine = get_face_engine()
    try:
        db.update_job(job_id, status="connecting", message="Connecting to Google Drive...")
        token_json = db.get_oauth_token(job_id)
        service = get_drive_service(token_json)

        # Retrieve job sources from database, or create them if they do not exist
        existing_sources = db.get_sources_for_job(job_id)
        if not existing_sources:
            for link in folder_links:
                db.create_job_source(job_id, link)
            existing_sources = db.get_sources_for_job(job_id)

        all_files = []
        for source in existing_sources:
            files = _process_source(job_id, source["id"], source["raw_link"])
            all_files.extend(files)

        # Merge and Deduplicate (Level 1)
        total_discovered = len(all_files)
        deduped_files, duplicate_count = _merge_and_dedupe(all_files)
        db.update_job(job_id, total_files=total_discovered, duplicate_files_skipped=duplicate_count)

        if not deduped_files:
            db.update_job(
                job_id,
                status="error",
                message="No images found in that folder. Check the folder ID and sharing access.",
            )
            return

        # Enforce MAX_FILES_PER_JOB limit
        if len(deduped_files) > MAX_FILES_PER_JOB:
            error_msg = f"Job exceeded the limit of {MAX_FILES_PER_JOB} files (found {len(deduped_files)} after deduplication)."
            db.update_job(job_id, status="error", message=error_msg)
            raise ValueError(error_msg)

        db.update_job(job_id, status="downloading", processed_files=0)

        seen_hashes = set()
        level_2_skips = 0
        processed_count = 0

        for i, f in enumerate(deduped_files, start=1):
            db.update_job(
                job_id,
                status="detecting",
                processed_files=processed_count,
                message=f"Processing {f['name']} ({i}/{len(deduped_files)})",
            )
            skipped = _process_one_file_sequential(service, face_engine, job_id, f, seen_hashes)
            if skipped:
                level_2_skips += 1
                db.update_job(job_id, duplicate_files_skipped=duplicate_count + level_2_skips)
            else:
                processed_count += 1
                db.update_job(job_id, processed_files=processed_count)

        db.update_job(job_id, status="clustering", message="Grouping faces into people...", processed_files=processed_count)

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
            message=f"Done. Found {people_count} distinct people across {processed_count} photos.",
        )

    except Exception as e:
        db.update_job(job_id, status="error", message=f"{type(e).__name__}: {e}")
        traceback.print_exc()
