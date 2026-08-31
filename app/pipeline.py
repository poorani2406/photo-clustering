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
import os
import cv2
import numpy as np

from app import db
from app.config import DATA_DIR, MAX_FILE_SIZE_BYTES, MAX_FILES_PER_JOB
from app.drive_service import (
    get_drive_service,
    list_images_in_folder,
    parse_drive_link,
    fetch_file_bytes,
    DriveApiError,
    DriveErrorCategory,
    InvalidDriveLinkError,
)
from app.face_engine import get_face_engine
from app.clustering import cluster_faces


def _process_source(job_id: int, source_id: int, raw_link: str) -> tuple[list[dict], str]:
    """Parses, connects, and lists files for a single source.
    Returns (files, error_message). If successful, error_message is empty."""
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

        service = get_drive_service()
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

        return files, ""

    except InvalidDriveLinkError as e:
        err_msg = "Invalid Google Drive folder link. Please provide a valid Google Drive folder URL or ID."
        print(f"[ERROR] [_process_source job_id={job_id} source_id={source_id}] {err_msg}")
        db.update_job_source(source_id, status="failed", message=err_msg)
        return [], err_msg

    except DriveApiError as e:
        if e.category == DriveErrorCategory.NOT_FOUND_OR_PRIVATE:
            err_msg = "This Google Drive folder is not publicly accessible. Please change the folder sharing setting to Anyone with the link -> Viewer and try again."
        elif e.category == DriveErrorCategory.DOMAIN_RESTRICTED:
            err_msg = "This Google Drive folder is restricted to specific organization domains. Please share as Anyone with the link -> Viewer."
        elif e.category == DriveErrorCategory.QUOTA_EXCEEDED:
            err_msg = "Google Drive API rate limit or quota exceeded. Please try again later."
        else:
            err_msg = f"Google Drive API error: {e}"

        print(f"[ERROR] [_process_source job_id={job_id} source_id={source_id}] {err_msg}")
        db.update_job_source(source_id, status="failed", message=err_msg)
        return [], err_msg

    except Exception as e:
        err_msg = f"Error accessing Drive folder: {e}"
        print(f"[ERROR] [_process_source job_id={job_id} source_id={source_id}] {err_msg}")
        db.update_job_source(source_id, status="failed", message=err_msg)
        return [], err_msg


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

    # 4. Save to per-job storage for fast thumbnails & ZIP streaming
    job_dir = DATA_DIR / "images" / str(job_id)
    job_dir.mkdir(parents=True, exist_ok=True)
    safe_name = f"{file_id}_{os.path.basename(filename)}"
    storage_path = str(job_dir / safe_name)
    with open(storage_path, "wb") as f_out:
        f_out.write(file_bytes)

    # 5. Decode image from bytes in memory
    img = cv2.imdecode(np.frombuffer(file_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        print(f"[WARNING] Skipping {filename} ({file_id}) because image decoding failed.")
        return False

    # 6. Pass image bytes to face_engine.detect_faces()
    faces = face_engine.detect_faces(file_bytes)

    # 7. Persist photo record
    photo_id = db.insert_photo(
        job_id=job_id,
        source_id=source_id,
        drive_file_id=file_id,
        filename=filename,
        mime_type=mime_type,
        size_bytes=size_bytes,
        content_hash=content_hash,
        storage_path=storage_path
    )

    # 8. Persist faces
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


def run_job(job_id: int, folder_links: list[str], public_token: Optional[str] = None):
    print(f"[WORKER START] job_id={job_id} token={public_token} starting processing for {len(folder_links)} folder link(s)", flush=True)
    
    def _update_status(**fields):
        db.update_job(job_id, **fields)
        if public_token:
            db.update_job_by_token(public_token, **fields)

    try:
        # 1. Immediately transition out of 'pending'
        _update_status(status="connecting", message="Connecting to Google Drive and initializing face engine...")

        # 2. Initialize Drive service
        try:
            service = get_drive_service()
        except Exception as e:
            err_msg = str(e)
            print(f"[WORKER EXCEPTION] job_id={job_id} drive_service_init_error={err_msg}", flush=True)
            _update_status(status="error", message=err_msg)
            return

        # 3. Initialize Face Engine inside try block
        try:
            face_engine = get_face_engine()
        except Exception as e:
            err_msg = f"Failed to initialize Face Engine: {e}"
            print(f"[WORKER EXCEPTION] job_id={job_id} face_engine_init_error={err_msg}", flush=True)
            traceback.print_exc()
            _update_status(status="error", message=err_msg)
            return

        # Retrieve job sources from database, or create them if they do not exist
        existing_sources = db.get_sources_for_job(job_id)
        if not existing_sources:
            for link in folder_links:
                db.create_job_source(job_id, link)
            existing_sources = db.get_sources_for_job(job_id)

        all_files = []
        last_error = ""
        for source in existing_sources:
            files, err_msg = _process_source(job_id, source["id"], source["raw_link"])
            all_files.extend(files)
            if err_msg:
                last_error = err_msg

        # Merge and Deduplicate (Level 1)
        total_discovered = len(all_files)
        deduped_files, duplicate_count = _merge_and_dedupe(all_files)
        _update_status(total_files=total_discovered, duplicate_files_skipped=duplicate_count)

        if not deduped_files:
            error_message = last_error or "No images found in that folder. Please make sure the folder contains images (JPEG, PNG, WEBP, HEIC) and is shared as Anyone with the link -> Viewer."
            _update_status(
                status="error",
                message=error_message,
            )
            print(f"[JOB COMPLETE] job_id={job_id} status=error msg='{error_message}'", flush=True)
            return

        # Enforce MAX_FILES_PER_JOB limit
        if len(deduped_files) > MAX_FILES_PER_JOB:
            error_msg = f"Job exceeded the limit of {MAX_FILES_PER_JOB} files (found {len(deduped_files)} after deduplication)."
            _update_status(status="error", message=error_msg)
            print(f"[JOB COMPLETE] job_id={job_id} status=error limit_exceeded={error_msg}", flush=True)
            raise ValueError(error_msg)

        print(f"[DOWNLOAD COMPLETE] job_id={job_id} prepared {len(deduped_files)} unique photos (discovered={total_discovered}, skipped_dupes={duplicate_count})", flush=True)
        _update_status(status="downloading", processed_files=0)


        seen_hashes = set()
        level_2_skips = 0
        processed_count = 0

        print(f"[FACE DETECTION START] job_id={job_id} starting CPU face detection on {len(deduped_files)} photos")
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

        print(f"[CLUSTERING START] job_id={job_id} grouping detected faces for {processed_count} photos")
        db.update_job(job_id, status="clustering", message="Grouping faces into people...", processed_files=processed_count)

        face_rows = db.get_faces_for_job(job_id)
        assignments = cluster_faces(face_rows)  # face_id -> cluster_index

        db.clear_people_for_job(job_id)

        cluster_to_person = {}
        for face in face_rows:
            cluster_idx = assignments[face["id"]]
            if cluster_idx not in cluster_to_person:
                person_id = db.create_person(job_id, f"Person {cluster_idx + 1}", face["id"])
                cluster_to_person[cluster_idx] = person_id
            db.set_face_person(face["id"], cluster_to_person[cluster_idx])

        people_count = len(cluster_to_person)
        db.update_job(
            job_id,
            status="done",
            message=f"Done. Found {people_count} distinct people across {processed_count} photos.",
        )
        print(f"[JOB COMPLETE] job_id={job_id} status=done (people={people_count}, photos={processed_count})")

    except Exception as e:
        err_msg = f"{type(e).__name__}: {e}"
        print(f"[WORKER EXCEPTION] job_id={job_id} error={err_msg}")
        db.update_job(job_id, status="error", message=err_msg)
        print(f"[JOB COMPLETE] job_id={job_id} status=error err={err_msg}")
        traceback.print_exc()
    except BaseException as be:
        err_msg = f"Fatal worker error ({type(be).__name__}): {be}"
        print(f"[WORKER FATAL EXCEPTION] job_id={job_id} error={err_msg}")
        db.update_job(job_id, status="error", message=err_msg)
        print(f"[JOB COMPLETE] job_id={job_id} status=error err={err_msg}")
        traceback.print_exc()
        raise






def run_direct_upload_job(job_id: int, image_items: list[tuple[str, str, str]]):
    """
    Orchestrates the facial detection and clustering pipeline for user-uploaded images.
    image_items: list of (filename, file_path_str, mime_type)
    """
    import os
    import hashlib
    face_engine = get_face_engine()
    try:
        total_discovered = len(image_items)
        db.update_job(
            job_id,
            status="detecting",
            total_files=total_discovered,
            duplicate_files_skipped=0,
            processed_files=0,
            message="Starting face detection on uploaded photos..."
        )

        if not image_items:
            db.update_job(
                job_id,
                status="error",
                message="No valid images were found in the upload.",
            )
            return

        seen_hashes = set()
        level_2_skips = 0
        processed_count = 0

        for i, (filename, file_path_str, mime_type) in enumerate(image_items, start=1):
            db.update_job(
                job_id,
                status="detecting",
                processed_files=processed_count,
                message=f"Processing {filename} ({i}/{len(image_items)})",
            )
            try:
                with open(file_path_str, "rb") as f:
                    file_bytes = f.read()

                size_bytes = len(file_bytes)
                if size_bytes > MAX_FILE_SIZE_BYTES:
                    print(f"[WARNING] Skipping {filename} because size {size_bytes} exceeds limit of {MAX_FILE_SIZE_BYTES} bytes.")
                    continue

                content_hash = hashlib.sha256(file_bytes).hexdigest()
                if content_hash in seen_hashes:
                    print(f"[DEBUG] Skipping content duplicate for job {job_id}: {filename} (hash={content_hash})")
                    level_2_skips += 1
                    db.update_job(job_id, duplicate_files_skipped=level_2_skips)
                    continue
                seen_hashes.add(content_hash)

                img = cv2.imdecode(np.frombuffer(file_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
                if img is None:
                    print(f"[WARNING] Skipping {filename} because image decoding failed.")
                    continue

                faces = face_engine.detect_faces(file_bytes)

                photo_id = db.insert_photo(
                    job_id=job_id,
                    source_id=0,
                    drive_file_id=os.path.basename(file_path_str),
                    filename=filename,
                    mime_type=mime_type,
                    size_bytes=size_bytes,
                    content_hash=content_hash,
                    storage_path=file_path_str
                )

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

                processed_count += 1
                db.update_job(job_id, processed_files=processed_count)

            except Exception as fe:
                print(f"[ERROR] Failed processing file {filename}: {fe}")

        db.update_job(
            job_id,
            status="clustering",
            message="Grouping faces into people...",
            processed_files=processed_count
        )

        face_rows = db.get_faces_for_job(job_id)
        assignments = cluster_faces(face_rows)

        db.clear_people_for_job(job_id)

        cluster_to_person = {}
        for face in face_rows:
            cluster_idx = assignments[face["id"]]
            if cluster_idx not in cluster_to_person:
                person_id = db.create_person(job_id, f"Person {cluster_idx + 1}", face["id"])
                cluster_to_person[cluster_idx] = person_id
            db.set_face_person(face["id"], cluster_to_person[cluster_idx])

        people_count = len(cluster_to_person)
        db.update_job(
            job_id,
            status="done",
            message=f"Done. Found {people_count} distinct people across {processed_count} photos.",
        )

    except Exception as e:
        db.update_job(job_id, status="error", message=f"{type(e).__name__}: {e}")
        traceback.print_exc()

