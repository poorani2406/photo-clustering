"""
Handles everything related to talking to Google Drive using API key access:
- Listing every image in a given publicly shared folder
- Downloading a file's bytes to memory
"""
import io
import re
import json
import os
from enum import Enum
from typing import NamedTuple, Optional
from urllib.parse import urlparse, parse_qs

from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
from googleapiclient.errors import HttpError

from app.config import (
    GOOGLE_DRIVE_API_KEY,
    IMAGE_MIME_TYPES,
)


class DriveLinkInfo(NamedTuple):
    folder_id: str
    resourcekey: Optional[str] = None


class InvalidDriveLinkError(ValueError):
    pass


class DriveErrorCategory(str, Enum):
    NOT_FOUND_OR_PRIVATE = "not_found_or_private"
    DOMAIN_RESTRICTED = "domain_restricted"
    API_ERROR = "api_error"
    QUOTA_EXCEEDED = "quota_exceeded"


class DriveApiError(Exception):
    def __init__(self, category: DriveErrorCategory, message: str, original_error: Optional[Exception] = None):
        super().__init__(message)
        self.category = category
        self.original_error = original_error


def parse_drive_link(raw_link: str) -> DriveLinkInfo:
    if not raw_link or not isinstance(raw_link, str):
        raise InvalidDriveLinkError("Drive link is empty or not a string")
    
    raw_link = raw_link.strip()
    
    # Check if it looks like a URL
    if "drive.google.com" in raw_link or "http://" in raw_link or "https://" in raw_link:
        try:
            parsed = urlparse(raw_link)
            folder_id = None
            resourcekey = None
            
            # Check path first (e.g. /folders/<id> or /drive/folders/<id>)
            path_parts = parsed.path.split('/')
            if "folders" in path_parts:
                idx = path_parts.index("folders")
                if idx + 1 < len(path_parts):
                    folder_id = path_parts[idx + 1]
            
            # If not in path, check query (e.g. ?id=<id>)
            query_params = parse_qs(parsed.query)
            if not folder_id and "id" in query_params:
                folder_id = query_params["id"][0]
            
            if "resourcekey" in query_params:
                resourcekey = query_params["resourcekey"][0]
            
            if not folder_id:
                raise InvalidDriveLinkError("Could not extract folder ID from URL")
            
            return DriveLinkInfo(folder_id=folder_id, resourcekey=resourcekey)
        except Exception as e:
            if isinstance(e, InvalidDriveLinkError):
                raise
            raise InvalidDriveLinkError(f"Error parsing Drive URL: {e}")
    else:
        # Bare folder ID
        if not re.match(r"^[a-zA-Z0-9\-_]+$", raw_link):
            raise InvalidDriveLinkError("Invalid characters in bare folder ID")
        return DriveLinkInfo(folder_id=raw_link, resourcekey=None)


def classify_drive_error(exception: HttpError) -> DriveErrorCategory:
    status = exception.resp.status
    reason = ""
    message = str(exception)
    
    try:
        error_data = json.loads(exception.content.decode("utf-8"))
        if "error" in error_data:
            err = error_data["error"]
            message = err.get("message", message)
            errors = err.get("errors", [])
            if errors:
                reason = errors[0].get("reason", "")
    except Exception:
        pass
    
    # 1. Quota Exceeded
    if (
        status == 429
        or reason in ("rateLimitExceeded", "userRateLimitExceeded", "quotaExceeded", "dailyLimitExceeded")
        or "quota" in message.lower()
        or "rate limit" in message.lower()
    ):
        return DriveErrorCategory.QUOTA_EXCEEDED
    
    # 2. Domain Restricted
    if reason == "domainRestricted" or "domain restricted" in message.lower():
        return DriveErrorCategory.DOMAIN_RESTRICTED
    
    # 3. Not found or private
    if (
        status == 404
        or reason in ("permissionDenied", "accessNotConfigured", "notFound", "fileNotFound")
        or "permission" in message.lower()
        or "not found" in message.lower()
        or status == 403
    ):
        return DriveErrorCategory.NOT_FOUND_OR_PRIVATE
    
    # 4. Default API Error
    return DriveErrorCategory.API_ERROR


def get_drive_service(api_key: Optional[str] = None):
    """Returns a Google Drive API client using a backend API key for public folder access."""
    key = api_key or os.getenv("GOOGLE_DRIVE_API_KEY") or GOOGLE_DRIVE_API_KEY
    if not key or not key.strip():
        raise RuntimeError(
            "GOOGLE_DRIVE_API_KEY environment variable is not configured on the server. "
            "Please set GOOGLE_DRIVE_API_KEY to access publicly shared Google Drive folders."
        )
    return build("drive", "v3", developerKey=key.strip(), static_discovery=False)



def list_images_in_folder(service, folder_id: str, resourcekey: Optional[str] = None) -> list:
    """Returns every image file (id, name, mimeType) directly inside the given folder."""
    files = []
    page_token = None
    query = f"'{folder_id}' in parents and trashed = false"

    while True:
        try:
            request = service.files().list(
                q=query,
                spaces="drive",
                fields="nextPageToken, files(id, name, mimeType)",
                pageToken=page_token,
                pageSize=1000,
                supportsAllDrives=True,
                includeItemsFromAllDrives=True,
            )
            if resourcekey:
                request.headers["X-Goog-Drive-Resource-Keys"] = f"{folder_id}/{resourcekey}"
            
            response = request.execute()
        except HttpError as e:
            category = classify_drive_error(e)
            raise DriveApiError(
                category,
                f"Failed to list images in folder {folder_id}: {e}",
                original_error=e
            )

        for f in response.get("files", []):
            if f["mimeType"] in IMAGE_MIME_TYPES or f["mimeType"].startswith("image/"):
                files.append(f)
        page_token = response.get("nextPageToken")
        if not page_token:
            break

    return files


def fetch_file_bytes(service, file_id: str, resourcekey: Optional[str] = None) -> bytes:
    """Downloads a Drive file's bytes completely to memory (returns bytes)."""
    try:
        request = service.files().get_media(fileId=file_id, supportsAllDrives=True)
        if resourcekey:
            request.headers["X-Goog-Drive-Resource-Keys"] = f"{file_id}/{resourcekey}"
        
        buffer = io.BytesIO()
        downloader = MediaIoBaseDownload(buffer, request)
        done = False
        while not done:
            _, done = downloader.next_chunk()
        return buffer.getvalue()
    except HttpError as e:
        category = classify_drive_error(e)
        raise DriveApiError(
            category,
            f"Failed to fetch file {file_id}: {e}",
            original_error=e
        )

