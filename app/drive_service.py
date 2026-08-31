"""
Handles everything related to talking to Google Drive using API key access:
- Listing every image in a given publicly shared folder
- Downloading a file's bytes to memory
"""
import io
import re
import json
import os
import urllib.request
import urllib.parse
import urllib.error
from enum import Enum
from typing import NamedTuple, Optional

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
            parsed = urllib.parse.urlparse(raw_link)
            folder_id = None
            resourcekey = None
            
            # Check path first (e.g. /folders/<id> or /drive/folders/<id>)
            path_parts = parsed.path.split('/')
            if "folders" in path_parts:
                idx = path_parts.index("folders")
                if idx + 1 < len(path_parts):
                    folder_id = path_parts[idx + 1]
            
            # If not in path, check query (e.g. ?id=<id>)
            query_params = urllib.parse.parse_qs(parsed.query)
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


def classify_drive_error(exception: Exception) -> DriveErrorCategory:
    status = getattr(exception, "code", 0) or getattr(exception, "status", 0)
    message = str(exception)
    
    if isinstance(exception, urllib.error.HTTPError):
        try:
            body = exception.read().decode("utf-8")
            error_data = json.loads(body)
            if "error" in error_data:
                err = error_data["error"]
                message = err.get("message", message)
                errors = err.get("errors", [])
                if errors:
                    reason = errors[0].get("reason", "")
                    if reason in ("rateLimitExceeded", "userRateLimitExceeded", "quotaExceeded", "dailyLimitExceeded"):
                        return DriveErrorCategory.QUOTA_EXCEEDED
                    if reason == "domainRestricted":
                        return DriveErrorCategory.DOMAIN_RESTRICTED
                    if reason in ("permissionDenied", "accessNotConfigured", "notFound", "fileNotFound"):
                        return DriveErrorCategory.NOT_FOUND_OR_PRIVATE
        except Exception:
            pass
    
    if status == 429 or "quota" in message.lower() or "rate limit" in message.lower():
        return DriveErrorCategory.QUOTA_EXCEEDED
    if "domain restricted" in message.lower():
        return DriveErrorCategory.DOMAIN_RESTRICTED
    if status in (404, 403) or "permission" in message.lower() or "not found" in message.lower() or "file not found" in message.lower():
        return DriveErrorCategory.NOT_FOUND_OR_PRIVATE
    
    return DriveErrorCategory.API_ERROR


def get_drive_service(api_key: Optional[str] = None):
    """Returns the API key string used to authenticate Google Drive REST requests."""
    key = api_key or os.getenv("GOOGLE_DRIVE_API_KEY") or GOOGLE_DRIVE_API_KEY
    if not key or not key.strip():
        raise RuntimeError(
            "GOOGLE_DRIVE_API_KEY environment variable is not configured on the server. "
            "Please set GOOGLE_DRIVE_API_KEY to access publicly shared Google Drive folders."
        )
    return key.strip()


def list_images_in_folder(service, folder_id: str, resourcekey: Optional[str] = None) -> list:
    """Returns every image file (id, name, mimeType) directly inside the given folder."""
    api_key = get_drive_service(service if isinstance(service, str) else None)
    files = []
    page_token = None
    query = f"'{folder_id}' in parents and trashed = false"

    while True:
        params_dict = {
            "q": query,
            "spaces": "drive",
            "fields": "nextPageToken, files(id, name, mimeType, resourceKey, size)",
            "pageSize": "1000",
            "supportsAllDrives": "true",
            "includeItemsFromAllDrives": "true",
            "key": api_key
        }
        if page_token:
            params_dict["pageToken"] = page_token
            
        url = f"https://www.googleapis.com/drive/v3/files?{urllib.parse.urlencode(params_dict)}"
        headers = {"User-Agent": "PhotoClustering/2.0"}
        if resourcekey:
            headers["X-Goog-Drive-Resource-Keys"] = f"{folder_id}/{resourcekey}"
            
        req = urllib.request.Request(url, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=10.0) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            category = classify_drive_error(e)
            raise DriveApiError(
                category,
                f"Failed to list images in folder {folder_id}: HTTP {e.code} {e.reason}",
                original_error=e
            )
        except Exception as e:
            category = classify_drive_error(e)
            raise DriveApiError(
                category,
                f"Network error accessing folder {folder_id}: {e}",
                original_error=e
            )

        for f in data.get("files", []):
            if f.get("mimeType") in IMAGE_MIME_TYPES or str(f.get("mimeType", "")).startswith("image/"):
                if f.get("resourceKey"):
                    f["resourcekey"] = f.get("resourceKey")
                elif resourcekey:
                    f["resourcekey"] = resourcekey
                files.append(f)
        page_token = data.get("nextPageToken")
        if not page_token:
            break

    return files


def fetch_file_bytes(service, file_id: str, resourcekey: Optional[str] = None) -> bytes:
    """Downloads a Drive file's bytes completely to memory (returns bytes)."""
    api_key = get_drive_service(service if isinstance(service, str) else None)
    url = f"https://www.googleapis.com/drive/v3/files/{file_id}?alt=media&supportsAllDrives=true&key={api_key}"
    headers = {"User-Agent": "PhotoClustering/2.0"}
    if resourcekey:
        headers["X-Goog-Drive-Resource-Keys"] = f"{file_id}/{resourcekey}"
        
    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=15.0) as resp:
            return resp.read()
    except urllib.error.HTTPError as e:
        category = classify_drive_error(e)
        raise DriveApiError(
            category,
            f"Failed to fetch file {file_id}: HTTP {e.code} {e.reason}",
            original_error=e
        )
    except Exception as e:
        category = classify_drive_error(e)
        raise DriveApiError(
            category,
            f"Network error fetching file {file_id}: {e}",
            original_error=e
        )


