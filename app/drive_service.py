"""
Handles everything related to talking to Google Drive:
- one-time OAuth login (cached to token.json after the first run)
- listing every image in a given folder
- downloading a file's bytes to disk

This is intentionally the ONLY file that imports Google client libraries,
so Stage 2+ can swap the storage backend without touching anything else.
"""
import io
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow

from app.config import GOOGLE_CREDENTIALS_FILE, GOOGLE_TOKEN_FILE, GOOGLE_SCOPES, IMAGE_MIME_TYPES


def get_drive_service():
    """Returns an authenticated Drive API client, running the OAuth consent
    flow in a browser the first time, and reusing the cached token after that."""
    creds = None

    if GOOGLE_TOKEN_FILE.exists():
        creds = Credentials.from_authorized_user_file(str(GOOGLE_TOKEN_FILE), GOOGLE_SCOPES)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            if not GOOGLE_CREDENTIALS_FILE.exists():
                raise FileNotFoundError(
                    f"Missing {GOOGLE_CREDENTIALS_FILE.name}. Download OAuth 'Desktop app' "
                    "credentials from Google Cloud Console and place them at this path. "
                    "See README.md for step-by-step instructions."
                )
            flow = InstalledAppFlow.from_client_secrets_file(
                str(GOOGLE_CREDENTIALS_FILE), GOOGLE_SCOPES
            )
            creds = flow.run_local_server(port=0)

        GOOGLE_TOKEN_FILE.write_text(creds.to_json())

    return build("drive", "v3", credentials=creds)


def list_images_in_folder(service, folder_id: str):
    """Returns every image file (id, name, mimeType) directly inside the given folder."""
    files = []
    page_token = None
    query = f"'{folder_id}' in parents and trashed = false"

    while True:
        response = (
            service.files()
            .list(
                q=query,
                spaces="drive",
                fields="nextPageToken, files(id, name, mimeType)",
                pageToken=page_token,
                pageSize=1000,
            )
            .execute()
        )
        for f in response.get("files", []):
            if f["mimeType"] in IMAGE_MIME_TYPES or f["mimeType"].startswith("image/"):
                files.append(f)
        page_token = response.get("nextPageToken")
        if not page_token:
            break

    return files


def download_file(service, file_id: str, dest_path):
    """Downloads a Drive file's bytes to dest_path (a Path object)."""
    request = service.files().get_media(fileId=file_id)
    buffer = io.BytesIO()
    downloader = MediaIoBaseDownload(buffer, request)
    done = False
    while not done:
        _, done = downloader.next_chunk()

    dest_path.write_bytes(buffer.getvalue())
