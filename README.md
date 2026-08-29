# Contact — Face Detection, Grouping & Secure Streaming

This application allows photographer clients to point the system at one or multiple Google Drive folders. The system downloads images to memory, detects faces using InsightFace, and clusters them into distinct individuals using DBSCAN.

---

## Key Features
- **In-memory Processing**: No downloaded images are stored on disk. Original photos are kept in transient memory buffers and garbage collected immediately.
- **Multiple Drive Folders**: Accepts multiple folder links, merging files and deduplicating them using `drive_file_id`.
- **Resource Keys support**: Supports shared folders that require resource keys (`resourcekey=...`).
- **Opaque Job Token Architecture**: Frontend routes are fully secured. Browser clients use opaque, secure public tokens (never raw integer IDs) to query jobs and access child assets (people, photos, and face thumbnails).
- **Cross-Job Protection**: Full ownership checks on face, photo, and person assets prevent enumeration of resources across jobs.
- **Streaming ZIP Downloads**: Allows downloading single photos, selected photos, or entire folders of a person grouped into a ZIP archive, generated on the fly one photo at a time.

---

## 1. Prerequisites & Dependencies

Create and activate a virtual environment, then install requirements:

```bash
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

---

## 2. Google Drive API Key Setup

This application uses API-key based authentication for server-side Google Drive API access.

1. Go to the [Google Cloud Console](https://console.cloud.google.com/).
2. Enable the **Google Drive API** for your project.
3. Go to **APIs & Services → Credentials → Create Credentials → API Key**.
4. Copy the generated key and add it to your `.env` configuration file:
   ```env
   GOOGLE_DRIVE_API_KEY=your_google_drive_api_key_here
   ```

### IMPORTANT Security Note: API Key Restrictions
To protect your key, you must restrict its access inside the **Google Cloud Console**:
- **API restriction**: Limit the key to only call the **Google Drive API**.
- **IP/HTTP restrictions**: Restrict calls to your deployment servers' IP addresses or web URLs.
- *Note*: These restrictions are configured and enforced in the Google Cloud Console, not in the application code.

---

## 3. Configuration (`.env`)

Create a `.env` file from `.env.example`:

```env
DATA_DIR=data
GOOGLE_DRIVE_API_KEY=AIzaSy...
MAX_FILE_SIZE_BYTES=26214400   # 25 MB limit per photo
MAX_FILES_PER_JOB=5000         # Maximum number of unique files per job
CLUSTER_EPS=0.9
```

---

## 4. How to Run & Use the Application

Start the FastAPI application:

```bash
python run.py
```

Open **http://127.0.0.1:8000** in your browser.

### Processing Folders:
Paster one or multiple folder links (separated by commas or newlines) in the input field:
- `https://drive.google.com/drive/folders/FOLDER_ID`
- `https://drive.google.com/drive/folders/FOLDER_ID?resourcekey=RESOURCE_KEY`
- Bare folder IDs

Click **Process folder** to start the job. Once completed, you will see a grid of all distinct people identified across the folders. Clicking a card will show all photos that person appears in.
