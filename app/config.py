import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / os.getenv("DATA_DIR", "data")
DB_PATH = DATA_DIR / "app.db"

GOOGLE_DRIVE_API_KEY = os.getenv("GOOGLE_DRIVE_API_KEY", "AIzaSyCHT-VLcAv11d1ACJBZVQyI16DxfIRgyNM")



MAX_FILE_SIZE_BYTES = int(
    os.getenv("MAX_FILE_SIZE_BYTES", "26214400")
)

MAX_FILES_PER_JOB = int(
    os.getenv("MAX_FILES_PER_JOB", "5000")
)

# Euclidean distance threshold for DBSCAN in clustering.py. Tuned for
# InsightFace/ArcFace's 512-d embeddings (roughly unit-normalized vectors,
# where same-person pairs typically land ~0.8-1.0 apart). This is NOT the
# same value that worked for face_recognition's 128-d embeddings - if you
# swap in a different FaceEngine, re-tune this via the CLUSTER_EPS env var.
CLUSTER_EPS = float(os.getenv("CLUSTER_EPS", "0.9"))
CLUSTER_MIN_SAMPLES = int(os.getenv("CLUSTER_MIN_SAMPLES", "2"))
CLUSTER_MERGE_THRESHOLD = float(os.getenv("CLUSTER_MERGE_THRESHOLD", "0.52"))
CLUSTER_FACE_THRESHOLD = float(os.getenv("CLUSTER_FACE_THRESHOLD", "0.52"))
CLUSTER_MIN_MERGE_SIZE = int(os.getenv("CLUSTER_MIN_MERGE_SIZE", "2"))

IMAGE_MIME_TYPES = {
    "image/jpeg",
    "image/png",
    "image/heic",
    "image/webp",
}

DATA_DIR.mkdir(parents=True, exist_ok=True)

