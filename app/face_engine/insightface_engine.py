"""
InsightFace implementation of FaceEngine.

Uses InsightFace's FaceAnalysis pipeline (the "buffalo_l" model pack),
which runs on onnxruntime and needs no C++ build step - unlike dlib,
it installs cleanly on Windows with plain pip.

FaceAnalysis.get(image) does detection + alignment + embedding in one
call and returns a list of `Face` objects, each with:
    - bbox: [x1, y1, x2, y2] (float32)
    - embedding: 512-d float32 ArcFace vector, NOT unit-normalized
      (magnitude is typically ~20-25, not ~1)
    - normed_embedding: the same vector, L2-normalized to unit length

We use normed_embedding here, not embedding. clustering.py's DBSCAN runs on
euclidean distance with eps=0.9 - a threshold that's only meaningful for
unit vectors, where euclidean distance and cosine similarity are directly
related (dist = sqrt(2 - 2*cos_sim)). Feeding it raw, magnitude-~23
embeddings would make every face's distance to every other face huge,
so nothing would ever cluster together.

Verified against insightface==1.0.1 (source-diffed against 0.7.3: the
FaceAnalysis constructor, prepare(), get(), and the Face object are
unchanged for this usage pattern - see project audit notes). 1.0.1 is
the pinned version specifically because, unlike 0.7.3, it doesn't build
the optional face3d C++/Cython extension by default, which is what was
requiring Visual Studio Build Tools on Windows.
"""
import numpy as np
from insightface.app import FaceAnalysis


from app.face_engine.base import FaceEngine


class InsightFaceEngine(FaceEngine):
    def __init__(self, model_name: str = "buffalo_l", ctx_id: int = -1, det_size=(640, 640)):
        """
        model_name: InsightFace model pack.
        ctx_id: -1 for CPU execution.
        allowed_modules: loads only detection + recognition (ArcFace), skipping unused 3D/landmark/genderage models.
        """
        print(f"[FACE ENGINE] Starting initialization (model: {model_name}, ctx_id: {ctx_id}, det_size: {det_size})...")
        print("[FACE ENGINE] Creating FaceAnalysis instance for CPU (modules: detection, recognition)...")
        # Explicitly configure CPU execution provider without searching for CUDA
        self._app = FaceAnalysis(
            name=model_name,
            allowed_modules=["detection", "recognition"],
            providers=["CPUExecutionProvider"]
        )
        print("[FACE ENGINE] FaceAnalysis created. Preparing model (downloading/loading from cache)...")
        self._app.prepare(ctx_id=-1, det_size=det_size)
        print("[FACE ENGINE] Model preparation completed. Running first test inference warmup...")
        # First test inference warmup on dummy image
        dummy_img = np.zeros((100, 100, 3), dtype=np.uint8)
        self._app.get(dummy_img)
        print("[FACE ENGINE] Face engine ready and pre-warmed.")


    def detect_faces(self, image_bytes: bytes) -> list[dict]:
        import cv2

        image = cv2.imdecode(
            np.frombuffer(image_bytes, dtype=np.uint8),
            cv2.IMREAD_COLOR
        )
        if image is None:
            return []

        orig_h, orig_w = image.shape[:2]
        # Optimize CPU detection speed and RAM by scaling down oversized raw photos
        max_dim = 1280
        scale = 1.0
        if max(orig_h, orig_w) > max_dim:
            scale = max_dim / float(max(orig_h, orig_w))
            new_w = int(round(orig_w * scale))
            new_h = int(round(orig_h * scale))
            det_image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        else:
            det_image = image

        detected = self._app.get(det_image)

        faces = []
        for face in detected:
            x1, y1, x2, y2 = face.bbox
            if scale != 1.0:
                x1, y1, x2, y2 = x1 / scale, y1 / scale, x2 / scale, y2 / scale

            faces.append(
                {
                    "top": max(0, int(round(y1))),
                    "right": min(orig_w, int(round(x2))),
                    "bottom": min(orig_h, int(round(y2))),
                    "left": max(0, int(round(x1))),
                    "embedding": np.asarray(face.normed_embedding, dtype=np.float32),
                }
            )
        return faces

