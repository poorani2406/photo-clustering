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
        model_name: InsightFace model pack. Downloaded once to ~/.insightface
            on first use, then cached.
        ctx_id: -1 = CPU, >=0 = GPU device index. Defaults to CPU so this
            works out of the box on any machine, including Windows laptops
            with no CUDA setup.
        det_size: input resolution for the detector. 640x640 is InsightFace's
            standard default and a good speed/accuracy tradeoff.
        """
        self._app = FaceAnalysis(name=model_name)
        self._app.prepare(ctx_id=ctx_id, det_size=det_size)

    def detect_faces(self, image_path: str) -> list[dict]:
        # cv2 reads images as BGR numpy arrays, which is exactly what
        # FaceAnalysis.get() expects.
        import cv2

        image = cv2.imread(image_path)
        if image is None:
            return []

        detected = self._app.get(image)

        faces = []
        for face in detected:
            x1, y1, x2, y2 = face.bbox
            faces.append(
                {
                    "top": int(round(y1)),
                    "right": int(round(x2)),
                    "bottom": int(round(y2)),
                    "left": int(round(x1)),
                    "embedding": np.asarray(face.normed_embedding, dtype=np.float32),
                }
            )
        return faces
