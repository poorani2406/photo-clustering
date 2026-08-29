"""
Abstract interface every face engine must implement.

The rest of the app (pipeline.py, db.py, clustering.py) only ever needs
one capability: "give me the faces in this image, as boxes + embeddings."
It should not need to know or care whether that comes from InsightFace,
face_recognition, AWS Rekognition, or anything else.

FaceEngine is that seam. Concrete engines (InsightFaceEngine, and any
future engine) subclass this and implement detect_faces(). Calling code
depends only on this abstract type.
"""
from abc import ABC, abstractmethod


class FaceEngine(ABC):
    @abstractmethod
    def detect_faces(self, image_bytes: bytes) -> list[dict]:
        """
        Detect every face in the image bytes supplied in memory.

        Must return a list of dicts, one per face, each with exactly these
        keys (this shape is the contract the rest of the app is built on):
            {
                "top": int,
                "right": int,
                "bottom": int,
                "left": int,
                "embedding": np.ndarray,  # 1-D float vector
            }

        Returns an empty list if no faces are found.
        """
        raise NotImplementedError
