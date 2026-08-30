"""
Public surface of the face_engine package.

Callers elsewhere in the app (pipeline.py) should only ever import
FaceEngine (the type) and get_face_engine() (the factory) from here -
never import a concrete engine class directly. That's what lets the
concrete engine be swapped later without touching pipeline.py.
"""
import threading
from app.face_engine.base import FaceEngine
from app.face_engine.insightface_engine import InsightFaceEngine

_engine: FaceEngine | None = None
_engine_lock = threading.Lock()


def get_face_engine() -> FaceEngine:
    """
    Returns the app-wide FaceEngine instance, creating it on first call in a thread-safe manner.
    """
    global _engine
    if _engine is None:
        with _engine_lock:
            if _engine is None:
                _engine = InsightFaceEngine()
    return _engine


__all__ = ["FaceEngine", "InsightFaceEngine", "get_face_engine"]

