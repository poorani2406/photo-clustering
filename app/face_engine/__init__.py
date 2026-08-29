"""
Public surface of the face_engine package.

Callers elsewhere in the app (pipeline.py) should only ever import
FaceEngine (the type) and get_face_engine() (the factory) from here -
never import a concrete engine class directly. That's what lets the
concrete engine be swapped later without touching pipeline.py.
"""
from app.face_engine.base import FaceEngine
from app.face_engine.insightface_engine import InsightFaceEngine

_engine: FaceEngine | None = None


def get_face_engine() -> FaceEngine:
    """
    Returns the app-wide FaceEngine instance, creating it on first call.

    Lazy + cached as a singleton because constructing InsightFaceEngine
    loads its ONNX models from disk, which is slow (roughly a second or
    more) and wasteful to repeat for every photo in a job.

    To swap engines in the future (e.g. a new NewEngine(FaceEngine)),
    this is the only place that needs to change:
        return NewEngine()
    """
    global _engine
    if _engine is None:
        _engine = InsightFaceEngine()
    return _engine


__all__ = ["FaceEngine", "InsightFaceEngine", "get_face_engine"]
