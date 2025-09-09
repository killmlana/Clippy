from __future__ import annotations
from functools import lru_cache
from qdrant_client import QdrantClient
from ..core.settings import settings

@lru_cache(maxsize=1)
def get_qdrant() -> QdrantClient:
    return QdrantClient(url=settings.QDRANT_URL, prefer_grpc=False)