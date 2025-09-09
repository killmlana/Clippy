from __future__ import annotations
import os
from pathlib import Path

class Settings:
    QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
    QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "safebooru_union_clip")
    IMAGES_ROOT = Path(os.getenv("IMAGES_ROOT", "./safebooru_union/ALL")).resolve()
    ALLOWED_ORIGINS = [o.strip() for o in os.getenv("ALLOWED_ORIGINS", "http://localhost:5173").split(",") if o.strip()]
    HNSW_EF = int(os.getenv("HNSW_EF", "128"))
    ALPHA = float(os.getenv("ALPHA", "0.2"))
    PREF_TOP_N = int(os.getenv("PREF_TOP_N", "100"))
    EMA_BETA = float(os.getenv("EMA_BETA", "0.9"))
    DELTA_CLICK = float(os.getenv("DELTA_CLICK", "0.05"))
    DELTA_SAVE = float(os.getenv("DELTA_SAVE", "0.10"))
    KG_STORE = Path(os.getenv("KG_STORE", "./data/kg_store.json")).resolve()
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

settings = Settings()