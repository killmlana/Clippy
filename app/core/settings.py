from future import annotations
import os
from pathlib import Path
from dotenv import load_dotenv; load_dotenv()

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
    MODEL_NAME = os.getenv("MODEL_NAME", "ViT-H-14")
    PRETRAINED = os.getenv("PRETRAINED", "laion2b_s32b_b79k")
    DEVICE = os.getenv("DEVICE", "cuda")
    PRECISION = os.getenv("PRECISION", "fp16")
    GOOGLE_GENAI_USE_VERTEXAI = os.getenv("GOOGLE_GENAI_USE_VERTEXAI", "false").lower() in ("true", "1", "t")
    PROMPT_LOG_LEVEL = os.getenv("PROMPT_LOG_LEVEL", "INFO").upper()
    GEMINI_VISION_MODEL = os.getenv("GEMINI_VISION_MODEL", "gemini-2.5-flash")

settings = Settings()