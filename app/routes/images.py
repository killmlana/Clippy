from __future__ import annotations
from pathlib import Path
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from ..core.settings import settings
from ..lib.qd_client import get_qdrant
from ..helpers.images import safe_under_root

router = APIRouter()

@router.get("/image/{image_id}")
def get_image(image_id: str):
    client = get_qdrant()
    pid = int(image_id) if image_id.isdigit() else image_id
    recs = client.retrieve(settings.QDRANT_COLLECTION, ids=[pid], with_payload=True)
    if not recs:
        raise HTTPException(status_code=404, detail="image_id not found")
    payload = recs[0].payload or {}
    raw_path = payload.get("path")
    if not raw_path:
        raise HTTPException(status_code=404, detail="No path in payload")

    abs_path = Path(raw_path).resolve()
    if not safe_under_root(abs_path, settings.IMAGES_ROOT):
        raise HTTPException(status_code=403, detail="Path outside IMAGES_ROOT")
    return FileResponse(path=str(abs_path))
