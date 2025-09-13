from __future__ import annotations
from pathlib import Path
import mimetypes
from typing import Optional

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from PIL import Image  # only used for fallback sniff

from ..core.settings import settings
from ..lib.qd_client import get_qdrant   # keep your module name
from ..helpers.images import safe_under_root

router = APIRouter()

# --- ensure common image types are registered (some envs miss .webp) ---
mimetypes.add_type("image/webp", ".webp")
mimetypes.add_type("image/jpeg", ".jpg")
mimetypes.add_type("image/jpeg", ".jpeg")
mimetypes.add_type("image/png",  ".png")

_FALLBACK_FMT_MAP = {
    "PNG":  "image/png",
    "JPEG": "image/jpeg",
    "JPG":  "image/jpeg",
    "WEBP": "image/webp",
    "GIF":  "image/gif",
    "BMP":  "image/bmp",
    "TIFF": "image/tiff",
}

def _detect_media_type(p: Path) -> str:
    # 1) extension-based
    mt, _ = mimetypes.guess_type(p.name, strict=False)
    if mt:
        return mt
    # 2) sniff via PIL (handles extensionless/odd names)
    try:
        with Image.open(p) as im:
            fmt = (im.format or "").upper()
        return _FALLBACK_FMT_MAP.get(fmt, "application/octet-stream")
    except Exception:
        return "application/octet-stream"

@router.get("/image/{image_id}")
def get_image(image_id: str):
    client = get_qdrant()
    pid = int(image_id) if image_id.isdigit() else image_id
    recs = client.retrieve(settings.QDRANT_COLLECTION, ids=[pid], with_payload=True)
    if not recs:
        raise HTTPException(status_code=404, detail="image_id not found")

    payload = recs[0].payload or {}
    raw_path: Optional[str] = payload.get("path")
    if not raw_path:
        raise HTTPException(status_code=404, detail="No path in payload")

    abs_path = Path(raw_path).resolve()
    if not safe_under_root(abs_path, settings.IMAGES_ROOT):
        raise HTTPException(status_code=403, detail="Path outside IMAGES_ROOT")

    media_type = _detect_media_type(abs_path)

    # inline display + basic caching
    headers = {
        "Content-Disposition": f'inline; filename="{abs_path.name}"',
        "Cache-Control": "public, max-age=604800, immutable",
    }

    return FileResponse(path=str(abs_path), media_type=media_type, headers=headers)