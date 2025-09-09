from __future__ import annotations
from fastapi import APIRouter, HTTPException, Query, Body
from fastapi.responses import JSONResponse
from ..core.settings import settings
from ..lib.qd_client import get_qdrant
from ..lib import kg_store
from ..schemas import FeedbackIn, ProfileOut

router = APIRouter()

@router.post("/feedback")
def post_feedback(payload: FeedbackIn = Body(...)):
    client = get_qdrant()
    pid = int(payload.image_id) if str(payload.image_id).isdigit() else payload.image_id
    recs = client.retrieve(settings.QDRANT_COLLECTION, ids=[pid], with_payload=True)
    if not recs:
        raise HTTPException(status_code=404, detail="image_id not found")
    p = recs[0].payload or {}
    tags_all = payload.tags or p.get("tags_all") or []
    style = p.get("style_cluster")

    updated_tags, updated_style = kg_store.update_feedback(payload.user_id, tags_all, style, payload.action)

    data = kg_store._load()
    u = data["users"][payload.user_id]
    edge_key = "saved" if payload.action == "save" else "clicked"
    u["edges"][edge_key][-1]["image_id"] = pid
    kg_store._save(data)

    return JSONResponse({"ok": True, "updated_tags": updated_tags, "updated_style": updated_style})

@router.get("/profile")
def get_profile(user_id: str = Query(..., min_length=1), top: int = Query(20, ge=1, le=200)):
    prof = kg_store.profile(user_id, top)
    return JSONResponse(ProfileOut(**prof).model_dump())
