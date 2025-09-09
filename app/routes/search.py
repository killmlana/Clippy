from __future__ import annotations
import io
from typing import Any, Dict, List, Optional, Tuple
from PIL import Image
import numpy as np
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from qdrant_client.models import SearchParams

from ..core.settings import settings
from ..lib.qd_client import get_qdrant
from ..encoders.openclip import encode_image_1280, encode_text_1280
from ..lib.encoders import ensure_vec, cos_norm
from ..lib import kg_store
from ..helpers.scoring import combine_scores
from ..helpers.filters import parse_tag_filters, or_filter

router = APIRouter()

TEXT_PROMPTS: Dict[str, List[str]] = {
    "apple": ["apple", "apple fruit", "apple drawing", "fruit still life"],
    "still life": ["still life", "fruit bowl still life", "tabletop still life"],
}


def _search(client, vector_name: str, vec: List[float], limit: int) -> Dict[str, Tuple[float, dict]]:
    out: Dict[str, Tuple[float, dict]] = {}
    res = client.query_points(
        collection_name=settings.QDRANT_COLLECTION,
        query=vec,  # list (not dict)
        using=vector_name,  # "image" | "edge" | "text"
        with_payload=True,
        with_vectors=False,
        limit=limit,
        search_params=SearchParams(hnsw_ef=settings.HNSW_EF),
    )
    for p in res.points:
        out[str(p.id)] = (float(p.score), dict(p.payload or {}))
    return out


@router.post("/hybrid")
async def hybrid_search(
        user_id: str = Form(...),
        top: int = Form(24),
        w_img: float = Form(0.7),
        w_edge: float = Form(0.3),
        w_txt: float = Form(0.2),
        query_text: Optional[str] = Form(None),
        tag_filters: Optional[str] = Form(None),
        sketch: Optional[UploadFile] = File(None),
):
    if top <= 0 or top > 200:
        raise HTTPException(status_code=400, detail="top must be 1..200")

    client = get_qdrant()

    img_q = edge_q = txt_q = None

    if sketch:
        raw = await sketch.read()
        try:
            pil = Image.open(io.BytesIO(raw)).convert("RGB")
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid image")
        v = cos_norm(ensure_vec(encode_image_1280(pil)))
        img_q = v
        edge_q = v

    if query_text:
        t = np.asarray(cos_norm(ensure_vec(encode_text_1280(query_text))), dtype=np.float32)
        prompts = TEXT_PROMPTS.get(query_text.strip().lower(), [])
        if prompts:
            acc = t.copy()
            for p in prompts:
                pv = np.asarray(cos_norm(ensure_vec(encode_text_1280(p))), dtype=np.float32)
                acc += pv
            t = (acc / (len(prompts) + 1)).astype(np.float32)
        txt_q = t.tolist()

    if not any([img_q, edge_q, txt_q]):
        raise HTTPException(status_code=400, detail="Provide sketch and/or query_text")

    over = max(top * 2, 50)
    scores_by_vec: Dict[str, Dict[str, Tuple[float, dict]]] = {}
    scores_by_vec["image"] = _search(client, "image", img_q, over) if (img_q and w_img > 0) else {}
    scores_by_vec["edge"] = _search(client, "edge", edge_q, over) if (edge_q and w_edge > 0) else {}
    scores_by_vec["text"] = _search(client, "text", txt_q, over) if (txt_q and w_txt > 0) else {}

    prefs = kg_store.load_user_prefs(user_id)
    user_tag_w = {k.lower(): float(v) for k, v in (prefs.get("tags") or {}).items()}
    fused = combine_scores(scores_by_vec, w_img, w_edge, w_txt, user_tag_w, settings.ALPHA)

    tags_req = parse_tag_filters(tag_filters)
    if tags_req:
        fused = [t for t in fused if or_filter(t[2], tags_req)]  # OR semantics (MVP)

    results = []
    for pid, score, payload in fused[:top]:
        results.append({
            "id": int(pid) if pid.isdigit() else pid,
            "score": round(float(score), 6),
            "payload": {
                "path": payload.get("path"),
                "tags_all": payload.get("tags_all", []),
                "source_post_url": payload.get("source_post_url"),
                "image_url": f"/image/{pid}",
            },
        })

    return JSONResponse({"results": results, "priors_size": len(user_tag_w)})
