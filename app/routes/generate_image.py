# app/routes/generate_image.py
from __future__ import annotations

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from app.lib.gemini.images import (
    run_customize_from_sketch_promptless,
    run_subject_from_sketch,
    run_edit_or_generate,
)

router = APIRouter()

@router.post("/generate_image")
async def generate_image(request: Request):
    body = await request.json()

    # Subject + Sketch path
    if body.get("subject_ref") and body.get("image_url"):
        resp = run_subject_from_sketch(
            subject_ref=body.get("subject_ref"),
            image_url=body.get("image_url"),
            style_tags=body.get("style_tags") or [],
            subject_type_default=body.get("subject_type_default") or "SUBJECT_TYPE_PERSON",
            n=int(body.get("n") or 1),
        )
        status = resp.get("status", 200)
        return JSONResponse(resp, status_code=status)

    # Style + Sketch customize path
    if body.get("refs") and body.get("image_url") and not body.get("mask_url"):
        resp = run_customize_from_sketch_promptless(
            refs=body.get("refs") or [],
            image_url=body.get("image_url"),
            style_tags=body.get("style_tags") or [],
            n=int(body.get("n") or 4),
        )
        status = resp.get("status", 200)
        return JSONResponse(resp, status_code=status)

    # Fallback: masked edit or prompt-generate (legacy)
    resp = run_edit_or_generate(
        subject=body.get("prompt_subject") or body.get("subject"),
        refs=body.get("refs") or [],
        image_url=body.get("image_url"),
        mask_url=body.get("mask_url"),
        n=int(body.get("n") or 1),
    )
    status = resp.get("status", 200)
    return JSONResponse(resp, status_code=status)
