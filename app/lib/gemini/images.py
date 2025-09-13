# app/lib/gemini/images.py
from __future__ import annotations

import base64
import io
import logging
import os
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image, ImageOps
import requests

from app.lib.gemini.client import VertexClients
from app.lib.gemini.prompting import (
    compose_prompt,
    default_negative_prompt,
    build_prompt_from_sketch_analysis,
)
from app.lib.gemini.utils import fetch_ref_bytes

log = logging.getLogger("images")

DATA_URL_RE_PREFIX = "data:"
ALLOWED_DATA_MIME_PREFIX = "image/"

# --- Tunables (via env) ---
AUTOSQUARE_MODE = os.getenv("IMAGEN_AUTOSQUARE", "expand").strip().lower()
IMAGEN_MAX_SIDE = int(os.getenv("IMAGEN_MAX_SIDE", "1536"))
IMAGEN_STYLE_MAX = int(os.getenv("IMAGEN_STYLE_MAX", "2"))  # only used in customize path cap

# --------- decoding & utils ---------
def data_url_to_bytes(s: str) -> Tuple[Optional[str], Optional[bytes]]:
    if not s or not isinstance(s, str) or not s.startswith(DATA_URL_RE_PREFIX):
        return None, None
    try:
        header, b64 = s.split(",", 1)
    except ValueError:
        return None, None
    if ";base64" not in header:
        return None, None
    mime = header.split(":")[1].split(";")[0].strip().lower()
    if not mime.startswith(ALLOWED_DATA_MIME_PREFIX):
        log.warning("images: rejecting non-image data URL (mime=%s)", mime)
        return None, None
    try:
        b = base64.b64decode(b64, validate=True)
        return mime, b
    except Exception:
        return None, None

def ensure_same_size(base_png: bytes, mask_png: bytes) -> Tuple[bytes, bytes]:
    base = Image.open(io.BytesIO(base_png)).convert("RGBA")
    mask = Image.open(io.BytesIO(mask_png))
    if mask.mode != "L":
        mask = mask.convert("L")
    if mask.size != base.size:
        mask = mask.resize(base.size, Image.NEAREST)
    mask = mask.point(lambda v: 255 if v >= 160 else 0).convert("L")
    bbuf = io.BytesIO(); base.save(bbuf, format="PNG")
    mbuf = io.BytesIO(); mask.save(mbuf, format="PNG")
    return bbuf.getvalue(), mbuf.getvalue()

def _make_full_white_mask_like(base_png: bytes) -> bytes:
    base = Image.open(io.BytesIO(base_png))
    m = Image.new("L", base.size, color=255)
    mbuf = io.BytesIO(); m.save(mbuf, format="PNG")
    return mbuf.getvalue()

def mask_has_editable(mask_png: bytes) -> bool:
    m = Image.open(io.BytesIO(mask_png)).convert("L")
    m = m.point(lambda v: 255 if v >= 200 else 0).convert("L")
    return m.getbbox() is not None

def _is_squareish(png_bytes: bytes, tol: float = 0.05) -> bool:
    img = Image.open(io.BytesIO(png_bytes))
    w, h = img.size
    if not w or not h:
        return False
    r = w / float(h)
    return (1.0 - tol) <= r <= (1.0 + tol)

def _expand_to_square(base_png: bytes, mask_png: Optional[bytes]) -> Tuple[bytes, Optional[bytes]]:
    base = Image.open(io.BytesIO(base_png)).convert("RGBA")
    w, h = base.size
    if w == h:
        return base_png, mask_png
    side = max(w, h)
    bg = (255, 255, 255, 255)
    new_base = Image.new("RGBA", (side, side), bg)
    off = ((side - w) // 2, (side - h) // 2)
    new_base.paste(base, off)

    new_mask = None
    if mask_png is not None:
        m = Image.open(io.BytesIO(mask_png)).convert("L")
        nm = Image.new("L", (side, side), 255)
        nm.paste(m, off)
        bufm = io.BytesIO(); nm.save(bufm, format="PNG"); new_mask = bufm.getvalue()

    buf = io.BytesIO(); new_base.save(buf, format="PNG")
    return buf.getvalue(), new_mask

def _crop_to_square(base_png: bytes, mask_png: Optional[bytes]) -> Tuple[bytes, Optional[bytes]]:
    base = Image.open(io.BytesIO(base_png)).convert("RGBA")
    w, h = base.size
    if w == h:
        return base_png, mask_png
    side = min(w, h)
    left = (w - side) // 2
    top = (h - side) // 2
    box = (left, top, left + side, top + side)
    new_base = base.crop(box)
    new_mask = None
    if mask_png is not None:
        m = Image.open(io.BytesIO(mask_png)).convert("L")
        new_mask_img = m.crop(box)
        bufm = io.BytesIO(); new_mask_img.save(bufm, format="PNG"); new_mask = bufm.getvalue()
    buf = io.BytesIO(); new_base.save(buf, format="PNG")
    return buf.getvalue(), new_mask

def _resize_max_side(base_png: bytes, mask_png: Optional[bytes], max_side: int) -> Tuple[bytes, Optional[bytes]]:
    base = Image.open(io.BytesIO(base_png)).convert("RGBA")
    w, h = base.size
    side = max(w, h)
    if side <= max_side:
        return base_png, mask_png
    scale = max_side / float(side)
    new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))
    base_resized = base.resize((new_w, new_h), Image.LANCZOS)

    new_mask_bytes: Optional[bytes] = None
    if mask_png is not None:
        m = Image.open(io.BytesIO(mask_png)).convert("L")
        m_resized = m.resize((new_w, new_h), Image.NEAREST)
        mbuf = io.BytesIO(); m_resized.save(mbuf, format="PNG"); new_mask_bytes = mbuf.getvalue()

    bbuf = io.BytesIO(); base_resized.save(bbuf, format="PNG")
    return bbuf.getvalue(), new_mask_bytes

def maybe_make_control_from_sketch(base_png: bytes) -> bytes:
    img = Image.open(io.BytesIO(base_png)).convert("L")
    img = ImageOps.autocontrast(img, cutoff=2)
    buf = io.BytesIO(); img.save(buf, format="PNG")
    return buf.getvalue()

def _b64(b: bytes) -> str:
    return base64.b64encode(b).decode("utf-8")

# ---------- Imagen helpers ----------
def _rest_customize(
    vc: VertexClients,
    *,
    prompt: str,
    refs: List[Dict[str, Any]],
    n: int,
    negative_extra: str = "",
) -> Tuple[List[bytes], float, str]:
    neg = default_negative_prompt()
    if negative_extra:
        neg = f"{neg}, {negative_extra}"
    instances = [{"prompt": prompt, "referenceImages": refs}]
    params = {"sampleCount": max(1, min(4, n)), "negativePrompt": neg, "language": "en"}
    imgs, ms = vc.rest_predict(instances=instances, parameters=params)
    return imgs, ms, "rest(customize)"

def _rest_subject_plus_control(
    vc: VertexClients,
    *,
    prompt: str,
    subject_png: bytes,
    subject_type: str,
    subject_desc: str,
    control_png: bytes,
    n: int,
    negative_extra: str = "",
) -> Tuple[List[bytes], float, str]:
    neg = default_negative_prompt()
    if negative_extra:
        neg = f"{neg}, {negative_extra}"
    refs = [
        {
            "referenceType": "REFERENCE_TYPE_SUBJECT",
            "referenceId": 1,
            "referenceImage": {"bytesBase64Encoded": _b64(subject_png)},
            "subjectImageConfig": {"subjectType": subject_type, "imageDescription": subject_desc[:200]},
        },
        {
            "referenceType": "REFERENCE_TYPE_CONTROL",
            "referenceId": 2,
            "referenceImage": {"bytesBase64Encoded": _b64(control_png)},
            "controlImageConfig": {"controlType": "CONTROL_TYPE_SCRIBBLE", "enableControlImageComputation": False},
        },
    ]
    instances = [{"prompt": prompt, "referenceImages": refs}]
    params = {"sampleCount": max(1, min(4, n)), "negativePrompt": neg, "language": "en"}
    imgs, ms = vc.rest_predict(instances=instances, parameters=params)
    return imgs, ms, "rest(subject+control)"

def _build_customize_refs_from_sketch(*, sketch_png: bytes, style_ref_bytes: List[bytes], style_desc: str) -> List[Dict[str, Any]]:
    refs: List[Dict[str, Any]] = []
    if style_ref_bytes:
        refs.append({
            "referenceType": "REFERENCE_TYPE_STYLE",
            "referenceId": 2,
            "referenceImage": {"bytesBase64Encoded": _b64(style_ref_bytes[0])},
            "styleImageConfig": {"styleDescription": (style_desc or "reference style")[:120]},
        })
    control_png = maybe_make_control_from_sketch(sketch_png)
    refs.append({
        "referenceType": "REFERENCE_TYPE_CONTROL",
        "referenceId": 3,
        "referenceImage": {"bytesBase64Encoded": _b64(control_png)},
        "controlImageConfig": {"controlType": "CONTROL_TYPE_SCRIBBLE", "enableControlImageComputation": False},
    })
    return refs[:4]

# --------- orchestration ---------
def run_customize_from_sketch_promptless(
    *,
    refs: List[Dict[str, Any]],
    image_url: str,
    style_tags: List[str],
    n: int = 4,
) -> Dict[str, Any]:
    """
    STYLE + CONTROL(scribble) path with robust sketch-understanding prompt (style-aware).
    """
    vc = VertexClients()
    if not vc.ready:
        return {"error": "Vertex AI is not configured. Set GOOGLE_GENAI_USE_VERTEXAI=true and ADC.", "status": 400}

    mime, sketch_png = data_url_to_bytes(image_url)
    if not sketch_png:
        return {"error": "Invalid or missing sketch image data URL.", "status": 400}

    style_ref_bytes: List[bytes] = []
    accepted_refs: List[Dict[str, Any]] = []
    for r in refs or []:
        b = fetch_ref_bytes(r)
        if b:
            style_ref_bytes.append(b); accepted_refs.append(r)

    if not style_ref_bytes:
        return {"error": "No valid style reference image bytes.", "status": 400}

    # Sketch understanding WITH style tags
    try:
        sketch_analysis = vc.gemini_understand_sketch_json(image_bytes=sketch_png, mime_type=mime or "image/png", style_tags=style_tags)
    except requests.HTTPError as e:
        text = getattr(e.response, "text", str(e))
        sketch_analysis = {}
        log.warning("gemini_understand_sketch_json failed: %s", text)

    style_desc = ", ".join([str(t).strip().replace("_"," ") for t in (style_tags or [])][:IMAGEN_STYLE_MAX]) or "reference style"
    packed_refs = _build_customize_refs_from_sketch(sketch_png=sketch_png, style_ref_bytes=style_ref_bytes, style_desc=style_desc)

    # Build style-aware prompt + negatives
    prompt, negatives = build_prompt_from_sketch_analysis(
        mode="style_customize",
        sketch_analysis=sketch_analysis,
        extra_style_words=[str(t).replace("_"," ") for t in (style_tags or [])][:IMAGEN_STYLE_MAX],
        subject_identity_in_ref=False,
        style_tags=style_tags,
    )
    neg_extra = ", ".join(negatives) if negatives else ""

    try:
        images, latency_ms, used = _rest_customize(vc, prompt=prompt, refs=packed_refs, n=n, negative_extra=neg_extra)
    except requests.HTTPError as e:
        text = getattr(e.response, "text", str(e))
        return {"error": f"Vertex request failed ({e.response.status_code if e.response else 'HTTPError'}): {text}",
                "status": e.response.status_code if e.response else 500}

    data = [{"b64_json": base64.b64encode(img).decode("utf-8")} for img in images]
    return {
        "data": data,
        "model_used": "imagen-3.0-capability-001",
        "latency_ms": int(latency_ms),
        "meta": {
            "source": used,
            "mode": "customize_promptless",
            "count": len(images),
            "constructed_prompt": prompt,
            "style_desc": style_desc,
            "style_refs_used": 1,
            "control": "scribble",
        },
    }

def run_subject_from_sketch(
    *,
    subject_ref: Dict[str, Any],
    image_url: str,
    style_tags: Optional[List[str]] = None,
    subject_type_default: str = "SUBJECT_TYPE_PERSON",
    n: int = 1,
) -> Dict[str, Any]:
    """
    SUBJECT + CONTROL(scribble) path with robust sketch-understanding prompt (style-aware).
    """
    vc = VertexClients()
    if not vc.ready:
        return {"error": "Vertex AI is not configured. Set GOOGLE_GENAI_USE_VERTEXAI=true and ADC.", "status": 400}

    mime, sketch_png = data_url_to_bytes(image_url)
    if not sketch_png:
        return {"error": "Invalid or missing sketch image data URL.", "status": 400}

    subject_bytes = fetch_ref_bytes(subject_ref)
    if not subject_bytes:
        return {"error": "Invalid subject reference image.", "status": 400}

    tags = style_tags or []

    # Sketch understanding WITH style tags
    try:
        sketch_analysis = vc.gemini_understand_sketch_json(image_bytes=sketch_png, mime_type=mime or "image/png", style_tags=tags)
    except requests.HTTPError as e:
        text = getattr(e.response, "text", str(e))
        sketch_analysis = {}
        log.warning("gemini_understand_sketch_json failed: %s", text)

    # Subject description: prefer caption or tags
    cap = (subject_ref.get("caption") or "").strip()
    tags_ref = [str(t).replace("_"," ") for t in (subject_ref.get("tags") or [])]
    subject_desc = cap or ", ".join(tags_ref) or "reference subject"

    # Control from sketch
    control_png = maybe_make_control_from_sketch(sketch_png)

    # Style-aware prompt + negatives
    prompt, negatives = build_prompt_from_sketch_analysis(
        mode="subject_transfer",
        sketch_analysis=sketch_analysis,
        extra_style_words=[str(t).replace("_"," ") for t in tags][:IMAGEN_STYLE_MAX] if tags else [],
        subject_identity_in_ref=True,
        style_tags=tags,
    )
    neg_extra = ", ".join(negatives) if negatives else ""

    try:
        images, latency_ms, used = _rest_subject_plus_control(
            vc,
            prompt=prompt,
            subject_png=subject_bytes,
            subject_type=subject_type_default,
            subject_desc=subject_desc,
            control_png=control_png,
            n=n,
            negative_extra=neg_extra,
        )
    except requests.HTTPError as e:
        text = getattr(e.response, "text", str(e))
        return {"error": f"Vertex request failed ({e.response.status_code if e.response else 'HTTPError'}): {text}",
                "status": e.response.status_code if e.response else 500}

    data = [{"b64_json": base64.b64encode(img).decode("utf-8")} for img in images]
    return {
        "data": data,
        "model_used": "imagen-3.0-capability-001",
        "latency_ms": int(latency_ms),
        "meta": {
            "source": used,
            "mode": "subject_plus_sketch",
            "count": len(images),
            "constructed_prompt": prompt,
            "subject_type": subject_type_default,
            "control": "scribble",
        },
    }

# ---- legacy edit/generate (unchanged) ----
def run_edit_or_generate(
    *,
    subject: Optional[str],
    refs: List[Dict[str, Any]],
    image_url: Optional[str],
    mask_url: Optional[str],
    n: int = 1,
) -> Dict[str, Any]:
    vc = VertexClients()
    if not vc.ready:
        return {"error": "Vertex AI is not configured. Set GOOGLE_GENAI_USE_VERTEXAI=true and ADC.", "status": 400}

    base_png: Optional[bytes] = None
    mask_png: Optional[bytes] = None

    if image_url and image_url.startswith(DATA_URL_RE_PREFIX):
        _, base_png = data_url_to_bytes(image_url)
        if base_png is None:
            log.warning("images: image_url is a non-image data URL; ignoring")
    if mask_url and mask_url.startswith(DATA_URL_RE_PREFIX):
        _, mask_png = data_url_to_bytes(mask_url)
        if mask_png is None:
            log.warning("images: mask_url is a non-image data URL; ignoring")

    accepted_refs = [r for r in (refs or []) if isinstance(r, dict)]
    palette: List[str] = []
    for r in accepted_refs:
        pal = r.get("palette") or []
        if isinstance(pal, list):
            palette.extend([str(x) for x in pal])

    prompt = compose_prompt(
        subject=subject or "sketch refinement",
        refs=accepted_refs,
        sketch_style_tokens=None,
        palette_hex=palette,
    )

    used = "vertex"
    latency_ms = 0.0
    images: List[bytes] = []
    mode = "generate"

    try:
        if base_png:
            if not mask_png:
                log.info("images: no mask provided — auto-generating full-white mask for masked edit")
                mask_png = _make_full_white_mask_like(base_png)

            base_png, mask_png = ensure_same_size(base_png, mask_png)

            pre_sq = _is_squareish(base_png)
            if not pre_sq and AUTOSQUARE_MODE in ("expand", "crop"):
                if AUTOSQUARE_MODE == "expand":
                    log.info("images: autosquare=expand — padding canvas to square (padded areas editable)")
                    base_png, mask_png = _expand_to_square(base_png, mask_png)
                else:
                    log.info("images: autosquare=crop — center-cropping to square")
                    base_png, mask_png = _crop_to_square(base_png, mask_png)
                base_png, mask_png = ensure_same_size(base_png, mask_png)

            base_png, mask_png = _resize_max_side(base_png, mask_png, IMAGEN_MAX_SIDE)

            if not mask_has_editable(mask_png):
                log.info("images: provided mask has no editable region — replacing with full-white mask")
                mask_png = _make_full_white_mask_like(base_png)

            images, latency_ms, used = _rest_edit_with_mask(
                vc,
                prompt=prompt,
                base_png=base_png,
                mask_png=mask_png,
                n=n,
            )
            mode = "edit"
        else:
            images, latency_ms, used = _rest_generate(vc, prompt=prompt, n=n)
            mode = "generate"

    except requests.HTTPError as e:
        text = getattr(e.response, "text", str(e))
        return {"error": f"Vertex request failed ({e.response.status_code if e.response else 'HTTPError'}): {text}",
                "status": e.response.status_code if e.response else 500}

    data = [{"b64_json": base64.b64encode(img).decode("utf-8")} for img in images]
    return {
        "data": data,
        "model_used": "imagen-3.0-capability-001",
        "latency_ms": int(latency_ms),
        "meta": {
            "source": used,
            "mode": mode,
            "count": len(images),
            "constructed_prompt": prompt,
            "autosquare": AUTOSQUARE_MODE,
            "max_side": IMAGEN_MAX_SIDE,
        },
    }
