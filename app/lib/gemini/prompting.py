# app/lib/gemini/prompting.py
from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple
import logging
import os
import re

_LOG_NAME = "prompting"
logger = logging.getLogger(_LOG_NAME)
if not logger.handlers:
    _level = os.getenv("PROMPT_LOG_LEVEL", "INFO").upper()
    try:
        level = getattr(logging, _level, logging.INFO)
    except Exception:
        level = logging.INFO
    logger.setLevel(level)
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("[%(levelname)s] %(name)s: %(message)s"))
    logger.addHandler(_handler)

_PEOPLE_WORDS = {
    "person","people","human","humans","figure","figures","character","characters",
    "portrait","face","faces","man","woman","men","women","male","female","boy","girl"
}
_WORD_SPLIT_RE = re.compile(r"[^\w]+")

def _short_list(items: List[str], max_items: int = 8) -> str:
    return ", ".join([x for x in items if x][:max_items])

def _tokenize(s: str) -> List[str]:
    return [t.lower() for t in _WORD_SPLIT_RE.split(s or "") if t]

def _has_people_in_text(s: Optional[str]) -> bool:
    if not s:
        return False
    toks = set(_tokenize(s))
    return any(w in toks for w in _PEOPLE_WORDS)

def _has_people_in_refs(refs: List[Dict[str, Any]]) -> bool:
    for r in refs:
        tags = r.get("tags") or []
        if isinstance(tags, list):
            tag_toks = {str(t).lower() for t in tags}
            if tag_toks & _PEOPLE_WORDS:
                return True
        cap = r.get("caption")
        if cap and _has_people_in_text(cap):
            return True
    return False

def _dedup(seq: List[str]) -> List[str]:
    out, seen = [], set()
    for s in seq:
        if s not in seen:
            seen.add(s); out.append(s)
    return out

def default_negative_prompt() -> str:
    return "no text overlays, no watermark, no signature, no border, no frame"

# ---- style enforcement helpers (stronger execution for certain tags)
_STYLE_HINTS: Dict[str, Dict[str, Any]] = {
    "pixel_art": {
        "positives": [
            "low-resolution pixel grid",
            "blocky 8-bit aesthetic",
            "nearest-neighbor crisp edges (no antialiasing)",
            "limited 16–32 color retro palette",
            "1–2 px outlines around forms",
        ],
        "negatives": [
            "smooth gradients",
            "photographic textures",
            "vector-like clean curves",
            "motion blur",
            "soft focus",
        ],
        "palette_hint": "small, vibrant retro palette; avoid smooth gradients",
    },
    "vector_art": {
        "positives": [
            "clean Bézier curves",
            "flat, fully filled shapes (no hatching)",
            "geometric simplification",
            "crisp, uniform edges",
            "limited shading with hard breakpoints",
        ],
        "negatives": [
            "outline-only look",
            "monochrome line drawing",
            "sketchy hand-drawn strokes",
            "textured or noisy fills",
            "photographic gradients",
        ],
        "palette_hint": "bold, flat color regions with high local contrast; minimal gradients",
    },
    "cel_shading": {
        "positives": ["hard-edged shadows", "2–3 tone shading", "comic/anime lighting"],
        "negatives": ["soft painterly gradients", "photoreal textures"],
    },
    "manga_style": {
        "positives": ["inked lineart", "screentones", "high contrast black & white or limited palette"],
        "negatives": ["photorealistic skin pores", "heavy color gradients"],
    },
    "line_art": {
        "positives": ["clean, confident lines", "minimal or no fill", "consistent line weight"],
        "negatives": ["painterly shading", "noisy textures"],
    },
}

def _normalize_tag(t: str) -> str:
    return t.strip().lower().replace(" ", "_")

def style_hints_for_tags(tags: List[str]) -> Tuple[List[str], List[str], Optional[str]]:
    pos: List[str] = []
    neg: List[str] = []
    pal: Optional[str] = None
    for raw in tags or []:
        k = _normalize_tag(raw)
        hints = _STYLE_HINTS.get(k)
        if not hints:
            continue
        pos.extend(hints.get("positives", []))
        neg.extend(hints.get("negatives", []))
        if not pal and hints.get("palette_hint"):
            pal = hints["palette_hint"]
    return _dedup(pos), _dedup(neg), pal

def enforce_style_overrides(style_tags: List[str], style_bundle: List[str], negatives: List[str]) -> Tuple[List[str], List[str]]:
    """
    When certain styles are chosen (e.g., vector_art), drop sketch-derived cues that fight the style
    and add reinforcing negatives.
    """
    tags = {t.replace(" ", "_").lower() for t in style_tags or []}
    sb = [s for s in style_bundle if s]
    neg = [n for n in negatives if n]

    if "vector_art" in tags:
        drop = {"line art", "outline only", "outline-only", "monochrome outline", "sketchy", "hand-drawn", "magenta color"}
        sb = [s for s in sb if s.lower() not in drop]
        neg.extend(["outline-only look", "monochrome line drawing", "sketchy hand-drawn strokes"])

    # dedup case-insensitively
    def _dedup_ci(x: List[str]) -> List[str]:
        out, seen = [], set()
        for v in x:
            k = v.lower()
            if k not in seen:
                seen.add(k); out.append(v)
        return out

    return _dedup_ci(sb), _dedup_ci(neg)

# ---------- Existing edit/generate prompt ----------
def compose_prompt(
    *,
    subject: Optional[str],
    refs: List[Dict[str, Any]],
    sketch_style_tokens: Optional[List[str]] = None,
    palette_hex: Optional[List[str]] = None,
) -> str:
    subj = (subject or "sketch refinement of the drawn subject, faithful to user lines").strip()

    style_words: List[str] = []
    captions: List[str] = []
    pal: List[str] = []

    for r in refs:
        tags = r.get("tags") or []
        if isinstance(tags, list):
            style_words.extend([str(t) for t in tags])
        cap = r.get("caption")
        if cap:
            captions.append(str(cap))
        pr = r.get("palette") or []
        if isinstance(pr, list):
            pal.extend([str(x) for x in pr])

    if palette_hex:
        pal.extend(palette_hex)

    style_words = _dedup(style_words)
    pal = _dedup(pal)

    subject_has_people = _has_people_in_text(subj)
    refs_have_people = _has_people_in_refs(refs)
    allow_people = subject_has_people or refs_have_people

    parts: List[str] = []
    parts.append(f"Subject: {subj}.")
    parts.append(
        "Task: edit and morph/improve the existing artwork/sketch using the references provided "
        "continuing background, lighting, perspective, materials and local detail seamlessly from the provided sketch and references."
    )
    parts.append("Preserve the original subject identity, silhouette, geometry, and camera framing from the sketch.")

    if not allow_people:
        parts.append(
            "Do not introduce any new subjects, characters, faces, or humans. Introduce background elements from references provided."
            "Prefer environmental continuation (background, sky, foliage, architecture, textures)."
        )
    else:
        parts.append("Do not add extra characters beyond what is implied by the sketch and references. Introduce background elements from references provided.")

    if style_words:
        parts.append(f"Style modifiers from references: {_short_list(style_words)}.")
    if captions:
        parts.append(f"Context from reference captions: {_short_list(captions, 6)}.")
    if sketch_style_tokens:
        parts.append(f"Sketch style cues: {_short_list(sketch_style_tokens)}.")
    if pal:
        parts.append(f"Color palette preference: {_short_list(pal, 10)}.")
    parts.append("Apply edits strictly inside the mask; keep unmasked pixels unchanged. "
                 "Maintain composition and line fidelity; avoid text overlays or watermarks.")

    prompt = " ".join(parts)

    try:
        logger.info("Prompt (len=%d): %s", len(prompt), prompt)
    except Exception:
        pass

    return prompt

# ---------- Build prompt from sketch analysis (style-aware) ----------
def build_prompt_from_sketch_analysis(
    *,
    mode: str,                                   # "style_customize" or "subject_transfer"
    sketch_analysis: Dict[str, Any],
    extra_style_words: Optional[List[str]] = None,
    subject_identity_in_ref: bool = False,
    style_tags: Optional[List[str]] = None,
) -> Tuple[str, List[str]]:
    """
    Construct a robust natural prompt from Gemini 2.5 Flash sketch analysis.
    Returns (prompt, negatives_list) so callers can also populate Imagen's negativePrompt.
    """
    extra_style_words = extra_style_words or []
    style_tags = style_tags or []
    sa = sketch_analysis or {}

    subject_summary = sa.get("subject_summary") or "the sketched subject"
    pose = sa.get("pose_orientation") or ""
    viewpoint = sa.get("viewpoint") or ""
    environment = sa.get("environment") or ""
    composition = sa.get("composition") or ""
    camera = sa.get("camera") or ""
    natural_description = sa.get("natural_description") or ""
    key_objects = sa.get("key_objects") or []

    style_cues = sa.get("style_cues_from_lines") or []
    style_notes = sa.get("style_notes") or ""
    style_key_traits = sa.get("style_key_traits") or []
    style_negative_notes = sa.get("style_negative_notes") or []
    style_palette_hint = sa.get("style_palette_hint") or ""

    hints_pos, hints_neg, hints_pal = style_hints_for_tags(style_tags)

    parts: List[str] = []

    if mode == "style_customize":
        parts.append("Follow the pose/lines of [3]. Adopt the style from [2].")
    else:  # subject_transfer
        parts.append("Preserve subject identity from [1]. Follow the pose/lines of [2].")

    parts.append(f"Subject: {subject_summary}.")
    if natural_description:
        parts.append(f"Sketch intent: {natural_description}")
    if pose or viewpoint:
        pieces = [p for p in [pose, viewpoint] if p]
        parts.append("Orientation & Viewpoint: " + ", ".join(pieces) + ".")
    if environment:
        parts.append(f"Environment: {environment}.")
    if key_objects:
        parts.append("Key objects: " + _short_list([str(x) for x in key_objects], 8) + ".")
    if composition:
        parts.append(f"Composition: {composition}.")
    if camera:
        parts.append(f"Camera: {camera}.")

    readable_tags = _short_list([t.replace("_", " ") for t in style_tags], 6)
    if readable_tags:
        parts.append(f"Render explicitly in the intended style: {readable_tags}.")

    style_bundle = _dedup([*style_cues, *style_key_traits, *extra_style_words, *hints_pos])
    negatives = _dedup([*style_negative_notes, *hints_neg])

    # Enforce overrides so chosen style wins over sketch-only cues
    style_bundle, negatives = enforce_style_overrides(style_tags or [], style_bundle, negatives)

    if style_notes:
        parts.append(f"Style guidance: {style_notes}.")
    if style_bundle:
        parts.append("Ensure the output clearly exhibits: " + _short_list(style_bundle, 12) + ".")

    palette_hint = style_palette_hint or hints_pal
    if palette_hint:
        parts.append(f"Palette hint: {palette_hint}.")

    if mode == "subject_transfer" and subject_identity_in_ref:
        parts.append("Maintain identity and distinctive features from [1]; do not invent new identity traits.")

    parts.append(default_negative_prompt())
    parts.append("High visual fidelity to the sketch geometry and layout. Clean, coherent, and artifact-free output.")

    prompt = " ".join(parts)
    try:
        logger.info("SketchPrompt (len=%d): %s", len(prompt), prompt)
    except Exception:
        pass
    return prompt, negatives
