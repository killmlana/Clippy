from __future__ import annotations

import base64
import json
import logging
import os
from typing import Dict, List, Optional, Tuple

import requests

log = logging.getLogger("vision")


def _b64(b: bytes) -> str:
    return base64.b64encode(b).decode("utf-8")


def _auth_header() -> Dict[str, str]:
    import google.auth
    from google.auth.transport.requests import Request

    creds, _ = google.auth.default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
    if not creds.valid:
        creds.refresh(Request())
    return {"Authorization": f"Bearer {creds.token}"}


def _model_candidates() -> List[str]:
    prefer = os.getenv("GEMINI_VISION_MODEL")
    if prefer:
        return [prefer]
    return [
        "gemini-2.5-flash",
        "gemini-2.0-flash",
    ]


def _generate_content_vertex(
    *, project: str, location: str, model: str, parts: List[Dict]
) -> Tuple[Optional[str], str]:
    url = (
        f"https://{location}-aiplatform.googleapis.com/v1/"
        f"projects/{project}/locations/{location}/publishers/google/models/{model}:generateContent"
    )
    body = {
        "contents": [{"role": "user", "parts": parts}],
        "generationConfig": {
            "temperature": 0.2,
            "maxOutputTokens": 512,
            "responseMimeType": "application/json",
        },
    }
    headers = _auth_header()
    headers["Content-Type"] = "application/json"

    resp = requests.post(url, headers=headers, data=json.dumps(body), timeout=60)
    if resp.status_code == 404:
        raise FileNotFoundError(resp.text)
    resp.raise_for_status()
    data = resp.json()
    try:
        txt = data["candidates"][0]["content"]["parts"][0].get("text")  # type: ignore[index]
    except Exception:
        txt = None
    return txt, model


def analyze_refs_from_bytes(
    *, project: str, location: str, refs_bytes: List[bytes]
) -> List[Dict[str, str]]:
    if not refs_bytes:
        return []

    models = _model_candidates()
    results: List[Dict[str, str]] = []

    for idx, b in enumerate(refs_bytes):
        parts = [
            {"inlineData": {"mimeType": "image/png", "data": _b64(b)}},
            {"text": (
                "You are a concise vision analyst. In 2–4 bullet points:\n"
                "1) Describe the overall art style & media,\n"
                "2) Note lighting/mood and palette,\n"
                "3) Mention notable textures/materials.\n"
                "Then provide ONE short caption (<= 20 words).\n"
                "Return JSON with keys: style, caption. No prose outside JSON."
            )},
        ]

        txt: Optional[str] = None
        last_err: Optional[Exception] = None
        for m in models:
            try:
                txt, _ = _generate_content_vertex(project=project, location=location, model=m, parts=parts)
                break
            except FileNotFoundError as e:
                log.warning("vision: model not found (%s); trying next", m); last_err = e
            except Exception as e:
                log.info("vision: analysis failed for ref[%d] on %s: %s", idx, m, e); last_err = e

        if not txt:
            if last_err:
                log.info("vision: analysis failed for ref[%d]: %s", idx, last_err)
            results.append({"style": "", "caption": ""})
            continue

        style, caption = "", ""
        try:
            j = json.loads(txt)
            style = str(j.get("style", "")).strip()
            caption = str(j.get("caption", "")).strip()
        except Exception:
            caption = txt.strip()

        results.append({"style": style, "caption": caption})

    return results
