# app/lib/gemini/utils.py
from __future__ import annotations

import base64
import logging
from typing import Dict, Optional, Tuple

log = logging.getLogger("gemini.utils")

DATA_URL_PREFIX = "data:"


def data_url_to_bytes(s: str) -> Tuple[Optional[str], Optional[bytes]]:
    """
    Decode a data: URL into (mime, bytes).

    Only 'image/*' data URLs are accepted. Returns (None, None) if invalid or non-image.

    Examples:
      - data:image/png;base64,iVBORw0...
      - data:image/jpeg;base64,/9j/4AAQ...
    """
    if not s or not isinstance(s, str) or not s.startswith(DATA_URL_PREFIX):
        return None, None
    try:
        head, b64 = s.split(",", 1)
        mime = "application/octet-stream"
        if head.startswith("data:"):
            head_rest = head[5:]
            semi = head_rest.split(";")[0]
            if "/" in semi:
                mime = semi.lower()
        if not mime.startswith("image/"):
            log.warning("data_url_to_bytes: non-image data URL rejected (mime=%s)", mime)
            return None, None
        return mime, base64.b64decode(b64)
    except Exception as e:
        log.debug("data_url_to_bytes failed: %s", e)
        return None, None


def fetch_ref_bytes(ref: Dict[str, object]) -> Optional[bytes]:
    """
    Load bytes for a reference image.

    Accepted forms:
      1) ref['image_b64'] or ref['image_bytes_b64']  (raw base64; no 'data:' header)
      2) ref['image_url']:
         - data: URL -> decoded here (must be image/*)
         - http(s) URL -> downloaded if Content-Type is image/*
         - anything else -> ignored (frontend should inline to data: URL)
    """
    # Raw base64 (no data: header)
    b64 = ref.get("image_b64") or ref.get("image_bytes_b64")
    if isinstance(b64, str) and b64:
        try:
            import base64 as _b
            return _b.b64decode(b64)
        except Exception as e:
            log.debug("fetch_ref_bytes: base64 decode failed: %s", e)

    # data: or http(s) URL
    url = ref.get("image_url")
    if not isinstance(url, str) or not url:
        return None

    if url.startswith(DATA_URL_PREFIX):
        mime, b = data_url_to_bytes(url)
        if not (mime and b):
            log.warning("fetch_ref_bytes: data URL present but not an image or invalid")
            return None
        return b

    if url.startswith("http://") or url.startswith("https://"):
        try:
            import requests
            r = requests.get(url, timeout=15, headers={"Accept": "image/*"})
            r.raise_for_status()
            ct = (r.headers.get("Content-Type") or "").lower()
            if not ct.startswith("image/"):
                log.warning("fetch_ref_bytes: remote URL not image (ct=%s): %s", ct, url)
                return None
            return r.content
        except Exception as e:
            log.debug("fetch_ref_bytes: HTTP fetch failed for %s: %s", url, e)
            return None

    # e.g. '/image/123' — should be resolved + inlined client-side
    log.debug("fetch_ref_bytes: non-absolute/unknown URL (inline as data: on client): %s", url)
    return None
