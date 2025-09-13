# app/lib/gemini/client.py
from __future__ import annotations

import base64
import json
import os
import time
from typing import Any, Dict, List, Optional, Tuple


class VertexClients:
    """
    Vertex AI client wrapper — REST ONLY.
    - Uses Google ADC to fetch an OAuth token.
    - Calls Imagen via :predict and Gemini via :generateContent.
    - No vertexai SDK imports or code paths.
    """

    def __init__(self) -> None:
        self.use_vertex = os.getenv("GOOGLE_GENAI_USE_VERTEXAI", "false").lower() == "true"
        self.project = os.getenv("GOOGLE_CLOUD_PROJECT")
        self.location = os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1")

        self._rest_ready = False
        if self.use_vertex:
            self._try_init_rest()

    # ---------- REST init / token ----------
    def _try_init_rest(self) -> None:
        try:
            import google.auth
            from google.auth.transport.requests import Request
            creds, _ = google.auth.default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
            if not creds.valid:
                creds.refresh(Request())
            self._rest_ready = True
        except Exception as e:
            print(f"[vertex] REST init failed: {e}")

    @property
    def ready(self) -> bool:
        return self.use_vertex and self._rest_ready

    def _obtain_token(self) -> str:
        import google.auth
        from google.auth.transport.requests import Request
        creds, _ = google.auth.default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
        if not creds.valid:
            creds.refresh(Request())
        return creds.token  # type: ignore[no-any-return]

    # ---------- REST predict for Imagen ----------
    def rest_predict(
        self,
        *,
        instances: List[Dict[str, Any]],
        parameters: Optional[Dict[str, Any]] = None,
        model_id: str = "imagen-3.0-capability-001",
        timeout: int = 120,
    ) -> Tuple[List[bytes], float]:
        """
        Call Imagen 3 via publisher predict endpoint.
        Returns (list_of_image_bytes, latency_ms).
        """
        assert self._rest_ready, "Vertex REST not initialized"
        import requests

        url = (
            f"https://{self.location}-aiplatform.googleapis.com/v1/"
            f"projects/{self.project}/locations/{self.location}/publishers/google/models/{model_id}:predict"
        )
        token = self._obtain_token()

        payload: Dict[str, Any] = {"instances": instances}
        if parameters:
            payload["parameters"] = parameters

        start = time.time()
        resp = requests.post(
            url,
            headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
            data=json.dumps(payload),
            timeout=timeout,
        )
        if not resp.ok:
            raise requests.HTTPError(f"{resp.status_code} {resp.text}", response=resp)
        body = resp.json()

        images: List[bytes] = []
        for pred in body.get("predictions", []):
            b64 = pred.get("bytesBase64Encoded")
            if b64:
                images.append(base64.b64decode(b64))
                continue
            for im in pred.get("images", []):
                b = im.get("bytesBase64Encoded")
                if b:
                    images.append(base64.b64decode(b))

        return images, (time.time() - start) * 1000.0

    # ---------- Gemini (vision) JSON analyzers ----------
    def gemini_analyze_image_json(
        self,
        *,
        image_bytes: bytes,
        mime_type: str = "image/png",
        model_id: Optional[str] = None,
        timeout: int = 60,
    ) -> Dict[str, Any]:
        """
        General style/content analyzer (used for style refs, etc.)
        Returns structured tags, palette, composition, camera, and a natural description.
        """
        assert self._rest_ready, "Vertex REST not initialized"
        import requests

        model = model_id or os.getenv("GEMINI_VISION_MODEL", "gemini-2.5-flash")
        url = (
            f"https://{self.location}-aiplatform.googleapis.com/v1/"
            f"projects/{self.project}/locations/{self.location}/publishers/google/models/{model}:generateContent"
        )
        token = self._obtain_token()

        prompt = (
            "You are an art/style analyst. Describe the image's artistic style and content.\n"
            "Return STRICT JSON with keys: "
            "{"
            "\"style_tags\": string[] (max 12, lowercase), "
            "\"palette_hex\": string[] (3-8 like #aabbcc), "
            "\"medium\": string, "
            "\"lighting\": string, "
            "\"textures\": string[], "
            "\"composition\": string, "
            "\"camera\": string, "
            "\"era\": string, "
            "\"mood\": string, "
            "\"natural_description\": string"
            "}\n"
            "Do not include any extra text outside the JSON."
        )

        payload = {
            "contents": [{
                "role": "user",
                "parts": [
                    {"text": prompt},
                    {"inlineData": {"mimeType": mime_type, "data": base64.b64encode(image_bytes).decode("utf-8")}},
                ],
            }],
            "generationConfig": {
                "temperature": 0.2,
                "responseMimeType": "application/json",
            },
        }

        resp = requests.post(
            url,
            headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
            data=json.dumps(payload),
            timeout=timeout,
        )
        if not resp.ok:
            raise requests.HTTPError(f"{resp.status_code} {resp.text}", response=resp)

        data = resp.json()
        try:
            text = (
                data.get("candidates", [{}])[0]
                    .get("content", {})
                    .get("parts", [{}])[0]
                    .get("text", "")
            )
            return json.loads(text) if text else {}
        except Exception:
            return {}

    def gemini_understand_sketch_json(
        self,
        *,
        image_bytes: bytes,
        mime_type: str = "image/png",
        timeout: int = 60,
        model_id: Optional[str] = None,
        style_tags: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Sketch-understanding (lines-on-white) analyzer for robust generation prompts.
        STYLE-AWARE: passes user style tags so guidance ties to them.
        """
        assert self._rest_ready, "Vertex REST not initialized"
        import requests

        model = model_id or os.getenv("GEMINI_VISION_MODEL", "gemini-2.5-flash")
        url = (
            f"https://{self.location}-aiplatform.googleapis.com/v1/"
            f"projects/{self.project}/locations/{self.location}/publishers/google/models/{model}:generateContent"
        )
        token = self._obtain_token()

        style_text = ""
        if style_tags:
            style_text = ", ".join([str(t).replace("_", " ") for t in style_tags][:6])

        prompt = (
            "You are analyzing a rough sketch (few lines on white). "
            "Treat the sketch stroke color as NON-BINDING (just a tool color). "
            "Do NOT infer the final palette from the stroke color unless the style tags call for a monochrome outline look. "
            "Extract structured facts useful for an image-generation prompt.\n"
            f"If provided, the intended style tags are: [{style_text}]. Use them to tailor style guidance.\n"
            "Return STRICT JSON with exactly these keys:\n"
            "{\n"
            "  \"subject_summary\": string,\n"
            "  \"subject_type_guess\": string,\n"
            "  \"pose_orientation\": string,\n"
            "  \"viewpoint\": string,\n"
            "  \"environment\": string,\n"
            "  \"key_objects\": string[],\n"
            "  \"composition\": string,\n"
            "  \"camera\": string,\n"
            "  \"style_cues_from_lines\": string[],\n"
            "  \"negative_cues\": string[],\n"
            "  \"natural_description\": string,\n"
            "  \"style_notes\": string,\n"
            "  \"style_key_traits\": string[],\n"
            "  \"style_negative_notes\": string[],\n"
            "  \"style_palette_hint\": string\n"
            "}\n"
            "No extra text outside the JSON."
        )

        payload = {
            "contents": [{
                "role": "user",
                "parts": [
                    {"text": prompt},
                    {"inlineData": {"mimeType": mime_type, "data": base64.b64encode(image_bytes).decode("utf-8")}},
                ],
            }],
            "generationConfig": {
                "temperature": 0.1,
                "responseMimeType": "application/json",
            },
        }

        resp = requests.post(
            url,
            headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
            data=json.dumps(payload),
            timeout=timeout,
        )
        if not resp.ok:
            raise requests.HTTPError(f"{resp.status_code} {resp.text}", response=resp)

        data = resp.json()
        try:
            text = (
                data.get("candidates", [{}])[0]
                    .get("content", {})
                    .get("parts", [{}])[0]
                    .get("text", "")
            )
            return json.loads(text) if text else {}
        except Exception:
            return {}
