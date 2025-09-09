from __future__ import annotations
import numpy as np

try:
    from app.encoders.openclip import encode_image_1280, encode_text_1280
except Exception as e:
    raise RuntimeError(
        "Missing encoders.openclip.encode_image_1280 / encode_text_1280. "
        "Provide your scaffolds."
    ) from e

def ensure_vec(lst_like) -> list[float]:
    if isinstance(lst_like, np.ndarray):
        return lst_like.astype(np.float32).tolist()
    return [float(x) for x in lst_like]

def cos_norm(vec: list[float]) -> list[float]:
    arr = np.asarray(vec, dtype=np.float32)
    n = float(np.linalg.norm(arr) + 1e-8)
    return (arr / n).astype(np.float32).tolist()