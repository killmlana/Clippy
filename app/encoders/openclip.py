from __future__ import annotations
from functools import lru_cache
from typing import Tuple, List
import os
import numpy as np
import torch
from PIL import Image
import open_clip

# Env-configurable; matches your locked choices by default
MODEL_NAME = os.getenv("OPENCLIP_MODEL", "ViT-bigG-14")
PRETRAINED = os.getenv("OPENCLIP_PRETRAINED", "laion2b_s39b_b160k")
DEVICE      = os.getenv("OPENCLIP_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
PRECISION   = os.getenv("OPENCLIP_PRECISION", "fp32")  # fp16|bf16|fp32

@lru_cache(maxsize=1)
def _get_model() -> Tuple[torch.nn.Module, any, str, str]:
    model, _, preprocess = open_clip.create_model_and_transforms(
        MODEL_NAME, pretrained=PRETRAINED, device=DEVICE
    )
    model.eval()
    return model, preprocess, DEVICE, PRECISION

def _amp_ctx(device: str, precision: str):
    use_amp = (device == "cuda") and precision in ("fp16", "bf16")
    if not use_amp:
        class _Noop:
            def __enter__(self): return None
            def __exit__(self, *a): return False
        return _Noop()
    dtype = torch.float16 if precision == "fp16" else torch.bfloat16
    return torch.amp.autocast("cuda", dtype=dtype)

def _to_unit(vec: torch.Tensor) -> np.ndarray:
    vec = vec / (vec.norm(dim=-1, keepdim=True) + 1e-8)
    arr = vec.squeeze(0).detach().to("cpu", dtype=torch.float32).numpy()
    return arr  # shape [1280]

@torch.no_grad()
def encode_image_1280(pil: Image.Image) -> List[float]:
    model, preprocess, device, precision = _get_model()
    try:
        with _amp_ctx(device, precision):
            x = preprocess(pil).unsqueeze(0).to(device)
            v = model.encode_image(x)
        arr = _to_unit(v)
    except RuntimeError as e:
        # Graceful CPU fallback on OOM
        if "out of memory" in str(e).lower() and device == "cuda":
            torch.cuda.empty_cache()
            os.environ["OPENCLIP_DEVICE"] = "cpu"
            _get_model.cache_clear()
            model, preprocess, device, precision = _get_model()
            with _amp_ctx(device, precision):
                x = preprocess(pil).unsqueeze(0).to(device)
                v = model.encode_image(x)
            arr = _to_unit(v)
        else:
            raise
    return arr.astype(np.float32).tolist()

@torch.no_grad()
def encode_text_1280(text: str) -> List[float]:
    model, preprocess, device, precision = _get_model()
    tokenizer = open_clip.get_tokenizer(MODEL_NAME)
    toks = tokenizer([text]).to(device)
    with _amp_ctx(device, precision):
        v = model.encode_text(toks)
    arr = _to_unit(v).astype(np.float32)
    return arr.tolist()
