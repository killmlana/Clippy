#!/usr/bin/env python3
# Deterministic similar-image search (OOM-safe)
import argparse, json
from typing import Optional, List
import numpy as np
from PIL import Image, ImageSequence
import torch
import open_clip
from qdrant_client import QdrantClient

# ------- PIL loader -------
def load_pil_rgb(path: str) -> Optional[Image.Image]:
    try:
        im = Image.open(path)
        if getattr(im, "is_animated", False):
            im = next(ImageSequence.Iterator(im))
        if im.mode in ("RGBA", "LA"):
            bg = Image.new("RGB", im.size, (255, 255, 255))
            bg.paste(im, mask=im.split()[-1])
            im.close()
            return bg
        return im.convert("RGB")
    except Exception:
        return None

# ------- OpenCLIP helpers -------
def load_clip(model_name: str, pretrained: str, device: str):
    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name, pretrained=pretrained, device=device
    )
    model.eval()
    return model, preprocess

@torch.no_grad()
def embed_image_once(model, preprocess, pil: Image.Image, device: str, precision: str):
    # precision: "fp16", "bf16", "fp32"
    use_amp = (device == "cuda") and (precision in ("fp16", "bf16"))
    amp_dtype = torch.float16 if precision == "fp16" else torch.bfloat16
    ctx = torch.autocast(device_type="cuda", dtype=amp_dtype) if use_amp else torch.cuda.amp.autocast(False)

    with ctx:
        x = preprocess(pil).unsqueeze(0)
        if device == "cuda":
            x = x.to(device, non_blocking=True)
        v = model.encode_image(x)
        v = v / v.norm(dim=-1, keepdim=True)
    vec = v.squeeze(0).detach().to("cpu").numpy().astype(np.float32)

    # free VRAM immediately
    del x, v
    if device == "cuda":
        torch.cuda.empty_cache()
    return vec

def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))

# ------- main -------
def main():
    ap = argparse.ArgumentParser(description="Deterministic similar-image search (image+tags vectors), OOM-safe")
    ap.add_argument("--qdrant-url", default="http://localhost:6333")
    ap.add_argument("--collection", required=True)
    ap.add_argument("--image", required=True)
    ap.add_argument("--topk", type=int, default=24)
    ap.add_argument("--candidates", type=int, default=200)
    ap.add_argument("--ef", type=int, default=1024)
    ap.add_argument("--exact-candidates", action="store_true")

    # model + device controls
    ap.add_argument("--model", default="ViT-bigG-14")
    ap.add_argument("--pretrained", default="laion2b_s39b_b160k")
    ap.add_argument("--device", default=("cuda" if torch.cuda.is_available() else "cpu"),
                    choices=["cuda","cpu"])
    ap.add_argument("--precision", default="fp16", choices=["fp16","bf16","fp32"],
                    help="GPU precision; ignored on CPU")
    ap.add_argument("--fallbackModels", default="ViT-H-14:laion2b_s32b_b79k,ViT-B-32:laion2b_s34b_b79k",
                    help="comma list of model:pretrained to try if OOM (on same device)")

    # fusion weights
    ap.add_argument("--w-image", type=float, default=0.8)
    ap.add_argument("--w-text", type=float, default=0.2)

    ap.add_argument("--json", dest="as_json", action="store_true")
    ap.add_argument("--exclude-path", action="store_true")
    args = ap.parse_args()

    # deterministic tie-breaks
    np.random.seed(0)
    if args.device == "cuda":
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.matmul.allow_tf32 = True  # speedup if supported

    pil = load_pil_rgb(args.image)
    if pil is None:
        raise SystemExit(f"Could not open image: {args.image}")

    # ---- Try to load model on requested device; fall back on OOM ----
    tried = []
    candidates_models = [(args.model, args.pretrained)]
    # if OOM, try same device with smaller models, then CPU with original and fallbacks
    fb = []
    if args.fallbackModels:
        for tok in args.fallbackModels.split(","):
            tok = tok.strip()
            if not tok: continue
            if ":" in tok:
                m, p = tok.split(":", 1)
                fb.append((m.strip(), p.strip()))
    candidates_models.extend(fb)

    def try_load_embed(device_choice: str):
        last_err = None
        for m, p in candidates_models:
            try:
                model, preprocess = load_clip(m, p, device_choice)
                vec = embed_image_once(model, preprocess, pil, device_choice, args.precision)
                # free weights ASAP
                del model
                if device_choice == "cuda":
                    torch.cuda.empty_cache()
                return vec, (m, p, device_choice)
            except RuntimeError as e:
                em = str(e)
                last_err = e
                tried.append((m, p, device_choice, em[:120]))
                # clear on OOM
                if device_choice == "cuda":
                    torch.cuda.empty_cache()
                # only treat CUDA out of memory as a fallback trigger
                if "out of memory" in em.lower() or "cublas" in em.lower():
                    continue
                else:
                    raise
        # if we’re here, all candidates on this device failed
        raise last_err if last_err else RuntimeError("Failed to load/embed")

    try:
        q_vec, used = try_load_embed(args.device)
    except Exception:
        if args.device == "cuda":
            # fall back to CPU for the same chain (original + fallbacks)
            q_vec, used = try_load_embed("cpu")
        else:
            # already on CPU: rethrow
            raise
    # print(f"Using model={used[0]} pretrained={used[1]} device={used[2]}")

    # ---- candidate fetch from Qdrant (image space only) ----
    client = QdrantClient(url=args.qdrant_url)
    ann = client.search(
        collection_name=args.collection,
        query_vector=("image", q_vec.tolist()),
        limit=max(args.candidates, args.topk),
        search_params={"hnsw_ef": args.ef, "exact": args.exact_candidates},
        with_payload=True,
    )
    cand_ids = [int(r.id) for r in ann]
    if not cand_ids:
        print("[]") if args.as_json else print("No candidates returned.")
        return

    # ---- retrieve candidate vectors for deterministic exact rerank ----
    pts = client.retrieve(
        collection_name=args.collection,
        ids=cand_ids,
        with_vectors=True,
        with_payload=True,
    )

    id_list: List[int] = []
    img_vecs: List[np.ndarray] = []
    txt_vecs: List[Optional[np.ndarray]] = []
    payloads = []
    for p in pts:
        vecs = getattr(p, "vectors", {}) or {}
        v_img = vecs.get("image")
        v_txt = vecs.get("text")
        if not v_img:
            continue
        v_img = np.asarray(v_img, dtype=np.float32)
        v_txt = np.asarray(v_txt, dtype=np.float32) if v_txt is not None else None
        id_list.append(int(p.id))
        img_vecs.append(v_img)
        txt_vecs.append(v_txt)
        payloads.append(p.payload or {})

    # exact fusion scoring
    scores = []
    for i, pid in enumerate(id_list):
        s_img = cosine(q_vec, img_vecs[i])
        s_txt = cosine(q_vec, txt_vecs[i]) if txt_vecs[i] is not None else 0.0
        s = args.w_image * s_img + args.w_text * s_txt
        scores.append((pid, s))

    # stable sort
    scores.sort(key=lambda t: (-t[1], t[0]))
    top = scores[:args.topk]

    # format
    pmap = {id_list[i]: payloads[i] for i in range(len(id_list))}
    out = []
    for pid, sc in top:
        pl = pmap.get(pid, {})
        out.append({
            "id": pid,
            "score": round(float(sc), 6),
            "path": None if args.exclude_path else pl.get("path"),
            "source_post_url": pl.get("source_post_url"),
            "tags_all": pl.get("tags_all"),
        })

    if args.as_json:
        print(json.dumps(out, indent=2, ensure_ascii=False))
    else:
        for r in out:
            print(f"{r['id']:>10}  score={r['score']:.6f}  "
                  f"path={r['path'] if not args.exclude_path else '…'}  url={r['source_post_url']}")
            if r.get("tags_all"):
                tags = ", ".join((r["tags_all"] or [])[:10])
                print(f"    tags: {tags}")

if __name__ == "__main__":
    main()
