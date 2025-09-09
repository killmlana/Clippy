#!/usr/bin/env python3
import argparse, json, sys
from typing import Optional, List
import numpy as np
from PIL import Image, ImageSequence, Image as PILImage
import torch
import open_clip
from qdrant_client import QdrantClient
from qdrant_client.models import SearchParams


# ---------- helpers ----------
def load_pil_rgb(path: str) -> Optional[Image.Image]:
    try:
        im = Image.open(path)
        if getattr(im, "is_animated", False):
            im = next(ImageSequence.Iterator(im))
        if im.mode in ("RGBA", "LA"):
            bg = PILImage.new("RGB", im.size, (255, 255, 255))
            bg.paste(im, mask=im.split()[-1]);
            im.close()
            return bg
        return im.convert("RGB")
    except Exception:
        return None


def load_clip(model_name: str, pretrained: str, device: str):
    model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained, device=device)
    model.eval()
    return model, preprocess


@torch.no_grad()
def embed_image(model, preprocess, pil: Image.Image, device: str, precision: str):
    use_amp = (device == "cuda") and (precision in ("fp16", "bf16"))
    amp_dtype = torch.float16 if precision == "fp16" else torch.bfloat16
    if use_amp:
        ctx = torch.amp.autocast("cuda", dtype=amp_dtype)
    else:
        # no AMP
        class _Noop:
            def __enter__(self): return None

            def __exit__(self, *a): return False

        ctx = _Noop()

    with ctx:
        x = preprocess(pil).unsqueeze(0)
        if device == "cuda":
            x = x.to(device, non_blocking=True)
        v = model.encode_image(x)
        v = v / v.norm(dim=-1, keepdim=True)
    vec = v.squeeze(0).detach().to("cpu").numpy().astype(np.float32)
    del x, v
    if device == "cuda":
        torch.cuda.empty_cache()
    return vec


def cosine(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def get_vector_sizes(client, collection_name: str):
    info = client.get_collection(collection_name)
    cfg = getattr(info, "config", None)
    if cfg and getattr(cfg, "params", None) and getattr(cfg.params, "vectors", None):
        vecs = cfg.params.vectors
        return {k: int(v.size) for k, v in vecs.items()} if isinstance(vecs, dict) else {"": int(vecs.size)}
    params = getattr(info, "params", None)
    if params and getattr(params, "vectors", None):
        vecs = params.vectors
        return {k: int(v.size) for k, v in vecs.items()} if isinstance(vecs, dict) else {"": int(vecs.size)}
    raise RuntimeError("Could not resolve collection vector schema")


# ---------- main ----------
def main():
    ap = argparse.ArgumentParser(description="Deterministic similar-image search (image + text) via query_points")
    ap.add_argument("--qdrant-url", default="http://localhost:6333")
    ap.add_argument("--collection", required=True)
    ap.add_argument("--image", required=True)
    ap.add_argument("--topk", type=int, default=24)
    ap.add_argument("--candidates", type=int, default=200)
    ap.add_argument("--ef", type=int, default=1024)
    ap.add_argument("--exact-candidates", action="store_true")
    ap.add_argument("--model", default="ViT-bigG-14")
    ap.add_argument("--pretrained", default="laion2b_s39b_b160k")
    ap.add_argument("--device", default=("cuda" if torch.cuda.is_available() else "cpu"), choices=["cuda", "cpu"])
    ap.add_argument("--precision", default="fp32", choices=["fp16", "bf16", "fp32"])
    ap.add_argument("--w-image", type=float, default=0.8)
    ap.add_argument("--w-text", type=float, default=0.2)
    ap.add_argument("--json", dest="as_json", action="store_true")
    ap.add_argument("--exclude-path", action="store_true")
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()

    np.random.seed(0)
    if args.device == "cuda":
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.matmul.allow_tf32 = True

    # 1) embed query
    pil = load_pil_rgb(args.image)
    if pil is None:
        print("Could not open image", file=sys.stderr)
        if args.as_json: print("[]")
        sys.exit(2)

    model, preprocess = load_clip(args.model, args.pretrained, args.device)
    try:
        q_vec = embed_image(model, preprocess, pil, args.device, args.precision)
    except RuntimeError as e:
        if "out of memory" in str(e).lower() and args.device == "cuda":
            model, preprocess = load_clip(args.model, args.pretrained, "cpu")
            q_vec = embed_image(model, preprocess, pil, "cpu", "fp32")
        else:
            raise
    del model
    if args.device == "cuda":
        torch.cuda.empty_cache()

    # 2) schema & count
    client = QdrantClient(url=args.qdrant_url)
    vec_sizes = get_vector_sizes(client, args.collection)
    need = vec_sizes.get("image")
    count = client.count(args.collection, exact=True).count
    if args.debug:
        print(f"[debug] query dim: {len(q_vec)}  schema(image): {need}  points: {count}")

    if need is None:
        print("[debug] No 'image' vector in schema", file=sys.stderr)
        if args.as_json: print("[]"); sys.exit(3)
    if len(q_vec) != need:
        print(f"[debug] DIM MISMATCH: query {len(q_vec)} vs schema {need}", file=sys.stderr)
        if args.as_json: print("[]"); sys.exit(4)
    if count == 0:
        print("[debug] Collection empty", file=sys.stderr)
        if args.as_json: print("[]"); sys.exit(5)

    # 3) candidate fetch with query_points (PASS LIST, not dict)
    sp = SearchParams(hnsw_ef=args.ef, exact=args.exact_candidates)
    res = client.query_points(
        collection_name=args.collection,
        query=q_vec.tolist(),  # <-- list
        using="image",  # <-- which named vector
        with_payload=True,
        with_vectors=False,
        limit=max(args.candidates, args.topk),
        search_params=sp  # <-- correct kwarg
    )
    cand_ids = [int(p.id) for p in res.points]
    if args.debug:
        print(f"[debug] candidates fetched: {len(cand_ids)} (ef={args.ef}, exact={args.exact_candidates})")

    if not cand_ids:
        # peek whether points really have image vectors
        pts, _ = client.scroll(collection_name=args.collection, with_vectors=True, limit=10)
        has_img = sum(1 for p in pts if (p.vectors or {}).get("image"))
        if args.debug:
            print(f"[debug] sample points with image vectors: {has_img}/10")
        if args.as_json:
            print("[]")
        else:
            print("No candidates returned.")
        return

    # 4) exact rerank (image + text)
    full = client.retrieve(
        args.collection,
        ids=cand_ids,
        with_vectors=["image", "text"],
        with_payload=True,
    )

    # Debug: how many came back with vectors?
    has_img = sum(1 for p in full if (getattr(p, "vector", {}) or {}).get("image"))
    has_txt = sum(1 for p in full if (getattr(p, "vector", {}) or {}).get("text"))
    if args.debug:
        print(f"[debug] retrieve: {len(full)} points; with image={has_img}, with text={has_txt}")

    id_list, img_vecs, txt_vecs, payloads = [], [], [], []
    for p in full:
        vecs = (getattr(p, "vector", {}) or {})
        v_img = vecs.get("image")
        if v_img is None:
            continue
        v_txt = vecs.get("text")
        id_list.append(int(p.id))
        img_vecs.append(np.asarray(v_img, dtype=np.float32))
        txt_vecs.append(np.asarray(v_txt, dtype=np.float32) if v_txt is not None else None)
        payloads.append(p.payload or {})

    scores = []
    for i, pid in enumerate(id_list):
        s_img = cosine(q_vec, img_vecs[i])
        s_txt = cosine(q_vec, txt_vecs[i]) if txt_vecs[i] is not None else 0.0
        s = args.w_image * s_img + args.w_text * s_txt
        scores.append((pid, s))
    scores.sort(key=lambda t: (-t[1], t[0]))
    top = scores[:args.topk]

    pmap = {id_list[i]: payloads[i] for i in range(len(id_list))}
    out = []
    for pid, sc in top:
        pl = pmap.get(pid, {})
        out.append({
            "id": pid,
            "score": round(float(sc), 6),
            "path": (None if args.exclude_path else pl.get("path")),
            "source_post_url": pl.get("source_post_url"),
            "tags_all": pl.get("tags_all"),
        })

    print(json.dumps(out, indent=2, ensure_ascii=False) if args.as_json else out)


if __name__ == "__main__":
    main()
