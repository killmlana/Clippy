#!/usr/bin/env python3
# Deterministic "find similar" using only image + tags(text) vectors
import argparse, os, json
import numpy as np
from typing import Optional, List
from PIL import Image, ImageSequence
import torch
import open_clip
from qdrant_client import QdrantClient

# ---------- PIL loader ----------
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

# ---------- OpenCLIP ----------
def load_clip(model_name: str, pretrained: str, device: str):
    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name, pretrained=pretrained, device=device
    )
    model.eval()
    return model, preprocess

@torch.no_grad()
def embed_image(model, preprocess, pil: Image.Image, device: str) -> np.ndarray:
    with torch.cuda.amp.autocast(enabled=(device == "cuda")):
        x = preprocess(pil).unsqueeze(0).to(device)
        v = model.encode_image(x)
        v = v / v.norm(dim=-1, keepdim=True)
    return v.squeeze(0).cpu().numpy().astype(np.float32)

# ---------- Cosine helpers ----------
def cosine(a: np.ndarray, b: np.ndarray) -> float:
    # expects non-zero vectors
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))

# ---------- Main search ----------
def main():
    ap = argparse.ArgumentParser(description="Deterministic similar-image search (image + tags vectors)")
    ap.add_argument("--qdrant-url", default="http://localhost:6333")
    ap.add_argument("--collection", required=True)
    ap.add_argument("--image", required=True, help="path to the query image")
    ap.add_argument("--topk", type=int, default=24, help="final results to return")
    ap.add_argument("--candidates", type=int, default=200, help="candidate pool before exact rerank")
    ap.add_argument("--model", default="ViT-bigG-14")
    ap.add_argument("--pretrained", default="laion2b_s39b_b160k")
    ap.add_argument("--device", default=("cuda" if torch.cuda.is_available() else "cpu"))
    ap.add_argument("--ef", type=int, default=1024, help="HNSW ef for candidate fetch")
    ap.add_argument("--exact-candidates", action="store_true",
                    help="ask Qdrant to compute exact distances when fetching candidates (more deterministic, slower)")
    ap.add_argument("--w-image", type=float, default=0.7, help="weight for image↔image cosine")
    ap.add_argument("--w-text", type=float, default=0.3, help="weight for image↔text cosine")
    ap.add_argument("--json", dest="as_json", action="store_true", help="print JSON instead of text")
    ap.add_argument("--exclude-path", action="store_true", help="omit local file paths from output")
    args = ap.parse_args()

    # deterministic numpy order
    np.random.seed(0)

    # 1) Embed the input image with the same model you used at ingest
    pil = load_pil_rgb(args.image)
    if pil is None:
        raise SystemExit(f"Could not open image: {args.image}")
    model, preprocess = load_clip(args.model, args.pretrained, args.device)
    q_vec = embed_image(model, preprocess, pil, args.device)

    # 2) Fetch candidates from Qdrant using the IMAGE vector space
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
        print("No candidates returned.")
        return

    # 3) Retrieve candidate vectors (image + text) exactly for deterministic rerank
    pts = client.retrieve(
        collection_name=args.collection,
        ids=cand_ids,
        with_vectors=True,
        with_payload=True,
    )

    # build arrays for scoring
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
        # text vector is optional; if missing, treat as zero contribution
        v_txt = np.asarray(v_txt, dtype=np.float32) if v_txt is not None else None

        id_list.append(int(p.id))
        img_vecs.append(v_img)
        txt_vecs.append(v_txt)
        payloads.append(p.payload or {})

    if not id_list:
        print("No candidates with usable vectors.")
        return

    # 4) Exact, deterministic fusion scoring
    scores = []
    for i in range(len(id_list)):
        s_img = cosine(q_vec, img_vecs[i])
        s_txt = cosine(q_vec, txt_vecs[i]) if txt_vecs[i] is not None else 0.0
        s = args.w_image * s_img + args.w_text * s_txt
        scores.append((id_list[i], s))

    # 5) Stable sort by (score desc, id asc) for determinism
    scores.sort(key=lambda t: (-t[1], t[0]))
    top = scores[:args.topk]

    # map to payloads
    out = []
    pay_by_id = {int(p.get("id", i_id)) if "id" in p else i_id: (p, i)
                 for i, (i_id, _) in enumerate(scores)}
    # Better: build a dict by actual id_list order
    pmap = {id_list[i]: payloads[i] for i in range(len(id_list))}
    for pid, sc in top:
        pl = pmap.get(pid, {})
        row = {
            "id": pid,
            "score": round(float(sc), 6),
            "path": None if args.exclude_path else pl.get("path"),
            "source_post_url": pl.get("source_post_url"),
            "tags_all": pl.get("tags_all"),
        }
        out.append(row)

    if args.as_json:
        print(json.dumps(out, indent=2, ensure_ascii=False))
    else:
        for r in out:
            print(f"{r['id']:>10}  score={r['score']:.6f}  "
                  f"path={r['path'] if not args.exclude_path else '…'}  url={r['source_post_url']}")
            if r.get("tags_all"):
                # show first few tags for context
                tags = ", ".join((r["tags_all"] or [])[:10])
                print(f"    tags: {tags}")

if __name__ == "__main__":
    main()
