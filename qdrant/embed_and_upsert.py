# Chunked OpenCLIP → Qdrant embedder
import argparse, json, os, gc, sys
from typing import Optional, List, Tuple
from PIL import Image, ImageSequence, ImageFilter
import numpy as np
import torch
import open_clip
from tqdm import tqdm
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct

# -------------------- PIL helpers --------------------
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

def canny_edge_pil(pil: Image.Image) -> Image.Image:
    try:
        import cv2
        arr = np.array(pil)[:, :, ::-1]  # RGB->BGR
        gray = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 100, 200)
        edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
        return Image.fromarray(edges)
    except Exception:
        return pil.convert("L").filter(ImageFilter.FIND_EDGES).convert("RGB")

# -------------------- CLIP wrappers --------------------
def load_clip(model_name: str, pretrained: str, device: str):
    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name, pretrained=pretrained, device=device
    )
    tok = open_clip.get_tokenizer(model_name)
    model.eval()
    dim = model.text_projection.shape[1] if hasattr(model, "text_projection") else 512
    return model, preprocess, tok, dim

@torch.no_grad()
def embed_images(model, preprocess, pils: List[Image.Image], device: str) -> np.ndarray:
    # minibatch encode; returns [B, D] normalized np.float32
    xs = [preprocess(p).unsqueeze(0) for p in pils]
    x = torch.cat(xs, dim=0).to(device)
    with torch.cuda.amp.autocast(enabled=(device == "cuda")):
        v = model.encode_image(x)
        v = v / v.norm(dim=-1, keepdim=True)
    out = v.detach().cpu().numpy().astype(np.float32)
    del x, v
    return out

@torch.no_grad()
def embed_texts(model, tokenizer, texts: List[str], device: str) -> np.ndarray:
    toks = tokenizer(texts)
    if isinstance(toks, list):  # some tokenizer versions return list
        toks = torch.tensor(toks)
    t = toks.to(device)
    with torch.cuda.amp.autocast(enabled=(device == "cuda")):
        v = model.encode_text(t)
        v = v / v.norm(dim=-1, keepdim=True)
    out = v.detach().cpu().numpy().astype(np.float32)
    del t, v
    return out

def build_text_from_tags(tags: List[str], template: str) -> str:
    return template.replace("{tags}", ", ".join(tags or []))

# -------------------- manifest chunking --------------------
def iter_manifest_lines(path: str, start_line: int = 0, max_lines: Optional[int] = None,
                        shard_index: Optional[int] = None, shard_count: Optional[int] = None):
    """
    Yields (lineno, parsed_json) honoring start/limit and shard slicing.
    Sharding rule: include line if (lineno % shard_count) == shard_index.
    """
    ln = -1
    emitted = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            ln += 1
            if ln < start_line:
                continue
            if shard_index is not None and shard_count is not None:
                if (ln % shard_count) != shard_index:
                    continue
            try:
                rec = json.loads(line)
            except Exception:
                continue
            yield (ln, rec)
            emitted += 1
            if max_lines is not None and emitted >= max_lines:
                break

# -------------------- main --------------------
def main():
    ap = argparse.ArgumentParser(description="Chunked CLIP embeddings → Qdrant")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--qdrant-url", default="http://localhost:6333")
    ap.add_argument("--collection", required=True)
    ap.add_argument("--model", default="ViT-H-14")
    ap.add_argument("--pretrained", default="laion2b_s32b_b79k")
    ap.add_argument("--device", default=("cuda" if torch.cuda.is_available() else "cpu"))

    # Memory & performance knobs
    ap.add_argument("--upsert-batch", type=int, default=64, help="how many points per Qdrant upsert")
    ap.add_argument("--clip-batch", type=int, default=1, help="images per CLIP forward (set >1 for speed, uses more RAM/VRAM)")
    ap.add_argument("--gc-every", type=int, default=256, help="force gc after this many images")
    ap.add_argument("--empty-cache", action="store_true", help="also call torch.cuda.empty_cache() after gc")

    # Vectors
    ap.add_argument("--no-text", action="store_true", help="skip text vectors")
    ap.add_argument("--add-edge-vector", action="store_true", help="store edge vectors for sketch search")
    ap.add_argument("--text-template", default="an illustration with {tags}")

    # Chunking options
    ap.add_argument("--start-line", type=int, default=0, help="start from this line number (0-based)")
    ap.add_argument("--max-lines", type=int, default=None, help="process at most this many lines")
    ap.add_argument("--shard-index", type=int, default=None, help="this shard index (0-based)")
    ap.add_argument("--shard-count", type=int, default=None, help="total number of shards")

    # Misc
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--fail-log", default=None, help="write failed IDs to this file")
    args = ap.parse_args()

    # Basic sanity for sharding
    if (args.shard_index is None) ^ (args.shard_count is None):
        print("If using sharding, provide BOTH --shard-index and --shard-count.", file=sys.stderr)
        sys.exit(2)

    client = QdrantClient(url=args.qdrant_url)
    model, preprocess, tokenizer, dim = load_clip(args.model, args.pretrained, args.device)

    upsert_buffer: List[PointStruct] = []
    n_emb, n_up, n_fail = 0, 0, 0
    fail_ids = []

    def flush():
        nonlocal upsert_buffer, n_up
        if upsert_buffer and not args.dry_run:
            client.upsert(collection_name=args.collection, points=upsert_buffer)
            n_up += len(upsert_buffer)
            upsert_buffer.clear()

    def maybe_gc():
        if n_emb and (n_emb % args.gc_every == 0):
            gc.collect()
            if args.empty_cache and args.device == "cuda":
                torch.cuda.empty_cache()

    # Accumulators for CLIP minibatches
    img_ids, img_pils, img_payloads, texts_for_batch, want_edge = [], [], [], [], []

    def encode_and_queue():
        """Encode accumulated minibatch, build points, clear memory."""
        nonlocal upsert_buffer, img_ids, img_pils, img_payloads, texts_for_batch, want_edge, n_emb
        if not img_ids:
            return
        # images -> vectors
        try:
            img_vecs = embed_images(model, preprocess, img_pils, args.device)
        except Exception as e:
            # fallback to per-item (helps isolate bad images)
            img_vecs = []
            for pil in img_pils:
                try:
                    v = embed_images(model, preprocess, [pil], args.device)[0]
                except Exception:
                    v = None
                img_vecs.append(v)
            img_vecs = np.stack([v if v is not None else np.zeros((dim,), np.float32) for v in img_vecs], axis=0)

        # text -> vectors (optional)
        if not args.no_text and texts_for_batch:
            try:
                txt_vecs = embed_texts(model, tokenizer, texts_for_batch, args.device)
            except Exception:
                txt_vecs = [None] * len(texts_for_batch)
        else:
            txt_vecs = [None] * len(img_ids)

        # edge -> vectors (optional)
        if True:
            edge_pils = [canny_edge_pil(p) for p in img_pils]
            try:
                edge_vecs = embed_images(model, preprocess, edge_pils, args.device)
            except Exception:
                edge_vecs = [None] * len(img_ids)
        else:
            edge_vecs = [None] * len(img_ids)

        # build points
        for i, pid in enumerate(img_ids):
            vecs = {"image": img_vecs[i]}
            if txt_vecs[i] is not None:
                vecs["text"] = txt_vecs[i]
            if edge_vecs[i] is not None:
                vecs["edge"] = edge_vecs[i]
            upsert_buffer.append(PointStruct(id=int(pid), vector=vecs, payload=img_payloads[i]))

            if len(upsert_buffer) >= args.upsert_batch:
                flush()

        # clear minibatch memory
        for p in img_pils:
            try:
                p.close()
            except Exception:
                pass
        img_ids.clear(); img_pils.clear(); img_payloads.clear(); texts_for_batch.clear(); want_edge.clear()
        n_emb += len(vecs["image"]) if isinstance(vecs["image"], np.ndarray) else len(img_ids)  # progress basis
        maybe_gc()

    # Iterate over the chosen slice of the manifest
    for ln, rec in iter_manifest_lines(
        args.manifest,
        start_line=args.start_line,
        max_lines=args.max_lines,
        shard_index=args.shard_index,
        shard_count=args.shard_count,
    ):
        img_path = rec.get("local_path")
        if not img_path or not os.path.exists(img_path):
            n_fail += 1
            if args.fail_log: fail_ids.append(rec.get("id"))
            continue

        pil = load_pil_rgb(img_path)
        if pil is None:
            n_fail += 1
            if args.fail_log: fail_ids.append(rec.get("id"))
            continue

        payload = {
            "id": rec["id"],
            "path": img_path,
            "tags_all": rec.get("tags_all"),
            "matched_tags": rec.get("matched_tags") or rec.get("matched_queries"),
            "rating": rec.get("rating"),
            "score": rec.get("score"),
            "width": rec.get("width"),
            "height": rec.get("height"),
            "md5": rec.get("md5"),
            "source_post_url": rec.get("source_post_url"),
        }

        img_ids.append(rec["id"])
        img_pils.append(pil)
        img_payloads.append(payload)
        texts_for_batch.append(build_text_from_tags(rec.get("tags_all") or [], args.text_template))

        if len(img_ids) >= args.clip_batch:
            encode_and_queue()

    # tail
    encode_and_queue()
    flush()

    if args.fail_log and fail_ids:
        with open(args.fail_log, "w", encoding="utf-8") as f:
            for i in fail_ids:
                f.write(str(i) + "\n")

    print(f"Done. Upserted {n_up} points; failed {n_fail}. "
          f"(slice: start={args.start_line}, max={args.max_lines}, shard={args.shard_index}/{args.shard_count})")

if __name__ == "__main__":
    main()
