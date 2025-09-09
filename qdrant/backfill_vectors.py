import argparse, json, os, gc, sys
from typing import Optional, List, Set, Dict
import numpy as np
from PIL import Image, ImageSequence, ImageFilter
import torch, open_clip
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
            bg.paste(im, mask=im.split()[-1]); im.close()
            return bg
        return im.convert("RGB")
    except Exception:
        return None

def canny_edge_pil(pil: Image.Image) -> Image.Image:
    # Prefer OpenCV Canny; fallback to PIL edges
    try:
        import cv2
        arr = np.array(pil)[:, :, ::-1]  # RGB->BGR
        gray = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 100, 200)
        edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
        return Image.fromarray(edges)
    except Exception:
        return pil.convert("L").filter(ImageFilter.FIND_EDGES).convert("RGB")

# -------------------- Qdrant helpers --------------------
def point_named_vectors(point) -> Dict[str, List[float]]:
    """
    Robustly extract named vectors from a Retrieved/Scored point across client versions.
    Prefers `point.vector` (as you noted), falls back to `point.vectors`.
    Returns dict-like {name: vector} or {} if none.
    """
    v = getattr(point, "vector", None)
    if v is None:
        v = getattr(point, "vector", None)
    if v is None:
        return {}
    # Already a dict?
    if isinstance(v, dict):
        return v
    # Some pydantic objects expose .vectors (again), .to_dict(), or .dict()
    inner = getattr(v, "vector", None)
    if isinstance(inner, dict):
        return inner
    if hasattr(v, "to_dict"):
        try: return v.to_dict()
        except Exception: pass
    if hasattr(v, "dict"):
        try: return v.dict()
        except Exception: pass
    # Unnamed vector — return under empty name (not useful for named spaces)
    try:
        if isinstance(v, (list, tuple, np.ndarray)):
            return {"": list(v)}
    except Exception:
        pass
    return {}

def iter_zero_ids(client: QdrantClient, collection: str, vector_name: str, page: int = 2048) -> Set[int]:
    """Find IDs with empty/zero vectors for `vector_name` by scrolling."""
    zeros: Set[int] = set()
    pts, off = client.scroll(collection_name=collection, with_vectors=[vector_name], with_payload=False, limit=page)
    while pts:
        for p in pts:
            vecs = point_named_vectors(p)
            v = vecs.get(vector_name)
            if (not v) or (np.linalg.norm(v) < 1e-6):
                zeros.add(int(p.id))
        if not off: break
        pts, off = client.scroll(collection_name=collection, with_vectors=[vector_name], with_payload=False, limit=page, offset=off)
    return zeros

def read_manifest_map(path: str) -> dict:
    m = {}
    with open(path, "r", encoding="utf-8") as f:
        for ln in f:
            try:
                rec = json.loads(ln)
                m[int(rec["id"])] = rec
            except Exception:
                pass
    return m

# -------------------- OpenCLIP --------------------
def load_clip(model_name: str, pretrained: str, device: str):
    model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained, device=device)
    model.eval()
    return model, preprocess

@torch.no_grad()
def embed_batch_images(model, preprocess, pils: List[Image.Image], device: str) -> np.ndarray:
    xs = [preprocess(p).unsqueeze(0) for p in pils]
    x = torch.cat(xs, dim=0)
    if device == "cuda":
        x = x.to(device, non_blocking=True)
    v = model.encode_image(x)
    v = v / v.norm(dim=-1, keepdim=True)
    out = v.detach().to("cpu").numpy().astype(np.float32)
    del x, v
    if device == "cuda":
        torch.cuda.empty_cache()
    return out

# -------------------- Backfill core --------------------
def backfill_one_space(
    kind: str,                       # "image" or "edge"
    model, preprocess,
    client: QdrantClient,
    collection: str,
    id2rec: dict,
    targets: Set[int],
    device: str,
    clip_batch: int,
    upsert_batch: int,
    gc_every: int,
    fail_log_path: str,
):
    assert kind in ("image", "edge")
    buf: List[PointStruct] = []
    fail_ids: List[int] = []
    done = fail = 0

    def flush():
        nonlocal buf
        if buf:
            client.upsert(collection_name=collection, points=buf)
            buf.clear()

    targets_list = list(targets)
    for i in range(0, len(targets_list), clip_batch):
        batch_ids = targets_list[i:i+clip_batch]
        pils, recs = [], []
        for pid in batch_ids:
            rec = id2rec.get(pid)
            if not rec:
                fail += 1; fail_ids.append(pid); continue
            pth = rec.get("local_path") or rec.get("path")
            if not pth or not os.path.exists(pth):
                fail += 1; fail_ids.append(pid); continue
            pil = load_pil_rgb(pth)
            if pil is None:
                fail += 1; fail_ids.append(pid); continue
            if kind == "edge":
                pil_edge = canny_edge_pil(pil)
                pils.append(pil_edge)
                try: pil.close()
                except Exception: pass
            else:
                pils.append(pil)
            recs.append(rec)

        if not pils:
            continue

        # Batch embed; if it fails, fall back per-item (and skip bad ones)
        try:
            vecs = embed_batch_images(model, preprocess, pils, device)
        except Exception:
            vecs = []
            for pil in pils:
                try:
                    v = embed_batch_images(model, preprocess, [pil], device)[0]
                except Exception:
                    v = None
                vecs.append(v)

        # Queue non-zero upserts
        for j, rec in enumerate(recs):
            pid = int(rec["id"])
            v = vecs[j]
            if v is None or float(np.linalg.norm(v)) < 1e-6:
                fail += 1; fail_ids.append(pid); continue
            buf.append(PointStruct(
                id=pid,
                vector={kind: v},                  # <-- IMPORTANT: use 'vector' (singular)
                payload={
                    "id": pid,
                    "path": rec.get("local_path") or rec.get("path"),
                    "tags_all": rec.get("tags_all"),
                    "source_post_url": rec.get("source_post_url"),
                }
            ))
            done += 1
            if len(buf) >= upsert_batch:
                flush()

        # free PILs
        for p in pils:
            try: p.close()
            except Exception: pass

        if (done + fail) % gc_every == 0:
            gc.collect()
            if device == "cuda":
                torch.cuda.empty_cache()

    flush()
    if fail_ids:
        with open(fail_log_path, "w", encoding="utf-8") as f:
            for pid in fail_ids:
                f.write(str(pid) + "\n")

    print(f"[{kind}] backfill complete: updated {done}, failed {fail}. "
          f"Fail list: {fail_log_path if fail_ids else '(none)'}")

# -------------------- CLI --------------------
def main():
    ap = argparse.ArgumentParser(description="Backfill non-zero CLIP vectors (image/edge) into Qdrant")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--qdrant-url", default="http://localhost:6333")
    ap.add_argument("--collection", required=True)

    # model/device
    ap.add_argument("--model", default="ViT-bigG-14")
    ap.add_argument("--pretrained", default="laion2b_s39b_b160k")
    ap.add_argument("--device", default=("cuda" if torch.cuda.is_available() else "cpu"), choices=["cuda","cpu"])

    # performance
    ap.add_argument("--clip-batch", type=int, default=4)
    ap.add_argument("--upsert-batch", type=int, default=64)
    ap.add_argument("--gc-every", type=int, default=512)

    # selection
    ap.add_argument("--which", choices=["image","edge","both"], default="both")
    ap.add_argument("--force-all", action="store_true", help="Re-embed ALL manifest IDs for the chosen spaces")
    ap.add_argument("--ids-file", default=None, help="Only backfill listed IDs (one per line)")

    # logs
    ap.add_argument("--fail-log-prefix", default="backfill_fail")

    args = ap.parse_args()

    client = QdrantClient(url=args.qdrant_url)
    id2rec = read_manifest_map(args.manifest)
    if not id2rec:
        print("Manifest is empty or unreadable.", file=sys.stderr); sys.exit(2)

    kinds = []
    if args.which in ("image", "both"): kinds.append("image")
    if args.which in ("edge", "both"):  kinds.append("edge")

    # Load model once for both spaces
    model, preprocess = load_clip(args.model, args.pretrained, args.device)

    # Resolve target IDs:
    #  - if ids-file: use it
    #  - elif force-all: all manifest IDs
    #  - else: detect zeros per kind (separately)
    ids_from_file: Optional[Set[int]] = None
    if args.ids_file and os.path.exists(args.ids_file):
        ids_from_file = set()
        with open(args.ids_file, "r") as f:
            for s in f:
                s = s.strip()
                if s: ids_from_file.add(int(s))

    for kind in kinds:
        if ids_from_file is not None:
            targets = set(ids_from_file)
        elif args.force_all:
            targets = set(id2rec.keys())
        else:
            print(f"Scanning Qdrant for zero/empty '{kind}' vectors...")
            targets = iter_zero_ids(client, args.collection, kind)
            if not targets:
                print(f"[{kind}] Nothing to backfill (no zero/empty vectors found).")
                continue

        print(f"[{kind}] Backfilling {len(targets)} items using {args.model}/{args.pretrained} on {args.device}")
        backfill_one_space(
            kind=kind,
            model=model, preprocess=preprocess,
            client=client, collection=args.collection,
            id2rec=id2rec, targets=targets,
            device=args.device,
            clip_batch=args.clip_batch,
            upsert_batch=args.upsert_batch,
            gc_every=args.gc_every,
            fail_log_path=f"{args.fail_log_prefix}_{kind}.txt",
        )

if __name__ == "__main__":
    main()
