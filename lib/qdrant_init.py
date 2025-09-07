import argparse
import torch
import open_clip
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance

def infer_dim(model_name: str, pretrained: str, device: str) -> int:
    model, _, _ = open_clip.create_model_and_transforms(
        model_name, pretrained=pretrained, device=device
    )
    return int(getattr(model, "text_projection", None).shape[1] if hasattr(model, "text_projection") else 512)

def main():
    ap = argparse.ArgumentParser(description="Init Qdrant collection for CLIP vectors")
    ap.add_argument("--qdrant-url", default="http://localhost:6333")
    ap.add_argument("--collection", required=True)
    ap.add_argument("--model", default="ViT-H-14")
    ap.add_argument("--pretrained", default="laion2b_s32b_b79k")
    ap.add_argument("--device", default=("cuda" if torch.cuda.is_available() else "cpu"))
    ap.add_argument("--add-edge-vector", action="store_true", help="Also create an 'edge' vector for sketch search")
    ap.add_argument("--overwrite", action="store_true", help="Recreate even if exists")
    args = ap.parse_args()

    dim = infer_dim(args.model, args.pretrained, args.device)
    client = QdrantClient(url=args.qdrant_url)

    vectors = {
        "image": VectorParams(size=dim, distance=Distance.COSINE),
        "text":  VectorParams(size=dim, distance=Distance.COSINE),
    }
    if args.add_edge_vector:
        vectors["edge"] = VectorParams(size=dim, distance=Distance.COSINE)

    if args.overwrite:
        client.recreate_collection(collection_name=args.collection, vectors_config=vectors)
    else:
        existing = [c.name for c in client.get_collections().collections]
        if args.collection in existing:
            print(f"Collection '{args.collection}' exists; leaving unchanged.")
        else:
            client.recreate_collection(collection_name=args.collection, vectors_config=vectors)

    print(f"OK: '{args.collection}' with vectors: {', '.join(vectors.keys())}")

if __name__ == "__main__":
    main()