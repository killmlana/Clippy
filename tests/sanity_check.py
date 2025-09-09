from qdrant_client import QdrantClient
import numpy as np

client = QdrantClient("http://localhost:6333")
coll = "safebooru_union_clip"

zero = nonzero = 0
scroll, next_page = client.scroll(collection_name=coll, with_vectors=["text"], limit=1024)
while scroll:
    for p in scroll:
        v = (getattr(p, "vector", {}) or {}).get("text")
        if not v:            # None or []
            zero += 1
        else:
            n = np.linalg.norm(v)
            if n < 1e-6:     # treat as zero
                zero += 1
            else:
                nonzero += 1
    if not next_page: break
    scroll, next_page = client.scroll(collection_name=coll, with_vectors=["text"], limit=1024, offset=next_page)
print("text vectors -> nonzero:", nonzero, "zero/empty:", zero)
