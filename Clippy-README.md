# Clippy — Style‑Aware Image Retrieval & (Optional) Editing

Clippy is a FastAPI service and script suite for **style‑aware visual retrieval** (hybrid of image/edge/text) backed by **Qdrant**. It can accept a sketch or text prompt, search a vector DB, and return relevant references. Optional edit/generation can be wired in via your preferred image model provider.

> These docs are generic and meant to drop into your repo as-is. Update paths/names if your tree differs.

---

## Quick start

### Requirements
- Python 3.10+
- Qdrant (Docker recommended)
- Optional: CUDA‑capable GPU for faster embedding
- A dataset (e.g., your Danbooru/Safebooru subset) and optional `tags.txt`

### 1) Run Qdrant
```bash
docker run -p 6333:6333 -p 6334:6334 -v $PWD/qdrant_storage:/qdrant/storage qdrant/qdrant:latest
# UI: http://localhost:6333/dashboard
```

### 2) Create venv & install
```bash
python -m venv .venv
source .venv/bin/activate   # .\.venv\Scripts\activate on Windows
pip install -U pip wheel
pip install -e .
```

### 3) Configure environment
Create `.env` in the repo root:
```env
# Qdrant
QDRANT_URL=http://localhost:6333
QDRANT_API_KEY=
QDRANT_COLLECTION=safebooru_union_clip

# Embeddings
MODEL_NAME=ViT-bigG-14
PRETRAINED=laion2b_s39b_b160k
DEVICE=cuda            # or cpu
PRECISION=fp32

# Ingestion
IMAGES_ROOT=/abs/path/to/images

# (Optional) Image editing/generation provider
OPENAI_API_KEY=
```

### 4) Create collection & ingest
See **Qdrant Scripts** doc for full details. Quick run:
```bash
python scripts/create_collection.py --url $QDRANT_URL --collection $QDRANT_COLLECTION --dim 1280

python scripts/ingest.py   --images-root "$IMAGES_ROOT" --collection $QDRANT_COLLECTION   --model "$MODEL_NAME" --pretrained "$PRETRAINED"   --device "$DEVICE" --precision "$PRECISION" --batch 128
```

### 5) Run the API
```bash
uvicorn main:app --reload --port 8000
```
Main search endpoint (no `/api` prefix):
- `POST /search/hybrid`

Open `test_main.http` for ready-made calls (VS Code REST Client), or see the **Retrieval API** doc.

---

## Concepts

### Hybrid retrieval
Clippy queries Qdrant with a **weighted mix** of embeddings:
- **image** (CLIP image embed)
- **edge** (edge-map embed for sketch‑like shape)
- **text** (CLIP text embed)

Weights `wImg`, `wEdge`, `wTxt` ∈ [0,1]; normalized per request.

### Sketch‑to‑reference
Send a sketch PNG + optional text; tune weights; receive paths/ids/scores of similar references.

### Generation/edit (optional)
If configured, you can pass retrieved refs + sketch to a model provider. Retrieval works standalone if none is configured.

---

## Project layout (typical)
```
app/           # FastAPI routes, request/response models
scripts/       # Qdrant collection + ingestion/backfill utilities
tests/         # CLI debuggers and/or pytest tests
qdrant/        # schema helpers or notes (optional)
main.py        # FastAPI entrypoint
pyproject.toml # deps and metadata
tags.txt       # example tag list
test_main.http # REST Client examples
```

---

## Troubleshooting

- **404 at `/api/search/hybrid`** → Use `/search/hybrid` (no `/api` prefix).
- **All scores 0** → That vector space might be missing in the collection. Re‑ingest or run backfill scripts.
- **“vector” vs “vectors” mismatch** → Single‑vector collections expect key `vector`; named‑vectors expect `vectors` (dict). Keep schema and payload consistent.
- **CUDA OOM** → Lower `--batch`, switch to CPU, or use a smaller model.

---

_Last updated: 2025-09-13_
