# Clippy — Style-Aware Image Retrieval & Generation

Clippy is a FastAPI service and script suite for **style-aware visual retrieval** backed by **Qdrant**. It accepts a sketch and/or text prompt, runs **hybrid similarity search** over image/edge/text embeddings, and returns reference images. Generative Image models typically perform much better when references are provided. Especially during style-transfer workflows (transferring one artstyle to the other), references help the model to understand better the differences. 
The project aims to refine the generative workflow of image generation models by making it easier for artists to retrieve references across various artstyles and use them further to generate better images without having any knowledge of prompt engineering.


---

## 🧰 Tech Stack

| Layer | Primary | Notes / Optional |
|---|---|---|
| Language & Runtime | **Python 3.10+** | venv/uv recommended |
| API Framework | **FastAPI**, **Uvicorn** | Pydantic for schema validation |
| Embeddings | **PyTorch**, **open-clip-torch** | `Pillow`, `opencv-python` for I/O & edge-maps |
| Vector Database | **Qdrant Server**, **qdrant-client** | HNSW index; supports named vectors |
| Storage | **Filesystem** under `IMAGES_ROOT` | Image paths stored in Qdrant payloads |
| Config | **python-dotenv** | `.env` in repo root |
| Observability/Utils | `tqdm` (progress), logging via Uvicorn/stdlib |
| Testing & Tools | **pytest**, **httpie/curl**, VS Code **REST Client** |
| **Frontend Client** | **Vite + React + TypeScript** | Tailwind CSS, **shadcn/ui**, **lucide-react**, **Fabric.js** for sketch UI;
| Edit/Gen | Vertex AI / SDXL workflow

---

## ⛳ Quick Start

### Requirements
- Python **3.10+**
- **Qdrant** (Docker recommended)
- Optional GPU for faster embedding (OpenCLIP)
- A dataset (e.g., your Danbooru/Safebooru subset) and `tags.txt`

### 1) Run Qdrant
```bash
docker run -p 6333:6333 -p 6334:6334 \
  -v $PWD/qdrant_storage:/qdrant/storage \
  qdrant/qdrant:latest
# UI: http://localhost:6333/dashboard
````

### 2) Create venv & install

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .\.venv\Scripts\activate
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

# Image Gen
GOOGLE_GENAI_USE_VERTEXAI=true
GOOGLE_CLOUD_PROJECT=
GOOGLE_CLOUD_LOCATION=
GOOGLE_APPLICATION_CREDENTIALS= #json key for service account
PROMPT_LOG_LEVEL=DEBUG
GEMINI_VISION_MODEL=gemini-2.5-flash
```

### 4) Create collection & ingest

```bash
# Retrieval Script retrieve_safebooru.py
python scripts/retrieve_safebooru.py \
  --tags-file ./tags.txt \
  --out ./data/safebooru \
  --union --workers 16 --max-pages 20 --resume


# Ingesting Dataset
python qdrant/embed_and_upsert.py \
  --manifest "$MANIFEST" \
  --qdrant-url "$QDRANT_URL" \
  --collection "$QDRANT_COLLECTION" \
  --model "$MODEL_NAME" --pretrained "$PRETRAINED" \
  --device "$DEVICE" \
  --clip-batch 4 --upsert-batch 64 --gc-every 512 \
  --text-template "an illustration with {tags}"

# Backfill missing vectors 
python scripts/backfill_vectors.py \
  --manifest "$MANIFEST" \
  --qdrant-url "$QDRANT_URL" \
  --collection "$QDRANT_COLLECTION" \
  --model "$MODEL_NAME" --pretrained "$PRETRAINED" \
  --device "$DEVICE" \
  --which both --clip-batch 4 --upsert-batch 64 --gc-every 512
```

### 5) Run the API

```bash
uvicorn main:app --reload --port 8000
```

---

## 🧭 How the Architecture Works

### Components

* **Client (Vite + React + TS + Fabric.js)**  
  Lets artists sketch, tune weights (`wImg`, `wEdge`, `wTxt`), and preview results.

* **FastAPI App (`main.py`, `app/`)**
  - Receives multipart or JSON requests.
  - Preprocesses inputs (e.g., edge-map from sketch).
  - Computes embeddings via OpenCLIP.
  - Assembles a **hybrid query** with normalized weights.
  - Calls **Qdrant** (ANN search) and optionally exact re-rank.
  - Resolves `path` payloads to serve preview images (e.g., `GET /image/{id}`).

* **Embedding Workers (OpenCLIP)**
  - **Image** embeddings from the original images.
  - **Edge** embeddings from edge-maps (sketch/shape signal).
  - **Text** embeddings from prompts/tags/captions.

* **Qdrant (Vector DB)**
  - Stores vectors + payloads (`path`, `tags`, etc.).
  - Prefer **named vectors** (`image|edge|text`) for flexible query-time weighting.
  - HNSW index for fast approximate search; `ef_search` tunes quality/speed.

* **Image Store (filesystem path rooted at `IMAGES_ROOT`)**
  - Original assets referenced by payload `path`.
  - API enforces safe path resolution (no escaping root).

* **Edit/Generation Provider**  
  Pluggable adapter that powers `/images/generate` (see `generate_image.py`). Supports **style references** and **subject references**:

  #### How to send references (provider-agnostic request shape)
  The API accepts a `refs` array; each ref declares a **role** and a **weight**:
  ```json
  {
    "prompt": "pop art still life, magenta outline",
    "count": 1,
    "refs": [
      { "role": "style",   "id": 12345, "weight": 0.7 },   // style reference from retrieval
      { "role": "subject", "url": "https://.../apple.png", "weight": 0.6 }
    ],
    "sketch_weight": 0.0
  }

Providers map these roles as follows:

* **Google Imagen 3 (Vertex AI)** – Use **style customization** with one or more **reference images** to steer look & feel; mask-based editing is also supported. The adapter converts `refs.role=="style"` into Imagen’s style reference inputs; `subject` refs can be fed as additional reference images or via mask+edit flows depending on the task. ([Google Cloud][1])
* **SDXL pipelines** – Use **image-to-image** for coarse subject retention and add **IP-Adapter** (style / face / composition variants) for stronger **style** and **identity** adherence. This approach improves controllability and preserves style/subjects more reliably than prompt-only generation. ([Hugging Face][2])

**Why references help:** Research on **IP-Adapter** shows that adding image prompts (style or subject) to diffusion models yields comparable or better results than fine-tuning for many tasks, and crucially **improves style/identity control** without changing the base model. In practice with SDXL, combining image-to-image + IP-Adapter often outperforms text-only prompts on style fidelity. ([arXiv][3])

#### Recommended patterns

* **Style-first:** 1–3 style refs, `style_weight ≈ 0.5–0.8`, moderate CFG/guidance; keep `subject_weight` low or 0.
* **Subject-first:** 1–2 subject refs (clean, centered), `subject_weight ≈ 0.6–0.9`; optional low `style_weight` for finish.
* **SDXL img2img:** start with a subject ref as the initial image, **low denoise/strength** (e.g., 0.2–0.35) to retain identity; layer IP-Adapter style refs for look. ([Hugging Face][2])
* **Imagen editing:** when you need to preserve a subject tightly, use **mask-based edit** with a subject ref and a mask to constrain changes. ([Google Cloud][4])

[1]: https://cloud.google.com/vertex-ai/generative-ai/docs/image/style-customization?utm_source=chatgpt.com "Style customization | Generative AI on Vertex AI"
[2]: https://huggingface.co/docs/diffusers/en/using-diffusers/img2img?utm_source=chatgpt.com "Image-to-image"
[3]: https://arxiv.org/abs/2308.06721?utm_source=chatgpt.com "IP-Adapter: Text Compatible Image Prompt Adapter for Text-to-Image Diffusion Models"
[4]: https://cloud.google.com/vertex-ai/generative-ai/docs/image/overview?utm_source=chatgpt.com "Imagen on Vertex AI | AI Image Generator"

### 🔎 Retrieval Results

I tried abstract sketches with very satisfactory results.

#### Input:

![retrieval-input](https://github.com/killmlana/Clippy/blob/master/assets/retrieval-input.png)

#### Output:

![retrieval-output](https://github.com/killmlana/Clippy/blob/master/assets/retrieval-output.png)

#### Nearest Output:

![nearest-retrieval-output](https://github.com/killmlana/Clippy/blob/master/assets/retrival-output-nearest.webp)

### Style transfer (from sketch)

Results while using style-transfer with Google Vertex Ai Imagen 3 model

#### Input:

![sketch-input](https://github.com/killmlana/Clippy/blob/master/assets/sketch-generation-input.png)

#### Output using pixel art style:

![sketch-generation-output](https://github.com/killmlana/Clippy/blob/master/assets/sketch-generation-output.png)

### Data Flow

**Ingestion (offline):**

1. **Scan dataset** → build file list + optional tags.
2. **Compute embeddings** (`image`, `edge`, `text`).
3. **Upsert to Qdrant** with payloads.
4. **Index/Optimize** HNSW for recall.

**Query (online):**

1. Client sends **multipart** (sketch/image) and/or **JSON** (`queryText`, weights, filters).
2. API **preprocesses** → **embeds** → **weights** → **hybrid query**.
3. Qdrant returns top-K; API optionally exact re-ranks; responses include `id`, `score`, `path`, `payload`.
4. Client previews images; edit/generation step can run thereafter.

---

## 🎨 Client (Vite + React + TypeScript)

### Why this stack?

* **Vite**: fast HMR and lean builds.
* **React + TS**: type-safety and a mature ecosystem.
* **Fabric.js**: rich HTML5 canvas sketching (brush, layers, undo).
* **Tailwind + shadcn/ui + lucide-react**: rapid, modern UI.

## 🛠️ API (brief)

### `POST /search/hybrid`

Hybrid similarity using image/edge/text with per-request weights.

### `GET /image/{image_id}`

Serves the image resolved from payload `path`, with safe path checks.

### `POST /images/generate`

Uses the provided references to generate images.

---

*Last updated: 2025-09-14*
