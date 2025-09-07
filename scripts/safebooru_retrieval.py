# -*- coding: utf-8 -*-
"""
Safebooru retrieval only:
- enumerate IDs by tag (robust DAPI paging)
- cap IDs to HF mirror ceiling
- download with CheeseChaser (WebP mirror)
- write ids_per_tag.json, manifest.jsonl, ATTRIBUTION.md
No Qdrant, no embeddings, no projector training.
"""

import argparse, os, sys, time, json, random, glob
from typing import Dict, List, Optional
from urllib.parse import quote_plus
import logging

import requests
from tqdm import tqdm
from cheesechaser.datapool import SafebooruWebpDataPool

# ------------------------ Config / Attribution ------------------------

SAFEBOORU_API = "https://safebooru.org/index.php"

# Mirror info (override on CLI if you want)
MIRROR_DATASET = "deepghs/safebooru-webp-4Mpixel"
MIRROR_LICENSE = "other"
MIRROR_UPDATED_AT = "see dataset card"
DATA_SOURCE = "safebooru.org"
POST_URL_FMT = "https://safebooru.org/index.php?page=post&s=view&id={id}"

DEFAULT_MAX_MIRROR_ID = 5974383  # adjust if mirror updates

# ------------------------ Logging ------------------------

logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
log = logging.getLogger("safebooru_retrieve")

# ------------------------ Robust HTTP ------------------------

from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

def _make_session() -> requests.Session:
    s = requests.Session()
    s.headers.update({
        "User-Agent": ("Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                       "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"),
        "Accept": "application/json, text/plain, */*",
        "Referer": "https://safebooru.org/",
    })
    retry = Retry(
        total=5, connect=5, read=5,
        backoff_factor=1.2,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset(["GET"]),
        raise_on_status=False,
    )
    s.mount("https://", HTTPAdapter(max_retries=retry))
    s.mount("http://", HTTPAdapter(max_retries=retry))
    return s

def _get_json_with_backoff(sess: requests.Session, params: dict, *, max_tries=5):
    delay = 0.8
    for attempt in range(1, max_tries + 1):
        r = sess.get(SAFEBOORU_API, params=params, timeout=30)
        ct = (r.headers.get("Content-Type") or "").lower()
        body = (r.text or "").strip()

        if body.startswith("[") or body.startswith("{") or "application/json" in ct:
            try:
                return r.json()
            except requests.exceptions.JSONDecodeError:
                pass

        # empty body on 200 → no results
        if r.status_code == 200 and not body:
            return []

        # Throttle/HTML page → retry with backoff
        if "<html" in body[:200].lower() or "cloudflare" in body.lower():
            if attempt == max_tries:
                log.warning("HTML throttle after %d tries; params=%s\n%s",
                            attempt, params, body[:200])
                return []
        else:
            if attempt == max_tries:
                log.warning("Non-JSON after %d tries; params=%s CT=%s\n%s",
                            attempt, params, ct, body[:200])
                return []

        time.sleep(delay + random.uniform(0, 0.4))
        delay *= 1.6
    return []

# ------------------------ Helpers ------------------------

def read_tags_file(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        tags = [ln.strip() for ln in f if ln.strip() and not ln.strip().startswith("#")]
    return [t.replace(" ", "_") for t in tags]

def write_attribution(out_dir: str, tags: List[str], max_mirror_id: int,
                      mirror_updated_at: str = MIRROR_UPDATED_AT):
    path = os.path.join(out_dir, "ATTRIBUTION.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write(
f"""# Attribution

**Source**: Safebooru (https://safebooru.org)  
**Mirror (downloaded from)**: Hugging Face – {MIRROR_DATASET}  
**Mirror license**: {MIRROR_LICENSE}  
**Mirror last updated**: {mirror_updated_at}  
**Mirror ID ceiling used**: ≤ {max_mirror_id}

**Retrieval tool**: CheeseChaser (`SafebooruWebpDataPool`)  
**Requested tags**: {", ".join(tags)}

> Images remain the property of their respective artists/uploader(s).  
> Respect the source site terms/policies and the mirror’s license notes.  
> This dataset is provided “as-is” for research/analysis; redistribution may be restricted.
"""
        )
    log.info("Wrote attribution: %s", path)

def resolve_local_path(dst_dir: str, post_id: int) -> Optional[str]:
    meta = os.path.join(dst_dir, f"{post_id}_metainfo.json")
    if os.path.exists(meta):
        try:
            j = json.load(open(meta, "r", encoding="utf-8"))
            for k in ("saved_path", "output_path", "dst", "filename", "path"):
                v = j.get(k)
                if v:
                    p = v if os.path.isabs(v) else os.path.join(dst_dir, v)
                    if os.path.exists(p) and not p.endswith(".json"):
                        return p
        except Exception:
            pass
    hits = glob.glob(os.path.join(dst_dir, "**", f"{post_id}.*"), recursive=True)
    if hits:
        hits.sort(key=lambda p: (p.count(os.sep), len(p)))
        return hits[0]
    return None

def scan_existing_ids(directories: List[str]) -> set:
    """Find already-downloaded IDs (by metainfo or filename pattern) for --resume."""
    found = set()
    for d in directories:
        for p in glob.glob(os.path.join(d, "**", "*_metainfo.json"), recursive=True):
            try:
                pid = int(os.path.basename(p).split("_", 1)[0])
                found.add(pid)
            except Exception:
                pass
        for p in glob.glob(os.path.join(d, "**", "*.*"), recursive=True):
            base = os.path.basename(p)
            if base.endswith(".json"):  # skip metainfo/others
                continue
            try:
                pid = int(base.split(".", 1)[0])
                found.add(pid)
            except Exception:
                pass
    return found

# ------------------------ Enumeration ------------------------

def fetch_all_posts_for_tag(
    tag: str,
    *,
    limit_per_page: int = 100,
    max_pages: int = 20,
    sleep: float = 0.25,
    max_mirror_id: int = DEFAULT_MAX_MIRROR_ID,
    older_than_id: Optional[int] = None,
) -> Dict[int, dict]:
    """Return {id -> minimal_meta} for a single tag, constrained to mirror range."""
    assert 1 <= limit_per_page <= 1000, "Use <=100; larger often triggers throttle."
    out: Dict[int, dict] = {}
    sess = _make_session()
    pid = 0
    with tqdm(desc=f"[IDs] {tag}", unit="page") as pbar:
        while True:
            if max_pages is not None and pid >= max_pages:
                break
            tag_query = f"{tag} id:<={max_mirror_id}"
            if older_than_id is not None:
                tag_query += f" id:<{older_than_id}"
            params = {
                "page": "dapi", "s": "post", "q": "index", "json": 1,
                "limit": limit_per_page, "pid": pid, "tags": tag_query,
            }
            data = _get_json_with_backoff(sess, params)
            if not data:
                break
            got = 0
            for post in data:
                if "id" not in post:
                    continue
                try:
                    pid_int = int(post["id"])
                except Exception:
                    continue
                if pid_int not in out:
                    out[pid_int] = {
                        "id": pid_int,
                        "tags_all": post.get("tags") or post.get("tag_string") or "",
                        "rating": post.get("rating"),
                        "score": post.get("score"),
                        "width": post.get("width"),
                        "height": post.get("height"),
                        "md5": post.get("md5"),
                        "ext": post.get("file_ext") or post.get("ext"),
                        "_raw": post,
                    }
                    got += 1
            pbar.update(1)
            pid += 1
            if got < limit_per_page:
                break
            time.sleep(sleep)
    return out

# ------------------------ Retrieval (download + manifest) ------------------------

def main():
    ap = argparse.ArgumentParser(description="Safebooru retrieval (CheeseChaser + manifest)")
    ap.add_argument("--tags-file", required=True, help="newline-separated tags")
    ap.add_argument("--out", required=True, help="output directory")
    ap.add_argument("--union", action="store_true", help="download union into OUT/ALL")
    ap.add_argument("--workers", type=int, default=16, help="CheeseChaser concurrency")
    ap.add_argument("--per-page", type=int, default=100, help="DAPI page size (<=100 recommended)")
    ap.add_argument("--max-pages", type=int, default=20, help="max pages per tag")
    ap.add_argument("--older-than-id", type=int, default=None, help="force older posts (adds id:<N)")
    ap.add_argument("--max-mirror-id", type=int, default=DEFAULT_MAX_MIRROR_ID, help="HF mirror ID ceiling")
    ap.add_argument("--ids-dump", default="ids_per_tag.json", help="filename to write tag->IDs map")
    ap.add_argument("--manifest", default="manifest.jsonl", help="filename to write manifest")
    ap.add_argument("--resume", action="store_true", help="skip IDs already downloaded in output dirs")
    ap.add_argument("--mirror-updated-at", default=MIRROR_UPDATED_AT,
                    help="free-form text for ATTRIBUTION.md (e.g., card timestamp)")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    tags = read_tags_file(args.tags_file)
    if not tags:
        log.error("No tags in file."); sys.exit(1)

    # Attribution
    write_attribution(args.out, tags, args.max_mirror_id, args.mirror_updated_at)

    # 1) Enumerate
    ids_per_tag: Dict[str, List[int]] = {}
    union_meta: Dict[int, dict] = {}

    for tag in tags:
        posts = fetch_all_posts_for_tag(
            tag,
            limit_per_page=args.per_page,
            max_pages=args.max_pages,
            max_mirror_id=args.max_mirror_id,
            older_than_id=args.older_than_id,
        )
        ids = sorted(posts.keys())
        ids_per_tag[tag] = ids
        for pid, meta in posts.items():
            if pid not in union_meta:
                union_meta[pid] = meta
            mt = set(union_meta[pid].get("matched_tags", []))
            mt.add(tag)
            union_meta[pid]["matched_tags"] = sorted(mt)
        log.info("Tag '%s': %d posts", tag, len(ids))

    # Save id mapping
    ids_dump_path = os.path.join(args.out, args.ids_dump)
    with open(ids_dump_path, "w", encoding="utf-8") as f:
        json.dump(ids_per_tag, f, indent=2)
    log.info("Wrote ID mapping: %s", ids_dump_path)

    # 2) Compute union and filter to mirror ceiling
    union_ids_all = sorted({i for v in ids_per_tag.values() for i in v})
    union_ids = [i for i in union_ids_all if i <= args.max_mirror_id]
    log.info("Enumerated IDs: %d; <= mirror max (%d): %d; > mirror: %d skipped",
             len(union_ids_all), args.max_mirror_id, len(union_ids),
             len(union_ids_all) - len(union_ids))

    # 3) Download
    pool = SafebooruWebpDataPool()

    if args.union:
        download_dirs = [os.path.join(args.out, "ALL")]
        os.makedirs(download_dirs[0], exist_ok=True)
        ids_for_dl = union_ids
        if args.resume:
            existing = scan_existing_ids(download_dirs)
            before = len(ids_for_dl)
            ids_for_dl = [i for i in ids_for_dl if i not in existing]
            log.info("Resume: %d/%d to download (skipped %d already present)",
                     len(ids_for_dl), before, before - len(ids_for_dl))
        log.info("Downloading UNION: %d → %s", len(ids_for_dl), download_dirs[0])
        if ids_for_dl:
            pool.batch_download_to_directory(
                resource_ids=ids_for_dl,
                dst_dir=download_dirs[0],
                max_workers=args.workers,
                save_metainfo=True,
                metainfo_fmt="{resource_id}_metainfo.json",
            )
    else:
        download_dirs = []
        for tag, ids in ids_per_tag.items():
            if not ids:
                log.info("Skipping '%s' (no results)", tag)
                continue
            dst = os.path.join(args.out, tag)
            os.makedirs(dst, exist_ok=True)
            ids_for_dl = [i for i in ids if i <= args.max_mirror_id]
            if args.resume:
                existing = scan_existing_ids([dst])
                before = len(ids_for_dl)
                ids_for_dl = [i for i in ids_for_dl if i not in existing]
                log.info("Resume '%s': %d/%d to download (skipped %d)", tag,
                         len(ids_for_dl), before, before - len(ids_for_dl))
            download_dirs.append(dst)
            log.info("Downloading %d for '%s' → %s", len(ids_for_dl), tag, dst)
            if ids_for_dl:
                pool.batch_download_to_directory(
                    resource_ids=ids_for_dl,
                    dst_dir=dst,
                    max_workers=args.workers,
                    save_metainfo=True,
                    metainfo_fmt="{resource_id}_metainfo.json",
                )

    # 4) Manifest (only for files we actually have locally)
    manifest_path = os.path.join(args.out, args.manifest)
    written = 0
    with open(manifest_path, "w", encoding="utf-8") as man:
        for pid in union_ids:
            rec = union_meta.get(pid)
            if rec is None:
                continue
            local_path = None
            for d in download_dirs:
                local_path = resolve_local_path(d, pid)
                if local_path: break
            if not local_path:
                continue  # keep manifest focused on available files
            tags_list = [t for t in (rec.get("tags_all") or "").split() if t]
            row = {
                "id": pid,
                "local_path": local_path,
                "tags_all": tags_list,
                "matched_tags": rec.get("matched_tags", []),
                "rating": rec.get("rating"),
                "score": rec.get("score"),
                "width": rec.get("width"),
                "height": rec.get("height"),
                "md5": rec.get("md5"),
                "source_ext": rec.get("ext"),
                # attribution
                "source_site": DATA_SOURCE,
                "source_post_url": POST_URL_FMT.format(id=pid),
                "mirror_dataset": MIRROR_DATASET,
                "mirror_license": MIRROR_LICENSE,
                "mirror_updated_at": args.mirror_updated_at,
            }
            man.write(json.dumps(row) + "\n")
            written += 1
    log.info("Wrote manifest: %s (rows=%d)", manifest_path, written)
    log.info("Done.")

if __name__ == "__main__":
    main()
