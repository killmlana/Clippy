from __future__ import annotations
import json, time
from pathlib import Path
from typing import Dict, Any, List, Tuple
from ..core.settings import settings

def _load() -> Dict[str, Any]:
    p = settings.KG_STORE
    if not p.exists():
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps({"users": {}}, ensure_ascii=False, indent=2))
    return json.loads(p.read_text())

def _save(data: Dict[str, Any]) -> None:
    settings.KG_STORE.parent.mkdir(parents=True, exist_ok=True)
    settings.KG_STORE.write_text(json.dumps(data, ensure_ascii=False, indent=2))

def load_user_prefs(user_id: str) -> dict:
    data = _load()
    return data.get("users", {}).get(user_id, {"tags": {}, "styles": {}, "edges": {"clicked": [], "saved": []}})

def _trim_top(d: Dict[str, float], n: int) -> Dict[str, float]:
    if len(d) <= n: return d
    return dict(sorted(d.items(), key=lambda kv: kv[1], reverse=True)[:n])

def _ema(cur: float, delta: float, beta: float) -> float:
    v = beta * cur + delta
    return max(0.0, min(1.0, float(v)))

def update_feedback(user_id: str, tags: List[str], style: str|None, action: str) -> Tuple[int, bool]:
    data = _load()
    u = data.setdefault("users", {}).setdefault(user_id, {"tags": {}, "styles": {}, "edges": {"clicked": [], "saved": []}})
    delta = settings.DELTA_SAVE if action == "save" else settings.DELTA_CLICK

    for t in tags:
        key = t.lower()
        u["tags"][key] = _ema(float(u["tags"].get(key, 0.0)), delta, settings.EMA_BETA)
    if style:
        u["styles"][style] = _ema(float(u["styles"].get(style, 0.0)), delta, settings.EMA_BETA)

    u["tags"] = _trim_top(u["tags"], settings.PREF_TOP_N)
    u["styles"] = _trim_top(u["styles"], settings.PREF_TOP_N)

    ts = int(time.time())
    u["edges"].setdefault("clicked", []); u["edges"].setdefault("saved", [])
    (u["edges"]["saved" if action == "save" else "clicked"]).append({"image_id": None, "ts": ts})  # image_id set by route

    _save(data)
    return len(tags), bool(style)

def profile(user_id: str, top: int) -> dict:
    u = load_user_prefs(user_id)
    def _top(d: Dict[str, float], n: int) -> Dict[str, float]:
        return {k: round(float(v), 6) for k, v in sorted(d.items(), key=lambda kv: kv[1], reverse=True)[:n]}
    return {
        "user_id": user_id,
        "tags": _top(u.get("tags", {}), top),
        "styles": _top(u.get("styles", {}), top),
        "counts": {"clicked": len(u.get("edges", {}).get("clicked", [])), "saved": len(u.get("edges", {}).get("saved", []))},
    }