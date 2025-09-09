from __future__ import annotations
from typing import List, Optional, Dict, Any

def parse_tag_filters(csv: Optional[str]) -> Optional[List[str]]:
    if not csv: return None
    return [t.strip().lower() for t in csv.split(",") if t.strip()]

def or_filter(payload: Dict[str, Any], tags_req: List[str]) -> bool:
    cand = [t.lower() for t in (payload.get("tags_all") or [])]
    return any(t in cand for t in tags_req)

def and_filter(payload: Dict[str, Any], tags_req: List[str]) -> bool:
    cand = set(t.lower() for t in (payload.get("tags_all") or []))
    return all(t in cand for t in tags_req)
