from __future__ import annotations
from typing import Dict, Tuple, Any, List

def combine_scores(
    scores_by_vec: Dict[str, Dict[str, Tuple[float, dict]]],
    w_img: float, w_edge: float, w_txt: float,
    user_tag_weights: Dict[str, float],
    alpha: float
) -> List[tuple[str, float, dict]]:
    ids = set().union(*[set(d.keys()) for d in scores_by_vec.values()]) if scores_by_vec else set()
    fused: List[tuple[str, float, dict]] = []
    for pid in ids:
        s_img, payload = scores_by_vec.get("image", {}).get(pid, (0.0, {}))
        s_edge, payload = scores_by_vec.get("edge",  {}).get(pid, (0.0, payload))
        s_txt, payload  = scores_by_vec.get("text",  {}).get(pid, (0.0, payload))
        base = w_img * s_img + w_edge * s_edge + w_txt * s_txt
        tags = [t.lower() for t in (payload.get("tags_all") or [])]
        prior = 0.0
        if tags and user_tag_weights:
            vals = [user_tag_weights.get(t, 0.0) for t in tags]
            prior = float(sum(vals) / len(vals)) if vals else 0.0
        fused.append((pid, base * (1.0 + alpha * prior), payload))
    fused.sort(key=lambda x: x[1], reverse=True)
    return fused
