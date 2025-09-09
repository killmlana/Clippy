from __future__ import annotations
from pathlib import Path

def safe_under_root(p: Path, root: Path) -> bool:
    p = p.resolve()
    try:
        return p.is_file() and p.is_relative_to(root)
    except AttributeError:
        try:
            p.relative_to(root)
            return p.is_file()
        except Exception:
            return False