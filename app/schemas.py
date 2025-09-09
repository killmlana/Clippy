from __future__ import annotations
from typing import Optional, List, Dict
from pydantic import BaseModel, Field

class FeedbackIn(BaseModel):
    user_id: str = Field(..., min_length=1)
    image_id: str = Field(..., min_length=1)
    action: str = Field(..., pattern="^(click|save)$")
    tags: Optional[List[str]] = None

class ProfileOut(BaseModel):
    user_id: str
    tags: Dict[str, float]
    styles: Dict[str, float]
    counts: Dict[str, int]