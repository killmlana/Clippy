from __future__ import annotations
import logging
from settings import settings

def configure_logging() -> None:
    logging.basicConfig(
        level=settings.LOG_LEVEL,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    logging.getLogger("qdrant_client").setLevel(logging.WARNING)