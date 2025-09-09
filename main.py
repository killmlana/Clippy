from __future__ import annotations
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.core.settings import settings
from app.core.logging import configure_logging

APP_NAME = "Clippy"
APP_VERSION = "0.2.0"

def create_app() -> FastAPI:
    configure_logging()
    app = FastAPI(title=APP_NAME, version=APP_VERSION)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.ALLOWED_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
        max_age=86400,
    )

    from .routes_search import router as search_router
    from .routes_feedback import router as feedback_router
    from .routes_images import router as images_router
    app.include_router(search_router, prefix="/search", tags=["search"])
    app.include_router(feedback_router, tags=["feedback"])
    app.include_router(images_router, tags=["images"])

    @app.get("/healthz")
    def healthz(): return {"ok": True, "name": APP_NAME, "version": APP_VERSION}

    return app

app = create_app()
