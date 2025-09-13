from __future__ import annotations
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.core.settings import settings
from app.core.logging import configure_logging
from dotenv import load_dotenv
load_dotenv()

APP_NAME = "Clippy"
APP_VERSION = "0.2.0"

def create_app() -> FastAPI:
    configure_logging()
    application = FastAPI(title=APP_NAME, version=APP_VERSION)

    application.add_middleware(
        CORSMiddleware,
        allow_origins=settings.ALLOWED_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
        max_age=86400,
    )

    from app.routes.search import router as search_router
    from app.routes.feedback import router as feedback_router
    from app.routes.images import router as images_router
    from app.routes.generate_image import router as generate_image_router
    application.include_router(search_router, prefix="/api/search", tags=["search"])
    application.include_router(feedback_router, prefix="/api", tags=["feedback"])
    application.include_router(images_router, prefix="/api", tags=["images"])
    application.include_router(generate_image_router, prefix="/api", tags=["images"])

    @application.get("/healthz")
    def healthz(): return {"ok": True, "name": APP_NAME, "version": APP_VERSION}

    return application

app = create_app()
