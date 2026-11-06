"""FastAPI application factory.

    uvicorn app.main:app --reload            (from the backend/ folder)
"""
from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .config import Settings
from .db import Base, make_engine, make_sessionmaker
from .routers import auth, projects

API_VERSION = "0.1.0"


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings()
    engine = make_engine(settings.database_url)
    Base.metadata.create_all(engine)

    app = FastAPI(title="3D-Room backend", version=API_VERSION,
                  description="Video -> 3D Gaussian splat room. Projects, uploads, tasks, worker API, scenes.")
    app.state.settings = settings
    app.state.engine = engine
    app.state.SessionLocal = make_sessionmaker(engine)

    app.add_middleware(CORSMiddleware, allow_origins=settings.cors_origins, allow_credentials=True,
                       allow_methods=["*"], allow_headers=["*"], expose_headers=["Content-Range", "Accept-Ranges"])
    app.include_router(auth.router)
    app.include_router(projects.router)

    @app.get("/health", tags=["meta"])
    def health():
        return {"status": "ok", "version": API_VERSION}

    return app


app = create_app()
