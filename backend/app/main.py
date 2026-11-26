"""FastAPI application entrypoint."""

from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.ratelimit import TokenBucketLimiter
from app.api.routes import router
from app.config import Settings, get_settings
from app.container import build_container
from app.voice.routes import router as voice_router

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("app")


async def _ingest_with_retry(app: FastAPI) -> None:
    """Index the KB; if it fails (e.g. embedding quota), keep retrying in the background
    while the app serves traffic in degraded mode (search tool reports KB unavailable)."""
    kb = app.state.container.kb
    delay = 10
    while True:
        status = await kb.ingest()
        if status.ready:
            log.info("KB ready: %d chunks from %d docs (reused=%s)", status.chunks, status.documents,
                     status.reused_index)
            return
        log.warning("KB ingest failed (%s); retrying in %ss", status.error, delay)
        await asyncio.sleep(delay)
        delay = min(delay * 2, 300)


def create_app(settings: Settings | None = None, container=None) -> FastAPI:
    settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.container = container or build_container(settings)
        app.state.chat_limiter = TokenBucketLimiter(settings.rate_limit_per_minute, settings.rate_limit_burst)
        app.state.voice_limiter = TokenBucketLimiter(settings.voice_sessions_per_minute, 2)
        app.state.active_voice = 0
        app.state.background_tasks = set()
        first = await app.state.container.kb.ingest()
        if not first.ready:
            t = asyncio.create_task(_ingest_with_retry(app))
            app.state.background_tasks.add(t)
        log.info("Provider=%s chat=%s live=%s kb_ready=%s", app.state.container.provider.name,
                 settings.chat_model, settings.live_model, first.ready)
        yield
        for t in list(app.state.background_tasks):
            t.cancel()

    app = FastAPI(title="AcmeCloud Support Agent", version="1.0.0", lifespan=lifespan)
    app.add_middleware(CORSMiddleware, allow_origins=settings.cors_origin_list, allow_credentials=False,
                       allow_methods=["GET", "POST"], allow_headers=["*"])
    app.include_router(router)
    app.include_router(voice_router)

    @app.exception_handler(Exception)
    async def unhandled(request: Request, exc: Exception):  # never leak stack traces
        log.exception("unhandled error on %s", request.url.path)
        return JSONResponse({"error": "internal_error", "message": "Something went wrong."}, status_code=500)

    @app.get("/")
    async def root():
        return {"service": "acme-support-agent", "docs": "/docs", "health": "/api/health"}

    return app


app = create_app()
