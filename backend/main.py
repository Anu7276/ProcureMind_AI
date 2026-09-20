"""
FastAPI application entry point.

Startup sequence (lifespan):
  1. Load all knowledge files into memory (knowledge_loader.load_all)
  2. Warm up LLM + embedder (first call downloads sentence-transformers model)
  3. Verify DB connectivity — log warnings if any store is unreachable

Routes:
  POST /ingest          → Document understanding + Node01/02
  POST /recommend       → Full pipeline or Nodes03-05 from structured req
  GET  /standard/{key}  → Direct lookup + graph neighbours
  GET  /health          → Store connectivity + config check
"""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.config.settings import settings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown lifecycle."""
    logger.info("=== BIS Recommendation Engine starting up ===")

    # 1. Load all static data into memory
    from ai.knowledge import knowledge_loader as kl
    kl.load_all()
    logger.info("Knowledge files loaded: %d standards in memory", len(kl.STANDARDS_BY_KEY))

    # 2. Warm up embedder (skip in mock mode to allow instant offline startup)
    import os
    if os.getenv("LLM_PROVIDER") != "mock":
        try:
            from ai.llm.llm_factory import get_embedder
            embedder = get_embedder()
            _ = embedder.encode_one("warm-up")
            logger.info("Embedder warmed up: %s / %s", settings.EMBEDDING_MODEL, settings.EMBEDDING_MODEL_NAME)
        except Exception as exc:
            logger.warning("Embedder warm-up failed (non-fatal): %s", exc)

    # 3. Log LLM provider
    logger.info("LLM provider: %s / %s", settings.LLM_PROVIDER, settings.llm_model_name)

    # 4. Check DB connectivity (non-blocking — warn only)
    from backend.services import postgres_service, qdrant_service, neo4j_service
    pg_ok = await postgres_service.is_available()
    qd_ok, _ = await qdrant_service.is_available()
    n4_ok, _ = await neo4j_service.is_available()
    logger.info("Store status — PostgreSQL: %s, Qdrant: %s, Neo4j: %s", pg_ok, qd_ok, n4_ok)
    if not any([pg_ok, qd_ok, n4_ok]):
        logger.warning("ALL stores are unreachable — only in-memory knowledge available")

    logger.info("=== Startup complete — API ready ===")
    yield

    # Shutdown
    logger.info("Shutting down connections...")
    await postgres_service.close()
    await qdrant_service.close()
    await neo4j_service.close()
    logger.info("=== Shutdown complete ===")


app = FastAPI(
    title="BIS Standards Recommendation Engine",
    description=(
        "AI-powered Indian Standards (IS code) recommendation engine. "
        "Submit a tender/specification document and receive verified, explainable "
        "IS code recommendations grounded in 1,380 BIS standards."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security: rate limiting + body-size cap + sanitised 500 errors
from backend.middleware.security import SecurityMiddleware  # noqa: E402
app.add_middleware(SecurityMiddleware)

# ── Mount routers (both root and /api prefix for reverse proxy / direct client compatibility) ──
from backend.api.routes import health, ingest, recommend, standard, graph  # noqa: E402

for r in [health.router, ingest.router, recommend.router, standard.router, graph.router]:
    app.include_router(r)
    app.include_router(r, prefix="/api")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "backend.main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=True,
    )
