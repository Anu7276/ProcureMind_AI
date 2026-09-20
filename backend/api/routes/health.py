"""GET /health — connectivity check for all three stores."""
from __future__ import annotations

import time
from typing import Dict

from fastapi import APIRouter

from ai.knowledge import knowledge_loader as kl
from backend.config.settings import settings
from backend.schemas.api_schemas import HealthResponse, StoreHealth

router = APIRouter(tags=["System"])


@router.get("/health", response_model=HealthResponse)
async def health_check():
    stores: Dict[str, StoreHealth] = {}

    # PostgreSQL
    try:
        from backend.services import postgres_service
        t0 = time.monotonic()
        pg_ok = await postgres_service.is_available()
        stores["postgres"] = StoreHealth(
            available=pg_ok,
            latency_ms=round((time.monotonic() - t0) * 1000, 1),
        )
    except Exception as exc:
        stores["postgres"] = StoreHealth(available=False, error=str(exc))

    # Qdrant
    try:
        from backend.services import qdrant_service
        qd_ok, qd_latency = await qdrant_service.is_available()
        stores["qdrant"] = StoreHealth(available=qd_ok, latency_ms=qd_latency)
    except Exception as exc:
        stores["qdrant"] = StoreHealth(available=False, error=str(exc))

    # Neo4j
    try:
        from backend.services import neo4j_service
        n4_ok, n4_latency = await neo4j_service.is_available()
        stores["neo4j"] = StoreHealth(available=n4_ok, latency_ms=n4_latency)
    except Exception as exc:
        stores["neo4j"] = StoreHealth(available=False, error=str(exc))

    # Determine overall status
    available_count = sum(1 for s in stores.values() if s.available)
    if available_count == 3:
        status = "ok"
    elif available_count == 0:
        status = "critical"
    else:
        status = "degraded"

    return HealthResponse(
        status=status,
        stores=stores,
        llm_provider=f"{settings.LLM_PROVIDER}/{settings.llm_model_name}",
        embedding_model=f"{settings.EMBEDDING_MODEL}/{settings.EMBEDDING_MODEL_NAME}",
        standards_loaded=len(kl.STANDARDS_BY_KEY),
    )


@router.get("/healthz")
async def healthz():
    """Liveness probe: returns 200 OK immediately if the service process is up."""
    return {"status": "alive"}


@router.get("/readyz")
async def readyz():
    """Readiness probe: returns 200 OK if in-memory knowledge is loaded, 503 otherwise."""
    from fastapi.responses import JSONResponse
    standards_count = len(kl.STANDARDS_BY_KEY)
    if standards_count > 0:
        return {"status": "ready", "standards_loaded": standards_count}
    return JSONResponse(
        status_code=503,
        content={"status": "not_ready", "error": "Standards catalog is not loaded into memory."},
    )
