"""
PostgreSQL service — async SQLAlchemy engine.
Exposes helper methods used by the pipeline nodes and API routes.
"""
from __future__ import annotations

import logging
import uuid
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from backend.config.settings import settings

logger = logging.getLogger(__name__)

# ── Engine (module-level singleton) ──────────────────────────────────────────

_engine = create_async_engine(
    settings.POSTGRES_URL,
    pool_pre_ping=True,
    pool_size=10,
    max_overflow=20,
    echo=False,
    connect_args={"timeout": 1.0},
)
_session_factory = async_sessionmaker(_engine, expire_on_commit=False)


@asynccontextmanager
async def get_session():
    async with _session_factory() as session:
        yield session


# ── Availability check ───────────────────────────────────────────────────────

async def is_available() -> bool:
    """Ping check — used for health endpoint and degraded-mode detection."""
    import asyncio
    try:
        async def _ping():
            async with get_session() as s:
                await s.execute(text("SELECT 1"))
            return True

        return await asyncio.wait_for(_ping(), timeout=1.0)
    except Exception as exc:
        logger.warning("PostgreSQL unavailable: %s", exc)
        return False


# ── Standards ─────────────────────────────────────────────────────────────────

async def get_standard_by_key(key: str) -> Optional[Dict[str, Any]]:
    try:
        async with get_session() as s:
            result = await s.execute(
                text("SELECT * FROM standards WHERE key = :key"), {"key": key}
            )
            row = result.mappings().first()
            return dict(row) if row else None
    except Exception as exc:
        logger.error("get_standard_by_key failed: %s", exc)
        return None


async def keyword_search_standards(query: str, limit: int = 10) -> List[Dict[str, Any]]:
    """
    Fallback for when Qdrant is unavailable (Node03 degraded path).
    Uses PostgreSQL full-text search on title + keyword match on display_code.
    """
    try:
        async with get_session() as s:
            result = await s.execute(
                text("""
                    SELECT key, display_code, title, status, category,
                           verification_level, flags, superseded_by
                    FROM standards
                    WHERE
                        to_tsvector('english', title) @@ plainto_tsquery('english', :q)
                        OR display_code ILIKE :code_pat
                    ORDER BY
                        CASE WHEN status = 'ACTIVE' THEN 0 ELSE 1 END,
                        CASE verification_level
                            WHEN 'verified_multi_source' THEN 0
                            WHEN 'needs_review' THEN 1
                            ELSE 2
                        END
                    LIMIT :lim
                """),
                {"q": query, "code_pat": f"%{query}%", "lim": limit},
            )
            return [dict(r) for r in result.mappings()]
    except Exception as exc:
        logger.error("keyword_search_standards failed: %s", exc)
        return []


# ── QCO / Compliance ─────────────────────────────────────────────────────────

async def get_qco_for_standard(standard_key: str) -> Optional[Dict[str, Any]]:
    try:
        async with get_session() as s:
            result = await s.execute(
                text("SELECT * FROM qco_orders WHERE standard_key = :key"),
                {"key": standard_key},
            )
            row = result.mappings().first()
            return dict(row) if row else None
    except Exception as exc:
        logger.error("get_qco_for_standard failed: %s", exc)
        return None


async def get_product_rules_for_standard(standard_key: str) -> List[Dict[str, Any]]:
    try:
        async with get_session() as s:
            result = await s.execute(
                text("SELECT * FROM product_rules WHERE standard_key = :key"),
                {"key": standard_key},
            )
            return [dict(r) for r in result.mappings()]
    except Exception as exc:
        logger.error("get_product_rules_for_standard failed: %s", exc)
        return []


async def get_certification_scheme(scheme_code: str) -> Optional[Dict[str, Any]]:
    try:
        async with get_session() as s:
            result = await s.execute(
                text("SELECT * FROM certification_schemes WHERE scheme_code = :code"),
                {"code": scheme_code},
            )
            row = result.mappings().first()
            return dict(row) if row else None
    except Exception as exc:
        logger.error("get_certification_scheme failed: %s", exc)
        return None


# ── Recommendation log ────────────────────────────────────────────────────────

async def write_recommendation_log(
    audit_id: str,
    query_text: str,
    input_type: str,
    structured_requirement: Optional[Dict],
    returned_codes: List[str],
    pipeline_warnings: List[str],
    top_confidence: Optional[float],
) -> None:
    try:
        async with get_session() as s:
            await s.execute(
                text("""
                    INSERT INTO recommendation_log
                        (audit_id, query_text, input_type, structured_requirement,
                         returned_codes, pipeline_warnings, top_confidence)
                    VALUES
                        (:audit_id, :query_text, :input_type, :structured_requirement::jsonb,
                         :returned_codes::jsonb, :pipeline_warnings::jsonb, :top_confidence)
                    ON CONFLICT (audit_id) DO NOTHING
                """),
                {
                    "audit_id": audit_id,
                    "query_text": query_text,
                    "input_type": input_type,
                    "structured_requirement": __import__("json").dumps(
                        structured_requirement or {}
                    ),
                    "returned_codes": __import__("json").dumps(returned_codes),
                    "pipeline_warnings": __import__("json").dumps(pipeline_warnings),
                    "top_confidence": top_confidence,
                },
            )
            await s.commit()
    except Exception as exc:
        logger.debug("write_recommendation_log failed (non-fatal): %s", exc)


async def close():
    await _engine.dispose()
