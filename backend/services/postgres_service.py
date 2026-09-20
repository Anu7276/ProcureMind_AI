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


_last_avail_check_time: float = 0.0
_last_avail_result: bool = False


# ── Availability check ───────────────────────────────────────────────────────

def _is_postgres_port_open(timeout: float = 0.1) -> bool:
    import socket
    from urllib.parse import urlparse
    try:
        url = settings.POSTGRES_URL.replace("postgresql+asyncpg://", "http://").replace("postgresql://", "http://")
        parsed = urlparse(url)
        host = parsed.hostname or "127.0.0.1"
        if host in ("localhost", "0.0.0.0"):
            host = "127.0.0.1"
        port = parsed.port or 5432
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(timeout)
            return s.connect_ex((host, port)) == 0
    except Exception:
        return False


async def is_available() -> bool:
    """Ping check — used for health endpoint and degraded-mode detection with 10s caching."""
    global _last_avail_check_time, _last_avail_result
    import time
    now = time.monotonic()
    if now - _last_avail_check_time < 10.0:
        return _last_avail_result

    # Fast probe to prevent connection timeout hangs when PostgreSQL is offline
    if not _is_postgres_port_open():
        _last_avail_result = False
        _last_avail_check_time = now
        return False

    import asyncio
    try:
        async def _ping():
            async with get_session() as s:
                await s.execute(text("SELECT 1"))
            return True

        res = await asyncio.wait_for(_ping(), timeout=1.0)
        _last_avail_result = res
        _last_avail_check_time = now
        return res
    except Exception as exc:
        logger.warning("PostgreSQL unavailable: %s", exc)
        _last_avail_result = False
        _last_avail_check_time = now
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


# ── Migrations ────────────────────────────────────────────────────────────────

async def run_pending_migrations() -> None:
    """Run pending SQL migrations from database/migrations/."""
    try:
        if not await is_available():
            logger.debug("PostgreSQL not available — skipping migrations")
            return
        from pathlib import Path
        migrations_dir = Path("database/migrations")
        if not migrations_dir.exists():
            return

        async with get_session() as s:
            await s.execute(text("""
                CREATE TABLE IF NOT EXISTS schema_migrations (
                    version VARCHAR(100) PRIMARY KEY,
                    applied_at TIMESTAMPTZ DEFAULT NOW()
                )
            """))
            await s.commit()

            res = await s.execute(text("SELECT version FROM schema_migrations"))
            applied = {r[0] for r in res.fetchall()}

            for sql_file in sorted(migrations_dir.glob("*.sql")):
                if sql_file.name not in applied:
                    logger.info("Applying database migration %s", sql_file.name)
                    sql_content = sql_file.read_text(encoding="utf-8")
                    # Split statements carefully
                    for statement in sql_content.split(";"):
                        stmt = statement.strip()
                        if stmt:
                            await s.execute(text(stmt))
                    await s.execute(
                        text("INSERT INTO schema_migrations (version) VALUES (:v)"),
                        {"v": sql_file.name},
                    )
                    await s.commit()
            logger.info("Database migrations check complete")
    except Exception as exc:
        logger.warning("run_pending_migrations failed: %s", exc)


# ── Recommendation log ────────────────────────────────────────────────────────

async def write_recommendation_log(
    audit_id: str,
    query_text: str,
    input_type: str,
    structured_requirement: Optional[Dict],
    returned_codes: List[str],
    pipeline_warnings: List[str],
    top_confidence: Optional[float],
    abstained: bool = False,
    abstain_reason: Optional[str] = None,
    top_match_strength: Optional[float] = None,
    closest_matches_codes: Optional[List[str]] = None,
) -> bool:
    """Write or update a structured audit row in recommendation_log.

    Returns True if successfully written/updated, False otherwise.
    """
    import json
    payload = {
        "audit_id": audit_id,
        "query_text": query_text[:4096],          # guard against oversized text
        "input_type": input_type,
        "structured_requirement": json.dumps(structured_requirement or {}),
        "returned_codes": json.dumps(returned_codes),
        "pipeline_warnings": json.dumps(pipeline_warnings),
        "top_confidence": top_confidence,
        "abstained": abstained,
        "abstain_reason": (abstain_reason or "")[:512],
        "top_match_strength": top_match_strength,
        "closest_matches_codes": json.dumps(closest_matches_codes or []),
    }
    try:
        async with get_session() as s:
            await s.execute(
                text("""
                    INSERT INTO recommendation_log
                        (audit_id, query_text, input_type, structured_requirement,
                         returned_codes, pipeline_warnings, top_confidence,
                         abstained, abstain_reason, top_match_strength, closest_matches_codes)
                    VALUES
                        (CAST(:audit_id AS uuid), :query_text, :input_type, CAST(:structured_requirement AS jsonb),
                         CAST(:returned_codes AS jsonb), CAST(:pipeline_warnings AS jsonb), :top_confidence,
                         :abstained, :abstain_reason, :top_match_strength, CAST(:closest_matches_codes AS jsonb))
                    ON CONFLICT (audit_id) DO UPDATE SET
                        returned_codes = EXCLUDED.returned_codes,
                        structured_requirement = EXCLUDED.structured_requirement,
                        pipeline_warnings = EXCLUDED.pipeline_warnings,
                        top_confidence = EXCLUDED.top_confidence,
                        abstained = EXCLUDED.abstained,
                        abstain_reason = EXCLUDED.abstain_reason,
                        top_match_strength = EXCLUDED.top_match_strength,
                        closest_matches_codes = EXCLUDED.closest_matches_codes
                """),
                payload,
            )
            await s.commit()
            return True
    except Exception as exc:
        # Try fallback for minimal schema if needed
        try:
            async with get_session() as s:
                await s.execute(
                    text("""
                        INSERT INTO recommendation_log
                            (audit_id, query_text, input_type, structured_requirement,
                             returned_codes, pipeline_warnings, top_confidence)
                        VALUES
                            (CAST(:audit_id AS uuid), :query_text, :input_type, CAST(:structured_requirement AS jsonb),
                             CAST(:returned_codes AS jsonb), CAST(:pipeline_warnings AS jsonb), :top_confidence)
                        ON CONFLICT (audit_id) DO UPDATE SET
                            returned_codes = EXCLUDED.returned_codes,
                            structured_requirement = EXCLUDED.structured_requirement,
                            pipeline_warnings = EXCLUDED.pipeline_warnings,
                            top_confidence = EXCLUDED.top_confidence
                    """),
                    {k: payload[k] for k in ["audit_id","query_text","input_type","structured_requirement",
                                              "returned_codes","pipeline_warnings","top_confidence"]},
                )
                await s.commit()
                return True
        except Exception as inner_exc:
            logger.warning("write_recommendation_log failed: %s | inner: %s", exc, inner_exc)
            return False


async def close():
    await _engine.dispose()
