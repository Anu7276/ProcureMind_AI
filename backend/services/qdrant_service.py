"""
Qdrant service — vector search for Node03.
Collection layout:
  standards_vectors       — one vector per standard (embedding_source_text)
  standards_fulltext_chunks — chunked full_text for 560 standards with text
"""
from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional, Tuple

from qdrant_client import AsyncQdrantClient, QdrantClient, models

from backend.config.settings import settings

logger = logging.getLogger(__name__)

# ── Client singletons ─────────────────────────────────────────────────────────

_async_client: Optional[AsyncQdrantClient] = None
_sync_client: Optional[QdrantClient] = None
_last_avail_check_time: float = 0.0
_last_avail_result: Tuple[bool, Optional[float]] = (False, None)


def get_sync_client() -> QdrantClient:
    """Used by the ingestion script (sync context)."""
    global _sync_client
    if _sync_client is None:
        _sync_client = QdrantClient(
            host=settings.QDRANT_HOST, port=settings.QDRANT_PORT
        )
    return _sync_client


async def get_async_client() -> AsyncQdrantClient:
    global _async_client
    if _async_client is None:
        _async_client = AsyncQdrantClient(
            host=settings.QDRANT_HOST,
            port=settings.QDRANT_PORT,
            timeout=settings.QDRANT_TIMEOUT_S,
        )
    return _async_client


# ── Availability check ────────────────────────────────────────────────────────

def _is_port_open(host: str, port: int, timeout: float = 0.1) -> bool:
    import socket
    try:
        resolved_host = "127.0.0.1" if host in ("localhost", "0.0.0.0") else host
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(timeout)
            return s.connect_ex((resolved_host, port)) == 0
    except Exception:
        return False


async def is_available() -> Tuple[bool, Optional[float]]:
    """Returns (available, latency_ms) with 10s caching."""
    global _last_avail_check_time, _last_avail_result
    now = time.monotonic()
    if now - _last_avail_check_time < 10.0:
        return _last_avail_result

    # Fast probe to prevent async connection hangs when daemon is offline
    if not _is_port_open(settings.QDRANT_HOST, settings.QDRANT_PORT):
        _last_avail_result = (False, None)
        _last_avail_check_time = now
        return _last_avail_result

    import asyncio
    t0 = time.monotonic()
    try:
        async def _ping():
            client = await get_async_client()
            return await client.get_collections()

        await asyncio.wait_for(_ping(), timeout=settings.QDRANT_TIMEOUT_S)
        latency = (time.monotonic() - t0) * 1000
        _last_avail_result = (True, latency)
        _last_avail_check_time = now
        return _last_avail_result
    except Exception as exc:
        logger.warning("Qdrant unavailable: %s", exc)
        _last_avail_result = (False, None)
        _last_avail_check_time = now
        return _last_avail_result


# ── Vector search — standards_vectors ────────────────────────────────────────

async def search_standards(
    query_vector: List[float],
    top_k: int = 10,
    filter_category: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Semantic search on standards_vectors collection.
    Returns list of hit dicts with score + payload.
    """
    try:
        client = await get_async_client()
        query_filter = None
        if filter_category:
            query_filter = models.Filter(
                must=[
                    models.FieldCondition(
                        key="category",
                        match=models.MatchValue(value=filter_category),
                    )
                ]
            )
        response = await client.query_points(
            collection_name=settings.QDRANT_COLLECTION_STANDARDS,
            query=query_vector,
            limit=top_k,
            query_filter=query_filter,
            with_payload=True,
        )
        hits = response.points
        return [
            {
                "score": h.score,
                "key": h.payload.get("key"),
                "display_code": h.payload.get("display_code"),
                "title": h.payload.get("title"),
                "category": h.payload.get("category"),
                "verification_level": h.payload.get("verification_level"),
                "source": "qdrant_vector",
            }
            for h in hits
        ]
    except Exception as exc:
        logger.error("search_standards failed: %s", exc)
        raise


async def search_fulltext_chunks(
    query_vector: List[float],
    top_k: int = 5,
) -> List[Dict[str, Any]]:
    """
    Clause-level search against chunked full_text.
    Returns chunk hits with parent standard key for citation.
    """
    try:
        client = await get_async_client()
        response = await client.query_points(
            collection_name=settings.QDRANT_COLLECTION_FULLTEXT,
            query=query_vector,
            limit=top_k,
            with_payload=True,
        )
        hits = response.points
        return [
            {
                "score": h.score,
                "parent_key": h.payload.get("parent_key"),
                "display_code": h.payload.get("display_code"),
                "chunk_index": h.payload.get("chunk_index"),
                "chunk_text": h.payload.get("chunk_text"),
                "source": "qdrant_fulltext",
            }
            for h in hits
        ]
    except Exception as exc:
        logger.error("search_fulltext_chunks failed: %s", exc)
        return []


# ── Collection setup (called by ingestion script) ─────────────────────────────

def ensure_collections(dim: int = 384) -> None:
    """Create collections if they don't exist. Called once during ingestion."""
    client = get_sync_client()
    existing = {c.name for c in client.get_collections().collections}

    if settings.QDRANT_COLLECTION_STANDARDS not in existing:
        client.create_collection(
            collection_name=settings.QDRANT_COLLECTION_STANDARDS,
            vectors_config=models.VectorParams(
                size=dim, distance=models.Distance.COSINE
            ),
        )
        logger.info("Created collection: %s", settings.QDRANT_COLLECTION_STANDARDS)

    if settings.QDRANT_COLLECTION_FULLTEXT not in existing:
        client.create_collection(
            collection_name=settings.QDRANT_COLLECTION_FULLTEXT,
            vectors_config=models.VectorParams(
                size=dim, distance=models.Distance.COSINE
            ),
        )
        logger.info("Created collection: %s", settings.QDRANT_COLLECTION_FULLTEXT)


def upsert_standard_vectors(
    records: List[Dict[str, Any]],  # each has 'vector' + payload fields
) -> None:
    """Batch upsert into standards_vectors. Called by ingestion."""
    client = get_sync_client()
    points = [
        models.PointStruct(
            id=idx,
            vector=rec["vector"],
            payload={
                "key": rec["key"],
                "display_code": rec["display_code"],
                "title": rec["title"],
                "category": rec.get("category"),
                "verification_level": rec.get("verification_level"),
            },
        )
        for idx, rec in enumerate(records)
    ]
    # Upload in batches of 200
    batch_size = 200
    for i in range(0, len(points), batch_size):
        client.upsert(
            collection_name=settings.QDRANT_COLLECTION_STANDARDS,
            points=points[i : i + batch_size],
        )


def upsert_fulltext_vectors(
    chunks: List[Dict[str, Any]],
) -> None:
    """Batch upsert into standards_fulltext_chunks. Called by ingestion."""
    client = get_sync_client()
    points = [
        models.PointStruct(
            id=idx,
            vector=chunk["vector"],
            payload={
                "parent_key": chunk["parent_key"],
                "display_code": chunk["display_code"],
                "chunk_index": chunk["chunk_index"],
                "chunk_text": chunk["chunk_text"],
            },
        )
        for idx, chunk in enumerate(chunks)
    ]
    batch_size = 200
    for i in range(0, len(points), batch_size):
        client.upsert(
            collection_name=settings.QDRANT_COLLECTION_FULLTEXT,
            points=points[i : i + batch_size],
        )


async def close():
    global _async_client
    if _async_client:
        await _async_client.close()
        _async_client = None
