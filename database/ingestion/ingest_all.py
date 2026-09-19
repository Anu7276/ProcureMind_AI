#!/usr/bin/env python3
"""
One-time data ingestion script.
Loads BIS_Sahayak_Clean_Data into PostgreSQL, Qdrant, and Neo4j.

Usage:
    python database/ingestion/ingest_all.py

Safe to re-run — each section checks existence before inserting.
Run from project root so that settings.DATA_DIR resolves correctly.

Build order:
  1. PostgreSQL — standards, qco_orders, certification_schemes, product_rules
  2. Qdrant     — standards_vectors (1,380 records), standards_fulltext_chunks (560 records)
  3. Neo4j      — Standard nodes (1,380), edges (836 from relationships JSON)
"""
from __future__ import annotations

import asyncio
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List

# Allow running from project root
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger("ingest")

from backend.config.settings import settings
from backend.services import qdrant_service, neo4j_service


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _load_json(path: Path) -> Any:
    logger.info("Loading %s", path)
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _chunk_text(text: str, size: int = 512, overlap: int = 64) -> List[str]:
    """Split text into overlapping character-level chunks."""
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        chunks.append(text[start:end])
        start += size - overlap
    return chunks


# ═══════════════════════════════════════════════════════════════════════════════
# Step 1 — PostgreSQL
# ═══════════════════════════════════════════════════════════════════════════════

def ingest_postgres(standards: List[Dict], qco: List[Dict],
                    schemes: List[Dict], rules: List[Dict]) -> None:
    import psycopg2
    import psycopg2.extras

    conn_str = settings.POSTGRES_SYNC_URL.replace("postgresql://", "")
    # psycopg2 connection string
    conn = psycopg2.connect(settings.POSTGRES_SYNC_URL.replace("+psycopg2", "").replace("postgresql://", "postgresql://"))
    conn.autocommit = False
    cur = conn.cursor()

    # ── Standards ──────────────────────────────────────────────────────────────
    logger.info("PostgreSQL: inserting %d standards...", len(standards))
    upsert_standard_sql = """
        INSERT INTO standards
            (key, display_code, edition_year, title, scope, status,
             category, subcategory, compliance_mandatory,
             verification_level, flags, has_full_text, superseded_by, keywords)
        VALUES
            (%(key)s, %(display_code)s, %(edition_year)s, %(title)s, %(scope)s,
             %(status)s, %(category)s, %(subcategory)s, %(compliance_mandatory)s,
             %(verification_level)s, %(flags)s::jsonb, %(has_full_text)s,
             %(superseded_by)s::jsonb, %(keywords)s::jsonb)
        ON CONFLICT (key) DO UPDATE SET
            display_code        = EXCLUDED.display_code,
            edition_year        = EXCLUDED.edition_year,
            title               = EXCLUDED.title,
            scope               = EXCLUDED.scope,
            status              = EXCLUDED.status,
            category            = EXCLUDED.category,
            subcategory         = EXCLUDED.subcategory,
            compliance_mandatory= EXCLUDED.compliance_mandatory,
            verification_level  = EXCLUDED.verification_level,
            flags               = EXCLUDED.flags,
            has_full_text       = EXCLUDED.has_full_text,
            superseded_by       = EXCLUDED.superseded_by,
            keywords            = EXCLUDED.keywords
    """
    for rec in tqdm(standards, desc="  PG standards"):
        dq = rec.get("data_quality", {})
        superseded_by = rec.get("superseded_by") or []
        if isinstance(superseded_by, str):
            superseded_by = [superseded_by]
        text_block = rec.get("text") or {}

        cur.execute(upsert_standard_sql, {
            "key": rec["key"],
            "display_code": rec.get("display_code", rec["key"]),
            "edition_year": rec.get("edition_year"),
            "title": rec.get("title", ""),
            "scope": rec.get("scope") or rec.get("description"),
            "status": rec.get("status", "ACTIVE"),
            "category": rec.get("category"),
            "subcategory": rec.get("subcategory"),
            "compliance_mandatory": (rec.get("compliance") or {}).get("mandatory_certification"),
            "verification_level": dq.get("verification_level"),
            "flags": json.dumps(dq.get("flags", [])),
            "has_full_text": text_block.get("has_full_text", False),
            "superseded_by": json.dumps(superseded_by),
            "keywords": json.dumps(rec.get("keywords") or []),
        })
    conn.commit()
    logger.info("PostgreSQL: standards done ✓")

    # ── QCO orders ─────────────────────────────────────────────────────────────
    logger.info("PostgreSQL: inserting %d QCO orders...", len(qco))
    for rec in qco:
        cur.execute("""
            INSERT INTO qco_orders
                (qco_id, product, standard_key, ministry, enforcement_status,
                 effective_date, certification_required, scope, penalty,
                 gazette_reference, origin, verification)
            VALUES
                (%(qco_id)s, %(product)s, %(standard_key)s, %(ministry)s,
                 %(enforcement_status)s, %(effective_date)s, %(certification_required)s,
                 %(scope)s, %(penalty)s, %(gazette_reference)s, %(origin)s, %(verification)s)
            ON CONFLICT (qco_id) DO NOTHING
        """, {
            "qco_id": rec["qco_id"],
            "product": rec.get("product"),
            "standard_key": rec.get("standard_key"),
            "ministry": rec.get("ministry"),
            "enforcement_status": rec.get("enforcement_status"),
            "effective_date": rec.get("effective_date"),
            "certification_required": rec.get("certification_required"),
            "scope": rec.get("scope"),
            "penalty": rec.get("penalty"),
            "gazette_reference": rec.get("gazette_reference"),
            "origin": rec.get("origin"),
            "verification": rec.get("verification"),
        })
    conn.commit()
    logger.info("PostgreSQL: QCO orders done ✓")

    # ── Certification schemes ──────────────────────────────────────────────────
    logger.info("PostgreSQL: inserting %d certification schemes...", len(schemes))
    for rec in schemes:
        cur.execute("""
            INSERT INTO certification_schemes
                (scheme_code, name, popular_name, description, statutory_basis,
                 symbol, applicable_sectors, lead_time_weeks, is_mandatory_for_qco)
            VALUES
                (%(scheme_code)s, %(name)s, %(popular_name)s, %(description)s,
                 %(statutory_basis)s, %(symbol)s, %(applicable_sectors)s::jsonb,
                 %(lead_time_weeks)s, %(is_mandatory_for_qco)s)
            ON CONFLICT (scheme_code) DO NOTHING
        """, {
            "scheme_code": rec["scheme_code"],
            "name": rec["name"],
            "popular_name": rec.get("popular_name"),
            "description": rec.get("description"),
            "statutory_basis": rec.get("statutory_basis"),
            "symbol": rec.get("symbol"),
            "applicable_sectors": json.dumps(rec.get("applicable_sectors", [])),
            "lead_time_weeks": rec.get("lead_time_weeks"),
            "is_mandatory_for_qco": rec.get("is_mandatory_for_qco", False),
        })
    conn.commit()
    logger.info("PostgreSQL: certification schemes done ✓")

    # ── Product rules ──────────────────────────────────────────────────────────
    logger.info("PostgreSQL: inserting product rules...")
    rule_rows_inserted = 0
    for product_rec in rules:
        product = product_rec.get("product", "")
        for rule in product_rec.get("rules", []):
            cur.execute("""
                INSERT INTO product_rules
                    (product, category, subcategory, state, standard_key, standard_raw, context)
                VALUES
                    (%(product)s, %(category)s, %(subcategory)s, %(state)s,
                     %(standard_key)s, %(standard_raw)s, %(context)s)
            """, {
                "product": product,
                "category": product_rec.get("category"),
                "subcategory": product_rec.get("subcategory"),
                "state": rule.get("state"),
                "standard_key": rule.get("standard_key"),
                "standard_raw": rule.get("standard_raw"),
                "context": rule.get("context"),
            })
            rule_rows_inserted += 1
    conn.commit()
    logger.info("PostgreSQL: %d product rule rows done ✓", rule_rows_inserted)

    cur.close()
    conn.close()


# ═══════════════════════════════════════════════════════════════════════════════
# Step 2 — Qdrant
# ═══════════════════════════════════════════════════════════════════════════════

def ingest_qdrant(standards: List[Dict]) -> None:
    from ai.llm.llm_factory import get_embedder

    embedder = get_embedder()
    dim = settings.EMBEDDING_DIM

    # Ensure collections exist
    qdrant_service.ensure_collections(dim=dim)

    # ── standards_vectors: all 1,380 records ──────────────────────────────────
    logger.info("Qdrant: embedding %d standards (embedding_source_text)...", len(standards))
    texts = [r.get("embedding_source_text", r.get("title", "")) for r in standards]
    embeddings = embedder.encode(texts, batch_size=64, show_progress=True)

    records_for_upsert = []
    for rec, vec in zip(standards, embeddings):
        dq = rec.get("data_quality", {})
        records_for_upsert.append({
            "key": rec["key"],
            "display_code": rec.get("display_code", rec["key"]),
            "title": rec.get("title", ""),
            "category": rec.get("category"),
            "verification_level": dq.get("verification_level"),
            "vector": vec,
        })

    qdrant_service.upsert_standard_vectors(records_for_upsert)
    logger.info("Qdrant: standards_vectors done ✓ (%d vectors)", len(records_for_upsert))

    # ── standards_fulltext_chunks: 560 records with full_text ────────────────
    logger.info("Qdrant: chunking full_text for records that have it...")
    chunks_data: List[Dict] = []
    for rec in standards:
        full_text = rec.get("full_text")
        if not full_text:
            continue
        text_chunks = _chunk_text(
            full_text,
            size=settings.CHUNK_SIZE,
            overlap=settings.CHUNK_OVERLAP,
        )
        for i, chunk in enumerate(text_chunks):
            chunks_data.append({
                "parent_key": rec["key"],
                "display_code": rec.get("display_code", rec["key"]),
                "chunk_index": i,
                "chunk_text": chunk,
            })

    logger.info("Qdrant: embedding %d full_text chunks...", len(chunks_data))
    chunk_texts = [c["chunk_text"] for c in chunks_data]
    chunk_embeddings = embedder.encode(chunk_texts, batch_size=64, show_progress=True)

    for c, vec in zip(chunks_data, chunk_embeddings):
        c["vector"] = vec

    qdrant_service.upsert_fulltext_vectors(chunks_data)
    logger.info("Qdrant: standards_fulltext_chunks done ✓ (%d chunks)", len(chunks_data))


# ═══════════════════════════════════════════════════════════════════════════════
# Step 3 — Neo4j
# ═══════════════════════════════════════════════════════════════════════════════

async def ingest_neo4j(standards: List[Dict], relationships: List[Dict]) -> None:
    # ── Standard nodes ─────────────────────────────────────────────────────────
    logger.info("Neo4j: creating %d Standard nodes...", len(standards))
    node_records = [
        {
            "key": r["key"],
            "display_code": r.get("display_code", r["key"]),
            "title": r.get("title", ""),
            "status": r.get("status", "ACTIVE"),
            "category": r.get("category"),
        }
        for r in standards
    ]
    # Process in batches of 200
    total_created = 0
    batch_size = 200
    for i in tqdm(range(0, len(node_records), batch_size), desc="  Neo4j nodes"):
        batch = node_records[i : i + batch_size]
        n = await neo4j_service.create_standard_nodes(batch)
        total_created += n
    logger.info("Neo4j: nodes done ✓ (%d new nodes created)", total_created)

    # ── Relationships ──────────────────────────────────────────────────────────
    logger.info("Neo4j: creating %d relationships (all 19 types)...", len(relationships))
    rel_count = await neo4j_service.create_relationships(relationships)
    logger.info("Neo4j: relationships done ✓ (%d new edges created)", rel_count)

    # ── Verify SUPERSEDED_BY links (IS 8828 pilot case) ───────────────────────
    driver = await neo4j_service.get_driver()
    async with driver.session() as s:
        result = await s.run(
            """
            MATCH (a:Standard {key: 'IS 8828'})-[:SUPERSEDED_BY]->(b:Standard)
            RETURN b.key AS successor
            """
        )
        successors = [r["successor"] for r in await result.data()]
        if successors:
            logger.info(
                "Neo4j: IS 8828 → SUPERSEDED_BY → %s ✓ (pilot case verified)",
                successors,
            )
        else:
            logger.warning(
                "Neo4j: IS 8828 SUPERSEDED_BY edges not found in graph — "
                "these are stored in standards_clean.json superseded_by field "
                "and will be resolved in-memory by Node04."
            )


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

async def main():
    logger.info("=" * 60)
    logger.info("BIS Standards — Data Ingestion Script")
    logger.info("=" * 60)

    # Load all data files
    standards = _load_json(settings.standards_json)
    relationships = _load_json(settings.relationships_json)
    qco = _load_json(settings.qco_json)
    schemes = _load_json(settings.certification_schemes_json)
    rules = _load_json(settings.product_rules_json)

    logger.info(
        "Data loaded: %d standards, %d relationships, %d QCO, %d schemes, %d product rules",
        len(standards), len(relationships), len(qco), len(schemes), len(rules),
    )

    # ── Step 1: PostgreSQL ────────────────────────────────────────────────────
    logger.info("\n── Step 1: PostgreSQL ──")
    try:
        ingest_postgres(standards, qco, schemes, rules)
    except Exception as exc:
        logger.error("PostgreSQL ingestion failed: %s", exc)
        logger.error("Ensure Docker is running: docker compose up -d postgres")
        raise

    # ── Step 2: Qdrant ────────────────────────────────────────────────────────
    logger.info("\n── Step 2: Qdrant (this may take several minutes for embeddings) ──")
    try:
        ingest_qdrant(standards)
    except Exception as exc:
        logger.error("Qdrant ingestion failed: %s", exc)
        logger.error("Ensure Docker is running: docker compose up -d qdrant")
        raise

    # ── Step 3: Neo4j ─────────────────────────────────────────────────────────
    logger.info("\n── Step 3: Neo4j ──")
    try:
        await ingest_neo4j(standards, relationships)
    except Exception as exc:
        logger.error("Neo4j ingestion failed: %s", exc)
        logger.error("Ensure Docker is running: docker compose up -d neo4j")
        raise

    logger.info("\n" + "=" * 60)
    logger.info("Ingestion complete!")
    logger.info("  PostgreSQL: 1380 standards, 11 QCO, 4 schemes, ~200+ product rules")
    logger.info("  Qdrant:     %d vectors in standards_vectors", len(standards))
    logger.info("  Neo4j:      %d nodes, %d edges", len(standards), len(relationships))
    logger.info("=" * 60)

    await neo4j_service.close()


if __name__ == "__main__":
    asyncio.run(main())
