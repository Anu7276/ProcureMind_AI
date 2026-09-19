"""
Neo4j service — knowledge graph traversal for Node03.
Manages a single async driver instance.
"""
from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional, Tuple

from neo4j import AsyncGraphDatabase, AsyncDriver

from backend.config.settings import settings

logger = logging.getLogger(__name__)

_driver: Optional[AsyncDriver] = None

# Relationship types loaded from data — all 19 from standards_relationships.json.
# We traverse ALL of them (not just the 5 named in spec) because the extra types
# (battery_safety, seismic_safety, etc.) carry real domain signal.
ALL_REL_TYPES = [
    "REFERENCES", "ALLIED", "REQUIRES_TESTING", "REQUIRES_SAFETY",
    "SUPERSEDED_BY", "SUPERSEDES", "PERFORMANCE", "EFFICIENCY_VERIFICATION",
    "INSTALLATION_SAFETY", "PERFORMANCE_RATING", "VFD_APPLICATION",
    "BATTERY_SAFETY", "BATTERY_STORAGE", "INVERTER_SYSTEM", "MAINTENANCE_CODE",
    "SEISMIC_SAFETY", "APPLICATION_CODE", "FOOD_CONTACT_SAFETY",
    "SUBMERSIBLE_MOTOR_SAFETY",
]


async def get_driver() -> AsyncDriver:
    global _driver
    if _driver is None:
        _driver = AsyncGraphDatabase.driver(
            settings.NEO4J_URI,
            auth=(settings.NEO4J_USER, settings.NEO4J_PASSWORD),
        )
    return _driver


# ── Availability check ────────────────────────────────────────────────────────

async def is_available() -> Tuple[bool, Optional[float]]:
    t0 = time.monotonic()
    try:
        driver = await get_driver()
        async with driver.session() as s:
            await s.run("RETURN 1")
        return True, (time.monotonic() - t0) * 1000
    except Exception as exc:
        logger.warning("Neo4j unavailable: %s", exc)
        return False, None


# ── Graph traversal — Node03 ──────────────────────────────────────────────────

async def expand_standard(
    key: str,
    max_hops: int = 2,
) -> List[Dict[str, Any]]:
    """
    Starting from `key`, traverse all relationship types up to `max_hops`
    and return all reachable Standard nodes with their edge type.
    """
    try:
        driver = await get_driver()
        async with driver.session() as s:
            result = await s.run(
                """
                MATCH (start:Standard {key: $key})
                CALL apoc.path.subgraphNodes(start, {
                    relationshipFilter: '>' ,
                    maxLevel: $hops
                })
                YIELD node
                WHERE node.key <> $key
                RETURN node.key       AS key,
                       node.display_code AS display_code,
                       node.title     AS title,
                       node.status    AS status,
                       node.category  AS category
                LIMIT 20
                """,
                key=key,
                hops=max_hops,
            )
            return [dict(r) for r in result]
    except Exception as exc:
        # APOC may not be loaded — fall back to 1-hop Cypher
        logger.warning("APOC expand failed (%s) — trying 1-hop fallback", exc)
        return await _expand_1hop(key)


async def _expand_1hop(key: str) -> List[Dict[str, Any]]:
    """Simple 1-hop neighbour traversal without APOC."""
    try:
        driver = await get_driver()
        async with driver.session() as s:
            result = await s.run(
                """
                MATCH (start:Standard {key: $key})-[r]->(neighbour:Standard)
                RETURN neighbour.key       AS key,
                       neighbour.display_code AS display_code,
                       neighbour.title     AS title,
                       neighbour.status    AS status,
                       neighbour.category  AS category,
                       type(r)             AS relationship_type
                LIMIT 15
                """,
                key=key,
            )
            return [dict(r) for r in result]
    except Exception as exc:
        logger.error("Neo4j 1-hop expand failed: %s", exc)
        return []


async def get_related_standards(key: str) -> List[Dict[str, Any]]:
    """
    Directed 1-hop neighbours with explicit relationship type.
    Used by GET /standard/{key} for the detail view.
    """
    try:
        driver = await get_driver()
        async with driver.session() as s:
            result = await s.run(
                """
                MATCH (s:Standard {key: $key})-[r]->(t:Standard)
                RETURN t.key           AS key,
                       t.display_code  AS display_code,
                       t.title         AS title,
                       type(r)         AS relationship_type
                UNION
                MATCH (s:Standard)<-[r]-(t:Standard {key: $key})
                RETURN s.key           AS key,
                       s.display_code  AS display_code,
                       s.title         AS title,
                       type(r)         AS relationship_type
                """,
                key=key,
            )
            return [dict(r) for r in result]
    except Exception as exc:
        logger.error("get_related_standards failed: %s", exc)
        return []


# ── Ingestion helpers (called by ingest_all.py) ───────────────────────────────

async def create_standard_nodes(records: List[Dict[str, Any]]) -> int:
    """MERGE Standard nodes — safe to re-run."""
    driver = await get_driver()
    async with driver.session() as s:
        result = await s.run(
            """
            UNWIND $records AS r
            MERGE (n:Standard {key: r.key})
            SET n.display_code = r.display_code,
                n.title        = r.title,
                n.status       = r.status,
                n.category     = r.category
            """,
            records=records,
        )
        summary = await result.consume()
        return summary.counters.nodes_created


async def create_relationships(edges: List[Dict[str, Any]]) -> int:
    """
    Create edges from standards_relationships.json.
    Relationship type is uppercased from the JSON `type` field.
    All 19 types are loaded — no filtering.
    """
    driver = await get_driver()
    created = 0
    async with driver.session() as s:
        for edge in edges:
            rel_type = edge["type"].upper().replace(" ", "_")
            result = await s.run(
                f"""
                MATCH (a:Standard {{key: $frm}}), (b:Standard {{key: $to}})
                MERGE (a)-[r:{rel_type}]->(b)
                """,
                frm=edge["frm"],
                to=edge["to"],
            )
            summary = await result.consume()
            created += summary.counters.relationships_created
    return created


async def close():
    global _driver
    if _driver:
        await _driver.close()
        _driver = None
