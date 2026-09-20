"""
Tests for Phase 5: True Hybrid Retrieval with RRF, Qdrant query_points, and Relationship Ordering.
"""
from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
import pytest

from ai.knowledge import knowledge_loader as kl
from ai.pipeline.nodes.node_03_retrieve import (
    _order_and_cap_relationships,
    _reciprocal_rank_fusion,
    node_03_retrieve,
)
from backend.services import qdrant_service


@pytest.fixture(scope="module", autouse=True)
def init_catalog():
    kl.load_all()


def test_order_and_cap_relationships():
    relations = [
        {"key": "IS 100", "relationship_type": "references"},
        {"key": "IS 200", "relationship_type": "requires_testing"},
        {"key": "IS 300", "relationship_type": "allied"},
        {"key": "IS 400", "relationship_type": "requires_safety"},
        {"key": "IS 500", "relationship_type": "performance"},
        {"key": "IS 600", "relationship_type": "battery_safety"},
        {"key": "IS 200", "relationship_type": "requires_testing"},  # duplicate
    ]
    ordered = _order_and_cap_relationships(relations, cap=5)

    assert len(ordered) == 5
    # Order: requires_testing (IS 200) > requires_safety (IS 400) > allied (IS 300) > performance (IS 500) > references (IS 100)
    expected_order = ["IS 200", "IS 400", "IS 300", "IS 500", "IS 100"]
    assert [r["key"] for r in ordered] == expected_order

    # Check labels
    for r in ordered:
        assert "label" in r
        assert r["label"] != ""
    assert ordered[0]["label"] == "Requires Testing"
    assert ordered[4]["label"] == "References"


def test_reciprocal_rank_fusion_scores_and_trace():
    sources_results = {
        "vector": [
            {"key": "IS 1", "score": 0.95},
            {"key": "IS 2", "score": 0.85},
            {"key": "IS 3", "score": 0.75},
        ],
        "lexical": [
            {"key": "IS 2", "score": 0.90},
            {"key": "IS 1", "score": 0.80},
            {"key": "IS 4", "score": 0.70},
        ],
        "clause": [
            {"key": "IS 1", "score": 0.88, "evidence_clause": "Clause 4.1 testing clause"},
        ],
    }

    fused, active_sources = _reciprocal_rank_fusion(sources_results, k=60)
    assert active_sources == ["vector", "lexical", "clause"]
    assert len(fused) == 4

    # IS 1 ranks #1 in vector, #2 in lexical, #1 in clause:
    # rrf = 1/61 + 1/62 + 1/61 = 0.016393 + 0.016129 + 0.016393 = 0.048916
    # IS 2 ranks #2 in vector, #1 in lexical:
    # rrf = 1/62 + 1/61 = 0.032522
    assert fused[0]["key"] == "IS 1"
    assert fused[1]["key"] == "IS 2"

    # Verify retrieval_trace
    is_1 = fused[0]
    assert is_1["retrieval_trace"] == {"vector": 1, "lexical": 2, "clause": 1}
    assert is_1["evidence_clause"] == "Clause 4.1 testing clause"
    assert is_1["source"] == "clause+lexical+vector"

    # Normalized scores should be in (0, 1]
    for cand in fused:
        assert 0.0 < cand["score"] <= 1.0


@pytest.mark.asyncio
async def test_qdrant_service_uses_query_points_never_search():
    """Verify qdrant_service uses client.query_points and never calls .search."""
    # Fake hit point
    mock_point = MagicMock()
    mock_point.score = 0.92
    mock_point.payload = {
        "key": "IS 1234",
        "display_code": "IS 1234:2000",
        "title": "Test Standard",
        "category": "ELECTRICAL",
        "verification_level": "verified_multi_source",
    }

    mock_response = MagicMock()
    mock_response.points = [mock_point]

    # Fake client that has query_points but NOT search (calling search raises AttributeError)
    fake_client = MagicMock(spec=["query_points", "get_collections"])
    fake_client.query_points = AsyncMock(return_value=mock_response)
    fake_client.get_collections = AsyncMock(return_value=MagicMock(collections=[]))

    with patch("backend.services.qdrant_service.get_async_client", AsyncMock(return_value=fake_client)):
        res = await qdrant_service.search_standards([0.1] * 384, top_k=5)
        assert len(res) == 1
        assert res[0]["key"] == "IS 1234"
        assert res[0]["score"] == 0.92
        assert fake_client.query_points.called
        # Check that .search does not exist on spec
        assert not hasattr(fake_client, "search")

        # Also check search_fulltext_chunks
        mock_chunk_point = MagicMock()
        mock_chunk_point.score = 0.87
        mock_chunk_point.payload = {
            "parent_key": "IS 1234",
            "display_code": "IS 1234:2000",
            "chunk_index": 2,
            "chunk_text": "Sample clause text",
        }
        mock_chunk_response = MagicMock()
        mock_chunk_response.points = [mock_chunk_point]
        fake_client.query_points = AsyncMock(return_value=mock_chunk_response)

        chunk_res = await qdrant_service.search_fulltext_chunks([0.1] * 384, top_k=3)
        assert len(chunk_res) == 1
        assert chunk_res[0]["parent_key"] == "IS 1234"
        assert chunk_res[0]["chunk_text"] == "Sample clause text"


@pytest.mark.asyncio
async def test_node_03_surfaces_retrieval_sources_used_and_traces():
    """Verify node_03 returns retrieval_sources_used in state."""
    state = {
        "raw_input": "Submersible pump 10 HP",
        "normalized_text": "submersible pump 10 hp",
        "structured_requirement": {
            "product": "submersible pump",
            "application": "water pumping",
        },
        "literal_codes": [],
        "thesaurus_expansions": [],
        "pipeline_warnings": [],
        "stages_completed": ["document_understanding", "ingest", "extract"],
    }

    out = await node_03_retrieve(state)
    assert "retrieval_sources_used" in out
    assert isinstance(out["retrieval_sources_used"], list)
    assert len(out["candidates"]) > 0

    top_cand = out["candidates"][0]
    assert "retrieval_trace" in top_cand
    assert "score" in top_cand
    assert 0.0 < top_cand["score"] <= 1.0
    assert "related_standards" in top_cand
